import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

d_columns = ["sex", "cp", "fbs", "restecg", "exang",
                    "slope", "ca", "thal", "num", "dataset"]
c_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
randomness = 1

df = pd.read_csv(Path(__file__).parent / "real_original.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"

#Configure these above lines; make sure to list all discrete and continuous columns

data = df.copy()

#NaN values are filled with medians; less susceptible to outliers
data[c_columns] = data[c_columns].fillna(data[c_columns].median())
for c in d_columns:
    #Turns missing data into a category
    data[c] = data[c].fillna(-1).astype(int).astype(str)

mu = data[c_columns].mean()
sd = data[c_columns].std() + 1e-6
#standardization of values/finds z-scores
cont_part = ((data[c_columns] - mu) / sd).to_numpy(np.float32)

#unique values in each categorical column
cats = {c: sorted(data[c].unique()) for c in d_columns}
sizes = [len(cats[c]) for c in d_columns]
disc_parts = []
for c in d_columns:
    #maps each category to an index and creates a one-hot encoding for the categorical data
    idx = data[c].map({v: i for i, v in enumerate(cats[c])}).to_numpy()
    disc_parts.append(np.eye(len(cats[c]), dtype=np.float32)[idx])

X = torch.tensor(np.hstack([cont_part] + disc_parts)).to(device)
continuousColumnNumber, in_dim = len(c_columns), X.shape[1]

idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=randomness)
X_train, X_test = X[train_idx], X[test_idx]

#length of noise vector to feed into the generator, which is then transformed into synthetic data
z_dim = 64

class Generator(nn.Module):
    #initialize and run a step
    def __init__(self, h=128):
        #h is the number of neurons in 1 of the hidden layers of the generator
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, in_dim))
    #move one batch forward
    def forward(self, z, hard=False):
        #hard=True means we use a one-hot vector; hard=False means we use a softmax vector
        #push noise in and have a batch of fake patients come out
        noise = self.net(z)
        #keep 5 continuous columns as they are
        parts, i = [noise[:, :continuousColumnNumber]], continuousColumnNumber
        #for each categorical column: create a one-hot vector of the unique values for that column
        for k in sizes:
            parts.append(F.gumbel_softmax(noise[:, i:i+k], tau=0.5, hard=hard))
            i += k
        #glue and concatenate the pieces back together for the fake patient
        return torch.cat(parts, 1)

class Critic(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.LeakyReLU(0.2),
            nn.Linear(h, h), nn.LeakyReLU(0.2),
            nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x)

def grad_penalty(Discriminator, real, fake):
    #a is a random number between 0 and 1 for each row in the batch
    a = torch.rand(real.size(0), 1, device=device)
    #mix is a random point along the line between real and fake data
    mix = (a * real + (1 - a) * fake).requires_grad_(True)
    #compute the gradient of the discriminator's output with respect to the mixed input
    g = torch.autograd.grad(Discriminator(mix).sum(), mix, create_graph=True)[0]
    #compute the gradient penalty as the mean squared difference between the norm of the gradient and 1
    #Pushing towards the Lipschitz Constant
    return ((g.norm(2, dim=1) - 1) ** 2).mean()

@torch.no_grad()
def critic_accuracy(Discriminator, Generator, real_batch):
    Discriminator.eval()
    Generator.eval()
    n = len(real_batch)
    fake = Generator(torch.randn(n, z_dim, device=device), hard=True)
    #discriminates between real and fake data, returning a score for each
    score_real = Discriminator(real_batch).squeeze(1)
    score_fake = Discriminator(fake).squeeze(1)

    # AUC shortcut: Is real > fake
    diff = score_real.unsqueeze(1) - score_fake.unsqueeze(0)
    auc = ((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())

    Discriminator.train(); Generator.train()
    return auc.item()

@torch.no_grad()
def generateFakeData(Generator, n, save_csv=False, filename="synthetic_data.csv"):
    Generator.eval()
    output = Generator(torch.randn(n, z_dim, device=device), hard=True).cpu().numpy()
    Generator.train()
    s = pd.DataFrame(output[:, :continuousColumnNumber] * sd.values + mu.values, columns=c_columns)
    i = continuousColumnNumber
    for c, k in zip(d_columns, sizes):
        s[c] = np.array(cats[c])[output[:, i:i+k].argmax(1)]
        i += k
    s = s[data.columns]                  # same column order as the real data
    if save_csv:
        s.to_csv(Path(__file__).parent / filename, index=False)
        print(f"Saved {len(s)} synthetic rows to {filename}")
    return s

torch.manual_seed(randomness)
Generate, Discriminate = Generator().to(device), Critic().to(device)
optG = torch.optim.Adam(Generate.parameters(), lr=2e-4, betas=(0.5, 0.9))
optD = torch.optim.Adam(Discriminate.parameters(), lr=2e-4, betas=(0.5, 0.9))

EPOCHS, BATCH, N_CRITIC, LAMBDA = 30, 64, 5, 10
CHECK = 10 #Checks the acc and auc every {CHECK} epochs
snapshot_epochs = {1, 10, 50, 100, 200, 300} #takes a graph of the generated data at these epochs
snapshots, d_hist, g_hist = {}, [], []
auc_hist = []

for ep in range(1, EPOCHS + 1):
    perm = torch.randperm(len(X_train), device=device) #shufflerows
    for b in range(0, len(X_train) - BATCH + 1, BATCH): #training data in batches
        real = X_train[perm[b:b+BATCH]]

        # critic: several steps per generator step
        for _ in range(N_CRITIC):
            fake = Generate(torch.randn(BATCH, z_dim, device=device)).detach()
            d_loss = Discriminate(fake).mean() - Discriminate(real).mean() + LAMBDA * grad_penalty(Discriminate, real, fake)
            optD.zero_grad(); d_loss.backward(); optD.step()

        # training the generator to make fakes
        g_loss = -Discriminate(Generate(torch.randn(BATCH, z_dim, device=device))).mean()
        optG.zero_grad(); g_loss.backward(); optG.step()

    d_hist.append(d_loss.item()); g_hist.append(g_loss.item())

    if (CHECK > 0) and (ep % CHECK == 0):
        auc_tr = critic_accuracy(Discriminate, Generate, X_train)
        auc_te = critic_accuracy(Discriminate, Generate, X_test)
        auc_hist.append((ep, auc_tr, auc_te))
        print(f"epoch {ep:4d} | train AUC={auc_tr:.3f} | test AUC={auc_te:.3f}")

    if ep in snapshot_epochs:
        snapshots[ep] = generateFakeData(Generate, 1000)

final = generateFakeData(Generate, 1000, save_csv=True, filename="synthetic_final.csv")

# real vs. synthetic histograms per snapshot
for col in ["age", "chol", "thalach"]:
    eps = sorted(snapshots)
    fig, axes = plt.subplots(1, len(eps) + 1, figsize=(3.5 * (len(eps) + 1), 3), sharex=True)
    axes[0].hist(data[col], bins=30); axes[0].set_title(f"real {col}")
    for ax, ep in zip(axes[1:], eps):
        ax.hist(snapshots[ep][col], bins=30); ax.set_title(f"epoch {ep}")
    plt.tight_layout(); plt.show()

# categorical proportions over time
cmp = pd.concat([data["num"].value_counts(normalize=True).rename("real")] +
                [snapshots[e]["num"].value_counts(normalize=True).rename(f"ep{e}") for e in sorted(snapshots)],
                axis=1)
#this is the comparison of how plentiful the "num"/diagnosis severity(0 no ill, 4 deathly ill) is; you can compare other discrete datas
print(cmp)

# losses
# plt.plot(d_hist, label="critic"); plt.plot(g_hist, label="generator"); plt.legend(); plt.show()

eps, tr, te = zip(*auc_hist)
plt.plot(eps, tr, label="train AUC")
plt.plot(eps, te, label="test AUC")
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("epoch"); plt.ylabel("critic AUC"); plt.legend(); plt.show()