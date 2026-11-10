"""Build a Colab-ready archive containing source, not datasets or credentials."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
output = root / "deliverables/gan_project_source.zip"
output.parent.mkdir(exist_ok=True)
patterns = ("src/gan_project/*.py", "configs/*.yaml", "scripts/*.py",
            "tests/*.py", "notebooks/*.ipynb", "docs/*.md", "pyproject.toml", "requirements*.txt", "README.md", "THIRD_PARTY_LICENSES.txt")
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for pattern in patterns:
        for file in sorted(root.glob(pattern)):
            archive.write(file, file.relative_to(root))
print(output)
