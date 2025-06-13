from pathlib import Path
import shutil

# Set absolute paths for your source (raw) and destination (dataset) folders
raw_dir = Path(r"F:\SmartBin\raw")
dataset_dir = Path(r"F:\SmartBin\dataset")
image_exts = {".jpg", ".jpeg"}

if not raw_dir.exists():
    raise FileNotFoundError(f"Source folder not found: {raw_dir}")
if not dataset_dir.exists():
    dataset_dir.mkdir(parents=True, exist_ok=True)

moved = 0
for file in raw_dir.rglob("*"):
    if file.is_file() and file.suffix.lower() in image_exts:
        dest = dataset_dir / file.name

        # Avoid overwriting: auto-increment if file exists
        counter = 1
        while dest.exists():
            dest = dataset_dir / f"{file.stem}_{counter}{file.suffix}"
            counter += 1

        shutil.move(str(file), str(dest))
        moved += 1
        print(f"Moved {file} -> {dest.name}")

print(f"\nDone. {moved} image(s) moved to {dataset_dir}")