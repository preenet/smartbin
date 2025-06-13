from pathlib import Path
import shutil

# Set the General folder path
general_dir = Path(r"F:\SmartBin\dataset\Recycle")
image_exts = {".jpg", ".jpeg"}

if not general_dir.exists():
    raise FileNotFoundError(f"General folder not found: {general_dir}")

moved = 0
skipped = 0

print(f"Searching for JPEG images in subfolders of: {general_dir}")
print("-" * 50)

# Find all image files in subdirectories (not in the root General folder)
for file in general_dir.rglob("*"):
    # Skip files that are already in the root General directory
    if file.parent == general_dir:
        continue
        
    if file.is_file() and file.suffix.lower() in image_exts:
        dest = general_dir / file.name

        # Avoid overwriting: auto-increment if file exists
        counter = 1
        original_dest = dest
        while dest.exists():
            dest = general_dir / f"{file.stem}_{counter}{file.suffix}"
            counter += 1

        try:
            # Move the file
            shutil.move(str(file), str(dest))
            moved += 1
            
            if dest != original_dest:
                print(f"Moved {file.relative_to(general_dir)} -> {dest.name} (renamed to avoid conflict)")
            else:
                print(f"Moved {file.relative_to(general_dir)} -> {dest.name}")
                
        except Exception as e:
            print(f"Error moving {file}: {e}")
            skipped += 1

print("-" * 50)
print(f"Done! {moved} image(s) moved to {general_dir}")
if skipped > 0:
    print(f"{skipped} file(s) skipped due to errors")

# Clean up empty subdirectories (optional)
print("\nCleaning up empty subdirectories...")
empty_dirs_removed = 0
for subdir in general_dir.iterdir():
    if subdir.is_dir():
        try:
            # Try to remove if empty
            subdir.rmdir()
            empty_dirs_removed += 1
            print(f"Removed empty directory: {subdir.name}")
        except OSError:
            # Directory not empty, skip
            continue

if empty_dirs_removed > 0:
    print(f"Removed {empty_dirs_removed} empty subdirectorie(s)")
else:
    print("No empty subdirectories to remove")