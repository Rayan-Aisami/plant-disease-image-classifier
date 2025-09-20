import os
import shutil
from sklearn.model_selection import train_test_split

# Get project root for absolute paths (robust on Windows)
project_root = os.path.dirname(os.path.dirname(__file__))  # Since script is in src/
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# Debug: List contents of 'data/' to see actual folders/files (e.g., plantvillage-dataset.zip, plantvillage-dataset/)
data_dir = os.path.join(project_root, 'data')
if os.path.exists(data_dir):
    print(f"Contents of 'data/': {os.listdir(data_dir)}")
else:
    print(f"Error: 'data/' folder not found at {data_dir}. Create it and unzip there.")
    raise FileNotFoundError("data/ folder missing.")

# Paths (adjusted for full dataset structure: hyphenated folder, 'color' for RGB images)
data_root = os.path.join(project_root, 'data', 'plantvillage-dataset', 'color')
train_dir = os.path.join(project_root, 'data', 'plantvillage-dataset', 'train')
val_dir = os.path.join(project_root, 'data', 'plantvillage-dataset', 'val')
test_dir = os.path.join(project_root, 'data', 'plantvillage-dataset', 'test')

# Debug: Check if data_root exists and list contents
if not os.path.exists(data_root):
    print(f"Error: {data_root} does not exist.")
    
    # Try alternative paths (e.g., direct classes, or other variants)
    alt_paths = [
        os.path.join(project_root, 'data', 'plantvillage-dataset'),  # If no 'color'
        os.path.join(project_root, 'data', 'plantvillage dataset', 'color'),  # Space variant
        os.path.join(project_root, 'data', 'plant_village', 'color')  # Underscore variant
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            print(f"Alternative path found: {alt}")
            print(f"Contents: {os.listdir(alt)}")  # Should show class folders or 'color' etc.
            print("Update data_root to this path in the script.")
            break
    else:
        print("No alternative paths found. Check unzip location in File Explorer.")
    
    raise FileNotFoundError(f"Path not found: {data_root}. Use printed info to adjust data_root.")

else:
    print(f"Data root found: {data_root}")
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    print(f"Classes found: {len(classes)} (should be 38). Examples: {classes[:5]}...")

# Create split folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# For each class
for class_name in os.listdir(data_root):
    class_path = os.path.join(data_root, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # Get image list (case-insensitive for .JPG/.jpg)
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg'))]
    print(f"Processing class '{class_name}': {len(images)} images")
    
    # Split: 80% train, 10% val, 10% test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # Create class subdirs in splits
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # Copy files
    for img in train_imgs:
        shutil.copy(img, os.path.join(train_dir, class_name, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy(img, os.path.join(val_dir, class_name, os.path.basename(img)))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_dir, class_name, os.path.basename(img)))

print("Data split complete! Check data/plantvillage-dataset/{train,val,test} for folders.")