
import os
import requests
from glob import glob
import zipfile

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/index-image/")
ZIP_PATH = os.environ.get("ZIP_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/archive.zip')))
EXTRACT_DIR = os.environ.get("EXTRACT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/images')))

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

# Extract images if not already extracted
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

IMAGE_DIR = EXTRACT_DIR

def find_images(root_dir):
    """Recursively find all image files and return (path, label) pairs.
    Label is derived from the immediate parent folder name (defect category).
    """
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(dirpath, fname)
                # Use parent folder as label (defect type), fall back to filename
                parent = os.path.basename(dirpath)
                label = parent if parent != os.path.basename(root_dir) else os.path.splitext(fname)[0]
                results.append((full_path, label))
    return results

def index_images():
    image_list = find_images(IMAGE_DIR)
    if not image_list:
        print(f"No images found in {IMAGE_DIR}. Make sure archive.zip is present and has images.")
        return
    print(f"Found {len(image_list)} images. Starting indexing...")
    success, failed = 0, 0
    for img_path, label in image_list:
        try:
            with open(img_path, "rb") as f:
                # Send both file and label as multipart form data
                files = {"file": (os.path.basename(img_path), f, "image/png")}
                # Pass label as form field, not data
                resp = requests.post(BACKEND_URL, files=files, data={"label": label}, timeout=30)
                if resp.ok:
                    print(f"[OK] {img_path} -> label: {label}")
                    success += 1
                else:
                    print(f"[FAIL] {img_path}: {resp.text}")
                    failed += 1
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            failed += 1
    print(f"\nIndexing complete. Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    index_images()

