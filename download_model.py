import urllib.request
import os

# Yahan apna Google Drive ya Hugging Face ka direct download link daalna
url = "TUMHARA_DIRECT_DOWNLOAD_LINK_YAHAN" 
filename = "yolov8m-world.pt"

print("Checking for model file...")
if not os.path.exists(filename):
    print("Downloading yolov8m-world.pt (this might take a few minutes)...")
    urllib.request.urlretrieve(url, filename)
    print("Model downloaded successfully! 🚀")
else:
    print("Model already exists. You are good to go!")
