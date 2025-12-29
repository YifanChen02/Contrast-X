import os
from huggingface_hub import hf_hub_download

# Set mirror endpoint
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Example: download a file from your dataset repo
file_path = hf_hub_download(
    repo_id="HaoChen2/PairContrastDataset",
    filename="PairedContrast.tar.gz",  # adjust to your file name
    repo_type="dataset"
)

print("Downloaded to:", file_path)




# pip install kaggle



