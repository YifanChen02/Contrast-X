from huggingface_hub import HfApi, upload_folder, upload_large_folder
import os
HF_TOKEN = os.getenv("HF_TOKEN")

import logging

# Enable logging (INFO shows progress, DEBUG shows full details)
logging.basicConfig(level=logging.INFO)


api = HfApi(token=HF_TOKEN)

print("Start uploading...")


upload_large_folder(
    folder_path="/date/hao/PairedContrast",
    repo_id="HaoChen2/PairContrastDataset",
    repo_type="dataset"
)