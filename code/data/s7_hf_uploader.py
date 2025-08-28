import os
from huggingface_hub import HfApi





HF_TOKEN = "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"

folder_path="/date/hao/PairedContrast/CT/low_256x256_2Dimension"


api = HfApi(token=os.getenv(HF_TOKEN))
api.upload_large_folder(
    folder_path=folder_path,
    repo_id="HaoChen2/PairContrastDataset",
    repo_type="dataset",
)