import os
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
hf_token = "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"



## Fetch and load dataset:
snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=hf_token, local_dir='semeval25-unlearning-data', repo_type="dataset")
retain_train_df = pd.read_parquet('semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet', engine='pyarrow') # Retain split: train set
retain_validation_df = pd.read_parquet('semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet', engine='pyarrow') # Retain split: validation set
forget_train_df = pd.read_parquet('semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet', engine='pyarrow') # Forget split: train set
forget_validation_df = pd.read_parquet('semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet', engine='pyarrow') # Forget split: validation set
os.makedirs("train", exist_ok=True)
os.makedirs("validation", exist_ok=True)

retain_train_df.to_json('train/retain.jsonl', orient='records', lines=True); forget_train_df.to_json('train/forget.jsonl', orient='records', lines=True)
retain_validation_df.to_json('validation/retain.jsonl', orient='records', lines=True); forget_validation_df.to_json('validation/forget.jsonl', orient='records', lines=True)