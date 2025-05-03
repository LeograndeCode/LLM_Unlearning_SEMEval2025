from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

hf_token = "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"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
## Fetch and load model:
snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-1B-model')
model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model').to(device)

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
pretrained_model = AutoModelForCausalLM.from_pretrained("semeval25-unlearning-1B-model").to(device)
