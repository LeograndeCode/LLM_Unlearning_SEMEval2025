import os
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
import torch
import logging
from torch.utils.data import DataLoader
import random
import numpy as np
from data.dataloader import create_forget_dataloader, create_retain_dataloader
from model.load_model import tokenizer, pretrained_model
from utils.loss_functions import get_answer_loss, get_rand_ans_loss, get_harmful_responses, compute_reverse_kl
from utils.task_vector import TaskVector


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

# Create folders
os.makedirs("semeval25-unlearning-model", exist_ok=True)
os.makedirs("semeval25-unlearning-model/task_vector", exist_ok=True)
# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
num_training_steps = 1000
bad_weight = 2.5
random_weight = 1
normal_weight = 0.5
batch_size = 2
lr = 2e-4

model_save_dir = "semeval25-unlearning-model"
task_vector_saving_path = "semeval25-unlearning-model/task_vector"


# Create dataloaders (implement create_*_dataloader functions)
forget_train_dl = create_forget_dataloader(tokenizer, forget_train_df, batch_size=batch_size)
retain_train_dl = create_retain_dataloader(tokenizer, retain_train_df, batch_size=batch_size)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model.train()

# Usage (add this before training loop)
bad_ans = get_harmful_responses(forget_train_df)


idx = 0
for _ in range(num_training_steps):
    for bad_batch, normal_batch in zip(forget_train_dl, retain_train_dl):
        # Move batches to device
        bad_batch = {k: v.to(device) for k, v in bad_batch.items()}
        normal_batch = {k: v.to(device) for k, v in normal_batch.items()}

        # Guided Distortion Module
        bad_loss = get_answer_loss("gd", bad_batch, model)

        # Random Disassociation Module
        random_loss = get_rand_ans_loss(bad_batch, tokenizer, bad_ans, model, K=5)

        # Preservation Divergence Module
        normal_loss = compute_reverse_kl(pretrained_model, model, normal_batch)

        # Total loss
        loss = (bad_weight * bad_loss +
               random_weight * random_loss +
               normal_weight * normal_loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Logging
        stats = (f"batch: {idx}, GD_loss: {bad_loss:.2f}, "
                f"RD_loss: {random_loss:.2f}, reversed_kl_loss: {normal_loss:.2f}, "
                f"combined_loss: {loss:.2f}")
        logging.info(stats)
        print(stats)
        idx += 1

# Save results
print("Saving model...")
model.save_pretrained(model_save_dir)
logging.info("Unlearning finished")

# Create and save task vector
task_vector = TaskVector(pretrained_model, model)
neg_task_vector = -task_vector
new_benign_model = neg_task_vector.apply_to(pretrained_model)
new_benign_model.save_pretrained(task_vector_saving_path)

print("Done saving task vector files!")