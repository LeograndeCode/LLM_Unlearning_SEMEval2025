import torch
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from rouge_score import rouge_scorer

class UnlearningDataset(Dataset):
    def __init__(self, data_source, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
            print(f"Caricati {len(self.data)} esempi dal DataFrame")
        elif isinstance(data_source, str):
            data_list = []
            with open(data_source, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data_list.append(item)
            self.data = pd.DataFrame(data_list)
            print(f"Caricati {len(self.data)} esempi da {data_source}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_text = item["input"]
        output_text = item["output"]

        # Tokenizzazione unica
        combined = f"{input_text} {output_text}"
        tokenized = self.tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenizzazione solo dell'input per start_locs
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "start_locs": input_ids.size(0),  # posizione dove finisce l'input
            "labels": tokenized["input_ids"].squeeze(0),
            "split": 1 if item.get("split", "retain") == "forget" else 0
        }

class DualTeacherTrainer:
    def __init__(self, model_path, tokenizer, teacher_lora_config, student_lora_config, device_map=None):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.teacher_lora_config = teacher_lora_config
        self.student_lora_config = student_lora_config
        self.device_map = device_map or {"student": "cuda:0", "teacher": "cuda:1"}
        
        self.good_teacher = None
        self.student_model = None
        self.initial_state_dict = {}
        
        # Validation tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        
    def setup_models(self, skip_teacher_setup=False):
        """Initialize and setup both teacher and student models"""
        print("🔧 Setting up models...")
        base_model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        if skip_teacher_setup is False:
            # Setup good teacher with LoRA (for training)
            
            self.good_teacher = get_peft_model(base_model, self.teacher_lora_config, local_files_only=True)
            self.good_teacher = self.good_teacher.to(self.device_map["teacher"])
            self.good_teacher.print_trainable_parameters()
            
        # Setup student model with LoRA
        self.student_model = get_peft_model(base_model, self.student_lora_config)
        self.student_model = self.student_model.to(self.device_map["student"])
        self.student_model.print_trainable_parameters()
        
        # Save initial state for task vector calculation
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                self.initial_state_dict[name] = param.data.clone()
        
        print("✅ Models setup completed")
        
    
    def create_bad_teacher_logits(self, good_teacher_logits: torch.Tensor):
        """Create more realistic bad teacher logits"""
        # Invece di -logits * 2.0, usa uniform distribution
        vocab_size = good_teacher_logits.size(-1)
        uniform_logits = torch.ones_like(good_teacher_logits) / vocab_size
        return uniform_logits + torch.randn_like(good_teacher_logits) * 0.1

        
    
    def compute_kl_divergence(self, batch):
        # Devices
        student_device = self.device_map["student"]
        teacher_device = self.device_map["teacher"]
    
        # Inputs
        input_ids_student = batch["input_ids"].to(student_device)
        attention_mask_student = batch["attention_mask"].to(student_device)
        labels_student = batch["labels"].to(student_device)
        split = batch["split"].float().to(student_device)
    
        # Student forward
        student_logits = self.student_model(input_ids_student, attention_mask=attention_mask_student).logits
        student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1).to(teacher_device)
    
        # Teacher forward (no grad)
        input_ids_teacher = batch["input_ids"].to(teacher_device)
        attention_mask_teacher = batch["attention_mask"].to(teacher_device)
    
        with torch.no_grad():
            good_teacher_logits = self.good_teacher(input_ids_teacher, attention_mask=attention_mask_teacher).logits
            bad_teacher_logits = self.create_bad_teacher_logits(good_teacher_logits)
            good_teacher_probs = torch.nn.functional.softmax(good_teacher_logits, dim=-1)
            bad_teacher_probs = torch.nn.functional.softmax(bad_teacher_logits, dim=-1)
    
        # Masks retain/forget
        retain_mask = (split <= 0.5).to(teacher_device)
        forget_mask = (split > 0.5).to(teacher_device)
    
        total_loss = 0.0
        if retain_mask.any():
            retain_kl = torch.nn.functional.kl_div(
                student_log_probs[retain_mask],
                good_teacher_probs[retain_mask.bool()],
                reduction="none",
                log_target=False
            ).sum(dim=-1)  # somma su vocab
            retain_kl = retain_kl.mean()
            total_loss += 1.5 * retain_kl  # retain_weight
    
        if forget_mask.any():
            forget_kl = torch.nn.functional.kl_div(
                student_log_probs[forget_mask],
                bad_teacher_probs[forget_mask.bool()],
                reduction="none",
                log_target=False
            ).sum(dim=-1)
            forget_kl = forget_kl.mean()
            total_loss += 5.0 * forget_kl  # forget_weight
    
        # Entropia student per regolarizzazione
        entropy_loss = -(student_log_probs.exp() * student_log_probs).sum(-1).mean()
    
        return total_loss + 0.2 * entropy_loss

    def validate_model(self, val_dataloader):
        """Validate with separate metrics for retain and forget samples"""
        self.student_model.eval()
        retain_losses = []
        forget_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                loss = self.compute_kl_divergence(batch)
                split = batch["split"]

                # Separate by sample type
                for i, s in enumerate(split):
                    if s == 0:  # retain
                        retain_losses.append(loss.item())
                    else:  # forget
                        forget_losses.append(loss.item())

        retain_avg = np.mean(retain_losses) if retain_losses else 0.0
        forget_avg = np.mean(forget_losses) if forget_losses else 0.0

        # Simple composite: retain_loss + (1 / (forget_loss + 1))
        # Lower retain_loss is better, higher forget_loss is better
        composite_score = retain_avg + 1.0 / (forget_avg + 1.0)

        self.student_model.train()
        return {
            'retain_loss': retain_avg,
            'forget_loss': forget_avg,
            'composite_score': composite_score
        }

    def validate_good_teacher(self, val_dataloader):
        """Validate the good teacher on validation set"""
        self.good_teacher.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Filter retain samples only for validation
                split = batch['split']
                retain_mask = (split == 0)
                
                if not retain_mask.any():
                    continue
                
                input_ids = batch['input_ids'][retain_mask].to(self.device_map["teacher"])
                attention_mask = batch['attention_mask'][retain_mask].to(self.device_map["teacher"])
                labels = batch['labels'][retain_mask].to(self.device_map["teacher"])
                
                if input_ids.size(0) == 0:
                    continue
                
                outputs = self.good_teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        self.good_teacher.train()
        return avg_val_loss
        
    def train_good_teacher(self, dataloader, val_dataloader=None, num_epochs=2, lr=1e-4, save_path="good_teacher_adapter", val_freq=1):
        """Train the good teacher on retain samples using LoRA with optional validation"""
        print("🚀 Training good teacher with LoRA...")

        self.good_teacher.to(self.device_map["teacher"])
        self.good_teacher.train()
        optimizer = torch.optim.AdamW(self.good_teacher.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            print(f"📅 Epoca {epoch + 1}/{num_epochs} - Good Teacher Training")
            
            epoch_losses = []
            retain_batches_processed = 0
            
            with tqdm(total=len(dataloader), desc=f"Good Teacher Epoca {epoch+1}") as pbar:
                for batch in dataloader:
                    # Filter retain samples only
                    split = batch['split']
                    retain_mask = (split == 0)
                    
                    if not retain_mask.any():
                        pbar.update(1)
                        continue
                    
                    # Extract retain samples
                    input_ids = batch['input_ids'][retain_mask].to(self.device_map["teacher"])
                    attention_mask = batch['attention_mask'][retain_mask].to(self.device_map["teacher"])
                    labels = batch['labels'][retain_mask].to(self.device_map["teacher"])
                    
                    if input_ids.size(0) == 0:
                        pbar.update(1)
                        continue
                    
                    optimizer.zero_grad()
                    outputs = self.good_teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    retain_batches_processed += 1
                    pbar.update(1)
                    
                    if retain_batches_processed % 100 == 0:
                        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"📊 Good Teacher Epoca {epoch+1} - Loss medio: {avg_loss:.4f}")
                
                # Validation if validation dataloader is provided
                if val_dataloader is not None and (epoch + 1) % val_freq == 0:
                    print("🔍 Running validation...")
                    val_loss = self.validate_good_teacher(val_dataloader)
                    print(f"📊 Validation Loss: {val_loss:.4f}")
        
        print("✅ Good teacher training completed")
        
        # Save only LoRA adapter
        self.save_good_teacher(save_path)
        print(f"💾 Good teacher adapter salvato in {save_path}")
        
    def train_student(self, dataloader, val_dataloader=None, num_epochs=4, lr=1e-4, val_freq=1, patience=3):
        """Train student model with dual teacher approach and validation"""
        print("🚀 Training student with LoRA against base model teacher...")
        
        self.student_model.train()
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr, weight_decay=0.01)
        
        # Early stopping
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            with tqdm(total=len(dataloader), desc=f"Student Epoca {epoch+1}") as pbar:
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss = self.compute_kl_divergence(batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)
            
            avg_loss = np.mean(epoch_losses)
            print(f"📊 Student Epoca {epoch+1} - Loss medio: {avg_loss:.4f}")
            
            # Validation if validation dataloader is provided
            if val_dataloader is not None and (epoch + 1) % val_freq == 0:
                print("🔍 Running validation...")
                val_results = self.validate_model(val_dataloader)
                val_loss = val_results['composite_score']
                print(f"📊 Validation - Retain: {val_results['retain_loss']:.4f}, Forget: {val_results['forget_loss']:.4f}, Composite: {val_loss:.4f}")

                # Check if this is the best model so far
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    patience_counter = 0
                    print(f"🎯 New best validation score: {val_loss:.4f}")

                    # Save best model
                    self.save_model(f"studentmodel_best_val")
                else:
                    patience_counter += 1
                    print(f"⚠️ No improvement for {patience_counter} validation checks")

                # Early stopping
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} validations)")
                    break
            
            # Save model after each epoch
            self.save_model(f"studentmodel_epoch_{epoch+1}")
        
        print("✅ Student training completed")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
    def save_model(self, save_path):
        """Save student model and tokenizer"""
        self.student_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
    def save_good_teacher(self, save_path):
        """Save only the LoRA adapter of the good teacher"""
        self.good_teacher.save_pretrained(save_path)

    def load_good_teacher_from_adapter(self, adapter_path):
        """Reload good teacher from base model + adapter"""
        print(f"📂 Caricamento good teacher da adapter {adapter_path} ...")
        base_model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        self.good_teacher = PeftModel.from_pretrained(base_model, adapter_path)
        self.good_teacher = self.good_teacher.to(self.device_map["teacher"])
        
        # Freeze params
        self.good_teacher.eval()
        for param in self.good_teacher.parameters():
            param.requires_grad = False
        print("✅ Good teacher caricato e congelato")

    def load_teacher(self, GOOD_TEACHER_PATH):
        self.good_teacher = AutoModelForCausalLM.from_pretrained(GOOD_TEACHER_PATH)
        self.good_teacher.eval()  # congelalo
        for param in self.good_teacher.parameters():
            param.requires_grad = False
        self.good_teacher.to(self.device_map["teacher"])
        print("✅ Good teacher caricato e congelato")

    def calculate_task_vector(self):
        """Calculate task vector from initial to final state"""
        task_vector = {}
        for name, param in self.student_model.named_parameters():
            if param.requires_grad and name in self.initial_state_dict:
                task_vector[name] = param.data - self.initial_state_dict[name]
        return task_vector

# Example usage
if __name__ == "__main__":
    # Configurazioni
    MODEL_PATH = "/kaggle/input/olmo-model/semeval25-unlearning-1B-model"
    DATA_PATH = "/kaggle/input/olmo-model/semeval25-unlearning-data"
    MIA_VAL_PATH = "/kaggle/input/mia-dataset-val"
    MIA_TRAIN_PATH = "/kaggle/input/mia-dataset"
    GOOD_TEACHER_PATH = "/kaggle/input/good-teacher"
    
    # Caricamento dataset
    retain_train_df = pd.read_parquet(f"{DATA_PATH}/data/retain_train-00000-of-00001.parquet", engine='pyarrow')
    retain_validation_df = pd.read_parquet(f"{DATA_PATH}/data/retain_validation-00000-of-00001.parquet", engine='pyarrow')
    forget_train_df = pd.read_parquet(f"{DATA_PATH}/data/forget_train-00000-of-00001.parquet", engine='pyarrow')
    forget_validation_df = pd.read_parquet(f"{DATA_PATH}/data/forget_validation-00000-of-00001.parquet", engine='pyarrow')
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create training and validation datasets
    batch_size = 4
    train_data = pd.concat([retain_train_df, forget_train_df], ignore_index=True)
    val_data = pd.concat([retain_validation_df, forget_validation_df], ignore_index=True)
    
    train_dataset = UnlearningDataset(train_data, tokenizer)
    val_dataset = UnlearningDataset(val_data, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    print(f"Training dataset creato con {len(train_dataset)} esempi")
    print(f"Validation dataset creato con {len(val_dataset)} esempi")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    
    # Initialize trainer
    trainer = DualTeacherTrainer(
        model_path=MODEL_PATH,
        tokenizer=tokenizer,
        teacher_lora_config=lora_config,
        student_lora_config=lora_config,
        device_map={"student": "cuda:0", "teacher": "cuda:1"}
    )
    
    # Setup models
    trainer.setup_models(skip_teacher_setup=True)
    trainer.load_good_teacher_from_adapter(GOOD_TEACHER_PATH)
    
    # Train student with validation
    trainer.train_student(train_dataloader, val_dataloader=val_dataloader, num_epochs=5, lr=3e-5)
    trainer.save_model("studentmodel_final")
