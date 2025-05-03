import torch
from torch.utils.data import DataCollatorForLanguageModeling
import random

def compute_reverse_kl(pretrained_model, current_model, batch, device):
    """
    Compute *backward* KL as the normal utility loss.
    """
    # Move entire batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    normal_outputs = current_model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    # Q: current model; P: pretrained model.
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, dim=-1)
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, dim=-1)

    # Negative KL divergence: sum(Q * log(Q/P))
    loss = - (prob_p * torch.log((prob_p + 1e-12) / prob_q)).sum(-1).mean()

    return loss

def get_answer_loss(operation, batch, model, device="cuda"):
    """
    Compute the loss on the answer (i.e. y) part.
    """
    # Move entire batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    start_locs = batch["start_locs"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none").to(device)
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :].to(device)
    shift_labels = labels[:, 1:].to(device)

    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp = input_ids[bid]
        one_st = start_locs[bid]

        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])

        if operation == "ga":
            position_loss = -position_loss

        position_weight = torch.zeros_like(one_inp).to(device)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1

        position_weight[one_inp == 1] = 0  # Ignore padding
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)

    return torch.stack(losses).mean()

def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda"):
    """
    Compute the loss of the random mismatch.
    """
    # Move entire batch to device
    bad_batch = {k: v.to(device) for k, v in bad_batch.items()}

    bad_input_ids = bad_batch["input_ids"]
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []

    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx]
        ori_text = tokenizer.decode(single_input_id)

        # Extract question
        question = ori_text.split("###")[1].split("Question:")[-1].strip()
        question_prefix = f"### Question: {question}\n ### Answer: "

        # Tokenize on device
        tokenized_question_prefix = tokenizer(
            question_prefix,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        start_loc = tokenized_question_prefix.input_ids.shape[1]

        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"
            tokenized_rs = tokenizer(
                random_sample,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            batch_random_features.append({
                "input_ids": tokenized_rs.input_ids.squeeze(0),
                "attention_mask": tokenized_rs.attention_mask.squeeze(0),
                "start_locs": torch.tensor([start_loc], device=device)
            })

    def get_harmful_responses(forget_train_df):
      """Extracts harmful responses from the forget training dataframe."""
      # Assuming your dataframe has a column called 'output' containing harmful responses
      return forget_train_df['output'].tolist()

    # Batchify on device
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)
    batch_random = {k: v.to(device) for k, v in batch_random.items()}

    return get_answer_loss("gd", batch_random, model, device) 