"""Loss functions for LLM Unlearning project."""

import logging
import random
import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def get_answer_loss(
    operation: str,
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_locs = batch["start_locs"]
    labels = batch["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []

    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])

        if operation == "ga":  # Negative the direction for GA
            position_loss = -position_loss

        # Simply put equal weights on all answers
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)

    final_loss = torch.stack(losses).mean()
    return final_loss


def compute_reverse_kl(
    pretrained_model: torch.nn.Module,
    current_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    Compute reverse KL divergence D_KL(P || Q) = sum_x P(x) * log(P(x)/Q(x))
    in a numerically stable way using log-softmax.

    Args:
        pretrained_model: reference model (P; fp32)
        current_model: model in training (Q; quantized+LoRA)
        batch: dict with input_ids, attention_mask
        device: 'cuda' or similar

    Returns:
        scalar loss (mean over batch and seq)
    """
    # Forward pass of current model (Q)
    out_q = current_model(
        batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
    )
    logits_q = out_q.logits  # [B, T, V]

    # Forward pass of pretrained model (P), without grad
    with torch.no_grad():
        out_p = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        logits_p = out_p.logits  # [B, T, V]

    # log-softmax (numerically stable)
    logp = F.log_softmax(logits_p, dim=-1)  # log P(x)
    logq = F.log_softmax(logits_q, dim=-1)  # log Q(x)

    # P(x) = exp(logp)
    p_prob = torch.exp(logp)

    # compute reverse KL = - sum_x P * (logp - logq)
    # (negative because we minimize)
    kl_per_token = -(p_prob * (logp - logq)).sum(dim=-1)  # [B, T]
    loss = kl_per_token.mean()  # scalar

    return loss


def get_rand_ans_loss(
    bad_batch: Dict[str, torch.Tensor],
    tokenizer: Any,
    normal_ans: List[str],
    model: torch.nn.Module,
    K: int = 5,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Random Disassociation: for each question in the batch, sample K answers from the retain set,
    create batch of texts `Question + Answer`, and call get_answer_loss("gd", ...).

    Args:
        bad_batch: batch of forget data
        tokenizer: tokenizer instance
        normal_ans: list of retain answers
        model: model instance
        K: number of random answers to sample
        device: device string

    Returns:
        loss tensor
    """
    # Decode questions from batch of input_ids
    # skip_special_tokens=True to remove pad/eos
    questions = tokenizer.batch_decode(bad_batch["input_ids"], skip_special_tokens=True)

    features = []
    for question in questions:
        prefix = question.strip()
        # Count real tokens of prefix (no pad)
        t_pref = tokenizer(prefix, truncation=True, padding=False)
        start_loc = len(t_pref["input_ids"])

        # For each question sample K random answers from retain set
        rand_samples = random.sample(normal_ans, K)
        for ans in rand_samples:
            text = prefix + ans
            tok = tokenizer(text, truncation=True, padding="max_length", max_length=128)
            features.append(
                {
                    "input_ids": tok["input_ids"],
                    "attention_mask": tok["attention_mask"],
                    "start_locs": start_loc,
                    "labels": tok["input_ids"],
                }
            )

    # Use the same DataCollator as training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(features)

    # Loss of gradient *descent* on the "answer" segment
    return get_answer_loss("gd", batch_random, model, device=device)
