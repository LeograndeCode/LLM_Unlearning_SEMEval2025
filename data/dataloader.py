from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

def create_forget_dataloader(tokenizer, df, batch_size=64):
    """
    Create a dataloader for the forget set from a pandas DataFrame.

    Args:
        tokenizer: Tokenizer for the model
        df: pandas DataFrame with columns ['id', 'input', 'output']
        batch_size: Batch size for the dataloader

    Returns:
        DataLoader for the forget set
    """
    def preprocess(examples):
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for prompt, response in zip(examples["input"], examples["output"]):
            # Format the text with question and answer
            text = f"### Question: {prompt}\n ### Answer: {response}"
            tokenized = tokenizer(text, truncation=True, padding="max_length")

            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])

            # Calculate start location of answer
            question_prefix = f"### Question: {prompt}\n ### Answer: "
            tokenized_prefix = tokenizer(question_prefix, truncation=True, padding="max_length")
            results["start_locs"].append(len(tokenized_prefix["input_ids"]) - 1)

        return results

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Apply preprocessing
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["id", "input", "output"]
    )

    # Set format to PyTorch
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True
    )

    return dataloader

def create_retain_dataloader(tokenizer, df, batch_size=64):
    """
    Create a dataloader for the retain set from a pandas DataFrame.

    Args:
        tokenizer: Tokenizer for the model
        df: pandas DataFrame with columns ['id', 'input', 'output']
        batch_size: Batch size for the dataloader

    Returns:
        DataLoader for the retain set
    """
    def preprocess(examples):
        results = {"input_ids": [], "attention_mask": []}

        for prompt, response in zip(examples["input"], examples["output"]):
            text = f"### Question: {prompt}\n ### Answer: {response}"
            tokenized = tokenizer(text, truncation=True, padding="max_length")

            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])

        return results

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Apply preprocessing
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["id", "input", "output"]
    )

    # Set format to PyTorch
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True
    )

    return dataloader
