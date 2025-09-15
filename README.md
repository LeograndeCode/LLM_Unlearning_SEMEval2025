# LLM Unlearning for SemEval 2025

This repository implements machine unlearning approaches for Large Language Models (LLMs) as part of the SemEval-2025 Task 4: "Unlearning Sensitive Content from Large Language Models". We explore and compare two different unlearning strategies: a Dual-Teacher framework and Selective Knowledge Negation Unlearning (SKU).

## 📋 Task Overview

The SemEval-2025 Task 4 challenges participants to selectively erase specific content from pretrained language models without compromising their overall functionality. The dataset comprises 2,780 samples across three subtasks:

- **Subtask 1** (17.0%): Long-form synthetic creative documents
- **Subtask 2** (55.5%): Short-form synthetic biographies with PII data
- **Subtask 3** (27.5%): Real Wikipedia documents from training data

Each subtask includes both regurgitation tasks (verbatim reproduction) and knowledge-based QA tasks (factual recall), evaluated on retain and forget splits.

## 🏗️ Repository Structure

```
LLM_Unlearning_SEMEval2025/
├── DualTeacher_seq.ipynb           # Sequential dual-teacher implementation
├── DualTeacher_noseq.ipynb         # Mixed (non-sequential) dual-teacher implementation
├── evaluation_notebook.ipynb       # Evaluation analysis
├── data/                          # Dataset files (parquet format)
└── README.md                     # This file
```

## 🚀 Methods Implemented

### 1. Dual-Teacher Framework

Our main approach leverages two contrasting teacher models:

- **Good Teacher**: Fine-tuned on retain data to preserve legitimate knowledge
- **Bad Teacher**: Generates uniform/noisy logits to encourage forgetting
- **Student Model**: Learns from both teachers using different loss functions

#### Two Implementation Variants:

**Sequential Implementation** (`DualTeacher_seq.ipynb`):
- Alternates between retain and forget training phases within each epoch
- Uses separate dataloaders and phase-specific loss functions
- Cleaner separation of objectives, avoiding interference
- Better privacy protection (MIA resistance)

**Non-Sequential Implementation** (`DualTeacher_noseq.ipynb`):
- Processes retain and forget samples within the same training batches
- Uses unified loss function with masking operations
- Superior sample balance per gradient update
- More stable training dynamics

### 2. Selective Knowledge Negation Unlearning (SKU)

Alternative approach using direct token-level loss engineering:
- Unlikelihood loss on answer spans to suppress memorization
- Context stabilization to prevent catastrophic forgetting
- L2 anchoring and entropy maximization for regularization

## 🔧 Technical Details

### Model Configuration
- **Base Model**: OLMo-1B (provided by organizers)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with r=16, α=32
- **Hardware**: Dual Tesla T4 GPUs (32GB total VRAM)


## 📊 Evaluation Metrics

Following the SemEval-2025 protocol:

1. **Regurgitation Score**: ROUGE-L for completion tasks, exact match for QA
2. **MIA Resistance**: Resistance to membership inference attacks
3. **MMLU Score**: Preservation of general knowledge across 57 academic domains
4. **Aggregate Score**: Arithmetic mean of the three component scores

## 🔍 Key Features

- **Parameter-Efficient**: LoRA reduces memory requirements by 90%
- **Dual-GPU Support**: Teacher and student models on separate GPUs
- **Early Stopping**: Automatic validation monitoring with patience=3
- **Comprehensive Evaluation**: Official SemEval evaluation script integration
- **Memory Optimized**: Gradient accumulation and efficient batch processing

## 📖 Paper

Our approach is documented in the ACL 2023 format paper located in:
```
_LLM_24_25___E4__Unlearning_sensitive_content_from_Large_Language_Models/acl2023.tex
```

The paper provides detailed methodology, experimental results, and analysis of both sequential and non-sequential dual-teacher implementations.

## 🤝 Contributors

- Giulio Desana (331445)
- Augusto Leogrande (326050)
- Silvia Polizzi (323914)

## 📄 License

This project is part of the SemEval-2025 shared task. Please refer to the competition guidelines for usage restrictions.

## 🔗 References

- SemEval-2025 Task 4: Unlearning Sensitive Content from Large Language Models
- HuggingFace Transformers and PEFT libraries
- OLMo: Open Language Model by Allen Institute for AI