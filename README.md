# AgriSmart Assistant

A domain-specific agricultural assistant built by fine-tuning **TinyLlama-1.1B-Chat** using **LoRA (Low-Rank Adaptation)** on the KisanVaani Agriculture QA dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kellenmurerwa/AgriSmart-Assistant/blob/main/AgriSmart_Assistant.ipynb)

## Project Overview

Smallholder farmers often lack access to timely agricultural advice. This project fine-tunes a lightweight LLM to serve as a specialized assistant that answers questions about crops, pests, soil management, and fertilizers.

**Key Features:**
- LoRA-based parameter-efficient fine-tuning (only 0.2% of parameters trained)
- 4-bit NF4 quantization for memory-efficient inference (~700MB GPU)
- Interactive Gradio chatbot for real-time Q&A
- Comprehensive evaluation with BLEU, ROUGE, and Perplexity metrics
- Pre-trained adapter on Hugging Face for instant testing (no retraining needed)

**Fine-Tuned Model:** [kellenmurerwa/AgriSmart-TinyLlama-LoRA](https://huggingface.co/kellenmurerwa/AgriSmart-TinyLlama-LoRA) (Hugging Face)

## Dataset

**Source:** [KisanVaani/agriculture-qa-english-only](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) (Hugging Face)

| Property | Value |
|---|---|
| Total samples | 22,615 |
| Training set | 20,353 (90%) |
| Evaluation set | 2,262 (10%) |
| Format | Question-Answer pairs |
| Domain | Agriculture (crops, pests, soil, fertilizers) |

**Preprocessing Steps:**
1. Filtered null/empty entries
2. Formatted into instruction-response templates (`### Instruction: / ### Response:`)
3. Tokenized using TinyLlama's tokenizer (max 512 tokens)
4. Train/test split with seed=42 for reproducibility

## Fine-Tuning Methodology

**Base Model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (1.1B parameters)

**Approach:** Parameter-Efficient Fine-Tuning with LoRA via the `peft` library

| Configuration | Value |
|---|---|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj |
| Dropout | 0.05 |
| Quantization | 4-bit NF4 (double quant) |
| Trainable Parameters | ~2.25M (0.2% of total) |

## Experiment Results

Two hyperparameter experiments were conducted:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| **Learning Rate** | 2e-4 | 5e-5 |
| **Batch Size** | 2 | 4 |
| **Grad Accumulation** | 4 | 2 |
| **Effective Batch** | 8 | 8 |
| **Epochs** | 1 | 2 |
| **Total Steps** | 2,545 | 5,090 |
| **Final Loss** | 0.8724 | 0.8169 |
| **Final Accuracy** | 79.6% | 81.1% |

**Analysis:** Experiment 2 (lower learning rate, more epochs) achieved better convergence with lower loss and higher accuracy. The lower learning rate prevents overshooting during optimization, while additional epochs allow the model to learn more nuanced patterns.

## Evaluation Metrics

| Metric | Score |
|---|---|
| **BLEU** | 0.1810 |
| **ROUGE-1** | 0.5129 |
| **ROUGE-2** | 0.3268 |
| **ROUGE-L** | 0.4826 |
| **ROUGE-Lsum** | 0.4868 |
| **Perplexity** | 2.2583 |

The ROUGE-1 score of 0.51 indicates strong unigram overlap with reference answers, while the low perplexity of 2.26 shows the model predicts evaluation data well.

## Base Model vs Fine-Tuned Comparison

**Question:** *"How do I prevent cassava mosaic disease?"*

| Model | Response |
|---|---|
| **Base** | Cassava mosaic disease is a serious disease... Here are some preventive measures: 1. Proper crop management practices such as crop rotation, crop spacing... 2. Use of resistant cassava varieties... |
| **Fine-Tuned** | Cassava mosaic disease (CMD) is a major threat to cassava production worldwide. Here are some measures that can be taken to prevent CMD... |

The fine-tuned model provides more domain-specific terminology (e.g., "CMD" abbreviation) and more structured agricultural advice.

**Question:** *"What is the best fertilizer for rice crops?"*

| Model | Response |
|---|---|
| **Base** | The best fertilizer for rice crops is NPK. NPK stands for Nitrogen, Phosphorus, and Potassium. |
| **Fine-Tuned** | The best fertilizer for rice crops is a balanced mixture of nitrogen (N), phosphorus (P), and potassium (K), commonly called NPK fertilizers. These fertilizers provide the necessary nutrients for plant growth and improve the quality of the rice crop. |

The fine-tuned model provides more detailed and practical explanations.

## How to Run

### Option 1: Quick Test (Load Pre-trained Model)

The fastest way to test the chatbot — no training required (~2 minutes):

1. Open the notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kellenmurerwa/AgriSmart-Assistant/blob/main/AgriSmart_Assistant.ipynb)
2. Set runtime to **T4 GPU** (Runtime > Change runtime type)
3. Run **Section 2** (Install Dependencies) and **Section 3** (Import Libraries)
4. Run the **Quick Start** cell (loads the pre-trained adapter from [Hugging Face](https://huggingface.co/kellenmurerwa/AgriSmart-TinyLlama-LoRA))
5. Skip to **Section 11** (Gradio Deployment) to launch the chatbot

### Option 2: Full Training Pipeline (Google Colab)

Click the badge below to open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kellenmurerwa/AgriSmart-Assistant/blob/main/AgriSmart_Assistant.ipynb)

1. Click "Runtime" > "Change runtime type" > Select "T4 GPU"
2. Run all cells sequentially (training takes ~30-60 min on T4)
3. The Gradio chatbot will launch with a public URL

### Option 3: Local Setup

```bash
# Clone the repository
git clone https://github.com/kellenmurerwa/AgriSmart-Assistant.git
cd AgriSmart-Assistant

# Create a conda environment
conda create -n agrismart python=3.10 -y
conda activate agrismart

# Install PyTorch (adjust CUDA version for your GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install transformers datasets peft accelerate bitsandbytes trl evaluate gradio rouge_score nltk absl-py

# Run the notebook
jupyter notebook AgriSmart_Assistant.ipynb
```

**Requirements:** NVIDIA GPU with at least 6GB VRAM (tested on RTX 5070 8GB and Colab T4)

## Project Structure

```
AgriSmart-Assistant/
├── AgriSmart_Assistant.ipynb   # Complete pipeline notebook
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## Technologies Used

- **Model:** TinyLlama-1.1B-Chat-v1.0
- **Fine-Tuning:** LoRA via PEFT library
- **Training:** SFTTrainer from TRL
- **Quantization:** BitsAndBytes (4-bit NF4)
- **Evaluation:** BLEU, ROUGE, Perplexity (via `evaluate` library)
- **Deployment:** Gradio
- **Framework:** PyTorch + Hugging Face Transformers
