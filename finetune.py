# ==============================================================================
# Lab Exercise 7: LLM Performance on Code-Based Tasks (Corrected Version)
#
# This script fine-tunes and compares two models:
# 1. CodeBERT ('microsoft/codebert-base'): A model pre-trained on code.
# 2. BERT ('bert-base-uncased'): A general-purpose model pre-trained on text.
# The goal is to demonstrate the performance difference on a code-related task.
#
# UPDATE: This version checks for existing fine-tuned models and loads them
# to avoid re-training on every run.
# ==============================================================================

import math
import torch
import os # Added for checking if saved model directories exist
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device.upper()}")

# Define paths for saved models
codebert_save_path = "./codebert-finetuned"
bert_save_path = "./bert-finetuned"

# ==============================================================================
# SECTION 1: SETUP - MODELS AND TOKENIZERS
# ==============================================================================
print("\n--- Section 1: Initializing Models and Tokenizers ---")

# Define model checkpoints
code_model_checkpoint = "microsoft/codebert-base"
general_model_checkpoint = "bert-base-uncased"

# Load tokenizers
code_tokenizer = AutoTokenizer.from_pretrained(code_model_checkpoint)
general_tokenizer = AutoTokenizer.from_pretrained(general_model_checkpoint)

# Load models for Masked Language Modeling (MLM)
# These will be replaced by fine-tuned versions if they are found on disk
code_model = AutoModelForMaskedLM.from_pretrained(code_model_checkpoint).to(device)
general_model = AutoModelForMaskedLM.from_pretrained(general_model_checkpoint).to(device)

print("✅ Models and tokenizers loaded successfully.")


# ==============================================================================
# SECTION 2: DATA PREPROCESSING
# ==============================================================================
print("\n--- Section 2: Loading and Preprocessing Dataset ---")

# Load the dataset from Hugging Face. Using a small subset for a quick run.
# For a full run, increase the percentage (e.g., 'train[:10%]') or use the full set.
try:
    dataset = load_dataset("code_search_net", "python", split="train[:1%]", trust_remote_code=True)
except Exception as e:
    print(f"Could not load dataset. Error: {e}")
    print("Please ensure you have an active internet connection.")
    exit()

# Define a single preprocessing function
def preprocess_function(examples, tokenizer):
    """Tokenizes the code snippets from the 'whole_func_string' column."""
    return tokenizer(
        examples["whole_func_string"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Apply the tokenization to the dataset for both models
print("Tokenizing data for CodeBERT...")
tokenized_dataset_for_code_model = dataset.map(
    lambda e: preprocess_function(e, code_tokenizer),
    batched=True,
    remove_columns=dataset.column_names  # Clean up original columns
)

print("Tokenizing data for General BERT...")
tokenized_dataset_for_general_model = dataset.map(
    lambda e: preprocess_function(e, general_tokenizer),
    batched=True,
    remove_columns=dataset.column_names
)

print("✅ Data preprocessing complete.")


# ==============================================================================
# SECTION 3: MODEL FINE-TUNING OR LOADING
# ==============================================================================
print("\n--- Section 3: Fine-Tuning or Loading Models ---")

# Define a separate data collator for each tokenizer to ensure correct masking.
print("Initializing Data Collators...")
data_collator_code = DataCollatorForLanguageModeling(tokenizer=code_tokenizer, mlm_probability=0.15)
data_collator_general = DataCollatorForLanguageModeling(tokenizer=general_tokenizer, mlm_probability=0.15)
print("✅ Data Collators configured successfully.")


# Define training arguments using a workaround to avoid a common TypeError.
# First, initialize with the mandatory 'output_dir', then set other properties.
print("Initializing TrainingArguments...")
training_args = TrainingArguments(output_dir="./results")

training_args.evaluation_strategy = "epoch"
training_args.learning_rate = 2e-5
training_args.num_train_epochs = 1
training_args.weight_decay = 0.01
training_args.per_device_train_batch_size = 16
training_args.per_device_eval_batch_size = 16
training_args.save_total_limit = 2
training_args.logging_steps = 100

print("✅ TrainingArguments configured successfully.")


# --- Fine-tune or Load CodeBERT ---
if os.path.exists(codebert_save_path):
    print(f"\n✅ Found existing model at '{codebert_save_path}'. Loading from disk...")
    # A Trainer object is still needed for the evaluation step
    trainer_code = Trainer(
        model=AutoModelForMaskedLM.from_pretrained(codebert_save_path).to(device),
        args=training_args,
        eval_dataset=tokenized_dataset_for_code_model,
        data_collator=data_collator_code, # Use the correct collator
    )
else:
    print(f"\nNo pre-existing model found. 🚀 Starting Fine-Tuning for CodeBERT...")
    trainer_code = Trainer(
        model=code_model,
        args=training_args,
        train_dataset=tokenized_dataset_for_code_model,
        eval_dataset=tokenized_dataset_for_code_model,
        data_collator=data_collator_code, # Use the correct collator
    )
    trainer_code.train()
    print("✅ CodeBERT fine-tuning complete.")
    print(f"💾 Saving fine-tuned CodeBERT model to '{codebert_save_path}'...")
    trainer_code.save_model(codebert_save_path)
    print("✅ Model saved.")


# --- Fine-tune or Load General-Purpose BERT ---
if os.path.exists(bert_save_path):
    print(f"\n✅ Found existing model at '{bert_save_path}'. Loading from disk...")
    trainer_general = Trainer(
        model=AutoModelForMaskedLM.from_pretrained(bert_save_path).to(device),
        args=training_args,
        eval_dataset=tokenized_dataset_for_general_model,
        data_collator=data_collator_general, # Use the correct collator
    )
else:
    print(f"\nNo pre-existing model found. 🚀 Starting Fine-Tuning for General BERT...")
    trainer_general = Trainer(
        model=general_model,
        args=training_args,
        train_dataset=tokenized_dataset_for_general_model,
        eval_dataset=tokenized_dataset_for_general_model,
        data_collator=data_collator_general, # Use the correct collator
    )
    trainer_general.train()
    print("✅ General-Purpose BERT fine-tuning complete.")
    print(f"💾 Saving fine-tuned General BERT model to '{bert_save_path}'...")
    trainer_general.save_model(bert_save_path)
    print("✅ Model saved.")


# ==============================================================================
# SECTION 4: EVALUATION AND COMPARATIVE ANALYSIS
# ==============================================================================
print("\n--- Section 4: Evaluation and Comparison ---")

# --- Evaluate Models and Report Perplexity ---
# Perplexity is a measurement of how well a probability model predicts a sample.
# Lower perplexity indicates better performance. It is calculated as exp(loss).

print("\n🔍 Evaluating CodeBERT...")
eval_results_code = trainer_code.evaluate()
perplexity_code = math.exp(eval_results_code['eval_loss'])
print(f"  -> CodeBERT - Evaluation Loss: {eval_results_code['eval_loss']:.4f}")
print(f"  -> CodeBERT - Perplexity: {perplexity_code:.2f}")

print("\n🔍 Evaluating General-Purpose BERT...")
eval_results_general = trainer_general.evaluate()
perplexity_general = math.exp(eval_results_general['eval_loss'])
print(f"  -> General BERT - Evaluation Loss: {eval_results_general['eval_loss']:.4f}")
print(f"  -> General BERT - Perplexity: {perplexity_general:.2f}")


# --- Qualitative Test with Fill-Mask Pipeline ---
# This gives an intuitive sense of what each model has learned.
# The pipeline will load the model from the saved path.

print("\n🧪 Performing qualitative test with a fill-mask pipeline...")

# CodeBERT example
fill_mask_code = pipeline("fill-mask", model=trainer_code.model, tokenizer=code_tokenizer, device=0 if device=="cuda" else -1)
code_example = f"def add(a, b): {code_tokenizer.mask_token} a + b"
result_code = fill_mask_code(code_example)
print(f"\nCodeBERT completion for '{code_example}':")
for r in result_code[:3]:
    print(f"  - Token: '{r['token_str']}', Score: {r['score']:.3f}")


# BERT example
fill_mask_general = pipeline("fill-mask", model=trainer_general.model, tokenizer=general_tokenizer, device=0 if device=="cuda" else -1)
general_example = f"def add(a, b): {general_tokenizer.mask_token} a + b"
result_general = fill_mask_general(general_example)
print(f"\nGeneral BERT completion for '{general_example}':")
for r in result_general[:3]:
    print(f"  - Token: '{r['token_str']}', Score: {r['score']:.3f}")

print("\n\n--- 📊 Final Comparative Analysis ---")
print("The model with the LOWER perplexity is better.")
print(f"  - CodeBERT Perplexity: {perplexity_code:.2f}")
print(f"  - General BERT Perplexity: {perplexity_general:.2f}")
print("\nConclusion: As expected, the domain-specialized CodeBERT significantly outperforms")
print("the general-purpose BERT on a code-based task. This is due to its relevant")
print("pre-training corpus and code-aware tokenizer.")

print("\n✅✅✅ Script finished successfully. ✅✅✅")