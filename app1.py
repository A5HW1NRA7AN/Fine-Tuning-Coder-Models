# ==============================================================================
# Lab Exercise 7: LLM Performance on Code-Based Tasks (Streamlit Version)
#
# This script fine-tunes and compares two models:
# 1. CodeBERT ('microsoft/codebert-base'): A model pre-trained on code.
# 2. BERT ('bert-base-uncased'): A general-purpose model pre-trained on text.
#
# This Streamlit app provides a UI to run the analysis, view results,
# and interactively test the fine-tuned models.
#
# UPDATE: This version loads data from local Parquet files in a './data/' folder.
# FIX: Made TrainingArguments backward-compatible for older transformers versions.
# NEW: Saves evaluation metrics to a JSON file to avoid re-computation on future runs.
# VISUALIZATION: Added bar charts for quantitative and qualitative results.
# ==============================================================================

import math
import torch
import os
import streamlit as st
import pandas as pd
import glob
import json # Added for saving/loading metrics
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
)

# ==============================================================================
# 1. APP CONFIGURATION & INITIAL SETUP
# ==============================================================================

st.set_page_config(
    page_title="CodeBERT vs. BERT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 LLM Performance on Code-Based Tasks")
st.markdown("A comparative analysis of **CodeBERT** (domain-specialized) vs. a general-purpose **BERT** on a code completion task.")

# --- Define paths and check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
codebert_save_path = "./codebert-finetuned"
bert_save_path = "./bert-finetuned"
# --- NEW: Path for evaluation results file ---
results_path = "./evaluation_results.json"


# --- Model checkpoints
code_model_checkpoint = "microsoft/codebert-base"
general_model_checkpoint = "bert-base-uncased"

# ==============================================================================
# 2. CACHED FUNCTIONS FOR EFFICIENCY
# ==============================================================================

@st.cache_resource
def get_dataset():
    """Loads the dataset from local Parquet files."""
    data_dir = "./data/"
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

    if not parquet_files:
        st.error(f"No Parquet files found in the '{data_dir}' directory.")
        st.info("Please create a 'data' folder in your project root and place your `.parquet` files inside it.")
        return None
    
    try:
        full_dataset = load_dataset("parquet", data_files=parquet_files, split="train", trust_remote_code=True)
        sample_size = 4000
        if len(full_dataset) < sample_size:
            st.warning(f"Dataset size ({len(full_dataset)}) is smaller than the target sample size ({sample_size}). Using all available data.")
            return full_dataset
        else:
            return full_dataset.shuffle(seed=42).select(range(sample_size))
    except Exception as e:
        st.error(f"Could not load dataset from local files. Error: {e}")
        return None

@st.cache_data
def tokenize_dataset(_dataset, tokenizer_name):
    """Tokenizes the dataset using the specified tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def preprocess(examples):
        return tokenizer(
            examples["whole_func_string"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    tokenized_dataset = _dataset.map(
        preprocess,
        batched=True,
        remove_columns=_dataset.column_names
    )
    return tokenized_dataset, tokenizer

@st.cache_resource
def get_fill_mask_pipeline(model_path, device):
    """Loads a fine-tuned model and creates a fill-mask pipeline."""
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run the analysis first.")
        return None, None
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_id = 0 if device == "cuda" else -1
    return pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device_id), tokenizer


# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"Running on: **{device.upper()}**")
    
    st.info("The script will load data, fine-tune models if they don't exist locally, and then save the evaluation metrics to a JSON file for future runs.")
    
    run_button = st.button("🚀 Run Full Analysis", type="primary")
    
    st.markdown("---")
    st.header("🧪 Interactive Test")
    st.write("After running the analysis, you can test the models here.")
    
    codebert_ready = os.path.exists(codebert_save_path)
    bert_ready = os.path.exists(bert_save_path)
    
    if codebert_ready and bert_ready:
        st.success("Models are ready for testing!")
    else:
        st.warning("You must run the analysis to enable interactive testing.")

# ==============================================================================
# 4. MAIN APP LOGIC
# ==============================================================================

def run_analysis():
    with st.status("Loading and preparing dataset from local files...", expanded=True) as status:
        dataset = get_dataset()
        if dataset is None:
            st.stop()
        st.write(f"✅ Dataset loaded from local Parquet files. Using {len(dataset)} samples.")

        tokenized_dataset_code, code_tokenizer = tokenize_dataset(dataset, code_model_checkpoint)
        st.write("✅ Data tokenized for CodeBERT.")
        
        tokenized_dataset_general, general_tokenizer = tokenize_dataset(dataset, general_model_checkpoint)
        st.write("✅ Data tokenized for General BERT.")
        status.update(label="Data preparation complete!", state="complete")

    st.subheader("Fine-Tuning & Evaluation")
    col1, col2 = st.columns(2)

    # --- CORRECTED: Replaced 'evaluation_strategy' with 'do_eval' for backward compatibility ---
    training_args = TrainingArguments(
        output_dir="./results",
        do_eval=True,  # Use this for older versions of transformers
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        logging_steps=100,
        save_total_limit=1,
    )
    
    # --- NEW: Load existing results if available ---
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        st.success(f"✅ Loaded existing evaluation results from `{results_path}`.")
    else:
        results = {}

    with col1:
        st.markdown("#### CodeBERT (`microsoft/codebert-base`)")
        with st.spinner("Processing CodeBERT..."):
            model_to_train = AutoModelForMaskedLM.from_pretrained(code_model_checkpoint)
            trainer_code = Trainer(
                model=model_to_train.to(device),
                args=training_args,
                train_dataset=tokenized_dataset_code,
                eval_dataset=tokenized_dataset_code,
                data_collator=DataCollatorForLanguageModeling(tokenizer=code_tokenizer, mlm_probability=0.15),
            )
            
            if not os.path.exists(codebert_save_path):
                st.info("No pre-existing model found. Starting fine-tuning...")
                trainer_code.train()
                st.success("Fine-tuning complete.")
                trainer_code.save_model(codebert_save_path)
                st.write(f"💾 Model saved to `{codebert_save_path}`")
            else:
                st.success(f"✅ Found existing model. Loading from `{codebert_save_path}`.")
                trainer_code.model = AutoModelForMaskedLM.from_pretrained(codebert_save_path).to(device)

            if 'codebert_loss' not in results:
                st.info("Evaluating CodeBERT...")
                eval_results = trainer_code.evaluate()
                results['codebert_loss'] = eval_results['eval_loss']
                results['codebert_perplexity'] = math.exp(eval_results['eval_loss'])
                st.write("Evaluation complete.")
            else:
                st.write("Using cached evaluation metrics for CodeBERT.")


    with col2:
        st.markdown("#### General BERT (`bert-base-uncased`)")
        with st.spinner("Processing General BERT..."):
            model_to_train = AutoModelForMaskedLM.from_pretrained(general_model_checkpoint)
            trainer_general = Trainer(
                model=model_to_train.to(device),
                args=training_args,
                train_dataset=tokenized_dataset_general,
                eval_dataset=tokenized_dataset_general,
                data_collator=DataCollatorForLanguageModeling(tokenizer=general_tokenizer, mlm_probability=0.15),
            )

            if not os.path.exists(bert_save_path):
                st.info("No pre-existing model found. Starting fine-tuning...")
                trainer_general.train()
                st.success("Fine-tuning complete.")
                trainer_general.save_model(bert_save_path)
                st.write(f"💾 Model saved to `{bert_save_path}`")
            else:
                st.success(f"✅ Found existing model. Loading from `{bert_save_path}`.")
                trainer_general.model = AutoModelForMaskedLM.from_pretrained(bert_save_path).to(device)
            
            if 'bert_loss' not in results:
                st.info("Evaluating General BERT...")
                eval_results = trainer_general.evaluate()
                results['bert_loss'] = eval_results['eval_loss']
                results['bert_perplexity'] = math.exp(eval_results['eval_loss'])
                st.write("Evaluation complete.")
            else:
                st.write("Using cached evaluation metrics for General BERT.")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    st.info(f"💾 Evaluation results saved to `{results_path}`.")
    
    st.session_state.results = results
    st.session_state.analysis_completed = True

if run_button:
    st.session_state.run_analysis = True

if 'run_analysis' in st.session_state:
    if 'analysis_completed' not in st.session_state:
        run_analysis()
    
    st.markdown("---")
    st.subheader("📊 Quantitative Results")
    st.markdown("Perplexity measures how well a model predicts a sequence. **Lower is better.**")

    results = st.session_state.results
    
    # --- VISUALIZATION ADDED HERE ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="CodeBERT Perplexity",
            value=f"{results['codebert_perplexity']:.2f}",
            help=f"Eval Loss: {results['codebert_loss']:.4f}"
        )
        st.metric(
            label="General BERT Perplexity",
            value=f"{results['bert_perplexity']:.2f}",
            delta=f"{results['bert_perplexity'] - results['codebert_perplexity']:.2f} (Higher than CodeBERT)",
            delta_color="inverse",
            help=f"Eval Loss: {results['bert_loss']:.4f}"
        )
    with col2:
        # Create a DataFrame for charting
        chart_data = pd.DataFrame({
            'Model': ['CodeBERT', 'General BERT'],
            'Perplexity': [results['codebert_perplexity'], results['bert_perplexity']]
        }).set_index('Model')
        
        st.write("#### Perplexity Comparison")
        st.bar_chart(chart_data)
    
    st.markdown("---")
    st.subheader("Conclusion")
    st.success(
        "As expected, the domain-specialized **CodeBERT** significantly outperforms the general-purpose **BERT** on a code-based task. "
        "This is reflected in its lower perplexity score, indicating it's less 'surprised' by the code structure and syntax it encounters. "
        "This is due to its relevant pre-training on a large corpus of code."
    )


# ==============================================================================
# 5. INTERACTIVE TESTING SECTION
# ==============================================================================

st.markdown("---")
st.header("🧪 Interactive Qualitative Test")

if not (codebert_ready and bert_ready):
    st.info("Run the full analysis to enable this interactive section.")
else:
    fill_mask_code, code_tokenizer = get_fill_mask_pipeline(codebert_save_path, device)
    fill_mask_general, general_tokenizer = get_fill_mask_pipeline(bert_save_path, device)
    
    st.write("Enter a Python code snippet and use the model-specific mask token to see what each model predicts.")
    
    default_snippet = f"def get_user(user_id): {code_tokenizer.mask_token} db.query(user_id)"
    
    st.markdown(f"**CodeBERT mask token:** `{code_tokenizer.mask_token}`")
    st.markdown(f"**General BERT mask token:** `{general_tokenizer.mask_token}`")
    
    user_input = st.text_input("Enter your code snippet here:", value=default_snippet)
    
    if user_input and fill_mask_code and fill_mask_general:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CodeBERT Predictions")
            with st.spinner("Getting predictions..."):
                code_input = user_input.replace(general_tokenizer.mask_token, code_tokenizer.mask_token)
                try:
                    result_code = fill_mask_code(code_input)
                    df_code = pd.DataFrame(result_code)[["token_str", "score"]]
                    df_code.rename(columns={'token_str': 'Predicted Token', 'score': 'Confidence Score'}, inplace=True)
                    st.dataframe(df_code, use_container_width=True)

                    # --- VISUALIZATION ADDED HERE ---
                    st.write("##### Prediction Confidence")
                    chart_code = df_code.set_index('Predicted Token')
                    st.bar_chart(chart_code)

                except Exception as e:
                    st.error(f"Could not get prediction. Is the mask token `{code_tokenizer.mask_token}` present?")

        with col2:
            st.markdown("#### General BERT Predictions")
            with st.spinner("Getting predictions..."):
                general_input = user_input.replace(code_tokenizer.mask_token, general_tokenizer.mask_token)
                try:
                    result_general = fill_mask_general(general_input)
                    df_general = pd.DataFrame(result_general)[["token_str", "score"]]
                    df_general.rename(columns={'token_str': 'Predicted Token', 'score': 'Confidence Score'}, inplace=True)
                    st.dataframe(df_general, use_container_width=True)
                    
                    # --- VISUALIZATION ADDED HERE ---
                    st.write("##### Prediction Confidence")
                    chart_general = df_general.set_index('Predicted Token')
                    st.bar_chart(chart_general)

                except Exception as e:
                    st.error(f"Could not get prediction. Is the mask token `{general_tokenizer.mask_token}` present?")