# Project Report: Comparative Analysis of CodeBERT and BERT for Code-Based Tasks

---

## 1. Project Overview

This project provides a comprehensive analysis comparing the performance of a **domain-specialized Large Language Model (LLM)**, `microsoft/codebert-base`, against a **general-purpose LLM**, `bert-base-uncased`. The goal is to empirically demonstrate the advantage of using a model pre-trained on a specific domain (source code) for tasks within that domain.

The core task is **Masked Language Modeling (MLM)**, where the models are fine-tuned to predict masked-out tokens in Python code snippets. The project includes scripts for fine-tuning and evaluation, culminating in an interactive **Streamlit dashboard** that allows for both quantitative analysis and qualitative, hands-on testing.

---

## 2. Project Objectives

* **Fine-Tune Models:** To fine-tune both CodeBERT and a general BERT model on a dataset of Python functions.
* **Quantitative Evaluation:** To measure and compare the performance of the fine-tuned models using a standard industry metric, **Perplexity**.
* **Qualitative Analysis:** To build an interactive user interface where users can input code snippets and directly compare the predictive capabilities of each model.
* **Efficiency:** To implement caching mechanisms for fine-tuned models and evaluation results to avoid redundant computation on subsequent runs.
* **Visualization:** To present the comparison results clearly using visual aids like bar charts.

---

## 3. Technology Stack

* **Programming Language:** Python 3.x
* **Core Libraries:**
    * **PyTorch:** For building and training the neural network models.
    * **Hugging Face `transformers`:** For accessing pre-trained models (CodeBERT, BERT), tokenizers, and the high-level `Trainer` API.
    * **Hugging Face `datasets`:** For loading and processing the `code_search_net` dataset.
* **Web Framework:**
    * **Streamlit:** To create the interactive web-based dashboard for analysis and model testing.
* **Data Handling:**
    * **Pandas:** For data manipulation and presentation within the Streamlit app.

---

## 4. Project Architecture & File Structure

The project is organized into two main Python scripts, along with directories for storing data, models, and results.

```
/project-root
|
├── 📂 data/                     # (Optional) For storing local .parquet dataset files
│   └── train-00000-of-00001.parquet
|
├── 📂 codebert-finetuned/        # Saved fine-tuned CodeBERT model and tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
|
├── 📂 bert-finetuned/            # Saved fine-tuned BERT model and tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
|
├── 📜 finetune.py               # Standalone script for model fine-tuning and evaluation
├── 📜 app1.py                    # The main Streamlit application for the dashboard
├── 📜 evaluation_results.json    # Cached JSON file storing perplexity scores
└── 📜 requirements.txt          # Project dependencies
```

* **`finetune.py`**: A command-line script that handles the entire pipeline: data loading, preprocessing, model fine-tuning, and evaluation. It saves the fine-tuned models to disk.
* **`app1.py`**: A Streamlit application that provides a user-friendly GUI. It can trigger the full evaluation pipeline and includes an "Interactive Playground" to test the models in real-time.
* **`evaluation_results.json`**: Caches the final evaluation metrics (loss and perplexity) to speed up subsequent launches of the app.

---

## 5. Methodology

The project follows a systematic workflow from data preparation to interactive analysis.

### 5.1. Data Loading and Preprocessing
The dataset used is **`code_search_net`** (Python subset), which contains a large collection of Python functions.
1.  **Loading:** The dataset is loaded from the Hugging Face Hub. A small subset (`train[:1%]`) is used to ensure the process completes quickly for demonstration purposes.
2.  **Tokenization:** Each function in the dataset is tokenized separately for CodeBERT and BERT, as they use different tokenizers. The code is truncated or padded to a fixed length of 128 tokens.

### 5.2. Model Fine-Tuning
The core of the project is the fine-tuning process on the **Masked Language Modeling (MLM)** task.
* **Objective:** The model learns to predict a `[MASK]` token based on the surrounding context of the code.
* **Process:** The Hugging Face `Trainer` API is used to manage the training loop. Both models are trained for one epoch with a learning rate of `2e-5`.
* **Efficiency:** The script first checks if a fine-tuned model already exists in the `./codebert-finetuned` or `./bert-finetuned` directories. If so, it skips the training step and loads the existing model, saving significant time.

### 5.3. Evaluation

Performance is assessed both quantitatively and qualitatively.

#### Quantitative Evaluation
* **Metric:** The primary metric is **Perplexity**, which is a measure of how well a probability model predicts a sample. It is calculated as the exponential of the evaluation loss (`math.exp(eval_loss)`). **A lower perplexity score indicates a better model.**
* **Caching:** After evaluation, the perplexity scores are saved to `evaluation_results.json`. The Streamlit app reads this file first, avoiding the need to re-evaluate the models every time it runs.

#### Qualitative Analysis
* **Interactive Playground:** The Streamlit app features a text area where users can input their own Python code and place a `[MASK]` token.
* **Live Predictions:** The app uses a `fill-mask` pipeline to generate the top 5 most likely token predictions from both models, displaying them side-by-side for immediate comparison.

### 5.4. Visualization

To make the results intuitive, the Streamlit app incorporates **bar charts**:
1.  **Perplexity Comparison:** A chart visually compares the final perplexity scores of the two models.
2.  **Prediction Confidence:** In the interactive playground, bar charts show the confidence scores for the top predictions, making it easy to see how certain each model is about its suggestions.

---

## 6. How to Run the Project

### Step 1: Clone the Repository and Install Dependencies
```bash
git clone <your-repo-url>
cd <project-directory>
pip install -r requirements.txt
```

### Step 2: Fine-Tune the Models
Run the `finetune.py` script. This will download the dataset, fine-tune both models, and save them to their respective directories (`./codebert-finetuned` and `./bert-finetuned`).

```bash
python finetune.py
```
*(This step can take a while, especially on a CPU. It only needs to be run once.)*

### Step 3: Launch the Streamlit Dashboard
Once the models are fine-tuned and saved, you can launch the interactive application.

```bash
streamlit run app1.py
```
The application will open in your web browser. You can then run the evaluation and test the models interactively.

---

## 7. Results and Conclusion

The analysis consistently shows that **CodeBERT significantly outperforms the general-purpose BERT** on the code completion task, as evidenced by its much lower perplexity score.

**Conclusion:** The results validate the hypothesis that domain-specific pre-training is highly effective. CodeBERT's exposure to a vast corpus of source code during its initial training phase equips it with a nuanced understanding of code syntax, structure, and semantics, making it far more adept at code-related tasks than a model trained on general-purpose text. This project serves as a practical demonstration of the importance of choosing the right model for a specialized task.
