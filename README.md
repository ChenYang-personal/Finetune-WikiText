# Fine-Tuning a Language Model on WikiText

This repository contains the complete code for **data preparation**, **fine-tuning**, and **evaluation** of a pre-trained language model (DistilGPT2) on the **WikiText** dataset.  
The workflow is implemented in the `finetuneLLM.ipynb` notebook.

---

## 📂 Project Structure

- `finetuneLLM.ipynb`: Main Jupyter Notebook for data loading, model fine-tuning, and evaluation.
- `requirements.txt`: (to be generated) List of required Python packages.

---

## 📋 Requirements

- Python 3.8+
- Jupyter Notebook
- PyTorch
- Hugging Face `transformers`
- Hugging Face `datasets`
- Hugging Face `evaluate`
- `accelerate`
- `math`
- `torchvision` (optional if GPU-based torch setup)

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not generated yet, install manually:

```bash
pip install torch torchvision
pip install transformers datasets evaluate accelerate
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **(Optional) Create and Activate a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

3. **Install Required Packages**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```
Then open `finetuneLLM.ipynb`.

---

## 🚀 How to Run

1. Open the `finetuneLLM.ipynb` file in Jupyter Notebook.
2. Run all the cells sequentially:
   - **Load WikiText dataset**
   - **Tokenize and prepare data blocks**
   - **Fine-tune the DistilGPT2 model**
   - **Evaluate model performance (Perplexity)**
3. At the end, the notebook reports the fine-tuned model’s performance using **Perplexity**.

---

## 📈 Results

- The fine-tuned model's **Perplexity** is computed and printed after training.
- Lower perplexity indicates better language modeling on the WikiText dataset.

---

## 📎 Notes

- Fine-tuning uses **mixed-precision (FP16)** if a GPU is available.
- Training is configured with **5 epochs** and **batch size 4**.

---

## 🛠 Future Improvements

- Add model saving/loading after fine-tuning.
- Perform hyperparameter tuning (batch size, learning rate).
- Extend evaluation with BLEU, ROUGE, or downstream tasks.

---

## 📄 Example `requirements.txt`

You should also create a `requirements.txt` file containing:

```
torch
torchvision
transformers
datasets
evaluate
accelerate
```

---
