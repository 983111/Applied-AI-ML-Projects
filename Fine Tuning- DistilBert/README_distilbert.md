# DistilBERT for IT Support Ticket Classification (Real Data)

Fine-tunes `distilbert-base-uncased` to route IT-support questions into **five support queues**. The training data is **real, human-written text** from Stack Exchange, not synthetic or LLM-generated.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/distilbert_it_support_classification.ipynb)

> Replace `<your-username>` and `<your-repo>` after pushing to GitHub.

## Classes

| Class | Source community | Represents |
|-------|------------------|------------|
| Linux | Ask Ubuntu | Linux desktop / OS issues |
| Apple | Ask Different | macOS / iOS issues |
| Android | Android Enthusiasts | Android phone issues |
| Server & Network | Server Fault | Sysadmin, servers, networking |
| Security | Information Security | Security incidents and hardening |

## Dataset

- **Source:** [`flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl`](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl), which holds Stack Exchange question and answer pairs split by community. Only the `title_body` field (question title plus body) is used.
- **Size used:** 4,000 questions per class (20,000 total), sampled with shuffling, de-duplicated, minimum length 40 characters.
- **Split:** stratified 80 / 10 / 10 (train / validation / test).

### Important limitation

Real labelled IT *tickets* are almost never public, and public "IT ticket" datasets are generally synthetic. This project uses real support questions as a proxy. **The label is the community a question was posted in, not a hand-annotated ticket category.** Some questions legitimately belong to more than one class (for example a VPN problem on macOS). Report this when you present results.

### Licence

Stack Exchange content is community-contributed and licensed under CC BY-SA, and the Hugging Face dataset card lists CC BY-NC-SA 4.0. Keep attribution and check the terms before any commercial use.

## Method

1. Load, clean and de-duplicate questions (streamed, so no giant download).
2. Stratified split.
3. **Baseline:** TF-IDF + Logistic Regression, plus a variant with brand words ("Ubuntu", "iPhone", "Android"...) masked to show how much the task relies on obvious keywords.
4. **Fine-tune DistilBERT** with the Hugging Face `Trainer`.
5. Evaluate with accuracy, macro-F1, per-class report, confusion matrix and a list of the most confident mistakes.
6. Inference demo on hand-written tickets and model export.

### Hyper-parameters

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-uncased` |
| Max sequence length | 256 |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Warm-up | 10% |
| Weight decay | 0.01 |
| Precision | fp16 on GPU |
| Model selection | best validation macro-F1 |

All are set in the configuration cell at the top of the notebook.

## How to Run

1. Open the notebook in Google Colab and set **Runtime → Change runtime type → T4 GPU**.
2. Run **Runtime → Run all**. Expect roughly 10 to 20 minutes on a T4.
3. Fill in the results table below and add your confusion matrix screenshot.

## Results

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression | _your result_ | _your result_ |
| TF-IDF, brand words masked | _your result_ | _your result_ |
| DistilBERT (fine-tuned) | _your result_ | _your result_ |

Add `![confusion matrix](images/confusion_matrix.png)` and a short note on which classes are confused most.

## Troubleshooting

- **`Could not load split ...`:** the dataset layout on Hugging Face may have changed. Open the dataset's *Files and versions* tab, check the split or file names, and adjust `load_site()`. If the dataset needs a loading script, install an older library with `pip install "datasets<4"` and restart the runtime.
- **`eval_strategy` or `processing_class` errors:** your `transformers` version is old. Run `pip install -U transformers` and restart the runtime.
- **Out of GPU memory:** lower `BATCH_SIZE` to 16 or `MAX_LEN` to 128.
- **Too slow:** lower `PER_CLASS` to 2000 or `EPOCHS` to 2.

## Extending

- Add a Windows class (`superuser` community) or other communities by editing `CLASSES`.
- Compare against `bert-base-uncased` or `roberta-base`.
- Add a confidence threshold with a "needs human triage" fallback.
- Evaluate on real tickets from your own organisation, which is the true test of a ticket router.

## Author

**Vishwajeet Adkine** — [Portfolio](https://prt-delta.vercel.app/)
