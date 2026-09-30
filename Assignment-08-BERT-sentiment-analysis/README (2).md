# Assignment 8 — Pre-trained BERT for Sentiment Analysis

## Practical Details
- **Practical No.:** 8
- **Subject:** Deep Learning
- **Department:** CSE-AI
- **Semester:** 5
- **Academic Year:** 2026-27
- **Division:** E
- **Batch:** 1
- **Roll No.:** 64
- **PRN:** 12412320
- **Student:** Samyak Shende

## Problem Statement
Implement a pre-trained BERT model for sentiment analysis or text classification on a sample dataset.

## Files
- `CSAIE_64_DL_Assignment 8.docx` — practical implementation sheet
- `Assignment_7_BERT_SST2.ipynb` — BERT implementation notebook
- `DL_Assignment8_Samyak_Shende.pdf` — formatted report
- `README.md` — assignment overview

## Implementation
The notebook:
1. Installs `transformers`, `datasets`, and `accelerate`.
2. Loads `textattack/bert-base-uncased-SST-2`.
3. Uses the SST-2 validation split from GLUE.
4. Samples up to 500 validation examples with seed 42.
5. Performs batched tokenization and inference.
6. Calculates accuracy, precision, recall, and F1-score.
7. Generates a BERT confusion matrix.
8. Reviews the ten highest-confidence misclassifications.
9. Performs inference on a custom sentence.

## Model
**Model:** `textattack/bert-base-uncased-SST-2`

The tokenizer uses padding, truncation, and a maximum sequence length of 128 tokens. Predictions are generated in batches of 32.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- High-confidence misclassification analysis

## Results
The practical sheet contains an empty Output section, so this repository does not fabricate numerical results. Execute the notebook to generate the actual metric values, confusion matrix, error analysis, and custom inference result.

## Conclusion
The experiment demonstrates how a pretrained BERT sequence-classification model can be used for sentiment analysis with minimal task-specific code. The workflow includes reproducible sampling, batched inference, evaluation, visualization, error analysis, and custom text prediction.
