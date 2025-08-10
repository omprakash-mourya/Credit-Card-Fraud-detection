# Dataset Directory

This directory should contain the credit card fraud dataset.

## Required File

Please download the dataset from Kaggle and place it here:

**File:** `creditcard.csv`  
**Source:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Size:** ~150 MB  
**Format:** CSV with 284,807 rows and 31 columns  

## Download Instructions

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click "Download" (requires Kaggle account)
3. Extract `creditcard.csv` from the ZIP file
4. Place `creditcard.csv` in this `data/` directory

## Dataset Structure

The dataset contains the following columns:
- `Time`: Number of seconds elapsed between transactions
- `V1` to `V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = Normal, 1 = Fraud)

## Expected Location

```
data/
├── README.md          (this file)
└── creditcard.csv     (download and place here)
```

## Note

The dataset file (`creditcard.csv`) is not included in this repository due to its size and licensing. You must download it separately from Kaggle.
