# HealthGuard Insurance - Medical Charges Prediction
# MiniProject02
## Noah Caney and Aakarsh Arora

## What this project does

HealthGuard Insurance currently prices plans using outdated actuarial tables, which underprices
high-risk customers and overprices low-risk ones. This project uses the company's historical
customer records to (1) explore what drives medical charges, (2) build and compare regression
models that predict a customer's annual charges, and (3) classify customers as "expensive" or not
relative to the median charge.

## Dataset

Kaggle - *Medical Cost Personal Datasets* (`mirichoi0218/insurance`): 1,338 rows, 7 columns
(`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).

The notebook downloads the data directly from Kaggle using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("mirichoi0218/insurance")
```

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Set up your Kaggle credentials so the notebook can download the dataset:
   - Log in to Kaggle, go to **Account -> Settings**, and click **Create New Token**.
     This downloads a `kaggle.json` file containing your username and API key.
   - In Colab, upload `kaggle.json` when prompted, or set the environment variables
     `KAGGLE_USERNAME` and `KAGGLE_KEY` to the values from that file.
3. Open `MiniProject_02_Insurance.ipynb` and run all cells top to
   bottom. The first data cell uses `kagglehub` to download the `mirichoi0218/insurance`
   dataset automatically.

The notebook is organized as: load & profile -> clean -> EDA (5 plots) -> seven regression models
with a summary table -> classification of expensive customers.

## Summary of findings

- **Smoking is the dominant driver of medical charges.** Most charges fall between roughly \$5,000
  and \$15,000; smokers incur substantially higher charges than non-smokers at similar ages and BMIs.
  By region, the Southeast has the highest average charges and the Southwest the lowest.
- **Best regression model: Decision Tree (max_depth = 4)**, with the highest accuracy (R² ≈ 0.90,
  RMSE ≈ \$4,346) while remaining easy to interpret. It makes the reasoning behind each prediction easy to follow.
- **Classification (expensive = charges above the median of \$9,386)**