# HealthGuard Insurance - Medical Charges Prediction
# MiniProject02
## Noah Caney and Aakarsh Arora

## Business problem

HealthGuard Insurance prices its plans using outdated actuarial tables, which underprices high-risk
customers (the company loses money on them) and overprices low-risk customers (who leave for cheaper
competitors). The goal of this project is to use historical customer records to (a) understand what
drives medical charges, (b) predict a customer's annual charges so premiums can be set fairly, and
(c) flag whether a customer is likely to be "expensive."

## Data and cleaning decisions

The dataset contains 1,338 customer records with seven fields: `age`, `sex`, `bmi`, `children`,
`smoker`, `region`, and `charges`. Data quality was assessed with `head`, `shape`,
`info`, `describe`, and `value_counts`.

Cleaning decisions:

- **Duplicates:** one exact duplicate row was found and dropped. Removing a single identical record
  out of 1,338 does not meaningfully shrink the dataset or shift any distribution, so it is a
  low-risk cleanup.
- **Missing values:** `isna().sum()` returned zero for every column, so no imputation was necessary.
- **Encoding:** the categorical columns (`sex`, `smoker`, `region`) were one-hot encoded rather than
  label encoded, so that no false numeric ordering is introduced among unordered categories.
  `drop_first=True` was used to avoid redundant, perfectly-correlated dummy columns.

The raw dataframe was kept separate from the cleaned and encoded versions so the original data
remains traceable.

## Exploratory data analysis

Five visualizations were produced: the distribution of charges, charges by smoking status, charges vs.
age (colored by smoker), charges vs. BMI (colored by smoker), and average charges by region.

Key insights:

- Most insurance charges fall between roughly \$5,000 and \$15,000.
- Individuals who smoke generally incur higher charges than non-smokers.
- Among people of similar age and BMI, smokers have significantly higher charges.
- By region, the Southeast has the highest average charges and the Southwest the lowest.

The single strongest predictor of charges was smoking status, which is used as the input for the
Simple Linear Regression model below.

## Predicting charges

Seven models were trained on a single 80/20 train/test split and compared on MAE, MSE, RMSE, and R².
Distance-based and regularized models (SVR, Ridge, Lasso) used standardized features; the linear,
polynomial, and tree models did not require scaling.

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Decision Tree (depth 4) | 2,621 | 18,886,632 | 4,346 | 0.897 |
| Polynomial (degree 2) | 2,867 | 21,585,840 | 4,646 | 0.883 |
| Polynomial (degree 3) | 3,049 | 23,719,530 | 4,870 | 0.871 |
| Multiple Linear | 4,177 | 35,478,020 | 5,956 | 0.807 |
| Ridge (L2) | 4,177 | 35,478,960 | 5,956 | 0.807 |
| Lasso (L1) | 4,177 | 35,478,960 | 5,956 | 0.807 |
| Polynomial (degree 4) | 3,756 | 36,697,660 | 6,058 | 0.800 |
| SVR (linear) | 3,506 | 40,596,240 | 6,372 | 0.779 |
| SVR (rbf) | 3,422 | 54,554,240 | 7,386 | 0.703 |
| Simple Linear (smoker) | 5,831 | 60,039,300 | 7,749 | 0.673 |

**Recommendation.** The Decision Tree (max_depth = 4) is recommended. It achieved the highest accuracy
(R² ≈ 0.90, RMSE ≈ \$4,346, meaning predictions are typically within about \$4,300 of actual charges)
while remaining easy to interpret. The model breaks the data into a series of simple decisions based on
factors such as smoking status, BMI, age, and number of children, making it straightforward to follow
how it reaches a prediction and which variables matter most.

## Flagging expensive customers

The target was defined as `charges` above the median (\$9,386): 1 = expensive, 0 = not. This produces a
balanced (~50/50) target. The three classifiers of logistic regression, a decision tree and a random forest
were trained on an 80/20 stratified split and evaluated with a confusion matrix and precision/recall.

Results: Logistic Regression ≈ 0.91, Decision Tree ≈ 0.94, Random Forest ≈ 0.94.