# Coffee Quality Predictor

A machine learning project that predicts the quality score of Arabica coffee samples (**Total Cup Points**) from their sensory ratings and physical attributes.

I built it as the final project for the Machine Learning course in my Software Development specialization. It covers the full workflow, from cleaning a messy real-world dataset to training and comparing regression models and explaining what they learned.

> The notebook's code comments and some plot labels are in Spanish.

## Dataset

- **Source:** [Coffee Quality Data (CQI)](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi) on Kaggle (`df_arabica_clean.csv`)
- **Size:** 207 Arabica samples, 41 original columns
- **Target:** `Total Cup Points`, the overall quality score given by certified Coffee Quality Institute graders (roughly 80–90 in this dataset)

The notebook downloads the dataset automatically with `kagglehub`.

## Workflow

### 1. Data cleaning and preprocessing
- Checked for duplicates and measured the share of missing values in each column
- Grouped inconsistent `Processing Method` labels into three categories: *Washed / Wet*, *Natural / Dry* and *Pulped natural / honey*
- Cleaned the free-text `Altitude` field: converted ranges such as `"1500-1800"` to their midpoint, fixed malformed entries by hand and filled the rest with the mean
- Engineered a `Coffee Age` feature (days between harvest year and certificate expiration)
- Dropped identifiers, contact fields, near-constant columns and columns that were mostly empty

### 2. Exploratory data analysis
- Distributions of the numeric features and the target
- Average score by country of origin
- Relationship between altitude, each sensory attribute and the final score
- Correlation matrix and correlation of each feature with the target

### 3. Feature engineering
- One-hot encoding for `Processing Method`
- Standardization of numeric features with `StandardScaler`

### 4. Modeling
Compared three tree-based ensemble regressors, each tuned with `GridSearchCV` (3-fold CV) over `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`:
- Random Forest
- Gradient Boosting
- Extra Trees

### 5. Explainability
- Permutation importance
- SHAP summary plots for all three models
- SHAP dependence plots for `Altitude`, `Quakers` and `Aroma`
- A small interactive widget (`ipywidgets`) that predicts a score from sliders for the main sensory inputs

## Results

Held-out test set (20% split), with errors reported on the original point scale:

| Model             | Test R² | Test MAE (points) | Best CV R² |
|-------------------|:-------:|:-----------------:|:----------:|
| Random Forest     | 0.95    | 0.22              | 0.960      |
| Gradient Boosting | 0.96    | 0.20              | 0.965      |
| Extra Trees       | **0.97**| **0.14**          | **0.969**  |

Extra Trees did best overall, with an average error of about 0.14 points.

**Main takeaway:** the sensory attributes (Aroma, Flavor, Aftertaste, Acidity, Balance, Overall) drive almost all of the prediction. Altitude, moisture, defects and processing method add comparatively little.

## Limitations and lessons learned

- **The high scores come with a caveat.** In the CQI grading protocol, `Total Cup Points` is mostly the *sum* of the sensory scores the models use as inputs. The models are largely learning that sum, which explains the high R². They are not finding hidden patterns. A harder and more interesting version of the problem would predict quality from non-sensory features only (origin, altitude, processing, moisture, defects).
- **Small dataset.** 207 samples is not much, and the gap between training and test scores shows some overfitting, especially for Gradient Boosting.
- **Preprocessing before the split.** The scaler and imputer were fit on the full dataset before the train/test split. Putting them inside the scikit-learn `Pipeline` would remove that small data leak.
- **Subjective labels.** The scores come from human cuppers, so they carry some subjectivity and grader-to-grader variation.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · SHAP · ipywidgets · kagglehub

## Running it

The notebook was developed in Google Colab. The easiest way to run it is to open it there. To run it locally instead:

```bash
pip install -r requirements.txt
jupyter notebook EDA_Coffe.ipynb
```

The dataset is public, so `kagglehub` downloads it on first run without a Kaggle account.
