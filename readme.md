# Freight Rate Prediction

Predicts the posted rate (USD) of a freight load from its distance, weight, equipment type,
day of week and market signals. Built for the Machine Learning Engineer assessment.

## Results

Time-based validation: train on Jan-Aug 2025, test on Sep-Oct 2025 (the model never sees the future).

| Model / feature set | MAPE | Median APE | MAE ($) | MAPE excl. outliers |
|---|---|---|---|---|
| Benchmark: median $/mile table | 6.36% | 3.36% | 140 | 3.89% |
| A: distance, weight, equipment, signals, geography | 6.26% | 3.74% | 147 | 3.90% |
| B: A without geography | 5.86% | 3.15% | 139 | 3.47% |
| C: A + daily market averages | 5.39% | 2.79% | 125 | 2.98% |
| **D: B + daily market averages (final)** | **5.27%** | **2.49%** | **122** | **2.85%** |

"Outliers" = loads with a rate under $1.20 or over $5.00 per mile (about 1.4% of loads).

## Project files

| File | What it does |
|---|---|
| `features.py` | Cleans the data and builds the model features. Run it alone for a before/after cleaning report. |
| `checks.py` | Data exploration: distance checks, rate outliers, day-of-week effect, market signals. |
| `train_step1.py` | Compares feature sets using the time-based split. |
| `predict_step2.py` | Trains the final model on all labelled data and writes both submission files. |
| `score.py` | Provided scorer: validates the output files and draws the December chart. |

## How to run

```bash
python -m pip install -r requirements.txt
```

Put the four provided CSV files in a `data/` folder with these names:

```
data/train-test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

Then run:

```bash
python train_step1.py        # validation results (feature comparison)
python predict_step2.py      # writes validation_predictions.csv and december_chart_inputs_completed.csv
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_completed.csv
```

The scorer checks both files and saves the chart to `scorer_results/candidate_december.png`.

## Approach

**Split.** Training data covers Jan-Oct 2025 and the loads to predict cover Nov-Dec, so a random
split would be misleading. I validate on the last two months of the labelled data instead.

**Data-quality fixes** (`features.py`)
- Negative weights (292 train, 145 validation) are sign errors, so I use the absolute value.
- Missing weights (300 train, 165 validation) are filled with the median for that equipment type,
  computed on training data only.
- Missing market_index (374 train, 249 validation) is filled with that day's median across other loads.
- About 1.4% of loads have extreme rates per mile. No feature predicts them, so I keep them and
  use an L1 loss that is robust to them.
- Distance has a floor at exactly 70 miles. A flag for it did not help, so it is not used.
- Eight validation cities never appear in training, and coordinate features reduced accuracy,
  so the final model uses no geography.

**Features.** Distance, log distance, weight, equipment, day of week, the row's market_index and
quote_signal, plus the daily average of both across all loads that day. The daily average is a
much cleaner read of the market than a single noisy row and gave the largest gain.

**Model.** LightGBM predicting log(rate) with an L1 loss, averaged over 5 random seeds.

**December chart.** The December file has no market_index or quote_signal, so each day uses the
daily average of those signals from `validation.csv`, which covers Nov-Dec.

## Limitations

- About 1.4% of loads have extreme rates that no feature explains.
- The lead over the simple benchmark is modest because most of the signal is distance.
- December predictions depend on the market signals in `validation.csv`.
- Validation uses a single time split.
