"""Step 1: compare feature sets with a time-based split.

Train on Jan-Aug, test on Sep-Oct (the model never sees the future).
Run:  py train_step1.py      (needs:  pip install lightgbm)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import clean, build_features

# ------------------------------------------------------------------ load + clean
train_raw = pd.read_csv("data/train-test.csv", parse_dates=["date"])
val_raw = pd.read_csv("data/validation.csv", parse_dates=["date"])

train, medians = clean(train_raw)
val, _ = clean(val_raw, medians)

X = build_features(train)
y = np.log(train["posted_rate"])          # predict log(rate): errors become percentages

# Daily market level = average over ALL loads that day (features only, no labels,
# so it is fine to use train + validation rows).
daily = pd.concat([train, val]).groupby("date")[["market_index", "quote_signal"]].mean()
X["day_market_index"] = train["date"].map(daily["market_index"]).values
X["day_quote_signal"] = train["date"].map(daily["quote_signal"]).values
X["distance_is_70"] = (train["distance"] == 70).astype(int)   # the floor value we found

# ------------------------------------------------------------------ time split
is_train = train["date"] < "2025-09-01"
is_test = ~is_train
print(f"train rows: {is_train.sum()}  test rows: {is_test.sum()}")

rpm = train["posted_rate"] / train["distance"]
typical = (rpm > 1.2) & (rpm < 5)         # rows without extreme-rate outliers


def report(name, pred):
    """pred = predicted dollars for the test rows."""
    actual = train.loc[is_test, "posted_rate"].values
    ape = np.abs(actual - pred) / actual
    t = typical[is_test].values
    print(f"{name:34s} MAPE {100*ape.mean():5.2f}%  MedAPE {100*np.median(ape):5.2f}%  "
          f"MAE ${np.abs(actual-pred).mean():6.0f} | typical-rows MAPE {100*ape[t].mean():5.2f}%")


# ------------------------------------------------------------------ benchmark
bands = [0, 150, 300, 600, 1000, 2000, 4000]
band = pd.cut(train["distance"], bands)
med_rpm = rpm[is_train].groupby([train.loc[is_train, "equipment"], band[is_train]], observed=True).median()
bench = [med_rpm[(e, b)] for e, b in zip(train.loc[is_test, "equipment"], band[is_test])]
report("Benchmark (median $/mile table)", np.array(bench) * train.loc[is_test, "distance"].values)

# ------------------------------------------------------------------ model
PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=15,
              min_child_samples=60, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=0, verbose=-1)

geo = ["haversine", "dist_ratio", "pickup_lat", "pickup_lon", "delivery_lat",
       "delivery_lon", "d_lat", "d_lon"]
core = ["distance", "log_distance", "weight", "equipment", "market_index", "quote_signal", "dow"]
daily_cols = ["day_market_index", "day_quote_signal"]

FEATURE_SETS = {
    "A: your original features":       core + geo,
    "B: without geography":            core,
    "C: original + daily averages":    core + geo + daily_cols,
    "D: no geography + daily avgs":    core + daily_cols,
    "E: D + distance_is_70 flag":      core + daily_cols + ["distance_is_70"],
}

for name, cols in FEATURE_SETS.items():
    model = lgb.LGBMRegressor(**PARAMS).fit(X.loc[is_train, cols], y[is_train])
    pred = np.exp(model.predict(X.loc[is_test, cols]))
    report(name, pred)
