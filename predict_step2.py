"""Step 2: train the final model (feature set D) on ALL labelled data and write the
two files you submit.

Run:  py predict_step2.py
Creates: validation_predictions.csv  and  december_chart_inputs_completed.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import clean, build_features

# ------------------------------------------------------------------ load + clean
train_raw = pd.read_csv("data/train-test.csv", parse_dates=["date"])
val_raw = pd.read_csv("data/validation.csv", parse_dates=["date"])
template = pd.read_csv("data/validation-predictions-template.csv")
dec = pd.read_csv("data/december-chart-inputs.csv", parse_dates=["date"])

train, medians = clean(train_raw)
val, _ = clean(val_raw, medians)

# Daily market level: average over all loads that day (features only, no labels).
daily = pd.concat([train, val]).groupby("date")[["market_index", "quote_signal"]].mean()

# ------------------------------------------------------------------ features (set D)
FEATURES = ["distance", "log_distance", "weight", "equipment", "market_index",
            "quote_signal", "dow", "day_market_index", "day_quote_signal"]


def make_features(df):
    X = build_features(df)
    X["day_market_index"] = df["date"].map(daily["market_index"]).values
    X["day_quote_signal"] = df["date"].map(daily["quote_signal"]).values
    return X[FEATURES]


X_train = make_features(train)
y_train = np.log(train["posted_rate"])

# ------------------------------------------------------------------ train (5 seeds, averaged)
PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=15,
              min_child_samples=60, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1)
models = [lgb.LGBMRegressor(**PARAMS, random_state=s).fit(X_train, y_train) for s in range(5)]


def predict(X):
    return np.exp(np.mean([m.predict(X) for m in models], axis=0))


# ------------------------------------------------------------------ 1) the 12,000 validation loads
val["predicted_rate"] = predict(make_features(val)).round(2)
out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
assert len(out) == 12000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
out.to_csv("validation_predictions.csv", index=False)
print("Wrote validation_predictions.csv:", out.shape)

# ------------------------------------------------------------------ 2) the 31 fixed December rows
d = dec.drop(columns="predicted_rate").copy()
d["market_index"] = d["date"].map(daily["market_index"])   # Dec values come from validation.csv
d["quote_signal"] = d["date"].map(daily["quote_signal"])
for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
    d[c] = np.nan                                          # not used by this model
d, _ = clean(d, medians)
dec["predicted_rate"] = predict(make_features(d)).round(2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec.to_csv("december_chart_inputs_completed.csv", index=False)
print("Wrote december_chart_inputs_completed.csv:", dec.shape)
print(dec[["date", "predicted_rate"]].describe().round(1).to_string())
