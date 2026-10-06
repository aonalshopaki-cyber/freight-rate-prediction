"""Cleaning and feature engineering, shared by training and prediction.

Run directly (python features.py) to see a before/after cleaning report.
"""
import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in miles between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean(df, weight_medians=None):
    """Fix data-quality issues. Uses only feature columns (never the target).

    weight_medians: median weight per equipment type. Compute it from the
    training data and pass it in when cleaning other data.
    Returns (cleaned_df, weight_medians).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Negative weights are sign errors: take the absolute value.
    df["weight"] = df["weight"].abs()

    # 2. Missing weights: fill with the median for that equipment type.
    if weight_medians is None:
        weight_medians = df.groupby("equipment")["weight"].median()
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))

    # 3. Missing market_index: it is a daily level, so use that day's median
    #    across the other loads. If a whole day is missing, interpolate.
    daily = df.groupby("date")["market_index"].median()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    daily = daily.interpolate(limit_direction="both")
    df["market_index"] = df["market_index"].fillna(df["date"].map(daily))

    return df, weight_medians


def build_features(df):
    """Turn a cleaned dataframe into the model's feature matrix."""
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    hav = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["haversine"] = hav
    X["dist_ratio"] = (df["distance"] / hav.clip(lower=1)).clip(upper=3)
    X["weight"] = df["weight"]
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    X["pickup_lat"] = df["pickup_lat"]
    X["pickup_lon"] = df["pickup_lon"]
    X["delivery_lat"] = df["delivery_lat"]
    X["delivery_lon"] = df["delivery_lon"]
    X["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
    X["d_lon"] = df["delivery_lon"] - df["pickup_lon"]
    X["market_index"] = df["market_index"]
    X["quote_signal"] = df["quote_signal"]
    X["dow"] = df["date"].dt.dayofweek
    return X


if __name__ == "__main__":
    train = pd.read_csv("data/train-test.csv")
    val = pd.read_csv("data/validation.csv")

    print("BEFORE cleaning")
    for name, d in (("train", train), ("val", val)):
        print(f"  {name}: negative weights={int((d.weight < 0).sum())}, "
              f"missing weights={int(d.weight.isna().sum())}, "
              f"missing market_index={int(d.market_index.isna().sum())}")

    train_c, medians = clean(train)
    val_c, _ = clean(val, medians)   # validation uses the TRAIN medians

    print("\nAFTER cleaning")
    for name, d in (("train", train_c), ("val", val_c)):
        print(f"  {name}: negative weights={int((d.weight < 0).sum())}, "
              f"missing weights={int(d.weight.isna().sum())}, "
              f"missing market_index={int(d.market_index.isna().sum())}")

    print("\nWeight medians used for filling:\n", medians.round(0).to_string())
    X = build_features(train_c)
    print(f"\nFeature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print("Features:", list(X.columns))
    print("Any missing values left in features?", bool(X.isna().any().any()))
