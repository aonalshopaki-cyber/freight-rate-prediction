
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

train = pd.read_csv("data/train-test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


for df in (train, val):
    df["hav"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["dist_ratio"] = df.distance / df.hav

# ---------------------------------------------------------------- 1. distance
print("=" * 70, "\n1. DISTANCE vs STRAIGHT-LINE (haversine)\n", "=" * 70)
for name, df in (("train", train), ("val", val)):
    r = df.dist_ratio
    print(f"{name}: ratio quantiles", r.quantile([.01, .5, .99, .999]).round(3).to_dict())
    print(f"{name}: rows with ratio > 1.6: {(r > 1.6).sum()}   ratio < 1.0: {(r < 1.0).sum()}")
bad = train[train.dist_ratio > 1.6]
print("\nExample train rows with suspicious distance:")
print(bad[["pickup", "delivery", "distance", "hav", "dist_ratio", "posted_rate"]].head(8).round(2))
print("\nDoes the rate follow the stated distance or the true (haversine) distance?")
print("  corr(rate, distance) on bad rows:", round(bad.posted_rate.corr(bad.distance), 3))
print("  corr(rate, haversine) on bad rows:", round(bad.posted_rate.corr(bad.hav), 3))

# ---------------------------------------------------------------- 2. rate outliers
print("\n", "=" * 70, "\n2. RATE-PER-MILE OUTLIERS (train)\n", "=" * 70)
train["rpm"] = train.posted_rate / train.distance
print("rows with rpm > 5:", (train.rpm > 5).sum(), "| rpm > 8:", (train.rpm > 8).sum(), "| rpm < 0.6:", (train.rpm < 0.6).sum())
hi = train[train.rpm > 5]
print("\nOutliers by distance bucket:")
print(hi.groupby(pd.cut(hi.distance, [0, 150, 300, 600, 1000, 4000])).size())
print("Share of all rows in those buckets:")
print(train.groupby(pd.cut(train.distance, [0, 150, 300, 600, 1000, 4000])).size() / len(train))
print("\nOutliers by equipment:", hi.equipment.value_counts().to_dict())
print("Outliers by month:", hi.groupby(hi.date.dt.to_period("M")).size().to_dict())

# ---------------------------------------------------------------- 3. day of week
print("\n", "=" * 70, "\n3. DAY-OF-WEEK EFFECT\n", "=" * 70)
core = train[train.rpm < 5].copy()
core["resid"] = core.rpm / core.groupby([pd.cut(core.distance, [0, 150, 300, 600, 1000, 1500, 2500, 4000]), "equipment"], observed=True).rpm.transform("median")
core["dow"] = core.date.dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print("Mean relative rate by weekday (1.00 = typical):")
print(core.groupby("dow").resid.mean().reindex(order).round(4))
print("\nLoads per weekday:", core.groupby("dow").size().reindex(order).to_dict())

# ---------------------------------------------------------------- 4. signals
print("\n", "=" * 70, "\n4. MARKET_INDEX / QUOTE_SIGNAL vs PRICE\n", "=" * 70)
print("Row-level corr with relative rate:",
      core[["resid", "market_index", "quote_signal"]].corr().loc["resid"].round(3).to_dict())
print("\nBy equipment:")
for eq, g in core.groupby("equipment"):
    print(f"  {eq:8s} market_index {g.resid.corr(g.market_index):.3f} | quote_signal {g.resid.corr(g.quote_signal):.3f}")
print("\nBy distance bucket:")
for b, g in core.groupby(pd.cut(core.distance, [0, 300, 800, 1500, 4000]), observed=True):
    print(f"  {str(b):14s} market_index {g.resid.corr(g.market_index):.3f} | quote_signal {g.resid.corr(g.quote_signal):.3f}")

daily = core.groupby("date").agg(resid=("resid", "mean"), mi=("market_index", "mean"), qs=("quote_signal", "mean"))
print("\nDAILY level: corr of average relative rate with that day's signals")
print(" ", daily.corr().loc["resid", ["mi", "qs"]].round(3).to_dict())
print("Does today's price follow the signal from earlier days? (lag in days)")
for lag in (0, 1, 3, 7, 14):
    print(f"  lag {lag:2d}: market_index {daily.resid.corr(daily.mi.shift(lag)):.3f} | quote_signal {daily.resid.corr(daily.qs.shift(lag)):.3f}")

# quote_signal vs price per mile in raw terms
print("\nquote_signal quintiles -> mean relative rate:")
print(core.groupby(pd.qcut(core.quote_signal, 5), observed=True).resid.mean().round(4))
print("market_index quintiles -> mean relative rate:")
print(core.groupby(pd.qcut(core.market_index, 5), observed=True).resid.mean().round(4))
