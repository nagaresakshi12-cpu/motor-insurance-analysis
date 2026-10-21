import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

OUT = Path("eda_charts")
CURRENT_YEAR = datetime.now().year
MIN_SUM_INSURED = 1000
CSV_PATH = r"C:\Users\Tejasvi\OneDrive\Desktop\CaseStudy Folder\motor_insurance.csv"

COLUMN_HINTS = {
    "insurance_type": ["insurance_type", "type_of_insurance", "policy_type", "cover", "insr_type", "insurance"],
    "vehicle_type": ["vehicle_type", "type_vehicle", "type_of_vehicle", "veh_type", "vehicle_class", "body"],
    "usage": ["usage", "use"],
    "make": ["make", "manufacturer", "brand"],
    "model": ["model"],
    "mfg_year": ["manufactur", "mfg", "prod_year", "year_of", "reg_year", "year"],
    "sum_insured": ["sum_insured", "suminsured", "insured_value", "idv", "insured"],
    "premium": ["premium"],
    "claim_paid": ["claim_paid", "claim"],
    "policy_start": ["insr_begin", "policy_start", "start_date", "begin"],
}


def snake(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def map_columns(df):
    print("Columns in file:", list(df.columns))
    df.columns = [snake(c) for c in df.columns]
    mapping, used = {}, set()
    for std, hints in COLUMN_HINTS.items():
        for h in hints:
            match = [c for c in df.columns if h in c and c not in used]
            if std == "model":
                match = [c for c in match if "year" not in c]
            if match:
                mapping[match[0]] = std
                used.add(match[0])
                break
    df = df.rename(columns=mapping)
    print("Column mapping:", mapping)

    for r in ["make", "mfg_year", "sum_insured", "premium"]:
        if r not in df.columns:
            sys.exit(f"Column for '{r}' not found")
    for opt in ["model", "usage", "vehicle_type", "insurance_type"]:
        if opt not in df.columns:
            print(f"'{opt}' not found, filled with 'Unknown'")
            df[opt] = "Unknown"
    return df


def clean(df):
    print("\n=== 1. DATA CLEANING ===")
    print("Shape (raw):", df.shape)
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("Duplicates:", df.duplicated().sum())

    df = df.drop_duplicates().copy()

    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(str).str.strip().str.title().replace({"Nan": np.nan, "": np.nan})

    for c in ["sum_insured", "premium", "mfg_year", "claim_paid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
            )

    df["insurance_type"] = "Type " + df["insurance_type"].astype(str)

    if "claim_paid" in df.columns:
        df["claim_paid"] = df["claim_paid"].fillna(0)

    if "policy_start" in df.columns:
        start = pd.to_datetime(df["policy_start"], format="%d-%b-%y", errors="coerce")
        df["policy_year"] = start.dt.year
        df = df.drop(columns=["policy_start"])

    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].fillna("Unknown")
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(df[c].median())

    before = len(df)
    df = df[(df["premium"] > 0) & (df["sum_insured"] >= MIN_SUM_INSURED)]
    df = df[df["mfg_year"].between(1980, CURRENT_YEAR)]
    df["mfg_year"] = df["mfg_year"].astype(int)
    print(f"Removed {before - len(df)} invalid rows")

    for c in ["premium", "sum_insured"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[f"{c}_outlier"] = ~df[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        print(f"{c}: {df[f'{c}_outlier'].sum()} outliers flagged")

    print("Shape (clean):", df.shape)
    return df.reset_index(drop=True)


def engineer(df):
    print("\n=== 3. FEATURE ENGINEERING ===")
    ref_year = df["policy_year"] if "policy_year" in df.columns else CURRENT_YEAR
    df["vehicle_age"] = (ref_year - df["mfg_year"]).clip(lower=0)
    df["age_band"] = pd.cut(
        df["vehicle_age"],
        bins=[-1, 2, 5, 10, 15, 100],
        labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-15 yrs", "15+ yrs"],
    )
    df["premium_rate_pct"] = df["premium"] / df["sum_insured"] * 100
    df["si_band"] = pd.qcut(df["sum_insured"], 4, labels=["Low", "Mid", "High", "Premium"])
    df["premium_band"] = pd.qcut(df["premium"], 4, labels=["Low", "Mid", "High", "Very High"])
    df["make_model"] = df["make"] + " " + df["model"]
    df["premium_vs_segment"] = df["premium"] / df.groupby("insurance_type")["premium"].transform("mean")

    rate_cut = df["premium_rate_pct"].quantile(0.75)
    df["high_risk_flag"] = ((df["vehicle_age"] > 10) | (df["premium_rate_pct"] > rate_cut)).astype(int)

    if "claim_paid" in df.columns:
        df["claim_flag"] = (df["claim_paid"] > 0).astype(int)
        df["loss_ratio"] = df["claim_paid"] / df["premium"]

    print(df[["vehicle_age", "age_band", "premium_rate_pct", "si_band", "high_risk_flag"]].head())
    return df


def eda(df):
    OUT.mkdir(exist_ok=True)
    print("\n=== 2. EXPLORATORY ANALYSIS ===")
    print(df[["sum_insured", "premium", "vehicle_age", "premium_rate_pct"]].describe().round(2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["premium"], bins=40, kde=True, ax=ax[0]).set(title="Premium distribution")
    sns.histplot(df["sum_insured"], bins=40, kde=True, ax=ax[1]).set(title="Sum insured distribution")
    plt.tight_layout()
    plt.savefig(OUT / "01_distributions.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="insurance_type", y="premium", ax=ax[0], showfliers=False).set(title="Premium by insurance type")
    sns.boxplot(data=df, x="vehicle_type", y="premium", ax=ax[1], showfliers=False).set(title="Premium by vehicle type")
    ax[1].tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(OUT / "02_premium_by_type.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sample = df.sample(min(len(df), 5000), random_state=1)
    sns.scatterplot(data=sample, x="sum_insured", y="premium", hue="insurance_type", alpha=0.4)
    plt.title("Premium vs Sum insured")
    plt.tight_layout()
    plt.savefig(OUT / "03_premium_vs_si.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby("mfg_year")["premium"].mean().plot(ax=ax[0], marker="o", title="Avg premium by manufacturing year")
    df.groupby("mfg_year")["sum_insured"].mean().plot(ax=ax[1], marker="o", title="Avg sum insured by manufacturing year")
    plt.tight_layout()
    plt.savefig(OUT / "04_year_trends.png", dpi=150)
    plt.close()

    top = (
        df.groupby("make")
        .agg(policies=("premium", "size"), avg_premium=("premium", "mean"), avg_si=("sum_insured", "mean"))
        .sort_values("policies", ascending=False)
        .head(10)
    )
    top["avg_premium"].sort_values().plot.barh(figsize=(7, 5), title="Avg premium - top 10 makes by volume")
    plt.tight_layout()
    plt.savefig(OUT / "05_make_premium.png", dpi=150)
    plt.close()
    print("\nTop 10 makes:\n", top.round(0))

    num = df[["premium", "sum_insured", "vehicle_age", "premium_rate_pct"]]
    plt.figure(figsize=(5, 4))
    sns.heatmap(num.corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(OUT / "06_correlation.png", dpi=150)
    plt.close()

    print("\n=== KEY INSIGHTS ===")
    corr = num.corr()["premium"]
    print(f"Corr(premium, sum insured) = {corr['sum_insured']:.2f} | Corr(premium, vehicle age) = {corr['vehicle_age']:.2f}")
    print("Avg premium by insurance type:\n", df.groupby("insurance_type")["premium"].mean().round(0).to_string())
    print("Avg premium by vehicle type:\n", df.groupby("vehicle_type")["premium"].mean().round(0).to_string())
    print("Avg premium rate (% of SI) by age band:\n",
          df.groupby("age_band", observed=True)["premium_rate_pct"].mean().round(2).to_string())
    print("Avg premium rate (% of SI) by insurance type:\n",
          df.groupby("insurance_type")["premium_rate_pct"].mean().round(2).to_string())
    print("Avg premium by usage (top 8):\n",
          df.groupby("usage")["premium"].mean().sort_values(ascending=False).head(8).round(0).to_string())

    if "claim_paid" in df.columns:
        for k in ["insurance_type", "vehicle_type", "age_band"]:
            t = df.groupby(k, observed=True).agg(
                policies=("premium", "size"),
                claim_freq_pct=("claim_flag", "mean"),
                claims=("claim_paid", "sum"),
                premium=("premium", "sum"),
            )
            t["claim_freq_pct"] = (t["claim_freq_pct"] * 100).round(1)
            t["loss_ratio_pct"] = (t["claims"] / t["premium"] * 100).round(1)
            print(f"Claim frequency and loss ratio by {k}:\n", t[["policies", "claim_freq_pct", "loss_ratio_pct"]].to_string())

    if "policy_year" in df.columns:
        print("Avg premium by policy year:\n", df.groupby("policy_year")["premium"].mean().round(0).to_string())


if __name__ == "__main__":
    path = str(Path(sys.argv[1] if len(sys.argv) > 1 else CSV_PATH).resolve())
    out_folder = Path(path).parent
    os.chdir(out_folder)

    raw = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    df = map_columns(raw)
    df = clean(df)
    df = engineer(df)
    eda(df)
    df.to_csv("motor_insurance_clean.csv", index=False)
    print("\nSaved:", out_folder / "motor_insurance_clean.csv")