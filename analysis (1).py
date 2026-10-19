"""Market entry analysis: coffee shop segmentation and regional prioritisation.
Usage: python analysis.py coffee_shops.csv
Expects the same columns as coffee_shops.csv.
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

path = sys.argv[1] if len(sys.argv) > 1 else "coffee_shops.csv"
raw = pd.read_csv(path)

# ---- Cleaning ----
df = raw.dropna(subset=["Rating", "Reviews"]).copy()
df = df[df["Place type"] != "Appliance repair service"]
df = df[df["Reviews"] < df["Reviews"].mean() + 10 * df["Reviews"].std()]  # drops the 17,937-review manufacturer
df["dine"] = (df["Dine in option"] == True).astype(int)
df["take"] = (df["Takeout option"] == True).astype(int)
df["deliv"] = (df["Delivery option"] == True).astype(int)
df["logrev"] = np.log10(df["Reviews"])
df["mid"] = (df["Price"] == "$$").astype(int)

# ---- Clustering ----
feats = ["Rating", "logrev", "dine", "take", "deliv"]
X = StandardScaler().fit_transform(df[feats])
sil = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=1).fit_predict(X)) for k in range(2, 9)}
km = KMeans(4, n_init=50, random_state=1).fit(X)
df["cluster"] = km.labels_
prof = df.groupby("cluster").agg(n=("Rating", "size"), rating=("Rating", "mean"), med_rev=("Reviews", "median"),
        dine=("dine", "mean"), take=("take", "mean"), deliv=("deliv", "mean"), mid=("mid", "mean"))
names = {}
for c, r in prof.iterrows():
    if r["deliv"] > 0.5: names[c] = "Established full-service leaders"
    elif r["dine"] > 0.9: names[c] = "Mainstream dine-in/takeout core"
    elif r["take"] > 0.9: names[c] = "Newcomer grab-and-go"
    else: names[c] = "Newcomer small-format / niche"
df["segment"] = df["cluster"].map(names)
seg = df.groupby("segment").agg(n=("Rating", "size"), rating=("Rating", "mean"), med_rev=("Reviews", "median"),
        dine=("dine", "mean"), take=("take", "mean"), deliv=("deliv", "mean")).round(2)

# ---- Regions ----
reg = df.groupby("Region").agg(n=("Rating", "size"), rating=("Rating", "mean"), med_rev=("Reviews", "median"),
        lt100=("Reviews", lambda s: (s < 100).mean()), gt1000=("Reviews", lambda s: (s > 1000).mean()),
        deliv=("deliv", "mean")).sort_values("med_rev")

# ---- Service format ----
fmt = df.assign(full=((df["dine"] == 1) & (df["take"] == 1))).groupby("full")["Reviews"].median()
dlv = df.groupby("deliv")["Reviews"].median()
rho = df[["Rating", "logrev"]].corr(method="spearman").iloc[0, 1]

# ---- Charts ----
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(reg.index, reg.med_rev, color=["#c0392b" if r in ("Odessa", "Dnipro") else "#9aa5b1" for r in reg.index])
ax.set_ylabel("Median Google reviews per shop"); ax.set_title("Market maturity by region", loc="left", fontweight="bold")
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig("charts/regions.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.2))
for s, g in df.groupby("segment"):
    ax.scatter(g["Reviews"], g["Rating"], label=f"{s} (n={len(g)})", alpha=.75, s=28)
ax.set_xscale("log"); ax.set_xlabel("Reviews (log scale)"); ax.set_ylabel("Rating")
ax.set_title("Four market segments", loc="left", fontweight="bold"); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout(); plt.savefig("charts/segments.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.2))
labels = ["Neither dine-in\nnor takeout", "Dine-in +\ntakeout", "No delivery", "Delivery"]
vals = [df[(df["dine"] == 0) & (df["take"] == 0)]["Reviews"].median(), fmt[True], dlv[0], dlv[1]]
ax.bar(labels, vals, color=["#9aa5b1", "#2c7fb8", "#9aa5b1", "#2c7fb8"])
for i, v in enumerate(vals): ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("Median reviews"); ax.set_title("Full-service formats and popularity", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("charts/format.png", dpi=200); plt.close()

# ---- Export numbers for the report ----
import json
json.dump({"n_raw": len(raw), "n_clean": len(df), "sil": sil, "rho": float(rho),
           "pct_mid": float(df[df.Price.notna()].mid.mean()), "pct_ge47": float((df.Rating >= 4.7).mean()),
           "pct_deliv": float(df.deliv.mean()), "fmt_full": float(fmt[True]),
           "neither": float(df[(df["dine"] == 0) & (df["take"] == 0)]["Reviews"].median()),
           "dlv0": float(dlv[0]), "dlv1": float(dlv[1])}, open("stats.json", "w"))
seg.to_csv("segments.csv"); reg.round(2).to_csv("regions.csv")
print(seg); print(reg.round(2)); print(sil, rho)
