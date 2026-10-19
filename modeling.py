"""Predictive modelling: what explains how established (review-heavy) a coffee shop is?
Target: log10(Reviews). Compares a naive baseline, linear, tree-ensemble and neural-network models
with repeated cross-validation, then inspects drivers via permutation importance.
Usage: python modeling.py coffee_shops.csv
"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

path = sys.argv[1] if len(sys.argv) > 1 else "coffee_shops.csv"
raw = pd.read_csv(path)
df = raw.dropna(subset=["Rating", "Reviews"]).copy()
df = df[df["Place type"] != "Appliance repair service"]
df = df[df["Reviews"] < df["Reviews"].mean() + 10 * df["Reviews"].std()].reset_index(drop=True)
core = ["Coffee shop", "Cafe", "Espresso bar", "Coffee store", "Coffee stand"]
df["core_type"] = df["Place type"].isin(core).astype(int)
df["dine"] = (df["Dine in option"] == True).astype(int)
df["take"] = (df["Takeout option"] == True).astype(int)
df["deliv"] = (df["Delivery option"] == True).astype(int)
df["price_mid"] = (df["Price"] == "$$").astype(int)
df["price_known"] = df["Price"].notna().astype(int)
df["y"] = np.log10(df["Reviews"])

num = ["Rating", "dine", "take", "deliv", "price_mid", "price_known", "core_type"]
X = df[num + ["Region"]]; y = df["y"]
pre = ColumnTransformer([("n", StandardScaler(), num), ("r", OneHotEncoder(handle_unknown="ignore"), ["Region"])])

models = {
 "Naive (predict the mean)": DummyRegressor(),
 "Ridge regression": RidgeCV(alphas=np.logspace(-2, 3, 30)),
 "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0),
 "Gradient boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=.8, random_state=0),
 "Neural network (2x16, L2)": MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1.0, max_iter=3000, early_stopping=False, random_state=0),
 "Neural network (2x64, no reg.)": MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-5, max_iter=3000, random_state=0),
}
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
res = {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("m", m)])
    r2s, maes = [], []
    for tr, te in cv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr]); p = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], p)); maes.append(mean_absolute_error(y.iloc[te], p))
    res[name] = dict(r2=float(np.mean(r2s)), r2_sd=float(np.std(r2s)), mae=float(np.mean(maes)))
    print(f"{name:34s} R2={np.mean(r2s):6.3f} (+/-{np.std(r2s):.3f})  MAE(log10)={np.mean(maes):.3f}  ~x{10**np.mean(maes):.1f}")

# ---- Drivers: permutation importance on the best non-naive model, out-of-fold ----
best = max((k for k in res if not k.startswith("Naive")), key=lambda k: res[k]["r2"])
print("best:", best)
imp = {}
feat_groups = {"Rating": ["Rating"], "Dine-in": ["dine"], "Takeout": ["take"], "Delivery": ["deliv"],
               "Price listed at all\n(listing completeness)": ["price_known"], "Mid-price band": ["price_mid"], "Place type": ["core_type"], "Region": ["Region"]}
rng = np.random.RandomState(0)
accum = {k: [] for k in feat_groups}
for tr, te in RepeatedKFold(n_splits=5, n_repeats=3, random_state=7).split(X):
    pipe = Pipeline([("pre", pre), ("m", models[best])]).fit(X.iloc[tr], y.iloc[tr])
    base = r2_score(y.iloc[te], pipe.predict(X.iloc[te]))
    for g, cols in feat_groups.items():
        drops = []
        for _ in range(10):
            Xp = X.iloc[te].copy(); perm = rng.permutation(len(Xp))
            for c in cols: Xp[c] = Xp[c].values[perm]
            drops.append(base - r2_score(y.iloc[te], pipe.predict(Xp)))
        accum[g].append(np.mean(drops))
imp = {g: float(np.mean(v)) for g, v in accum.items()}
print({k: round(v, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])})

# ---- Adjusted regional effect (ridge, format held constant) ----
ridge = Pipeline([("pre", pre), ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))]).fit(X, y)
names = ridge.named_steps["pre"].get_feature_names_out()
coef = pd.Series(ridge.named_steps["m"].coef_, index=names)
region_eff = coef[[n for n in names if n.startswith("r__Region_")]]
region_eff.index = [n.replace("r__Region_", "") for n in region_eff.index]
region_eff = region_eff.sort_values()
print((10 ** region_eff).round(2))
fmt_eff = {k: float(coef["n__" + c]) for k, c in [("Dine-in", "dine"), ("Takeout", "take"), ("Delivery", "deliv"), ("Rating (per 1 SD)", "Rating")]}
print({k: round(10 ** v, 2) for k, v in fmt_eff.items()})

# ---- Ablation: remove listing-completeness proxies (price listed, dine-in/takeout flags) ----
def cv_r2(cols_num, model):
    pre2 = ColumnTransformer([("n", StandardScaler(), cols_num), ("r", OneHotEncoder(handle_unknown="ignore"), ["Region"])])
    sc = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=42).split(X):
        pp = Pipeline([("pre", pre2), ("m", model)]).fit(X.iloc[tr], y.iloc[tr]); sc.append(r2_score(y.iloc[te], pp.predict(X.iloc[te])))
    return float(np.mean(sc))
gb = lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=.8, random_state=0)
abl = {"All features": cv_r2(num, gb()),
       "Without price-listed flag": cv_r2([c for c in num if c != "price_known"], gb()),
       "Without price-listed, dine-in, takeout flags": cv_r2(["Rating", "deliv", "core_type"], gb()),
       "Rating + region only": cv_r2(["Rating"], gb())}
print(abl)

# ---- Charts ----
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 3.3))
nm = list(res); r2v = [res[k]["r2"] for k in nm]
ax.barh(nm[::-1], r2v[::-1], color=["#c0392b" if "Neural" in k else "#9aa5b1" for k in nm[::-1]])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Cross-validated R-squared (0 = no better than the mean)")
ax.set_title("Model comparison on unseen shops", loc="left", fontweight="bold"); plt.tight_layout()
plt.savefig("charts/models.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.6))
ii = pd.Series(imp).sort_values()
ax.barh(ii.index, ii.values, color="#2c7fb8"); ax.set_xlabel("Drop in R-squared when the feature is scrambled")
ax.set_title("What drives popularity (review count)?", loc="left", fontweight="bold"); plt.tight_layout()
plt.savefig("charts/importance.png", dpi=200); plt.close()

# out-of-fold predicted vs actual for the best model
oof = cross_val_predict(Pipeline([("pre", pre), ("m", models[best])]), X, y, cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=1))
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(10 ** y, 10 ** oof, s=16, alpha=.6, color="#2c7fb8")
lim = [3, 5000]; ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Actual reviews"); ax.set_ylabel("Predicted reviews (out-of-sample)")
ax.set_title(f"{best}", loc="left", fontsize=10, fontweight="bold"); plt.tight_layout()
plt.savefig("charts/pred_vs_actual.png", dpi=200); plt.close()

json.dump({"abl": abl, "res": res, "best": best, "imp": imp,
           "region_mult": {k: float(10 ** v) for k, v in region_eff.items()},
           "fmt_mult": {k: float(10 ** v) for k, v in fmt_eff.items()}, "n": len(df)}, open("model_stats.json", "w"), indent=1)
