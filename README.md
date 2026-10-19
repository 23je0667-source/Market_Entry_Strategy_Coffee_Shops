# Market Entry Strategy for a Coffee Company

A consulting-style market entry analysis of coffee shops, using Google business listings. The project answers two questions for a coffee company planning to enter a new market:

1. **How is the market structured?** (customer-facing shop segments)
2. **Where should the company enter first, and what drives shop popularity?** (regional prioritisation and predictive modelling)

The full write-up is in [`Market_Entry_Strategy_Coffee_Shops.pdf`](Market_Entry_Strategy_Coffee_Shops.pdf) (6 pages).

## Key findings

- **Four market segments** (K-means on rating, log-reviews, dine-in, takeout, delivery):

| Segment | Shops | Median reviews | Avg rating | Defining features |
|---|---|---|---|---|
| Established full-service leaders | 33 | 789 | 4.63 | 100% delivery, 85% dine-in |
| Mainstream dine-in/takeout core | 92 | 560 | 4.59 | ~100% dine-in, 95% takeout, no delivery |
| Newcomer grab-and-go | 33 | 24 | 4.75 | 100% takeout, little dine-in |
| Newcomer small-format / niche | 38 | 40 | 4.78 | Little dine-in, no takeout shown |

- **Rating and price do not differentiate shops.** About half of shops are rated 4.7 or higher, and 95% of priced shops are mid-range ($$). A new entrant should compete on service consistency and format breadth.
- **Regional recommendation:** enter **Odessa** first (youngest market), **Dnipro** second (lowest average rating, suggesting a service gap), and **Kyiv** later for scale. Poltava, which appears in some earlier drafts, is not supported by the data.
- **Predictive modelling:** review count can be predicted for unseen shops (cross-validated R² of 0.65 with gradient boosting, typical error about 2x). **Neural networks did not beat simpler models** (R² 0.52 to 0.56) because 196 shops is too few for deep learning.
- **Hidden driver:** much of the predictive power comes from *listing completeness* (for example, whether a price is listed), which signals an established shop and is not a lever a new entrant can pull. Without those flags, R² falls from 0.65 to 0.36.

## Repository contents

| File | Description |
|---|---|
| `Market_Entry_Strategy_Coffee_Shops.pdf` | Full consulting report: findings, models, recommendations, limitations |
| `analysis.py` | Cleaning, segmentation, regional comparison; writes charts and summary tables |
| `modeling.py` | Model comparison (baseline, ridge, random forest, gradient boosting, neural networks), feature importance, regional effects |
| `coffee_shops.csv` | Input data (200 shop listings, 10 regions) |
| `charts/` | Generated figures used in the report |
| `segments.csv`, `regions.csv`, `stats.json`, `model_stats.json` | Generated summary outputs |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
python analysis.py coffee_shops.csv     # segments, regions, charts
python modeling.py coffee_shops.csv     # model comparison and drivers
```

Both scripts expect a CSV with these columns: `Region`, `Place name`, `Place type`, `Rating`, `Reviews`, `Price`, `Delivery option`, `Dine in option`, `Takeout option`. Run them from the project folder so the `charts/` directory is found.

## Method summary

- **Cleaning:** dropped 2 listings with no rating or reviews, 1 appliance-repair business, and 1 extreme outlier (a coffee manufacturer with 17,937 reviews), leaving 196 shops. Dine-in and takeout flags are only shown when true, so missing values are treated as not offered.
- **Segmentation:** K-means (k = 4) on standardised features. Silhouette scores are nearly flat for k = 3 to 5 (about 0.40), so four segments is a useful lens rather than a hard fact.
- **Modelling:** target is log10(reviews). Models were evaluated with 5-fold cross-validation repeated 5 times. Feature importance used out-of-fold permutation importance, and an ablation test measured the effect of listing-completeness flags.

## Limitations

- The dataset has exactly 20 shops per region by design, so shop counts say nothing about market size.
- Review count is a proxy for how established a shop is, combining shop age and customer volume. The data cannot separate the two, so a region with few reviews could mean immature competition or smaller demand.
- The data appears to pre-date 2022 and includes cities that are now in conflict zones. It must be refreshed, and current security conditions checked, before any real investment decision.
- Findings about drivers are associations, not proven causal effects.
- Neural networks were implemented with scikit-learn's `MLPRegressor`; a deeper comparison would need more data.

## Suggested next steps

- Add population, income, rent and footfall per city, plus chain competitors and unit economics (break-even analysis).
- Collect shop opening dates or review timestamps (reviews per month) to separate shop age from true demand.
- Analyse review text for specific service gaps; this is where deep learning or language models become worthwhile.

## Data attribution

The analysis uses `coffee_shops.csv`, a set of 200 Google business listings for coffee shops across 10 Ukrainian regions, supplied with a market-entry exercise. The original dataset author is not stated in the file; credit belongs to whoever compiled it. The analysis, models and report in this repository are my own work built on that data.

## License

Code in this repository is released under the [MIT License](LICENSE). The dataset is not covered by this license and remains subject to its original terms.
