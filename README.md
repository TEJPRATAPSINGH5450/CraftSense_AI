# CraftSense AI — SIH Prototype

An end-to-end Streamlit prototype for artisan market intelligence.

## Features

1. Data upload (CSV / XLSX) and data-quality reporting.
2. Transaction cleaning: invalid quantities/prices, missing required fields, cancellation invoices and duplicate lines.
3. NLP craft-category discovery using TF-IDF + K-Means.
4. Customer segmentation using RFM-style features + StandardScaler + K-Means.
5. Segment-aware Apriori association rule mining.
6. Support-threshold sensitivity analysis.
7. Bootstrap rule recurrence / stability analysis.
8. Monthly category trend visualization.
9. What-if support/confidence simulator.
10. CSV exports and responsible-use notes.

## Quick start (Windows)

Open Command Prompt or the VS Code terminal inside this folder.

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python generate_demo_data.py
streamlit run app.py
```

If `py` is not recognized, use `python` instead.

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_demo_data.py
streamlit run app.py
```

The app normally opens at http://localhost:8501.

## Expected transaction file

One row represents one product line within an invoice.

Required columns:
- `InvoiceNo`: transaction/invoice ID
- `Product`: product title or description
- `Quantity`: purchased quantity
- `UnitPrice`: unit price
- `InvoiceDate`: transaction date

Recommended:
- `CustomerID`: pseudonymized customer identifier
- `Category`: verified craft category

Column aliases such as `Description`, `Price`, `Date`, and `Customer ID` are normalized by the backend.

## Using UCI Online Retail

The UCI Online Retail dataset has columns such as `InvoiceNo`, `StockCode`,
`Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, and `CustomerID`.
It does not provide artisan craft categories. You can use `Description` as
the product text and perform NLP category discovery, but do not claim the
resulting records are Indian artisan sales. Treat it as a general retail
benchmark. For artisan-domain validation, collect authorized, anonymized
orders from participating artisans or cooperatives.

Dataset: https://archive.ics.uci.edu/dataset/352/online+retail

## Important methodological notes

- The included demo data is simulated. Never present it as actual market evidence.
- Category discovery is unsupervised text grouping, not guaranteed semantic classification.
  Review the suggested labels with a domain expert.
- For association mining, each invoice is treated as one basket containing unique
  categories. If the business question is product-level bundling, mine `Product`
  instead of `Category`.
- The customer segmentation uses RFM-style features. It requires at least as many
  known customers as selected clusters. Rows with unknown customer IDs are not
  used for customer clustering.
- Support is the fraction of baskets containing an itemset. Confidence is
  directional. Lift > 1 indicates positive association relative to independence.
  Conviction can be infinite when confidence is 1 and the consequent is not universal.
- Bootstrap stability is the recurrence of a rule under resampling; it is not
  a formal guarantee of future performance.
- Association does not imply causation.

## Suggested SIH extension roadmap

1. Add verified artisan catalogue metadata (materials, region, craft tradition).
2. Compare TF-IDF with sentence-transformer embeddings for category discovery.
3. Add human-in-the-loop category approval and a persistent category mapping file.
4. Evaluate cluster quality using expert-labelled product examples.
5. Add a privacy-preserving seller-level dashboard and role-based access.
6. Test recommendations on a later time period to evaluate temporal generalization.
