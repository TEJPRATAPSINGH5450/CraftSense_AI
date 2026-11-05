from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


ALIASES = {
    "Invoice": "InvoiceNo",
    "Invoice No": "InvoiceNo",
    "Invoice Number": "InvoiceNo",
    "Customer ID": "CustomerID",
    "Customer": "CustomerID",
    "Description": "Product",
    "Item": "Product",
    "CategoryName": "Category",
    "Price": "UnitPrice",
    "Date": "InvoiceDate",
    "TransactionDate": "InvoiceDate",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    rename = {c: ALIASES[c] for c in data.columns if c in ALIASES}
    return data.rename(columns=rename)


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a transaction-line dataset without inventing missing purchases."""
    data = standardize_columns(df)
    required = ["InvoiceNo", "Product", "Quantity", "UnitPrice", "InvoiceDate"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Required: InvoiceNo, Product, Quantity, UnitPrice, InvoiceDate. "
            + "CustomerID and Category are strongly recommended."
        )

    report = {
        "input_rows": len(data),
        "missing_before": int(data.isna().sum().sum()),
    }

    if "CustomerID" not in data.columns:
        data["CustomerID"] = "UNKNOWN"
    if "Category" not in data.columns:
        data["Category"] = ""

    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["Product"] = data["Product"].astype("string").str.strip()
    data["Category"] = data["Category"].astype("string").str.strip()
    data["CustomerID"] = data["CustomerID"].astype("string").str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")

    # Remove cancelled invoices (common convention: invoice starts with C).
    cancellation = data["InvoiceNo"].str.upper().str.startswith("C", na=False)
    data = data.loc[~cancellation].copy()

    # Returns/negative quantities and zero/negative prices are excluded from
    # purchase-pattern mining. Keep this policy documented in the report.
    data = data.dropna(subset=["InvoiceNo", "Product", "Quantity", "UnitPrice", "InvoiceDate"])
    data = data[
        (data["InvoiceNo"] != "")
        & (data["Product"] != "")
        & (data["Quantity"] > 0)
        & (data["UnitPrice"] > 0)
    ].copy()

    data["CustomerID"] = data["CustomerID"].replace({"<NA>": "UNKNOWN", "nan": "UNKNOWN"})
    data["Category"] = data["Category"].replace({"<NA>": "", "nan": ""})
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data = data.drop_duplicates()

    report["output_rows"] = len(data)
    report["removed_rows"] = report["input_rows"] - len(data)
    report["missing_after"] = int(data.isna().sum().sum())
    report["unique_invoices"] = int(data["InvoiceNo"].nunique())
    report["unique_products"] = int(data["Product"].nunique())
    report["unique_customers"] = int(data.loc[data["CustomerID"] != "UNKNOWN", "CustomerID"].nunique())
    return data.reset_index(drop=True), report


def _clean_text(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _suggest_label(keywords: list[str]) -> str:
    text = " ".join(keywords).lower()
    mapping = {
        "Pottery / Ceramics": ["pottery", "ceramic", "clay", "terracotta", "vase", "bowl"],
        "Textiles / Embroidery": ["textile", "fabric", "cotton", "silk", "shawl", "dupatta", "embroidery", "phulkari", "woven"],
        "Wooden Crafts": ["wood", "wooden", "carved", "elephant", "box", "toy"],
        "Jewellery": ["jewel", "necklace", "earring", "bracelet", "bead", "pendant"],
        "Paintings / Art": ["painting", "art", "canvas", "madhubani", "warli", "miniature"],
        "Home Decor": ["decor", "wall", "basket", "macrame", "lamp", "hanging"],
        "Metal Crafts": ["metal", "brass", "copper", "iron", "bell"],
        "Leather Crafts": ["leather", "wallet", "bag", "footwear"],
    }
    for label, words in mapping.items():
        if any(word in text for word in words):
            return label
    return "Discovered Craft Group"


def discover_product_categories(
    df: pd.DataFrame, n_clusters: int = 6, text_column: str = "Product"
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Unsupervised TF-IDF + KMeans over distinct product descriptions."""
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found.")
    products = df[[text_column]].dropna().drop_duplicates().copy()
    products[text_column] = products[text_column].astype(str)
    products["clean_text"] = products[text_column].map(_clean_text)
    products = products[products["clean_text"].str.len() > 0].copy()

    if len(products) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} distinct non-empty product descriptions; "
            f"found {len(products)}."
        )

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=5000,
        min_df=1, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(products["clean_text"])
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    products["DiscoveredCluster"] = model.fit_predict(matrix)

    terms = vectorizer.get_feature_names_out()
    cluster_rows = []
    label_map = {}
    for cluster_id in range(n_clusters):
        center = model.cluster_centers_[cluster_id]
        top = center.argsort()[::-1][:8]
        keywords = [terms[i] for i in top if center[i] > 0]
        label = _suggest_label(keywords)
        label_map[cluster_id] = f"{label} ({cluster_id + 1})"
        cluster_rows.append({
            "Cluster": cluster_id,
            "SuggestedLabel": label_map[cluster_id],
            "TopKeywords": ", ".join(keywords),
            "ProductCount": int((products["DiscoveredCluster"] == cluster_id).sum()),
        })

    products["DiscoveredCategory"] = products["DiscoveredCluster"].map(label_map)
    score = None
    if len(products) > n_clusters and len(set(products["DiscoveredCluster"])) > 1:
        score = float(silhouette_score(matrix, products["DiscoveredCluster"], metric="cosine"))

    mapping = products[[text_column, "DiscoveredCluster", "DiscoveredCategory"]]
    result = df.merge(mapping, on=text_column, how="left")
    return result, pd.DataFrame(cluster_rows), score


def build_customer_features(df: pd.DataFrame, n_clusters: int = 3):
    known = df[df["CustomerID"].astype(str) != "UNKNOWN"].copy()
    if known["CustomerID"].nunique() < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} known customers for customer clustering. "
            "Upload data with CustomerID or reduce the number of clusters."
        )

    reference_date = known["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = known.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        CategoryDiversity=("Category", "nunique"),
        BasketCount=("InvoiceNo", "nunique"),
    ).reset_index()

    features = ["Recency", "Frequency", "Monetary", "Quantity", "CategoryDiversity"]
    x = np.log1p(rfm[features].clip(lower=0))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(x_scaled)

    silhouette = None
    if len(rfm) > n_clusters and len(set(rfm["Cluster"])) > 1:
        silhouette = float(silhouette_score(x_scaled, rfm["Cluster"]))

    transactions = df.merge(rfm[["CustomerID", "Cluster"]], on="CustomerID", how="inner")
    summary = rfm.groupby("Cluster").agg(
        Customers=("CustomerID", "nunique"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgSpend=("Monetary", "mean"),
        AvgCategoryDiversity=("CategoryDiversity", "mean"),
    ).round(2).reset_index()
    return transactions, rfm, summary, silhouette


def _rule_name(row) -> tuple[str, str]:
    left = ", ".join(sorted(list(row["antecedents"])))
    right = ", ".join(sorted(list(row["consequents"])))
    return left, right


def mine_association_rules(
    df: pd.DataFrame,
    support: float = 0.05,
    confidence: float = 0.30,
    lift: float = 1.0,
    category_column: str = "Category",
    max_len: int = 3,
) -> pd.DataFrame:
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found.")
    baskets = (
        df.groupby("InvoiceNo")[category_column]
        .apply(lambda values: sorted(set(str(v) for v in values if pd.notna(v) and str(v).strip())))
    )
    transactions = [items for items in baskets.tolist() if items]
    if len(transactions) < 2:
        return pd.DataFrame()

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket_matrix = pd.DataFrame(encoded, columns=encoder.columns_)

    if basket_matrix.shape[1] < 2:
        return pd.DataFrame()

    frequent = apriori(
        basket_matrix, min_support=support, use_colnames=True,
        max_len=max_len, low_memory=True
    )
    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent, metric="confidence", min_threshold=confidence
    )
    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules["lift"] >= lift].copy()
    if rules.empty:
        return pd.DataFrame()

    pairs = rules.apply(_rule_name, axis=1)
    rules["Antecedents"] = [p[0] for p in pairs]
    rules["Consequents"] = [p[1] for p in pairs]
    rules["Rule"] = rules["Antecedents"] + " → " + rules["Consequents"]
    rules["conviction"] = rules["conviction"].replace([np.inf, -np.inf], np.inf)
    keep = [
        "Antecedents", "Consequents", "Rule", "support", "confidence",
        "lift", "leverage", "conviction"
    ]
    return rules[keep].sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)


def compare_support_thresholds(
    df: pd.DataFrame, thresholds: list[float], confidence: float,
    category_column: str = "Category"
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        rules = mine_association_rules(
            df, support=threshold, confidence=confidence,
            lift=1.0, category_column=category_column
        )
        rows.append({
            "SupportThreshold": threshold,
            "RuleCount": len(rules),
            "MeanConfidence": float(rules["confidence"].mean()) if len(rules) else 0.0,
            "MeanLift": float(rules["lift"].mean()) if len(rules) else 0.0,
        })
    return pd.DataFrame(rows)


def bootstrap_rule_stability(
    df: pd.DataFrame, support: float, confidence: float,
    category_column: str = "Category", iterations: int = 20,
    sample_fraction: float = 0.8, random_state: int = 42
) -> pd.DataFrame:
    """Rule recurrence in bootstrap resamples; stability is descriptive, not proof."""
    baskets = (
        df.groupby("InvoiceNo")[category_column]
        .apply(lambda values: sorted(set(str(v) for v in values if pd.notna(v) and str(v).strip())))
        .tolist()
    )
    baskets = [b for b in baskets if b]
    if len(baskets) < 5:
        return pd.DataFrame(columns=["Rule", "Stability", "Appearances", "Iterations"])

    base_rows = []
    for i, basket in enumerate(baskets):
        for category in basket:
            base_rows.append({"InvoiceNo": str(i), category_column: category})
    base_rules = mine_association_rules(
        pd.DataFrame(base_rows), support, confidence, 1.0, category_column
    )
    base_rules_set = set(base_rules["Rule"]) if len(base_rules) else set()
    if not base_rules_set:
        return pd.DataFrame(columns=["Rule", "Stability", "Appearances", "Iterations"])

    rng = np.random.default_rng(random_state)
    counts = Counter()
    sample_size = max(2, int(len(baskets) * sample_fraction))

    for iteration in range(iterations):
        indices = rng.choice(len(baskets), size=sample_size, replace=True)
        sampled_rows = []
        for new_invoice, index in enumerate(indices):
            for category in baskets[index]:
                sampled_rows.append({
                    "InvoiceNo": f"boot_{iteration}_{new_invoice}",
                    category_column: category
                })
        sample_df = pd.DataFrame(sampled_rows)
        sampled_rules = mine_association_rules(
            sample_df, support, confidence, 1.0, category_column
        )
        found = set(sampled_rules["Rule"]) if len(sampled_rules) else set()
        for rule in base_rules_set.intersection(found):
            counts[rule] += 1

    result = pd.DataFrame([
        {
            "Rule": rule,
            "Appearances": counts[rule],
            "Iterations": iterations,
            "Stability": counts[rule] / iterations,
        }
        for rule in base_rules_set
    ])
    return result.sort_values("Stability", ascending=False).reset_index(drop=True)


def monthly_category_trends(df: pd.DataFrame, category_column: str = "Category"):
    data = df.copy()
    data["Month"] = data["InvoiceDate"].dt.to_period("M").astype(str)
    return (
        data.groupby(["Month", category_column])["InvoiceNo"]
        .nunique().reset_index(name="Transactions")
    )


def recommend_from_category(
    rules: pd.DataFrame, selected_antecedent: str, top_n: int = 5
) -> pd.DataFrame:
    if rules.empty:
        return rules
    result = rules[rules["Antecedents"] == selected_antecedent].copy()
    return result.sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).head(top_n)


def what_if_rules(
    df: pd.DataFrame, support: float, confidence: float,
    category_column: str = "Category"
):
    return mine_association_rules(
        df, support=support, confidence=confidence, lift=1.0,
        category_column=category_column
    )
