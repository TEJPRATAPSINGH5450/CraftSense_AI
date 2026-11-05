import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import (
    clean_transactions,
    discover_product_categories,
    build_customer_features,
    mine_association_rules,
    compare_support_thresholds,
    bootstrap_rule_stability,
    monthly_category_trends,
    recommend_from_category,
    what_if_rules,
)

st.set_page_config(
    page_title="CraftSense AI",
    page_icon="🧵",
    layout="wide",
)

st.title("🧵 CraftSense AI")
st.subheader("Artisan Market Intelligence & Association Discovery")
st.caption(
    "NLP category discovery • K-Means customer segmentation • "
    "Apriori market-basket analysis • explainable recommendations"
)

with st.sidebar:
    st.header("Project controls")
    st.write("Upload transaction-line data to begin.")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        help="One row should represent one purchased product line."
    )
    st.divider()
    st.markdown("**Expected columns**")
    st.code(
        "InvoiceNo, CustomerID, Product, Category,\n"
        "Quantity, UnitPrice, InvoiceDate",
        language="text"
    )
    st.caption(
        "CustomerID and Category may be missing, but are needed "
        "for some analysis features."
    )

@st.cache_data(show_spinner=False)
def read_upload(file_bytes, filename):
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))

def load_demo():
    path = Path("data/artisan_demo_transactions.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

if uploaded is not None:
    raw = read_upload(uploaded.getvalue(), uploaded.name)
else:
    raw = load_demo()
    if raw is None:
        st.info(
            "Upload your transaction CSV/Excel, or run "
            "`python generate_demo_data.py` to create a demonstration dataset."
        )
        st.stop()
    st.info("Using the included simulated demonstration dataset.")

try:
    clean, quality = clean_transactions(raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if clean.empty:
    st.error("No valid purchase records remain after cleaning.")
    st.stop()

# Session state holds the reviewed category mapping and analysis outputs.
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = clean.copy()
if "category_mode" not in st.session_state:
    st.session_state.category_mode = "Existing categories"

tabs = st.tabs([
    "Overview",
    "AI Category Discovery",
    "Customer Segmentation",
    "Association Mining",
    "Stability & Trends",
    "What-if Simulator",
    "Downloads & Ethics",
])

# ---------------- OVERVIEW ----------------
with tabs[0]:
    st.header("1. Dataset quality and overview")
    a, b, c, d = st.columns(4)
    a.metric("Valid transaction lines", f"{len(clean):,}")
    b.metric("Invoices", f"{quality['unique_invoices']:,}")
    c.metric("Products", f"{quality['unique_products']:,}")
    d.metric("Known customers", f"{quality['unique_customers']:,}")

    q1, q2, q3 = st.columns(3)
    q1.metric("Rows removed", f"{quality['removed_rows']:,}")
    q2.metric("Missing values before", f"{quality['missing_before']:,}")
    q3.metric("Missing values after", f"{quality['missing_after']:,}")

    with st.expander("Cleaning report"):
        st.json(quality)
    with st.expander("Preview cleaned data"):
        st.dataframe(clean.head(100), use_container_width=True)

    category_counts = (
        clean.groupby("Category")["InvoiceNo"].nunique()
        .sort_values(ascending=False).head(15).reset_index(name="Invoices")
    )
    if category_counts["Category"].astype(str).str.strip().ne("").any():
        fig = px.bar(
            category_counts, x="Category", y="Invoices",
            title="Top categories by number of invoices"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No existing category labels. Use AI Category Discovery.")

# ---------------- CATEGORY DISCOVERY ----------------
with tabs[1]:
    st.header("2. AI-powered artisan category discovery")
    st.write(
        "TF-IDF converts product text into numeric vectors. K-Means groups "
        "similar product descriptions. The suggested labels are hints and "
        "must be reviewed by a person before being used as final categories."
    )
    text_col = st.selectbox(
        "Product text column",
        [c for c in ["Product", "Description"] if c in clean.columns],
        index=0
    )
    n_product_clusters = st.slider(
        "Number of product groups", 2, 12, 6
    )
    if st.button("Run NLP + K-Means category discovery", type="primary"):
        try:
            discovered, cluster_summary, score = discover_product_categories(
                clean, n_clusters=n_product_clusters, text_column=text_col
            )
            st.session_state.discovered = discovered
            st.session_state.cluster_summary = cluster_summary
            st.session_state.discovery_score = score
        except Exception as exc:
            st.error(str(exc))

    if "discovered" in st.session_state:
        st.metric(
            "Product clustering silhouette score",
            "Not available" if st.session_state.discovery_score is None
            else f"{st.session_state.discovery_score:.3f}"
        )
        st.caption(
            "Silhouette is a geometric separation measure, not an accuracy score."
        )
        st.dataframe(st.session_state.cluster_summary, use_container_width=True)
        st.subheader("Review and rename discovered groups")
        summary = st.session_state.cluster_summary.copy()
        edited = st.data_editor(
            summary,
            disabled=["Cluster", "TopKeywords", "ProductCount"],
            use_container_width=True,
            key="category_editor"
        )
        st.session_state.cluster_summary = edited

        if st.button("Apply discovered categories to analysis"):
            mapping = st.session_state.discovered[
                [text_col, "DiscoveredCluster"]
            ].drop_duplicates()
            labels = dict(zip(
                st.session_state.cluster_summary["Cluster"],
                st.session_state.cluster_summary["SuggestedLabel"]
            ))
            mapping["DiscoveredCategory"] = mapping["DiscoveredCluster"].map(labels)
            chosen = clean.merge(mapping[[text_col, "DiscoveredCategory"]], on=text_col, how="left")
            chosen["DiscoveredCategory"] = chosen["DiscoveredCategory"].fillna("Unclassified")
            st.session_state.analysis_df = chosen
            st.session_state.category_mode = "Discovered categories"
            st.success("Discovered categories applied. Continue to Association Mining.")

        st.subheader("Product-to-cluster assignments")
        st.dataframe(
            st.session_state.discovered[
                [text_col, "DiscoveredCluster", "DiscoveredCategory"]
            ].head(200),
            use_container_width=True
        )

# ---------------- CUSTOMER SEGMENTATION ----------------
with tabs[2]:
    st.header("3. Customer segmentation using RFM + K-Means")
    st.write(
        "Features: recency, purchase frequency, monetary value, quantity "
        "and category diversity. Numerical values are log-transformed and scaled."
    )
    cluster_count = st.slider("Number of customer segments", 2, 8, 3)
    if st.button("Run customer segmentation", type="primary"):
        try:
            tx, rfm, summary, sil = build_customer_features(
                st.session_state.analysis_df, cluster_count
            )
            st.session_state.segmented_transactions = tx
            st.session_state.rfm = rfm
            st.session_state.segment_summary = summary
            st.session_state.customer_silhouette = sil
            st.success("Customer segmentation completed.")
        except Exception as exc:
            st.error(str(exc))

    if "rfm" in st.session_state:
        sil = st.session_state.customer_silhouette
        st.metric(
            "Customer clustering silhouette score",
            "Not available" if sil is None else f"{sil:.3f}"
        )
        st.dataframe(st.session_state.segment_summary, use_container_width=True)
        fig = px.scatter(
            st.session_state.rfm,
            x="Frequency", y="Monetary", color="Cluster",
            hover_data=["CustomerID"],
            title="Customer segments: frequency vs spend"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Cluster IDs are arbitrary. Interpret segments using their average "
            "RFM values; do not assume a cluster is automatically 'premium' or 'loyal'."
        )

# ---------------- ASSOCIATION MINING ----------------
with tabs[3]:
    st.header("4. Apriori association rule mining")
    data_for_rules = st.session_state.analysis_df.copy()
    category_options = ["Category"]
    if "DiscoveredCategory" in data_for_rules.columns:
        category_options.append("DiscoveredCategory")
    category_col = st.selectbox(
        "Category field for market-basket analysis",
        category_options,
        index=1 if st.session_state.category_mode == "Discovered categories"
        and "DiscoveredCategory" in category_options else 0
    )
    support = st.slider("Minimum support", 0.01, 0.50, 0.05, 0.01)
    confidence = st.slider("Minimum confidence", 0.10, 1.00, 0.30, 0.05)
    minimum_lift = st.slider("Minimum lift", 1.0, 5.0, 1.0, 0.1)
    max_len = st.slider("Maximum itemset size", 2, 4, 3)

    segment_filter = "All customers"
    if "segmented_transactions" in st.session_state:
        segment_filter = st.selectbox(
            "Customer segment",
            ["All customers"] + [
                f"Cluster {x}" for x in sorted(
                    st.session_state.segmented_transactions["Cluster"].unique()
                )
            ]
        )

    if st.button("Mine association rules", type="primary"):
        selected_data = data_for_rules
        if segment_filter != "All customers":
            if "segmented_transactions" not in st.session_state:
                st.error("Run customer segmentation first.")
                st.stop()
            cluster_id = int(segment_filter.split()[-1])
            selected_data = st.session_state.segmented_transactions
            selected_data = selected_data[
                selected_data["Cluster"] == cluster_id
            ]
            # Use discovered category mapping if selected in the main dataset.
            if "DiscoveredCategory" in data_for_rules.columns:
                selected_data = selected_data.drop(
                    columns=["DiscoveredCategory"], errors="ignore"
                ).merge(
                    data_for_rules[["InvoiceNo", "Product", "DiscoveredCategory"]].drop_duplicates(),
                    on=["InvoiceNo", "Product"], how="left"
                )
        try:
            rules = mine_association_rules(
                selected_data, support, confidence, minimum_lift,
                category_column=category_col, max_len=max_len
            )
            st.session_state.rules = rules
            st.session_state.rules_category_col = category_col
            st.success(f"Found {len(rules)} rules.")
        except Exception as exc:
            st.error(str(exc))

    if "rules" in st.session_state:
        rules = st.session_state.rules
        if rules.empty:
            st.warning("No rules found. Try lowering support/confidence or check your baskets.")
        else:
            st.dataframe(rules.round(4), use_container_width=True)
            st.subheader("Explain the metrics")
            st.markdown(
                "- **Support:** fraction of invoices containing both sides of the rule.\n"
                "- **Confidence:** fraction of invoices containing the antecedent that also contain the consequent.\n"
                "- **Lift:** confidence divided by the consequent's overall support; above 1 indicates positive association.\n"
                "- **Conviction:** directional implication measure; infinity can occur at 100% confidence."
            )
            fig = px.scatter(
                rules, x="support", y="confidence", size="lift",
                color="lift", hover_name="Rule",
                title="Association rule quality"
            )
            st.plotly_chart(fig, use_container_width=True)

            antecedents = sorted(rules["Antecedents"].unique())
            selected_antecedent = st.selectbox(
                "Select an antecedent to explain recommendations",
                antecedents
            )
            recommendations = recommend_from_category(
                rules, selected_antecedent
            )
            st.dataframe(recommendations, use_container_width=True)
            st.caption(
                "These are observed associations in this dataset, not causal "
                "effects or guaranteed future purchases."
            )

# ---------------- STABILITY & TRENDS ----------------
with tabs[4]:
    st.header("5. Rule stability and seasonal trends")
    data_for_analysis = st.session_state.analysis_df.copy()
    trend_category = "DiscoveredCategory" if (
        st.session_state.category_mode == "Discovered categories"
        and "DiscoveredCategory" in data_for_analysis.columns
    ) else "Category"

    st.subheader("Support-threshold sensitivity")
    thresholds = st.multiselect(
        "Thresholds to compare",
        [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        default=[0.02, 0.05, 0.08, 0.10]
    )
    sensitivity_conf = st.slider(
        "Confidence for threshold comparison", 0.10, 1.00, 0.30, 0.05
    )
    if st.button("Compare support thresholds"):
        comparison = compare_support_thresholds(
            data_for_analysis, thresholds, sensitivity_conf, trend_category
        )
        st.session_state.sensitivity = comparison
    if "sensitivity" in st.session_state:
        st.dataframe(st.session_state.sensitivity, use_container_width=True)
        fig = px.line(
            st.session_state.sensitivity,
            x="SupportThreshold", y="RuleCount", markers=True,
            title="Number of rules at different support thresholds"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Bootstrap stability")
    st.write(
        "Repeatedly resample invoices and measure the fraction of runs in "
        "which each baseline rule reappears."
    )
    boot_support = st.slider(
        "Bootstrap minimum support", 0.01, 0.40, 0.05, 0.01
    )
    iterations = st.slider("Bootstrap iterations", 5, 50, 20)
    if st.button("Run bootstrap stability"):
        stability = bootstrap_rule_stability(
            data_for_analysis, boot_support, sensitivity_conf,
            trend_category, iterations=iterations
        )
        st.session_state.stability = stability
    if "stability" in st.session_state:
        st.dataframe(st.session_state.stability, use_container_width=True)
        if not st.session_state.stability.empty:
            fig = px.bar(
                st.session_state.stability.head(20),
                x="Rule", y="Stability",
                title="Most repeatable association rules"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Seasonal / monthly category trends")
    trends = monthly_category_trends(data_for_analysis, trend_category)
    if not trends.empty:
        fig = px.line(
            trends, x="Month", y="Transactions", color=trend_category,
            markers=True, title="Monthly invoice counts by craft category"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough date/category information for trend analysis.")

# ---------------- WHAT IF ----------------
with tabs[5]:
    st.header("6. What-if market-basket simulator")
    st.write(
        "Change the support and confidence thresholds to see how strictness "
        "changes the discovered rule set. This is a sensitivity tool, not a "
        "forecast of causal impact."
    )
    what_support = st.slider("What-if support", 0.01, 0.50, 0.08, 0.01)
    what_conf = st.slider("What-if confidence", 0.10, 1.00, 0.40, 0.05)
    what_category = "DiscoveredCategory" if (
        st.session_state.category_mode == "Discovered categories"
        and "DiscoveredCategory" in st.session_state.analysis_df.columns
    ) else "Category"
    if st.button("Run what-if analysis"):
        result = what_if_rules(
            st.session_state.analysis_df,
            what_support, what_conf, what_category
        )
        st.session_state.whatif = result
    if "whatif" in st.session_state:
        st.metric("Rules under selected settings", len(st.session_state.whatif))
        st.dataframe(st.session_state.whatif, use_container_width=True)

# ---------------- DOWNLOADS & ETHICS ----------------
with tabs[6]:
    st.header("7. Downloads, reproducibility and responsible use")
    st.download_button(
        "Download cleaned transactions",
        clean.to_csv(index=False).encode("utf-8"),
        "cleaned_transactions.csv",
        "text/csv"
    )
    if "rfm" in st.session_state:
        st.download_button(
            "Download customer segments",
            st.session_state.rfm.to_csv(index=False).encode("utf-8"),
            "customer_segments.csv",
            "text/csv"
        )
    if "rules" in st.session_state and not st.session_state.rules.empty:
        st.download_button(
            "Download association rules",
            st.session_state.rules.to_csv(index=False).encode("utf-8"),
            "association_rules.csv",
            "text/csv"
        )
    if "discovered" in st.session_state:
        st.download_button(
            "Download product category discovery",
            st.session_state.discovered.to_csv(index=False).encode("utf-8"),
            "discovered_product_categories.csv",
            "text/csv"
        )

    st.subheader("Limitations and ethics")
    st.markdown(
        """
        - Product listings and descriptions are not proof of completed purchases.
        - Synthetic demonstration data must never be reported as real customer evidence.
        - A small set of sellers or regions can create sampling bias.
        - K-Means depends on feature scaling, feature selection and the chosen number of clusters.
        - Low-support rules may be unstable or driven by a few invoices.
        - Association does not establish causation.
        - Pseudonymize customer IDs, obtain permission for seller data, and do not expose personal information.
        - Review automatically suggested craft labels with domain experts before making business decisions.
        """
    )
    st.info(
        "For a reproducible report, record the dataset source, date downloaded, "
        "cleaning exclusions, feature definitions, random_state, cluster count, "
        "support/confidence thresholds and software versions."
    )

st.divider()
st.caption(
    "CraftSense AI — SIH prototype. Interpret results as descriptive "
    "associations within the supplied dataset."
)
