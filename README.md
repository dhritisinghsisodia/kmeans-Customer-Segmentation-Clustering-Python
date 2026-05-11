# 🛍️ Customer Segmentation Using RFM Analysis & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-KMeans-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![Domain](https://img.shields.io/badge/Domain-E--Commerce-purple?style=flat)

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Data Cleaning](#-data-cleaning)
- [RFM Feature Engineering](#-rfm-feature-engineering)
- [Customer Segments](#-customer-segments)
- [Revenue Contribution Analysis](#-revenue-contribution-analysis)
- [Key Findings](#-key-findings)
- [Business Recommendations](#-business-recommendations)
- [Tools Used](#-tools-used)
- [How to Run](#-how-to-run)

---

## 🎯 Project Overview

An online retail business has thousands of customers with vastly different purchasing behaviours — some buy frequently in large amounts, some haven't purchased in months, and some are occasional high-value buyers. Treating all of them the same way wastes marketing budget and misses retention opportunities.

**Business Problem:** How do we identify distinct customer groups so the business can target each with the right strategy?

**Approach:** RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering to segment customers into 6 actionable groups — from VIP customers to at-risk churners.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| **Source** | [UCI ML Repository — Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) |
| **Period** | December 2009 – December 2011 |
| **Records** | ~1 million transactions (pre-cleaning) |
| **Customers** | ~5,900 unique customers (post-cleaning) |
| **Geography** | UK-based online retailer, ships internationally |
| **Features** | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |

---

## 🔄 Project Workflow

```
Raw Transaction Data (1M+ rows)
        ↓
   Data Exploration
        ↓
   Data Cleaning (removed ~23% of records)
        ↓
   RFM Feature Engineering
        ↓
   Outlier Detection & Separation (IQR method)
        ↓
   StandardScaler → KMeans Clustering (k=3 on non-outliers)
        ↓
   Outlier Segments assigned separately (3 segments)
        ↓
   6 Final Customer Segments
        ↓
   Revenue Contribution Analysis (original)
        ↓
   Business Recommendations
```

---

## 🔧 Data Cleaning

The raw dataset required significant cleaning before analysis. Key steps:

**1. Invoice Filtering**
- Removed ~10,000 cancelled transactions (Invoice starting with `C`)
- Removed invalid `A`-prefix invoices (bad debt adjustments)
- Kept only standard 6-digit numerical invoices

**2. StockCode Cleaning**
- Performed a detailed character-by-character investigation of all non-standard StockCodes
- Excluded non-product entries: `POST`, `DOT`, `M`, `BANK CHARGES`, `TEST`, `GIFT`, `ADJUST`, `AMAZONFEE`, `DCGS`, `C2`, `C3`, `PADS`, `S`
- Standardised inconsistent letter cases (e.g., `bl` → `BL`, `f` → `F` at end of code)

| Code | Description | Decision |
|------|-------------|----------|
| POST / DOT | Postage charges | Exclude |
| M / m | Manual transactions | Exclude |
| BANK CHARGES | Bank fees | Exclude |
| TEST | Test entries | Exclude |
| gift_XXX | Gift card purchases (no Customer ID) | Exclude |
| ADJUST | Admin adjustments | Exclude |
| AMAZONFEE | Amazon shipping fees | Exclude |
| `\d{5}[a-zA-Z]+` | Legitimate product variants | Keep |

**3. Other Cleaning**
- Dropped rows with null `Customer ID` (20.5% of records)
- Removed zero and negative price entries
- Converted all columns to memory-efficient dtypes (`int32`, `float32`, `category`, `string`)

**Result:** ~23% of records removed, leaving a clean dataset of actual product sales with known customers.

---

## 📐 RFM Feature Engineering

RFM transforms transaction-level data into one row per customer with 3 behavioural features:

| Feature | Definition | Business Meaning |
|---------|-----------|-----------------|
| **Recency** | Days since last purchase | How recently active is this customer? |
| **Frequency** | Number of unique invoices | How often do they buy? |
| **Monetary** | Total spend (£) | How much have they spent? |

```python
aggregated_df = cleaned_df.groupby("Customer ID").agg(
    MonetaryValue=("SalesLineTotal", "sum"),
    Frequency=("Invoice", "nunique"),
    LastInvoiceDate=("InvoiceDate", "max")
)

aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days
```

**Why separate outliers before clustering?**

RFM distributions are highly right-skewed — a handful of super-buyers would distort the K-Means centroids and make clusters meaningless for the 95%+ of normal customers. Outliers were separated using the IQR method and assigned to their own dedicated segments.

---

## 👥 Customer Segments

### Choosing k=3

The Elbow Method and Silhouette Score were both used to select the optimal number of clusters for the non-outlier group. While inertia suggested k=4, the silhouette score peaked at **k=3** — indicating better-defined, more separable clusters.

### Non-Outlier Segments (K-Means, k=3)

| Segment | Avg Monetary | Avg Frequency | Avg Recency | Description |
|---------|-------------|--------------|-------------|-------------|
| **RETAIN** (Cluster 0) | £2,110 | 6.2 purchases | ~41 days | High-value regulars — loyal but not the most frequent |
| **NURTURE** (Cluster 2) | £606 | 2.2 purchases | ~51 days | Large group with low engagement — growth opportunity |
| **RE-ENGAGE** (Cluster 1) | £382 | 1.4 purchases | ~248 days | Long-inactive customers — highest churn risk |

### Outlier Segments (Assigned separately via IQR)

| Segment | Avg Monetary | Avg Frequency | Avg Recency | Description |
|---------|-------------|--------------|-------------|-------------|
| **DELIGHT** (Monetary + Frequency outliers) | £17,148 | 25.9 purchases | ~14 days | Top VIP customers — highest spend AND frequency |
| **PAMPER** (Monetary outliers only) | £6,498 | 7.2 purchases | ~48 days | High spenders, less frequent — luxury buyers |
| **UPSELL** (Frequency outliers only) | £2,735 | 15.0 purchases | ~23 days | Frequent buyers with moderate spend — upsell opportunity |

---

## 💰 Revenue Contribution Analysis

*This section is original analysis not present in the base tutorial.*

Understanding which segments actually drive revenue is crucial for prioritising CRM budget.

![Revenue Share by Segment](revenue_share.png)

| Segment | Customers | Revenue Share | Avg Order Value | Avg Recency |
|---------|-----------|--------------|-----------------|-------------|
| DELIGHT | Smallest | Highest % | Highest | ~14 days |
| PAMPER | Small | High % | Very High | ~48 days |
| UPSELL | Small | Moderate % | Moderate | ~23 days |
| RETAIN | Medium | Moderate % | £2,110 | ~41 days |
| NURTURE | Largest | Lower % | £606 | ~51 days |
| RE-ENGAGE | Medium | Lowest % | £382 | ~248 days |

**Key Insight:** DELIGHT + PAMPER + UPSELL represent a small fraction of customers but drive a disproportionately large share of revenue — classic 80/20 principle in action.

---

## 🔍 Key Findings

1. **VIP customers (DELIGHT) are tiny in count but massive in value** — losing one DELIGHT customer costs more than losing 10 NURTURE customers
2. **NURTURE is the largest segment** — this is the biggest opportunity for growth; converting even 15% of NURTURE to RETAIN would significantly increase revenue
3. **RE-ENGAGE customers haven't purchased in ~248 days on average** — well past typical win-back windows; require strong incentives or should be deprioritised
4. **UPSELL customers are already buying frequently but spending less per visit** — bundle deals or minimum order incentives could increase their average order value significantly
5. **Outlier segments (DELIGHT + PAMPER + UPSELL) warrant dedicated account management** — they are too valuable for generic marketing campaigns

---

## ✅ Business Recommendations

| Segment | Priority | Strategy |
|---------|----------|---------|
| 🟢 **DELIGHT** | 🔴 Critical | VIP programme, dedicated account manager, early access to new products, exclusive previews |
| 🟣 **PAMPER** | 🔴 High | Luxury/premium promotions, curated high-value bundles, white-glove service |
| 🟤 **UPSELL** | 🟠 High | Loyalty points system, bundle deals, minimum order threshold incentives |
| 🔵 **RETAIN** | 🟠 Medium | Monthly personalised newsletters, loyalty tier rewards, birthday/anniversary offers |
| 🟠 **NURTURE** | 🟡 Medium | Onboarding email sequence, first-repeat-purchase discount, category recommendations |
| 🔴 **RE-ENGAGE** | ⚪ Low | Win-back campaign with strong one-time discount; if no response within 30 days, deprioritise |

> **Budget allocation recommendation:** Direct 80% of CRM and retention budget toward DELIGHT + PAMPER + UPSELL. They deliver the highest ROI per pound spent.

---

## 🛠️ Tools Used

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core analysis language |
| Pandas | Data manipulation and RFM aggregation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualisation |
| Scikit-learn | StandardScaler, KMeans, Silhouette Score |
| Google Colab | Development environment |
| openpyxl | Reading Excel source data |

---

## ▶️ How to Run

1. Clone this repository
```bash
git clone https://github.com/yourusername/customer-segmentation-clustering.git
```

2. Open the notebook in Google Colab
```
File → Open Notebook → GitHub → paste repo URL
```

3. Upload the dataset
```
Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
Upload to Colab session when prompted
```

4. Run all cells in order (`Runtime → Run All`)

---

## 📁 Repository Structure

```
customer-segmentation-clustering/
│
├── Customer_Segmentation_Clustering.ipynb   ← Main analysis notebook
├── README.md                                ← This file
└── online_retail_II.xlsx                    ← Source dataset (download separately)
```

---

*Dataset: UCI Online Retail II | Domain: E-Commerce | Technique: RFM + K-Means Clustering*
