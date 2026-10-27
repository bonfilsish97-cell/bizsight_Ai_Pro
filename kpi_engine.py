"""
kpi_engine.py  (Stage 1: detection only)
Guesses which column plays which business role (revenue, cost, product...)
and lists which KPIs and charts the dataset can support.
KPI calculations and charts are added in Stage 2.
"""
from __future__ import annotations

import re

NUMERIC = {"numeric"}
LABEL_LIKE = {"categorical", "identifier", "text"}

# Each role: what it means, which column types are allowed, words that suggest it,
# and words that rule a column OUT (e.g. "unit_price" is not total revenue).
ROLES = {
    "revenue": {
        "label": "Revenue / Sales", "icon": "💰", "types": NUMERIC,
        "keywords": ["revenue", "sales", "sale", "turnover", "income", "amount", "total", "value", "gmv"],
        "avoid": ["cost", "cogs", "profit", "margin", "price", "quantity", "qty", "units", "unit",
                  "tax", "discount", "percent", "pct", "rate", "id", "count", "expense",
                  "target", "budget", "volume"],
        "help": "The money earned from sales. Used for revenue, growth and average order value.",
    },
    "cost": {
        "label": "Cost", "icon": "🧾", "types": NUMERIC,
        "keywords": ["cost", "cogs", "expense", "expenses", "spend", "purchase"],
        "avoid": ["revenue", "sales", "profit", "margin", "price", "percent", "pct", "rate", "id", "unit"],
        "help": "The total money spent to make or buy what was sold.",
    },
    "profit": {
        "label": "Profit", "icon": "📈", "types": NUMERIC,
        "keywords": ["profit", "profits", "earning", "earnings"],
        "avoid": ["margin", "percent", "pct", "rate", "ratio", "id"],
        "help": "Money left after costs. If missing, it can be calculated as revenue minus cost.",
    },
    "quantity": {
        "label": "Quantity / Units", "icon": "📦", "types": NUMERIC,
        "keywords": ["quantity", "qty", "units", "unit", "pcs", "pieces", "volume", "count"],
        "avoid": ["price", "cost", "revenue", "profit", "amount", "id", "percent", "rate"],
        "help": "How many items were sold.",
    },
    "date": {
        "label": "Date", "icon": "📅", "types": {"date"},
        "keywords": ["date", "time", "day", "month", "year", "period", "timestamp", "created", "ordered"],
        "avoid": [],
        "help": "When each sale or event happened. Needed for trends and growth.",
    },
    "product": {
        "label": "Product / Item", "icon": "🏷️", "types": LABEL_LIKE, "prefer": "categorical",
        "keywords": ["product", "products", "item", "items", "sku", "goods", "merchandise", "article"],
        "avoid": ["category", "segment", "type", "group", "department", "customer", "order"],
        "help": "The name or code of what was sold.",
    },
    "category": {
        "label": "Category", "icon": "🗂️", "types": LABEL_LIKE, "prefer": "categorical",
        "keywords": ["category", "categories", "segment", "department", "class", "group", "type"],
        "avoid": ["customer", "id"],
        "help": "A group that products belong to (for example Electronics).",
    },
    "customer": {
        "label": "Customer", "icon": "👥", "types": LABEL_LIKE,
        "keywords": ["customer", "customers", "client", "clients", "buyer", "consumer", "member"],
        "avoid": ["city", "region", "country", "state", "segment", "type", "age", "gender", "category"],
        "help": "The name or ID of the buyer. Used to count unique customers.",
    },
    "location": {
        "label": "Branch / Location", "icon": "📍", "types": LABEL_LIKE, "prefer": "categorical",
        "keywords": ["branch", "location", "store", "shop", "outlet", "city", "region", "country",
                     "state", "province", "district", "area", "site", "market"],
        "avoid": ["manager", "customer"],
        "help": "Where the sale happened (branch, shop, city, region).",
    },
    "order": {
        "label": "Order / Invoice ID", "icon": "🧮", "types": {"identifier", "categorical"},
        "keywords": ["order", "orders", "invoice", "transaction", "receipt", "bill"],
        "avoid": ["date", "time", "amount", "total", "value", "status", "customer", "product",
                  "item", "qty", "quantity", "price", "cost"],
        "help": "A number that identifies each order. Used to count orders separately from rows.",
    },
}


def _normalise(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return re.sub(r"[^a-z0-9]+", " ", spaced.lower()).strip()


def _score_column(name: str, info: dict) -> float:
    text = _normalise(name)
    tokens = set(text.split())
    compact = text.replace(" ", "")
    if tokens & set(info["avoid"]):
        return 0.0
    score = 0.0
    for keyword in info["keywords"]:
        if keyword in tokens:
            score += 3        # the whole word appears in the column name
        elif keyword in compact:
            score += 1        # the word appears inside a longer name
    return score


def detect_columns(df, col_types: dict[str, str]) -> dict[str, str | None]:
    """Pick the best column for each role. A column is used for only one role."""
    candidates = []
    for r_idx, (role, info) in enumerate(ROLES.items()):
        for c_idx, col in enumerate(df.columns):
            if col_types.get(col) not in info["types"]:
                continue
            score = _score_column(col, info)
            if score > 0:
                if col_types[col] == info.get("prefer"):
                    score += 0.5
                candidates.append((-score, r_idx, c_idx, role, col))
    candidates.sort()

    mapping: dict[str, str | None] = {role: None for role in ROLES}
    used: set[str] = set()
    for _, _, _, role, col in candidates:
        if mapping[role] is None and col not in used:
            mapping[role] = col
            used.add(col)
    return mapping


def available_kpis(mapping: dict[str, str | None]) -> list[dict]:
    """Which KPIs can be calculated with the current column mapping, and how they will be calculated."""
    has = lambda role: bool(mapping.get(role))
    can_profit = has("profit") or (has("revenue") and has("cost"))

    return [
        {"name": "Total Revenue", "icon": "💰", "available": has("revenue"),
         "needs": "A revenue or sales column.",
         "how": "Adds up all values in the revenue column."},
        {"name": "Total Orders / Transactions", "icon": "🧮", "available": True,
         "needs": "Nothing, but an order ID column gives a more accurate count.",
         "how": "Counts unique order IDs if an order column is chosen; otherwise counts rows as transactions."},
        {"name": "Total Units Sold", "icon": "📦", "available": has("quantity"),
         "needs": "A quantity column.",
         "how": "Adds up all values in the quantity column."},
        {"name": "Total Costs", "icon": "🧾", "available": has("cost"),
         "needs": "A cost column.",
         "how": "Adds up all values in the cost column."},
        {"name": "Total Profit", "icon": "📈", "available": can_profit,
         "needs": "A profit column, or both revenue and cost columns.",
         "how": "Uses the profit column, or revenue minus cost when no profit column exists."},
        {"name": "Profit Margin", "icon": "🎯", "available": can_profit and has("revenue"),
         "needs": "Revenue plus either profit or cost.",
         "how": "Total profit divided by total revenue, shown as a percentage (skipped if revenue is zero)."},
        {"name": "Average Order Value", "icon": "🛒", "available": has("revenue"),
         "needs": "A revenue column.",
         "how": "Total revenue divided by the number of orders (or transactions)."},
        {"name": "Unique Customers", "icon": "👥", "available": has("customer"),
         "needs": "A customer column.",
         "how": "Counts different customers, not rows. One customer who buys 5 times counts once."},
        {"name": "Number of Products", "icon": "🏷️", "available": has("product"),
         "needs": "A product column.",
         "how": "Counts the different products in the product column."},
        {"name": "Sales Growth", "icon": "🚀", "available": has("revenue") and has("date"),
         "needs": "Revenue and date columns, with at least two comparable periods.",
         "how": "Compares revenue in the latest period with the period before it."},
        {"name": "Best-Selling Product", "icon": "🏆", "available": has("product") and (has("revenue") or has("quantity")),
         "needs": "A product column plus revenue or quantity.",
         "how": "The product with the highest total revenue (or quantity if there is no revenue)."},
        {"name": "Top Branch / Location", "icon": "📍", "available": has("location") and has("revenue"),
         "needs": "A location column and a revenue column.",
         "how": "The location with the highest total revenue."},
        {"name": "Best-Performing Period", "icon": "📅", "available": has("date") and has("revenue"),
         "needs": "Date and revenue columns.",
         "how": "The month with the highest total revenue."},
    ]


def suggest_charts(mapping: dict[str, str | None], col_types: dict[str, str]) -> list[dict]:
    """Which charts would make sense for this dataset."""
    has = lambda role: bool(mapping.get(role))
    n_numeric = sum(1 for t in col_types.values() if t == "numeric")
    return [
        {"name": "Revenue trend over time (line chart)", "available": has("revenue") and has("date"),
         "needs": "revenue + date"},
        {"name": "Top products (bar chart)", "available": has("product") and (has("revenue") or has("quantity")),
         "needs": "product + revenue or quantity"},
        {"name": "Category performance (bar / donut chart)", "available": has("category") and has("revenue"),
         "needs": "category + revenue"},
        {"name": "Sales by location (bar chart)", "available": has("location") and has("revenue"),
         "needs": "location + revenue"},
        {"name": "Revenue vs cost vs profit", "available": has("revenue") and (has("cost") or has("profit")),
         "needs": "revenue + cost or profit"},
        {"name": "Customer purchasing summary", "available": has("customer") and (has("revenue") or has("order")),
         "needs": "customer + revenue or order ID"},
        {"name": "Distribution of numbers (histogram)", "available": n_numeric >= 1,
         "needs": "at least one numeric column"},
        {"name": "Relationships between numbers (scatter / correlation heatmap)", "available": n_numeric >= 2,
         "needs": "at least two numeric columns"},
    ]
