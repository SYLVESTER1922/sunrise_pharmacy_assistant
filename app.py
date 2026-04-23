import os
import re
import psycopg2
import pandas as pd
from psycopg2 import pool
import gradio as gr
from neo4j import GraphDatabase
from openai import OpenAI
from difflib import SequenceMatcher
from datetime import datetime, date
import json

# ── Credentials ───────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Supabase PostgreSQL connection ───────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")

# Connection pool — thread safe, reuses connections efficiently
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(1, 10, SUPABASE_URL)
    return _pool

def get_conn():
    """
    Returns a live PostgreSQL connection from the pool.
    Every query reads directly from Supabase — always current data.
    To update data: edit Google Sheets → run the loader script → data updates instantly.
    """
    return get_pool().getconn()

def release_conn(conn):
    get_pool().putconn(conn)

print("Supabase connection pool ready ✓")

# ── Load drug list ────────────────────────────────────────────
def get_all_drugs():
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT generic_name, brand_name, category FROM inventory ORDER BY generic_name",
            conn
        )
    finally:
        release_conn(conn)
    return df

DRUGS_DF   = get_all_drugs()
DRUG_NAMES = DRUGS_DF["generic_name"].tolist()

# ── Fuzzy drug name matching ──────────────────────────────────
def fuzzy_match_drug(text, threshold=78):
    text = re.sub(r"['\u2019\u2018`]", "", text.lower().strip())
    best_score = 0
    best_match = None
    for drug in DRUG_NAMES:
        drug_lower = drug.lower()
        if text in drug_lower or drug_lower in text:
            return drug
        score = SequenceMatcher(None, text, drug_lower).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = drug
    return best_match if best_score >= threshold else None

def fuzzy_correct_question(question):
    skip_words = {
        "what", "when", "where", "which", "have", "does", "there",
        "that", "this", "with", "from", "give", "will", "about",
        "tell", "show", "list", "find", "drug", "stock", "price",
        "cost", "batch", "expiry", "supplier", "interact", "used",
        "alternative", "recommend", "please", "could", "would",
        "anything", "something", "medicine", "medication", "tablet",
        "capsule", "syrup", "injection", "soon", "selling", "sales"
    }
    words = re.sub(r"['\u2019?!,.]", "", question).split()
    corrections = []
    corrected_words = list(words)
    for i, word in enumerate(words):
        w = word.lower()
        if len(w) < 4 or w in skip_words:
            continue
        match = fuzzy_match_drug(w, threshold=78)
        if match and match.lower() != w:
            corrected_words[i] = match
            corrections.append(f"'{word}' → '{match}'")
    corrected = " ".join(corrected_words)
    note = f"*(Auto-corrected: {', '.join(corrections)})*" if corrections else ""
    return corrected, note

# ── Intent classification ─────────────────────────────────────
GREETINGS = {
    "hi", "hey", "hello", "good morning", "good afternoon",
    "good evening", "help", "what can you do", "how are you",
    "what do you do", "who are you", "what are you", "start"
}
THANKS    = {"thank you", "thanks", "thank", "cheers", "appreciated"}
FAREWELLS = {"bye", "goodbye", "see you", "later", "exit", "quit"}

def classify_intent(question, conversation_history=None):
    q = question.lower().strip()
    q_clean = re.sub(r"['\u2019?!,.]", "", q)

    # System intents
    if any(q_clean == g or q_clean.startswith(g) for g in GREETINGS):
        return "greeting"
    if any(g in q_clean for g in THANKS):
        return "thanks"
    if any(g in q_clean for g in FAREWELLS):
        return "farewell"

    # Follow-up detection
    followup_refs = [
        "it", "this", "that", "these", "those", "them", "they",
        "the same", "above", "mentioned", "you said", "tell me more",
        "more about", "elaborate", "go on", "continue", "what else",
        "expand", "in detail", "and what about", "how about",
        "what about", "also", "another"
    ]
    if (conversation_history and len(q.split()) <= 8 and
            any(ref in q_clean for ref in followup_refs)):
        return "followup"

    # LOW STOCK — must come before stock_price
    if any(w in q for w in [
        "low stock", "running low", "almost out", "reorder",
        "need to order", "below reorder", "critical stock",
        "running low on stock", "low on stock", "need reorder",
        "stock alert", "drugs running", "which drugs are",
        "what drugs are", "what products are"
    ]):
        return "low_stock"

    # STOCK / PRICE
    if any(w in q for w in [
        "stock", "have", "available", "availability", "quantity",
        "how many", "price", "cost", "how much", "in stock",
        "do we have", "do you have", "shelf"
    ]):
        return "stock_price"

    # CATEGORY BROWSE
    if any(w in q for w in [
        "antibiotics", "antibiotic", "analgesics", "analgesic",
        "antihypertensives", "antihypertensive",
        "antidiabetics", "antidiabetic",
        "antimalarials", "antimalarial",
        "vitamins", "vitamin", "supplements",
        "antifungals", "antifungal",
        "gi medications", "gastrointestinal",
        "respiratory", "antiretrovirals", "antiretroviral",
        "arvs", "arv", "hiv drugs",
        "list all", "show all", "all drugs", "all medicines"
    ]):
        return "category_browse"

    # STATS — must check before stock_price
    if any(w in q for w in [
        "how many drugs", "how many types", "how many medicines",
        "how many categories", "how many drug", "drug categories",
        "total drugs", "drug types", "categories we have",
        "how many do we have", "inventory summary", "stock summary",
        "inventory overview", "stock overview"
    ]):
        return "stats"

    # EXPIRY
    if any(w in q for w in [
        "expir", "expire", "expiry", "batch", "shelf life",
        "best before", "use by", "when does",
        "most urgent", "urgent batch", "needs action", "critical batch"
    ]):
        return "expiry"

    # INTERACTIONS — CLINICAL
    if any(w in q for w in [
        "interact", "interaction", "together", "combine", "mix",
        "safe with", "take with", "used with", "combined with",
        "contraindic", "avoid with"
    ]):
        return "interaction"

    # SUPPLIER
    if any(w in q for w in [
        "supplier", "order from", "who supply", "distributor",
        "vendor", "supplies", "supply", "who provides",
        "where do we order", "procurement", "purchase from",
        "buy from", "source"
    ]):
        return "supplier"

    # SALES
    if any(w in q for w in [
        "sold", "sales", "revenue", "dispensed", "transaction",
        "top selling", "best selling", "most popular", "most sold",
        "highest revenue", "performance", "turnover",
        "customer type", "customer breakdown", "by customer",
        "breakdown", "split by", "prescription sales",
        "walk-in", "insurance sales"
    ]):
        return "sales"

    # ALTERNATIVES
    if any(w in q for w in [
        "alternative", "substitute", "instead of", "replace",
        "similar to", "other option", "other drug", "swap",
        "equivalent", "whats another", "what else can"
    ]):
        return "alternative"

    # DRUG INFO — CLINICAL (default)
    return "drug_info"

# ── Keyword extractor ─────────────────────────────────────────
STOPWORDS = {
    "what", "is", "the", "for", "do", "we", "have", "any", "of",
    "tell", "me", "about", "price", "cost", "stock", "interact",
    "with", "how", "much", "many", "does", "supplier", "supplies",
    "supply", "who", "interacts", "use", "used", "expiry", "expire",
    "when", "a", "an", "drug", "medicine", "medication", "our",
    "give", "alternative", "substitute", "instead", "similar",
    "whats", "can", "you", "recommend", "please", "there", "get",
    "find", "show", "list", "all", "are", "in", "to", "from",
    "that", "this", "on", "at", "by", "or", "and", "also", "its",
    "which", "where", "would", "could", "should", "will", "was",
    "been", "being", "has", "had", "not", "no", "yes"
}

def extract_keywords(question):
    clean = re.sub(r"['\u2019?!,.]", "", question.lower())
    return [w for w in clean.split() if w not in STOPWORDS and len(w) > 2]

def get_search_term(question):
    keywords = extract_keywords(question)
    return keywords[0] if keywords else question.lower()

# ═══════════════════════════════════════════════════════════════
# OPERATIONAL FORMATTERS — Pure data, no GPT, no hallucination
# ═══════════════════════════════════════════════════════════════

def format_stock_price(question):
    keywords = extract_keywords(question)
    if not keywords:
        return "❌ Please specify a drug name to check stock."
    conditions = " OR ".join(
        ["LOWER(generic_name) LIKE %s OR LOWER(brand_name) LIKE %s" for _ in keywords]
    )
    params = []
    for k in keywords:
        params.extend([f"%{k}%", f"%{k}%"])
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category
        FROM inventory WHERE {conditions} LIMIT 5
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        release_conn(conn)
    if df.empty:
        return "❌ Drug not found in inventory. Please check the spelling or use the Drug Lookup."
    lines = []
    for _, r in df.iterrows():
        stock_flag = "⚠️ LOW STOCK" if r['quantity_in_stock'] <= r['reorder_level'] else "✅ In Stock"
        lines.append(f"""**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}
| Field | Value |
|---|---|
| Stock | {r['quantity_in_stock']} units — {stock_flag} |
| Reorder Level | {r['reorder_level']} units |
| Selling Price | ${r['selling_price_usd']} |
| Cost Price | ${r['cost_price_usd']} |
| Shelf Location | {r['shelf_location']} |
| Category | {r['category']} |
""")
    return "\n".join(lines)

def format_category_browse(question):
    q = question.lower()
    categories = {
        "antibiotic":       "Antibiotics",
        "analgesic":        "Analgesics",
        "antihypertensive": "Antihypertensives",
        "antidiabetic":     "Antidiabetics",
        "antimalarial":     "Antimalarials",
        "vitamin":          "Vitamins/Supplements",
        "supplement":       "Vitamins/Supplements",
        "antifungal":       "Antifungals",
        "gi":               "GI medications",
        "gastrointestinal": "GI medications",
        "respiratory":      "Respiratory",
        "antiretroviral":   "Antiretrovirals",
        "arv":              "Antiretrovirals",
        "hiv":              "Antiretrovirals",
    }
    matched = None
    for keyword, category in categories.items():
        if keyword in q:
            matched = category
            break
    if not matched:
        return "❌ Category not recognised. Try: Antibiotics, Analgesics, Antiretrovirals, etc."
    sql = """
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, selling_price_usd, shelf_location
        FROM inventory WHERE category = %s
        ORDER BY generic_name
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=(matched,))
    finally:
        release_conn(conn)
    if df.empty:
        return f"❌ No drugs found in category: {matched}"
    header = f"**{matched}** — {len(df)} drugs\n\n"
    header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        stock_icon = "⚠️" if r['quantity_in_stock'] <= 0 else ""
        rows.append(
            f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
            f"{r['strength']} | {r['quantity_in_stock']}{stock_icon} | "
            f"${r['selling_price_usd']} | {r['shelf_location']} |"
        )
    return header + "\n".join(rows)

def format_stats():
    sql = """
        SELECT category,
               COUNT(*)               AS drug_count,
               SUM(quantity_in_stock) AS total_units,
               ROUND(AVG(selling_price_usd), 2) AS avg_price,
               ROUND(SUM(quantity_in_stock * cost_price_usd), 2) AS inventory_value
        FROM inventory
        GROUP BY category ORDER BY inventory_value DESC
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        release_conn(conn)
    total_drugs = df['drug_count'].sum()
    total_value = df['inventory_value'].sum()
    header = f"**Inventory Summary** — {total_drugs} products across {len(df)} categories\n\n"
    header += f"Total inventory value: **${total_value:,.2f}**\n\n"
    header += "| Category | Drugs | Total Units | Avg Price | Inv. Value |\n"
    header += "|---|---|---|---|---|\n"
    rows = [
        f"| {r['category']} | {r['drug_count']} | {r['total_units']} | "
        f"${r['avg_price']} | ${r['inventory_value']:,.2f} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows)

def format_low_stock():
    sql = """
        SELECT generic_name, brand_name, quantity_in_stock,
               reorder_level, category,
               ROUND((CAST(quantity_in_stock AS FLOAT) / reorder_level) * 100, 0)
               AS stock_pct
        FROM inventory
        WHERE quantity_in_stock <= reorder_level
        ORDER BY stock_pct ASC
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        release_conn(conn)
    if df.empty:
        return "✅ **Good news** — all products are currently above their reorder levels. No drugs are running low at this time."
    header = f"⚠️ **{len(df)} drug(s) at or below reorder level:**\n\n"
    header += "| Drug | Brand | Stock | Reorder Level | % of Reorder | Category |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | "
        f"{r['reorder_level']} | {r['stock_pct']:.0f}% | {r['category']} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows)

def format_expiry(question):
    q = question.lower()
    today = date.today()
    if any(w in q for w in [
        "soon", "this month", "next month", "90 days", "expiring",
        "about to expire", "near expiry", "earliest", "most urgent",
        "urgent", "action", "critical"
    ]):
        sql = """
            SELECT i.generic_name, i.brand_name, b.batch_number,
                   b.expiry_date, b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER)
                   AS days_remaining
            FROM batches b
            JOIN inventory i ON b.product_id = i.product_id
            WHERE julianday(b.expiry_date) - julianday('now') <= 90
            ORDER BY b.expiry_date ASC
        """
        conn = get_conn()
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            release_conn(conn)
        if df.empty:
            return "✅ No batches expiring within the next 90 days."
        header = f"**Batches expiring within 90 days** — {len(df)} batch(es):\n\n"
        header += "| Drug | Brand | Batch | Expiry Date | Days Left | Qty | Status |\n"
        header += "|---|---|---|---|---|---|---|\n"
        rows = []
        for _, r in df.iterrows():
            if r['days_remaining'] <= 30:
                status = "🚨 URGENT"
            elif r['days_remaining'] <= 60:
                status = "⚠️ Warning"
            else:
                status = "📅 Monitor"
            rows.append(
                f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | "
                f"{r['expiry_date']} | **{r['days_remaining']}** | "
                f"{r['quantity_remaining']} | {status} |"
            )
        return header + "\n".join(rows)
    else:
        keywords = extract_keywords(question)
        if not keywords:
            return "❌ Please specify a drug name to check expiry."
        conditions = " OR ".join(["LOWER(i.generic_name) LIKE %s" for _ in keywords])
        params = [f"%{k}%" for k in keywords]
        sql = f"""
            SELECT i.generic_name, i.brand_name, b.batch_number,
                   b.expiry_date, b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER)
                   AS days_remaining
            FROM batches b
            JOIN inventory i ON b.product_id = i.product_id
            WHERE {conditions}
            ORDER BY b.expiry_date ASC
        """
        conn = get_conn()
        try:
            df = pd.read_sql_query(sql, conn, params=params)
        finally:
            release_conn(conn)
        if df.empty:
            return "❌ No batch records found for that drug."
        header = "| Drug | Brand | Batch | Expiry Date | Days Left | Qty |\n"
        header = f"**Expiry records:**\n\n{header}|---|---|---|---|---|---|\n"
        rows = []
        for _, r in df.iterrows():
            days = r['days_remaining']
            flag = " 🚨 URGENT" if days <= 30 else (" ⚠️" if days <= 60 else "")
            rows.append(
                f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | "
                f"{r['expiry_date']} | **{days}**{flag} | {r['quantity_remaining']} |"
            )
        return header + "\n".join(rows)

def format_sales(question):
    q = question.lower()
    if any(w in q for w in [
        "customer type", "customer breakdown", "by customer",
        "breakdown", "split", "prescription", "walk-in",
        "walkin", "insurance", "type of customer"
    ]):
        sql = """
            SELECT customer_type,
                   COUNT(*)                    AS num_transactions,
                   SUM(quantity_sold)          AS total_units,
                   ROUND(SUM(total_amount), 2) AS total_revenue,
                   ROUND(SUM(total_amount) * 100.0 /
                       (SELECT SUM(total_amount) FROM transactions), 1)
                   AS revenue_pct
            FROM transactions
            GROUP BY customer_type
            ORDER BY total_revenue DESC
        """
        conn = get_conn()
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            release_conn(conn)
        header = "**Sales by Customer Type** (Last 30 days)\n\n"
        header += "| Customer Type | Transactions | Units Sold | Revenue | % of Total |\n"
        header += "|---|---|---|---|---|\n"
        rows = [
            f"| {r['customer_type']} | {r['num_transactions']} | "
            f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['revenue_pct']}% |"
            for _, r in df.iterrows()
        ]
        total = df['total_revenue'].sum()
        footer = f"\n**Total Revenue: ${total:,.2f}**"
        return header + "\n".join(rows) + footer
    else:
        sql = """
            SELECT i.brand_name, i.generic_name,
                   SUM(t.quantity_sold)          AS total_units,
                   ROUND(SUM(t.total_amount), 2) AS total_revenue,
                   COUNT(*)                       AS num_transactions
            FROM transactions t
            JOIN inventory i ON t.product_id = i.product_id
            GROUP BY i.brand_name, i.generic_name
            ORDER BY total_revenue DESC LIMIT 10
        """
        conn = get_conn()
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            release_conn(conn)
        header = "**Top 10 Selling Drugs** (Last 30 days)\n\n"
        header += "| Rank | Brand | Generic | Units | Revenue | Transactions |\n"
        header += "|---|---|---|---|---|---|\n"
        rows = [
            f"| {i+1} | {r['brand_name']} | {r['generic_name']} | "
            f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['num_transactions']} |"
            for i, (_, r) in enumerate(df.iterrows())
        ]
        return header + "\n".join(rows)

def format_supplier(question):
    keywords = extract_keywords(question)
    search_term = keywords[0] if keywords else get_search_term(question)
    cypher = """
        MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
        WHERE toLower(d.generic_name) CONTAINS toLower($search)
        RETURN d.generic_name AS drug, s.name AS supplier,
               s.contact AS contact, s.phone AS phone,
               s.city AS city, s.lead_time AS lead_time_days,
               s.payment_terms AS payment_terms
        LIMIT 5
    """
    with driver.session() as session:
        results = [dict(r) for r in session.run(cypher, search=search_term)]
    if not results:
        return "❌ No supplier information found for that drug."
    lines = [f"**Supplier information for {results[0]['drug']}:**\n"]
    seen = set()
    for r in results:
        key = r['supplier']
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"""| Field | Value |
|---|---|
| Supplier | **{r['supplier']}** |
| Contact | {r['contact']} |
| Phone | {r['phone']} |
| City | {r['city']} |
| Lead Time | {r['lead_time_days']} days |
| Payment Terms | {r['payment_terms']} |
""")
    return "\n".join(lines)

def format_alternative(question):
    keywords = extract_keywords(question)
    search_param = None
    for k in keywords:
        sql_check = "SELECT generic_name, category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1"
        conn = get_conn()
        try:
            result = pd.read_sql_query(sql_check, conn, params=(f"%{k}%",))
        finally:
            release_conn(conn)
        if not result.empty:
            search_param = f"%{k}%"
            drug_name = result.iloc[0]['generic_name']
            category = result.iloc[0]['category']
            break
    if not search_param:
        return "❌ Drug not found in inventory. Please check the spelling."
    sql = """
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, selling_price_usd, shelf_location
        FROM inventory
        WHERE category = (
            SELECT category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1
        )
        AND LOWER(generic_name) NOT LIKE %s
        AND quantity_in_stock > 0
        ORDER BY generic_name
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=(search_param, search_param))
    finally:
        release_conn(conn)
    if df.empty:
        return f"❌ No alternatives found in the same category as {drug_name}."
    header = f"**Alternatives to {drug_name}** (same category: {category})\n\n"
    header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
        f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | "
        f"{r['shelf_location']} |"
        for _, r in df.iterrows()
    ]
    disclaimer = ("\n\n⚠️ **Clinical Note:** Therapeutic substitution must be approved "
                  "by a qualified pharmacist. This list shows drugs in the same category only.")
    return header + "\n".join(rows) + disclaimer

def format_drug_summary(drug_name):
    sql = """
        SELECT i.generic_name, i.brand_name, i.formulation, i.strength,
               i.quantity_in_stock, i.reorder_level,
               i.selling_price_usd, i.cost_price_usd,
               i.shelf_location, i.category,
               MIN(b.expiry_date) AS nearest_expiry,
               CAST(MIN(julianday(b.expiry_date) - julianday('now')) AS INTEGER)
               AS days_to_expiry
        FROM inventory i
        LEFT JOIN batches b ON i.product_id = b.product_id
        WHERE LOWER(i.generic_name) LIKE LOWER(%s)
        GROUP BY i.product_id LIMIT 1
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=(f"%{drug_name}%",))
    finally:
        release_conn(conn)
    if df.empty:
        return f"❌ **{drug_name}** not found in inventory. Please check the spelling."
    r = df.iloc[0]
    stock_status = "⚠️ LOW STOCK — reorder needed" if r['quantity_in_stock'] <= r['reorder_level'] else "✅ In Stock"
    expiry_line = ""
    if r.get("days_to_expiry") is not None:
        d = r['days_to_expiry']
        if d <= 30:
            expiry_line = f"\n🚨 **URGENT:** Nearest batch expires in {d} days ({r['nearest_expiry']})"
        elif d <= 90:
            expiry_line = f"\n⚠️ Nearest expiry: {r['nearest_expiry']} ({d} days)"
        else:
            expiry_line = f"\n📅 Nearest expiry: {r['nearest_expiry']} ({d} days)"
    return f"""**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}

| Field | Value |
|---|---|
| **Stock** | {r['quantity_in_stock']} units — {stock_status} |
| **Reorder Level** | {r['reorder_level']} units |
| **Selling Price** | ${r['selling_price_usd']} |
| **Cost Price** | ${r['cost_price_usd']} |
| **Shelf Location** | {r['shelf_location']} |
| **Category** | {r['category']} |
{expiry_line}"""

# ═══════════════════════════════════════════════════════════════
# CLINICAL MODE — GPT with strict grounding + disclaimer
# ═══════════════════════════════════════════════════════════════

CLINICAL_DISCLAIMER = (
    "\n\n---\n⚠️ **Clinical Disclaimer:** This information is sourced from the pharmacy "
    "knowledge base. Always verify drug interactions, dosages and contraindications "
    "with a qualified pharmacist before dispensing."
)

CLINICAL_SYSTEM_PROMPT = """You are a pharmacy data assistant at Sunrise Pharmacy, Harare, Zimbabwe.
You are given STRUCTURED DATA retrieved from the pharmacy knowledge graph.
Your ONLY job is to summarise that data clearly for pharmacy staff.

ABSOLUTE RULES — violating these is not permitted under any circumstances:
1. Use ONLY the data provided below. Never add information from your training knowledge.
2. If the data does not contain the answer, say exactly: "This information is not available in our knowledge base."
3. Never invent, guess or infer drug names, doses, quantities, interactions or clinical facts.
4. Never add interactions, contraindications or side effects not explicitly present in the data.
5. Keep the answer to 3-5 sentences. Be precise and factual.
6. For interactions, always state the exact severity level from the data (Minor/Moderate/Major).
7. End every answer with: "Source: [data source name]"
"""

def generate_clinical_answer(question, intent, source, data, conversation_history=None):
    """
    GPT is ONLY called for clinical queries (interactions, drug_info).
    All other queries use direct data formatters.
    """
    if not data:
        if intent == "interaction":
            return ("No interactions found for that drug in our knowledge base. "
                    "Please consult a clinical pharmacist or reference guide." +
                    CLINICAL_DISCLAIMER)
        return ("No information found for that drug in our knowledge base. "
                "Please check the drug name." + CLINICAL_DISCLAIMER)

    messages = [{"role": "system", "content": CLINICAL_SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = f"""
RETRIEVED DATA FROM KNOWLEDGE BASE:
{json.dumps(data, indent=2)}

QUESTION FROM PHARMACY STAFF: {question}

Summarise the above data to answer the question. Use ONLY what is in the data above.
"""
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=400
    )
    return response.choices[0].message.content + CLINICAL_DISCLAIMER

def query_neo4j_interaction(question):
    keywords = extract_keywords(question)
    search_term = keywords[0] if keywords else get_search_term(question)
    cypher = """
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) CONTAINS toLower($search)
           OR toLower(b.generic_name) CONTAINS toLower($search)
        RETURN a.generic_name AS drug_a, b.generic_name AS drug_b,
               r.severity AS severity, r.description AS description,
               r.recommendation AS recommendation
        ORDER BY CASE r.severity
            WHEN 'Major' THEN 1 WHEN 'Moderate' THEN 2
            WHEN 'Minor' THEN 3 ELSE 4 END
        LIMIT 5
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, search=search_term)]

def query_neo4j_drug_info(question):
    keywords = extract_keywords(question)
    search_term = keywords[0] if keywords else get_search_term(question)
    cypher = """
        MATCH (d:Drug)-[:IN_CATEGORY]->(c:Category)
        WHERE toLower(d.generic_name) CONTAINS toLower($search)
        RETURN d.generic_name AS name, d.drug_class AS drug_class,
               d.indications AS indications,
               d.contraindications AS contraindications,
               d.side_effects AS side_effects,
               d.adult_dose AS adult_dose,
               d.pediatric_dose AS pediatric_dose,
               d.prescription AS prescription,
               d.controlled AS controlled,
               c.name AS category
        LIMIT 3
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, search=search_term)]

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTER — dispatches to operational or clinical mode
# ═══════════════════════════════════════════════════════════════

GREETING_RESPONSE = """👋 Hello! I am the **Netrisyl Pharmacy Assistant** for Sunrise Pharmacy.

I can help pharmacy staff with:
- 📦 **Stock & Prices** — *"Do we have amoxicillin in stock?"*
- ⚠️ **Drug Interactions** — *"What interacts with metformin?"* *(Clinical — verified data only)*
- 📅 **Expiry Alerts** — *"Which batches are expiring soon?"*
- 🚚 **Suppliers** — *"Who supplies ciprofloxacin?"*
- 💊 **Drug Information** — *"What is ibuprofen used for?"* *(Clinical — verified data only)*
- 🔄 **Alternatives** — *"What is an alternative to amoxicillin?"*
- 💰 **Sales Summary** — *"What are the top selling drugs?"*
- 📊 **Category Browse** — *"List all antibiotics"*
- 🔴 **Low Stock Alerts** — *"Which drugs are running low?"*
- 📈 **Inventory Stats** — *"How many drug categories do we have?"*

⚠️ **Important:** Clinical answers (interactions, dosages) are sourced strictly from our knowledge base and always include a pharmacist verification reminder.

How can I help you today?"""

THANKS_RESPONSE   = "You're welcome! Feel free to ask anytime. 😊"
FAREWELL_RESPONSE = "Goodbye! Come back anytime you need help. 👋"

def route_and_respond(question, intent, conversation_history=None):
    """
    Routes to operational formatter (no GPT) or clinical GPT mode.
    Returns (answer, source, mode) where mode is 'operational' or 'clinical'.
    """
    # ── System responses ──────────────────────────────────────
    if intent == "greeting":
        return GREETING_RESPONSE, "system", "system"
    if intent == "thanks":
        return THANKS_RESPONSE, "system", "system"
    if intent == "farewell":
        return FAREWELL_RESPONSE, "system", "system"

    # ── Follow-up — GPT with conversation context only ────────
    if intent == "followup":
        if not conversation_history:
            return "Could you clarify your question?", "system", "system"
        messages = [{"role": "system", "content": CLINICAL_SYSTEM_PROMPT}]
        for turn in conversation_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            temperature=0.0, max_tokens=300
        )
        return response.choices[0].message.content, "conversation history", "clinical"

    # ── OPERATIONAL MODE — direct data formatting, no GPT ─────
    if intent == "stock_price":
        return format_stock_price(question), "inventory database", "operational"
    if intent == "category_browse":
        return format_category_browse(question), "inventory database", "operational"
    if intent == "stats":
        return format_stats(), "inventory database", "operational"
    if intent == "low_stock":
        return format_low_stock(), "inventory database", "operational"
    if intent == "expiry":
        return format_expiry(question), "batch records", "operational"
    if intent == "sales":
        return format_sales(question), "transaction records", "operational"
    if intent == "supplier":
        return format_supplier(question), "supplier knowledge graph", "operational"
    if intent == "alternative":
        return format_alternative(question), "inventory database", "operational"

    # ── CLINICAL MODE — GPT with strict grounding ─────────────
    if intent == "interaction":
        data = query_neo4j_interaction(question)
        answer = generate_clinical_answer(
            question, intent, "drug interaction knowledge graph",
            data, conversation_history
        )
        return answer, "drug interaction knowledge graph", "clinical"

    if intent == "drug_info":
        data = query_neo4j_drug_info(question)
        answer = generate_clinical_answer(
            question, intent, "drug knowledge graph",
            data, conversation_history
        )
        return answer, "drug knowledge graph", "clinical"

    return "I could not process that request. Please try rephrasing.", "system", "system"

# ── Export chat ───────────────────────────────────────────────
def export_chat(chat_history):
    if not chat_history:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"pharmacy_chat_{timestamp}.txt"
    lines = [
        "Netrisyl Pharmacy Assistant — Chat Export",
        "Sunrise Pharmacy | Harare, Zimbabwe",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60, ""
    ]
    for msg in chat_history:
        role = "Staff" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{msg['content']}\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return filename

# ── Drug search filter ────────────────────────────────────────
def filter_drugs(search_text):
    if not search_text or len(search_text) < 2:
        return gr.update(choices=DRUG_NAMES[:20])
    matches = [d for d in DRUG_NAMES if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches if matches else DRUG_NAMES[:20])

# ── Core respond function ─────────────────────────────────────
def respond(message, chat_history, search_history):
    if not message or message.strip() == "":
        return "", chat_history, search_history, gr.update(), gr.update()

    conversation_history = [
        {"role": t["role"], "content": t["content"]}
        for t in (chat_history or [])
    ]

    try:
        corrected_message, correction_note = fuzzy_correct_question(message)
        intent = classify_intent(corrected_message, conversation_history)
        answer, source, mode = route_and_respond(
            corrected_message, intent, conversation_history
        )

        if mode == "system":
            full_answer = answer
        elif mode == "operational":
            header = f"*📦 Operational data — {source}*\n\n"
            full_answer = header + answer
            if correction_note:
                full_answer = f"{correction_note}\n\n{full_answer}"
        else:  # clinical
            header = f"*🧪 Clinical data — {source}*\n\n"
            full_answer = header + answer
            if correction_note:
                full_answer = f"{correction_note}\n\n{full_answer}"

    except Exception as e:
        full_answer = f"An error occurred: {str(e)}\nPlease try rephrasing your question."

    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})

    search_history = list(search_history or [])
    if message not in search_history:
        search_history.insert(0, message)
    search_history = search_history[:15]
    history_md = "\n".join([f"- {h}" for h in search_history])

    return (
        "",
        chat_history,
        search_history,
        gr.update(choices=search_history, value=None),
        gr.update(value=history_md)
    )

def drug_summary_respond(drug_name, chat_history, search_history):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update()
    try:
        answer = format_drug_summary(drug_name)
        header = "*📦 Operational data — inventory + batch records*\n\n"
        full_answer = header + answer
    except Exception as e:
        full_answer = f"Error: {str(e)}"
    label = f"Quick summary: {drug_name}"
    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": label})
    chat_history.append({"role": "assistant", "content": full_answer})
    search_history = list(search_history or [])
    if label not in search_history:
        search_history.insert(0, label)
    search_history = search_history[:15]
    history_md = "\n".join([f"- {h}" for h in search_history])
    return (
        chat_history,
        search_history,
        gr.update(choices=search_history, value=None),
        gr.update(value=history_md)
    )

def click_quick_question(question, chat_history, search_history):
    return respond(question, chat_history, search_history)

def reask_from_history(selected_question, chat_history, search_history):
    if not selected_question:
        return "", chat_history, search_history, gr.update(), gr.update()
    return respond(selected_question, chat_history, search_history)

# ── Featured drugs & quick questions ─────────────────────────
FEATURED_DRUGS = [
    "Amoxicillin", "Paracetamol", "Metformin", "Ibuprofen",
    "Ciprofloxacin", "Azithromycin", "Amlodipine", "Losartan",
    "Artemether/Lumefantrine", "Co-trimoxazole"
]

QUICK_QUESTIONS = [
    "Do we have amoxicillin in stock?",
    "What interacts with metformin?",
    "Which batches are expiring soon?",
    "Who supplies ciprofloxacin?",
    "What are the top selling drugs?",
    "Which drugs are running low on stock?"
]

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Netrisyl Pharmacy Assistant") as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #0d1b2a, #1a3a5c);
                padding: 16px 24px; border-radius: 10px; margin-bottom: 16px;
                display: flex; align-items: center; justify-content: space-between;">
        <img src="https://huggingface.co/spaces/Sylvester1922/Netrisyl_pharmacy_assistant/resolve/main/NI_Logo.png"
             style="height: 70px; object-fit: contain;" alt="Netrisyl Insights"
             onerror="this.style.display='none'"/>
        <div style="text-align: center; flex: 1;">
            <h1 style="color: white; margin: 0; font-size: 24px;">
                💊 Pharmacy Assistant
            </h1>
            <p style="color: #aed6f1; margin: 4px 0 0 0; font-size: 13px;">
                Powered by Neo4j Knowledge Graph + GPT-4o-mini | Harare, Zimbabwe
            </p>
        </div>
        <div style="width: 180px;"></div>
    </div>
    """)

    with gr.Row():

        # ── LEFT sidebar — Drug Lookup ────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🔍 Drug Lookup")
            drug_search = gr.Textbox(
                placeholder="Type e.g. amox...",
                label="Search drug name"
            )
            drug_dropdown = gr.Dropdown(
                choices=DRUG_NAMES[:20],
                label="Select drug",
                interactive=True
            )
            drug_lookup_btn = gr.Button("📋 Get Summary", variant="primary", size="sm")
            gr.Markdown("---")
            gr.Markdown("""
**Data Sources:**
- 📦 Inventory & Pricing
- 🧪 Drug Knowledge Graph
- ⚠️ Drug Interactions
- 📅 Batch & Expiry Records
- 🚚 Supplier Network
- 💰 30-Day Transactions
            """)
            gr.Markdown("---")
            gr.Markdown("""
**Modes:**
- 📦 *Operational* — direct data, no AI interpretation
- 🧪 *Clinical* — AI summary + pharmacist disclaimer
            """)

        # ── CENTRE — Chat ─────────────────────────────────────
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(label="Pharmacy Assistant", height=460)
            gr.Markdown("**💊 Quick Drug Lookup** — click for instant summary:")
            with gr.Row():
                drug_chips = [gr.Button(d, variant="secondary", size="sm")
                              for d in FEATURED_DRUGS[:5]]
            with gr.Row():
                drug_chips2 = [gr.Button(d, variant="secondary", size="sm")
                               for d in FEATURED_DRUGS[5:]]
            with gr.Row():
                msg    = gr.Textbox(
                    placeholder="Ask by drug name e.g. 'Do we have Amoxicillin?' or 'What interacts with Metformin?'",
                    label="",
                    scale=5
                )
                submit = gr.Button("Ask", variant="primary", scale=1)
            with gr.Row():
                export_btn  = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file = gr.File(label="Download", scale=2, visible=False)

        # ── RIGHT sidebar — Questions & History ───────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 💡 Quick Questions")
            quick_btns = [gr.Button(q, variant="secondary", size="sm")
                          for q in QUICK_QUESTIONS]
            gr.Markdown("---")
            gr.Markdown("### 🕘 Search History")
            history_dropdown = gr.Dropdown(
                choices=[],
                label="Select a past question to re-ask",
                interactive=True
            )
            history_display = gr.Markdown("*No searches yet*")

    gr.HTML("""
    <div style="text-align:center; margin-top:16px; color:#7f8c8d; font-size:12px;">
        Netrisyl Insights · Harare, Zimbabwe · Data. Analytics. Intelligence.
    </div>
    """)

    search_history_state = gr.State([])

    submit.click(respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display])
    msg.submit(respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display])

    history_dropdown.change(reask_from_history,
        [history_dropdown, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display])

    for btn, question in zip(quick_btns, QUICK_QUESTIONS):
        btn.click(click_quick_question,
            [gr.Textbox(value=question, visible=False), chatbot, search_history_state],
            [msg, chatbot, search_history_state, history_dropdown, history_display])

    all_chips = list(zip(drug_chips, FEATURED_DRUGS[:5])) + \
                list(zip(drug_chips2, FEATURED_DRUGS[5:]))
    for chip, drug_name in all_chips:
        chip.click(drug_summary_respond,
            [gr.Textbox(value=drug_name, visible=False), chatbot, search_history_state],
            [chatbot, search_history_state, history_dropdown, history_display])

    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])
    drug_lookup_btn.click(drug_summary_respond,
        [drug_dropdown, chatbot, search_history_state],
        [chatbot, search_history_state, history_dropdown, history_display])

    def do_export(chat_history):
        f = export_chat(chat_history)
        return gr.update(value=f, visible=True) if f else gr.update(visible=False)

    export_btn.click(do_export, [chatbot], [export_file])

demo.launch()
