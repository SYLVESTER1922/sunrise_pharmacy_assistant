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

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def run_cypher(cypher, params=None):
    """Run Cypher with auto-reconnect on stale connection."""
    global driver
    for attempt in range(2):
        try:
            with driver.session() as session:
                return [dict(r) for r in session.run(cypher, **(params or {}))]
        except Exception:
            if attempt == 0:
                driver = get_driver()
            else:
                raise
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

driver = get_driver()
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
    """Use GPT to classify intent — handles any natural language phrasing"""
    q = question.lower().strip()

    # Fast path — greetings handled without GPT call
    GREETINGS = {"hi","hello","hey","good morning","morning","good afternoon",
                 "good evening","howzit","how are you","whats up","what's up"}
    THANKS = {"thanks","thank you","thank","cheers","appreciated","great","ok","okay","cool","perfect","noted"}
    FAREWELLS = {"bye","goodbye","see you","see ya","later","ciao","take care"}

    q_clean = q.strip("?!.,'’ ")
    if q_clean in GREETINGS or any(q_clean.startswith(g) for g in GREETINGS):
        return "greeting"
    if any(g in q_clean for g in ["good morning", "good afternoon", "good evening", "good night"]):
        return "greeting"
    if q_clean in THANKS:
        return "thanks"
    if q_clean in FAREWELLS:
        return "farewell"
    if q.startswith("quick summary:"):
        return "drug_summary"

    # GPT intent classification
    intent_list = """
- stock_price: checking if a drug is available, stock levels, price, cost, shelf location
- expiry: batch expiry dates, days until expiry, which batches expire soon, urgent batches
- low_stock: drugs below or near reorder level, running low, need reordering, stock alerts
- sales: top/bottom selling drugs, revenue, sales by day/week, customer type breakdown, transactions
- supplier: who supplies a drug, order from, lead times, suppliers by city, procurement
- stats: inventory summary, how many categories, total products, inventory overview
- category_browse: list all drugs in a category e.g. all antibiotics, all antiretrovirals
- interaction: drug interactions, safe to combine, avoid with, contraindications between drugs
- drug_info: drug usage, dosage, side effects, indications, drug class, clinical information
- alternative: alternatives or substitutes for a drug, same category replacements
- briefing: daily morning briefing, daily summary, start of day summary
- reorder: procurement action list, what to order, reorder report
- forecast: revenue forecast, stock forecast, days of stock remaining, projections
- reconciliation: stock discrepancies, missing stock, stock audit, losses
- followup: follow-up on previous answer, more details, elaborate on last response
"""

    # Include last intent for followup detection
    last_intent = ""
    if conversation_history:
        last_intent = f"\nThe previous question's intent was: {conversation_history[-1].get('intent','unknown')}"

    prompt = f"""You are a pharmacy chatbot intent classifier. Given a question, return ONLY the intent name from this list:{intent_list}

Rules:
- Return ONLY the intent name, nothing else
- stock_price: "do we have X", "is X available", "price of X", "how much is X", "X in stock"
- expiry: "when does X expire", "batches of X", "how many batches", "expiry of X", "X expiring", "how many batches of X", "batch count for X"
- low_stock: "running low", "below reorder", "need restocking", "out of stock", "critical stock"
- sales: "top selling", "bottom selling", "slow moving", "revenue", "transactions", "sold most"
- category_browse: "list all X", "what X do we have", "show all X" where X is a drug category
- stats: "how many categories", "inventory summary", "total products", "overview of stock", "how many drug types" — NOT for specific drug questions
- drug_info: "what is X used for", "dosage of X", "side effects of X", "what does X treat"
- interaction: "interacts with", "safe with", "combine X and Y", "X and Y together"
- supplier: "who supplies X", "where do we order X", "supplier for X", "lead time"
- If the question references "it", "them", "those", "the same" AND there is conversation history → followup
- Default to stock_price if a specific drug name is mentioned without other context{last_intent}

Question: {question}
Intent:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().lower()
        # Validate it's a known intent
        valid_intents = {
            "stock_price","expiry","low_stock","sales","supplier","stats",
            "category_browse","interaction","drug_info","alternative",
            "briefing","reorder","forecast","reconciliation","followup","drug_summary"
        }
        return intent if intent in valid_intents else "drug_info"
    except Exception:
        # Fallback to keyword matching if GPT fails
        if any(w in q for w in ["low","running low","reorder","below reorder","critical","out of"]):
            return "low_stock"
        if any(w in q for w in ["expir","batch","batches","days until","how many batches"]):
            return "expiry"
        if any(w in q for w in ["sold","sales","revenue","top selling","bottom","slow moving"]):
            return "sales"
        if any(w in q for w in ["supplier","order from","lead time","who supplies"]):
            return "supplier"
        if any(w in q for w in ["interact","safe with","combine","avoid"]):
            return "interaction"
        if any(w in q for w in ["alternative","substitute","instead of","replace"]):
            return "alternative"
        if any(w in q for w in ["categories","how many categories","inventory summary"]):
            return "stats"
        if any(w in q for w in ["stock","have","available","price","cost","cheapest"]):
            return "stock_price"
        return "drug_info"

# ── Keyword extractor ─────────────────────────────────────────
SKIP_WORDS = {
    "what", "which", "who", "where", "when", "how", "why", "is", "are",
    "was", "were", "do", "does", "did", "have", "has", "had", "will",
    "can", "could", "should", "would", "the", "a", "an", "in", "on",
    "at", "for", "of", "to", "and", "or", "but", "with", "from",
    "about", "we", "our", "us", "i", "my", "me", "stock", "drug",
    "drugs", "medicine", "medicines", "pharmacy", "pharmacist",
    "please", "tell", "show", "give", "find", "get", "check",
    "supplier", "supply", "supplies", "order", "source",
    "batch", "expiry", "soon", "selling", "sales", "name",
    "information", "info", "details", "need", "want",
    "medication", "medications", "tablet", "capsule", "injection",
    "anything", "something", "everything"
}

def extract_keywords(question):
    """Extract likely drug names and key terms from a question"""
    import re
    words = re.sub(r"['\u2019?!,.]", "", question.lower()).split()
    keywords = []
    for w in words:
        if len(w) >= 4 and w not in SKIP_WORDS and not w.isdigit():
            keywords.append(w)
    return keywords

def get_search_term(question):
    """Get primary search term from question"""
    keywords = extract_keywords(question)
    return keywords[0] if keywords else question.lower()

def format_stock_price(question):
    q = question.lower()
    categories = {
        "antibiotic":       "Antibiotics",
        "analgesic":        "Analgesics",
        "antihypertensive": "Antihypertensives",
        "antidiabetic":     "Antidiabetics",
        "antimalarial":     "Antimalarials",
        "antifungal":       "Antifungals",
        "antiretroviral":   "Antiretrovirals",
        "respiratory":      "Respiratory",
        "vitamin":          "Vitamins/Supplements",
        "gi medication":    "GI medications",
    }
    # Cheapest / most expensive — handle BEFORE drug name search
    if any(w in q for w in ["cheapest", "lowest price", "least expensive"]):
        cat_match = None
        for kw, cat in categories.items():
            if kw in q:
                cat_match = cat
                break
        sql_c = "SELECT generic_name, brand_name, selling_price_usd, quantity_in_stock, shelf_location, category FROM inventory"
        sql_c += (" WHERE category = %s" if cat_match else "")
        _nums = re.findall(r'\b(\d+)\b', q)
        _lim = int(_nums[0]) if _nums else 5
        sql_c += f" ORDER BY selling_price_usd ASC LIMIT {_lim}"
        conn = get_conn()
        try:
            df_c = pd.read_sql_query(sql_c, conn, params=(cat_match,) if cat_match else None)
        finally:
            release_conn(conn)
        label = f"cheapest {cat_match}" if cat_match else "cheapest drugs"
        header = f"**Top 5 {label}:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
        rows = [f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | {r['quantity_in_stock']} | {r['shelf_location']} |" for _, r in df_c.iterrows()]
        return header + "\n".join(rows)
    if any(w in q for w in ["most expensive", "highest price", "most costly"]):
        conn = get_conn()
        try:
            _nums_e = re.findall(r'\b(\d+)\b', q)
            _lim_e = int(_nums_e[0]) if _nums_e else 5
            df_e = pd.read_sql_query(f"SELECT generic_name, brand_name, selling_price_usd, quantity_in_stock, shelf_location, category FROM inventory ORDER BY selling_price_usd DESC LIMIT {_lim_e}", conn)
        finally:
            release_conn(conn)
        header = "**Top 5 most expensive drugs:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
        rows = [f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | {r['quantity_in_stock']} | {r['shelf_location']} |" for _, r in df_e.iterrows()]
        return header + "\n".join(rows)
    # Default drug name search
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
               ROUND(AVG(selling_price_usd)::numeric, 2) AS avg_price,
               ROUND(SUM(quantity_in_stock * cost_price_usd)::numeric, 2) AS inventory_value
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
               ROUND((quantity_in_stock::numeric / reorder_level) * 100, 0)
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

    # Specific drug batch query — "how many batches of X" or "when does X expire"
    # Check if a specific drug name is mentioned alongside batch/expiry keywords
    drug_keywords = extract_keywords(question)
    specific_drug = None
    if drug_keywords:
        for kw in drug_keywords:
            if kw not in {"batch","batches","expiry","expire","expires","days","soon","urgent","date","when","many","how"}:
                conn = get_conn()
                try:
                    result = pd.read_sql_query(
                        "SELECT generic_name FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1",
                        conn, params=(f"%{kw}%",)
                    )
                finally:
                    release_conn(conn)
                if not result.empty:
                    specific_drug = result.iloc[0]["generic_name"]
                    break

    if specific_drug:
        conn = get_conn()
        try:
            df = pd.read_sql_query("""
                SELECT b.batch_number, b.expiry_date,
                       b.quantity_remaining,
                       (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days
                FROM batches b
                JOIN inventory i ON b.product_id = i.product_id
                WHERE LOWER(i.generic_name) LIKE %s
                ORDER BY b.expiry_date ASC
            """, conn, params=(f"%{specific_drug.lower()}%",))
        finally:
            release_conn(conn)
        if df.empty:
            return f"❌ No batch records found for {specific_drug}."
        header = f"**{specific_drug} — {len(df)} batch(es) on record:**\n\n"
        header += "| Batch | Expiry Date | Days Left | Qty | Status |\n|---|---|---|---|---|\n"
        rows = []
        for _, r in df.iterrows():
            days = r["days"]
            flag = "🚨 URGENT" if days < 30 else ("⚠️ Warning" if days < 90 else "✅ OK")
            rows.append(f"| {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{days}** | {r['quantity_remaining']} | {flag} |")
        return header + "\n".join(rows)

    if any(w in q for w in [
        "soon", "this month", "next month", "90 days", "expiring",
        "about to expire", "near expiry", "earliest", "most urgent",
        "urgent", "action", "critical"
    ]):
        sql = """
            SELECT i.generic_name, i.brand_name, b.batch_number,
                   b.expiry_date, b.quantity_remaining,
                   (b.expiry_date::date - CURRENT_DATE)::INTEGER
                   AS days_remaining
            FROM batches b
            JOIN inventory i ON b.product_id = i.product_id
            WHERE (b.expiry_date::date - CURRENT_DATE) <= 90
            ORDER BY b.expiry_date ASC
            LIMIT %s
        """
        nums = re.findall(r'\b(\d+)\b', question)
        exp_limit = int(nums[0]) if nums else 10
        exp_limit = max(1, min(exp_limit, 50))
        conn = get_conn()
        try:
            df = pd.read_sql_query(sql, conn, params=(exp_limit,))
        finally:
            release_conn(conn)
        if df.empty:
            return "✅ No batches expiring within the next 90 days."
        header = f"**Top {exp_limit} batches expiring soonest** — {len(df)} found:\n\n"
        header += "| Drug | Brand | Batch | Expiry Date | Days Left | Qty | Status |\n"
        header += "|---|---|---|---|---|---|---|\n"
        rows = []
        for _, r in df.iterrows():
            if r['days_remaining'] < 30:
                status = "🚨 URGENT"
            elif r['days_remaining'] <= 60:
                status = "⚠️ Warning"
            else:
                status = "📅 Monitor"
            rows.append(
                f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | "
                f"{str(r['expiry_date'])[:10]} | **{r['days_remaining']}** | "
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
                   (b.expiry_date::date - CURRENT_DATE)::INTEGER
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
                f"{str(r['expiry_date'])[:10]} | **{days}**{flag} | {r['quantity_remaining']} |"
            )
        return header + "\n".join(rows)

def format_sales(question):
    q = question.lower()

    # Customer type breakdown
    if any(w in q for w in [
        "customer type", "customer breakdown", "by customer",
        "breakdown", "split", "prescription", "walk-in",
        "walkin", "insurance", "type of customer"
    ]):
        sql = """
            SELECT customer_type,
                   COUNT(*)                    AS num_transactions,
                   SUM(quantity_sold)          AS total_units,
                   ROUND(SUM(total_amount)::numeric, 2) AS total_revenue,
                   ROUND((SUM(total_amount) * 100.0 /
                       (SELECT SUM(total_amount) FROM transactions))::numeric, 1)
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
        return header + "\n".join(rows) + f"\n\n**Total Revenue: ${total:,.2f}**"

    # Day of week query
    day_map = {
        "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
        "friday": 5, "saturday": 6, "sunday": 0
    }
    matched_day = None
    matched_day_name = None
    for day_name, day_num in day_map.items():
        if day_name in q:
            matched_day = day_num
            matched_day_name = day_name.capitalize()
            break
    if matched_day is not None:
        sql_day = """
            SELECT date, COUNT(*) AS num_transactions,
                   SUM(quantity_sold) AS total_units,
                   ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
            FROM transactions
            WHERE EXTRACT(DOW FROM date::date) = %s
            GROUP BY date ORDER BY date DESC LIMIT 5
        """
        conn = get_conn()
        try:
            df_day = pd.read_sql_query(sql_day, conn, params=(matched_day,))
        finally:
            release_conn(conn)
        if df_day.empty:
            return f"No transactions found for {matched_day_name}s."
        header = f"**{matched_day_name} Sales**\n\n| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
        rows = [f"| {str(r['date'])[:10]} | {r['num_transactions']} | {r['total_units']} | ${r['total_revenue']:,.2f} |" for _, r in df_day.iterrows()]
        return header + "\n".join(rows) + f"\n\n**Total {matched_day_name} Revenue: ${df_day['total_revenue'].sum():,.2f}**"
    # Last week query
    if any(w in q for w in ["last week", "this week", "past week"]):
        sql_week = """
            SELECT date, COUNT(*) AS num_transactions,
                   SUM(quantity_sold) AS total_units,
                   ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
            FROM transactions
            WHERE date::date >= (SELECT MAX(date::date) - 7 FROM transactions)
            GROUP BY date ORDER BY date DESC
        """
        conn = get_conn()
        try:
            df_week = pd.read_sql_query(sql_week, conn)
        finally:
            release_conn(conn)
        if df_week.empty:
            return "No transactions found for last week."
        header = "**Last Week Sales**\n\n| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
        rows = [f"| {str(r['date'])[:10]} | {r['num_transactions']} | {r['total_units']} | ${r['total_revenue']:,.2f} |" for _, r in df_week.iterrows()]
        return header + "\n".join(rows) + f"\n\n**Total: ${df_week['total_revenue'].sum():,.2f}**"
    # Last day / most recent day sales
    if any(w in q for w in [
        "last day", "yesterday", "latest day", "most recent day",
        "last transaction", "recent sales", "today"
    ]):
        sql_date = """
            SELECT date,
                   COUNT(*)                           AS num_transactions,
                   SUM(quantity_sold)                 AS total_units,
                   ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
            FROM transactions
            WHERE date = (SELECT MAX(date) FROM transactions)
            GROUP BY date
        """
        sql_drugs = """
            SELECT i.brand_name, i.generic_name,
                   SUM(t.quantity_sold)                   AS units,
                   ROUND(SUM(t.total_amount)::numeric, 2) AS revenue
            FROM transactions t
            JOIN inventory i ON t.product_id = i.product_id
            WHERE t.date = (SELECT MAX(date) FROM transactions)
            GROUP BY i.brand_name, i.generic_name
            ORDER BY revenue DESC
        """
        sql_ctype = """
            SELECT customer_type,
                   ROUND(SUM(total_amount)::numeric, 2) AS revenue
            FROM transactions
            WHERE date = (SELECT MAX(date) FROM transactions)
            GROUP BY customer_type ORDER BY revenue DESC
        """
        conn = get_conn()
        try:
            df_d  = pd.read_sql_query(sql_date, conn)
            df_dr = pd.read_sql_query(sql_drugs, conn)
            df_ct = pd.read_sql_query(sql_ctype, conn)
        finally:
            release_conn(conn)
        if df_d.empty:
            return "No transactions found."
        r = df_d.iloc[0]
        header  = f"**Sales for {str(r['date'])[:10]}** (Last recorded day)\n\n"
        header += f"Transactions: **{r['num_transactions']}** | "
        header += f"Units Sold: **{r['total_units']}** | "
        header += f"Revenue: **${r['total_revenue']:,.2f}**\n\n"
        header += "**By Drug:**\n\n| Brand | Generic | Units | Revenue |\n|---|---|---|---|\n"
        drug_rows = [
            f"| {row['brand_name']} | {row['generic_name']} | "
            f"{row['units']} | ${row['revenue']:,.2f} |"
            for _, row in df_dr.iterrows()
        ]
        ctype = "\n\n**By Customer Type:** " + " | ".join(
            [f"{row['customer_type']}: ${row['revenue']:,.2f}"
             for _, row in df_ct.iterrows()]
        )
        return header + "\n".join(drug_rows) + ctype

    # Default — top/bottom selling with number and direction extraction
    number_words = {
        "one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10
    }
    numbers = re.findall(r'\b(\d+)\b', question)
    limit = int(numbers[0]) if numbers else None
    if not limit:
        for word, num in number_words.items():
            if word in q:
                limit = num
                break
    limit = limit or 10
    limit = max(1, min(limit, 50))

    # Detect direction
    if any(w in q for w in [
        "least", "lowest", "bottom", "worst", "slow",
        "poor", "less", "fewest", "minimum"
    ]):
        order = "ASC"
        direction_label = f"Bottom {limit}"
    else:
        order = "DESC"
        direction_label = f"Top {limit}"

    # Detect sort column
    if any(w in q for w in ["unit", "quantity", "volume", "dispensed"]):
        sort_col = "total_units"
        sort_label = "by units sold"
    elif any(w in q for w in ["transaction", "frequency", "times"]):
        sort_col = "num_transactions"
        sort_label = "by transactions"
    else:
        sort_col = "total_revenue"
        sort_label = "by revenue"

    sql = f"""
        SELECT i.brand_name, i.generic_name,
               SUM(t.quantity_sold)          AS total_units,
               ROUND(SUM(t.total_amount)::numeric, 2) AS total_revenue,
               COUNT(*)                       AS num_transactions
        FROM transactions t
        JOIN inventory i ON t.product_id = i.product_id
        GROUP BY i.brand_name, i.generic_name
        ORDER BY {sort_col} {order}
        LIMIT %s
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=(limit,))
    finally:
        release_conn(conn)
    header  = f"**{direction_label} Selling Drugs** {sort_label} (Last 30 days)\n\n"
    header += "| Rank | Brand | Generic | Units | Revenue | Transactions |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = [
        f"| {i+1} | {r['brand_name']} | {r['generic_name']} | "
        f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['num_transactions']} |"
        for i, (_, r) in enumerate(df.iterrows())
    ]
    return header + "\n".join(rows)




def format_alternative(question):
    """Find drugs in same category as the named drug"""
    keywords = extract_keywords(question)
    if not keywords:
        return "❌ Please specify a drug name."
    search_param = None
    drug_name = None
    category = None
    for k in keywords:
        conn = get_conn()
        try:
            result = pd.read_sql_query(
                "SELECT generic_name, category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1",
                conn, params=(f"%{k}%",)
            )
        finally:
            release_conn(conn)
        if not result.empty:
            search_param = f"%{k}%"
            drug_name = result.iloc[0]["generic_name"]
            category = result.iloc[0]["category"]
            break
    if not search_param:
        return "❌ Drug not found in inventory."
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT generic_name, brand_name, formulation, strength,
                   quantity_in_stock, selling_price_usd, shelf_location
            FROM inventory
            WHERE category = (
                SELECT category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1
            )
            AND LOWER(generic_name) NOT LIKE %s
            AND quantity_in_stock > 0
            ORDER BY generic_name
        """, conn, params=(search_param, search_param))
    finally:
        release_conn(conn)
    if df.empty:
        return f"❌ No alternatives found for {drug_name} in category {category}."
    header = f"**Alternatives to {drug_name}** (category: {category})\n\n"
    header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
        f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows) + "\n\n⚠️ **Clinical Note:** Therapeutic substitution requires pharmacist approval."
def format_supplier(question):
    """Supplier lookup — handles drug lookup, lead time queries, and city queries"""
    q = question.lower()
    # Shortest lead time query
    if any(w in q for w in ["shortest", "fastest", "quickest", "best lead", "minimum lead"]):
        cypher = """
            MATCH (s:Supplier)
            RETURN s.name AS supplier, s.lead_time AS lead_time_days,
                   s.city AS city, s.contact AS contact
            ORDER BY s.lead_time ASC LIMIT 5
        """
        results = run_cypher(cypher)
        if not results:
            return "❌ No supplier information found."
        header = "**Suppliers with shortest lead times:**\n\n| Supplier | Lead Time | City | Contact |\n|---|---|---|---|\n"
        rows = [f"| {r['supplier']} | {r['lead_time_days']} days | {r['city']} | {r['contact']} |" for r in results]
        return header + "\n".join(rows)
    # City/count query
    if any(w in q for w in ["harare", "bulawayo", "mutare", "how many suppliers", "suppliers in", "city"]):
        cypher = """
            MATCH (s:Supplier)
            RETURN s.city AS city, count(s) AS supplier_count,
                   collect(s.name) AS suppliers
            ORDER BY supplier_count DESC
        """
        results = run_cypher(cypher)
        if not results:
            return "❌ No supplier information found."
        header = "**Suppliers by City:**\n\n| City | Count | Suppliers |\n|---|---|---|\n"
        rows = [f"| {r['city']} | {r['supplier_count']} | {', '.join(r['suppliers'])} |" for r in results]
        return header + "\n".join(rows)
    # Drug supplier lookup
    supplier_stopwords = {"order", "supplier", "supply", "supplies", "distributor",
                          "source", "procure", "purchase", "buy", "vendor", "where", "who"}
    keywords = extract_keywords(question)
    drug_keywords = [k for k in keywords if k not in supplier_stopwords]
    search_term = drug_keywords[0] if drug_keywords else (keywords[0] if keywords else "")
    if not search_term:
        return "❌ Please specify a drug name."
    cypher = """
        MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
        WHERE toLower(d.generic_name) CONTAINS toLower($search)
        RETURN d.generic_name AS drug, s.name AS supplier,
               s.contact AS contact, s.phone AS phone,
               s.city AS city, s.lead_time AS lead_time_days,
               s.payment_terms AS payment_terms
        LIMIT 5
    """
    results = run_cypher(cypher, {"search": search_term})
    if not results:
        return "❌ No supplier information found for that drug."
    lines = [f"**Supplier information for {results[0]['drug']}:**\n"]
    seen = set()
    for r in results:
        if r['supplier'] in seen:
            continue
        seen.add(r['supplier'])
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

def format_drug_summary(drug_name):
    sql = """
        SELECT i.generic_name, i.brand_name, i.formulation, i.strength,
               i.quantity_in_stock, i.reorder_level,
               i.selling_price_usd, i.cost_price_usd,
               i.shelf_location, i.category,
               MIN(b.expiry_date) AS nearest_expiry,
               (MIN(b.expiry_date::date) - CURRENT_DATE)::INTEGER
               AS days_to_expiry
        FROM inventory i
        LEFT JOIN batches b ON i.product_id = b.product_id
        WHERE LOWER(i.generic_name) LIKE LOWER(%s)
        GROUP BY i.product_id, i.generic_name, i.brand_name, i.formulation,
                 i.strength, i.quantity_in_stock, i.reorder_level,
                 i.selling_price_usd, i.cost_price_usd, i.shelf_location,
                 i.category
        LIMIT 1
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
            expiry_line = f"\n🚨 **URGENT:** Nearest batch expires in {d} days ({str(r['nearest_expiry'])[:10]})"
        elif d <= 90:
            expiry_line = f"\n⚠️ Nearest expiry: {str(r['nearest_expiry'])[:10]} ({d} days)"
        else:
            expiry_line = f"\n📅 Nearest expiry: {str(r['nearest_expiry'])[:10]} ({d} days)"
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
7. End every answer with the actual source name, for example: "Source: drug knowledge graph" or "Source: drug interaction knowledge graph".
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
    return run_cypher(cypher, {"search": search_term})

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
    return run_cypher(cypher, {"search": search_term})


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 1: DAILY BRIEFING
# ═══════════════════════════════════════════════════════════════
def format_daily_briefing():
    """Morning briefing — low stock + urgent expiry + yesterday revenue"""
    from datetime import date as date_obj
    today = date_obj.today().strftime("%A, %d %B %Y")

    # Low stock
    conn = get_conn()
    try:
        df_stock = pd.read_sql_query("""
            SELECT generic_name, brand_name, quantity_in_stock, reorder_level,
                   ROUND((quantity_in_stock::numeric/reorder_level)*100,0) AS pct
            FROM inventory WHERE quantity_in_stock <= reorder_level
            ORDER BY pct ASC LIMIT 5
        """, conn)
    finally:
        release_conn(conn)

    # Urgent expiry (< 30 days)
    conn = get_conn()
    try:
        df_exp = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, b.batch_number,
                   (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days_left,
                   b.quantity_remaining
            FROM batches b JOIN inventory i ON b.product_id = i.product_id
            WHERE (b.expiry_date::date - CURRENT_DATE) <= 30
            ORDER BY days_left ASC LIMIT 5
        """, conn)
    finally:
        release_conn(conn)

    # Yesterday revenue
    conn = get_conn()
    try:
        df_rev = pd.read_sql_query("""
            SELECT ROUND(SUM(total_amount)::numeric,2) AS revenue,
                   COUNT(*) AS txns, SUM(quantity_sold) AS units
            FROM transactions
            WHERE date = (SELECT MAX(date) FROM transactions)
        """, conn)
        df_avg = pd.read_sql_query("""
            SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily
            FROM (SELECT date, SUM(total_amount) AS daily_rev
                  FROM transactions GROUP BY date) t
        """, conn)
    finally:
        release_conn(conn)

    from datetime import datetime as _dt
    hour = _dt.now().hour
    tod = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 17 else "Good Evening")
    lines = [f"# 🌅 {tod}! Daily Briefing — {today}\n"]

    # Revenue summary
    rev = df_rev.iloc[0]
    avg = df_avg.iloc[0]['avg_daily']
    trend = "📈 above" if rev['revenue'] > avg else "📉 below"
    lines.append(f"## 💰 Yesterday's Revenue")
    lines.append(f"**${rev['revenue']:,.2f}** ({rev['txns']} transactions, {rev['units']} units sold)")
    lines.append(f"30-day average: **${avg:,.2f}** — Yesterday was {trend} average\n")

    # Low stock
    if df_stock.empty:
        lines.append("## ✅ Stock Levels\nAll products above reorder level. No action needed.\n")
    else:
        lines.append(f"## 🔴 Low Stock Alert — {len(df_stock)} drug(s) need reordering")
        lines.append("| Drug | Brand | Stock | Reorder | % |\n|---|---|---|---|---|")
        for _, r in df_stock.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['pct']:.0f}% |")
        lines.append("")

    # Urgent expiry
    if df_exp.empty:
        lines.append("## ✅ Expiry Status\nNo batches expiring within 30 days.\n")
    else:
        lines.append(f"## 🚨 Urgent Expiry — {len(df_exp)} batch(es) expiring within 30 days")
        lines.append("| Drug | Brand | Batch | Days Left | Qty |\n|---|---|---|---|---|")
        for _, r in df_exp.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | **{r['days_left']}** | {r['quantity_remaining']} |")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 2: REORDER ACTION LIST
# ═══════════════════════════════════════════════════════════════
def format_reorder_list():
    """Complete procurement action list with suggested order quantities"""
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, i.quantity_in_stock,
                   i.reorder_level, i.category,
                   COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,1), 0) AS avg_daily_sales,
                   CASE WHEN COALESCE(SUM(t.quantity_sold),0) > 0
                        THEN (i.reorder_level * 2 - i.quantity_in_stock)
                        ELSE (i.reorder_level * 2 - i.quantity_in_stock)
                   END AS suggested_order
            FROM inventory i
            LEFT JOIN transactions t ON i.product_id = t.product_id
            WHERE i.quantity_in_stock <= i.reorder_level
            GROUP BY i.product_id, i.generic_name, i.brand_name,
                     i.quantity_in_stock, i.reorder_level, i.category
            ORDER BY (i.quantity_in_stock::float/i.reorder_level) ASC
        """, conn)
    finally:
        release_conn(conn)

    if df.empty:
        return "✅ All products are above reorder level. No procurement action needed."

    header = f"## 📋 Procurement Action List — {len(df)} drug(s) to reorder\n\n"
    header += "| Drug | Brand | Current Stock | Reorder Level | Avg Daily Sales | Suggested Order | Category |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | "
            f"{r['reorder_level']} | {r['avg_daily_sales']} units/day | "
            f"**{max(int(r['suggested_order']),1)}** units | {r['category']} |"
        )
    return header + "\n".join(rows) + "\n\n*Suggested order = 2x reorder level minus current stock*"


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 3: REVENUE FORECAST
# ═══════════════════════════════════════════════════════════════
def format_revenue_forecast():
    """Project revenue and stock depletion at current sales rate"""
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, i.quantity_in_stock,
                   COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,2), 0) AS avg_daily,
                   i.selling_price_usd
            FROM inventory i
            LEFT JOIN transactions t ON i.product_id = t.product_id
            GROUP BY i.product_id, i.generic_name, i.brand_name,
                     i.quantity_in_stock, i.selling_price_usd
            ORDER BY (i.quantity_in_stock * i.selling_price_usd) DESC
            LIMIT 15
        """, conn)
        df_daily = pd.read_sql_query("""
            SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily_revenue
            FROM (SELECT date, SUM(total_amount) AS daily_rev
                  FROM transactions GROUP BY date) t
        """, conn)
    finally:
        release_conn(conn)

    avg_daily_rev = float(df_daily.iloc[0]['avg_daily_revenue'])
    forecast_30 = round(avg_daily_rev * 30, 2)
    forecast_90 = round(avg_daily_rev * 90, 2)

    lines = ["## 📈 Revenue & Stock Forecast\n"]
    lines.append(f"**Average Daily Revenue:** ${avg_daily_rev:,.2f}")
    lines.append(f"**30-Day Revenue Forecast:** ${forecast_30:,.2f}")
    lines.append(f"**90-Day Revenue Forecast:** ${forecast_90:,.2f}\n")
    lines.append("**Days of Stock Remaining (Top 10 by value):**\n")
    lines.append("| Drug | Brand | Stock | Avg Daily Sales | Days Remaining |\n|---|---|---|---|---|")
    for _, r in df.iterrows():
        if r['avg_daily'] > 0:
            days = round(r['quantity_in_stock'] / r['avg_daily'])
            flag = "🔴" if days < 30 else ("🟡" if days < 60 else "🟢")
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | {r['avg_daily']}/day | {flag} **{days} days** |")
        else:
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | No sales | ∞ |")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 4: MULTI-DRUG INTERACTION CHECK
# ═══════════════════════════════════════════════════════════════
def format_multi_interaction(question):
    """Check interactions between multiple drugs mentioned in one question"""
    keywords = extract_keywords(question)
    if len(keywords) < 2:
        return None  # fall through to single drug interaction
    cypher = """
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) IN $drugs OR toLower(b.generic_name) IN $drugs
        RETURN a.generic_name AS drug_a, b.generic_name AS drug_b,
               r.severity AS severity, r.description AS description,
               r.recommendation AS recommendation
        ORDER BY CASE r.severity WHEN 'Major' THEN 1
                 WHEN 'Moderate' THEN 2 WHEN 'Minor' THEN 3 ELSE 4 END
        LIMIT 10
    """
    drugs_lower = [k.lower() for k in keywords]
    results = run_cypher(cypher, {"drugs": drugs_lower})
    return results


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 5: CONTROLLED SUBSTANCE CHECK (built into drug info)
# ═══════════════════════════════════════════════════════════════
# This is handled in format_drug_summary and query_neo4j_drug_info
# The drug_knowledge table has a 'controlled' field — already displayed


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 6: SALES vs INVENTORY RECONCILIATION
# ═══════════════════════════════════════════════════════════════
def format_reconciliation(question):
    """Compare sales vs stock movement to flag discrepancies"""
    keywords = extract_keywords(question)
    drug_filter = ""
    params = []
    if keywords:
        drug_filter = "WHERE LOWER(i.generic_name) LIKE %s"
        params = [f"%{keywords[0]}%"]
    conn = get_conn()
    try:
        df = pd.read_sql_query(f"""
            SELECT i.generic_name, i.brand_name,
                   SUM(b.quantity_received) AS total_received,
                   SUM(t.quantity_sold)     AS total_sold,
                   i.quantity_in_stock      AS current_stock,
                   (SUM(b.quantity_received) - SUM(t.quantity_sold) - i.quantity_in_stock)
                       AS discrepancy
            FROM inventory i
            LEFT JOIN batches b      ON i.product_id = b.product_id
            LEFT JOIN transactions t ON i.product_id = t.product_id
            {drug_filter}
            GROUP BY i.product_id, i.generic_name, i.brand_name, i.quantity_in_stock
            HAVING ABS(SUM(b.quantity_received) - COALESCE(SUM(t.quantity_sold),0)
                       - i.quantity_in_stock) > 5
            ORDER BY ABS(SUM(b.quantity_received) - COALESCE(SUM(t.quantity_sold),0)
                         - i.quantity_in_stock) DESC
            LIMIT 10
        """, conn, params=params if params else None)
    finally:
        release_conn(conn)
    if df.empty:
        return "✅ Stock reconciliation looks clean — no significant discrepancies found."
    header = "## ⚠️ Stock Reconciliation — Discrepancies Found\n\n"
    header += "| Drug | Brand | Received | Sold | Current Stock | Discrepancy |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        flag = "🔴" if abs(r['discrepancy']) > 20 else "🟡"
        rows.append(
            f"| {r['generic_name']} | {r['brand_name']} | {r['total_received']:.0f} | "
            f"{r['total_sold']:.0f} | {r['current_stock']} | {flag} **{r['discrepancy']:.0f}** |"
        )
    return header + "\n".join(rows) + "\n\n*Discrepancy = Received − Sold − Current Stock. Non-zero may indicate theft, wastage or data entry errors.*"

# ═══════════════════════════════════════════════════════════════
# MAIN ROUTER — dispatches to operational or clinical mode
# ═══════════════════════════════════════════════════════════════

def get_greeting():
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 12:
        tod = "Good morning"
    elif hour < 17:
        tod = "Good afternoon"
    else:
        tod = "Good evening"
    return tod

GREETING_RESPONSE = f"""👋 {get_greeting()}! I am the **Netrisyl Pharmacy Assistant** for Sunrise Pharmacy.

**Type *Good morning* for your daily briefing** — low stock, urgent expiries, and yesterday's revenue in one message.

I can help with:
- 🌅 **Daily Briefing** — *"Good morning"*
- 📋 **Reorder List** — *"What is the reorder list?"*
- 📈 **Revenue Forecast** — *"Revenue forecast"*
- 📦 **Stock & Prices** — *"Do we have amoxicillin?"*
- ⚠️ **Drug Interactions** — *"What interacts with metformin?"*
- 📅 **Expiry Alerts** — *"Which batches are expiring soon?"*
- 🚚 **Suppliers** — *"Who supplies ciprofloxacin?"*
- 💊 **Drug Information** — *"What is ibuprofen used for?"*
- 🔄 **Alternatives** — *"What is an alternative to amoxicillin?"*
- 💰 **Sales Summary** — *"Top selling drugs"*
- 🔴 **Low Stock Alerts** — *"Which drugs are running low?"*
- 🔍 **Stock Reconciliation** — *"Check stock discrepancies"*

⚠️ Clinical answers always include a pharmacist verification reminder.

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
    if intent == "drug_summary":
        drug_name = question.replace("quick summary:", "").replace("Quick summary:", "").strip()
        answer = format_drug_summary(drug_name)
        return answer, "inventory + batch records", "operational"
    if intent == "briefing":
        return format_daily_briefing(), "inventory + batch + transaction records", "operational"
    if intent == "reorder":
        return format_reorder_list(), "inventory + transaction records", "operational"
    if intent == "forecast":
        return format_revenue_forecast(), "inventory + transaction records", "operational"
    if intent == "reconciliation":
        return format_reconciliation(question), "inventory + batch + transaction records", "operational"

    # ── CLINICAL MODE — GPT with strict grounding ─────────────
    if intent == "interaction":
        # Check if multiple drugs mentioned — use multi-drug checker
        keywords = extract_keywords(question)
        if len(keywords) >= 2:
            data = format_multi_interaction(question)
            if data is None:
                data = query_neo4j_interaction(question)
        else:
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

# ── Text to Speech ───────────────────────────────────────────
import tempfile

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
        elif intent == "followup":
            full_answer = answer
        elif mode == "operational":
            header = f"*📦 Operational data — {source}*\n\n"
            body = answer
            full_answer = f"{correction_note}\n\n{header}{body}" if correction_note else f"{header}{body}"
        else:  # clinical
            header = f"*🧪 Clinical data — {source}*\n\n"
            body = answer
            full_answer = f"{correction_note}\n\n{header}{body}" if correction_note else f"{header}{body}"

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
        gr.update(value=history_md),
        ""  # clear brief_box on new question
    )

def drug_summary_respond(drug_name, chat_history, search_history):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update(), ""
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
        gr.update(value=history_md),
        ""  # clear brief_box
    )

def click_quick_question(question, chat_history, search_history):
    return respond(question, chat_history, search_history)

def reask_from_history(selected_question, chat_history, search_history):
    if not selected_question:
        return "", chat_history, search_history, gr.update(), gr.update(), ""
    return respond(selected_question, chat_history, search_history)

# ── Featured drugs & quick questions ─────────────────────────
FEATURED_DRUGS = [
    "Amoxicillin", "Paracetamol", "Metformin", "Ibuprofen",
    "Ciprofloxacin", "Azithromycin", "Amlodipine", "Losartan",
    "Artemether/Lumefantrine", "Co-trimoxazole"
]

QUICK_QUESTIONS = [
    "Good morning",
    "Which drugs are running low on stock?",
    "Which batches are expiring soon?",
    "What are the reorder list?",
    "What are the top selling drugs?",
    "Revenue forecast",
    "Do we have amoxicillin in stock?",
    "What interacts with metformin?"
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
                💊 💊💊💊Pharmacy Assistant
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
            chatbot = gr.Chatbot(label="Pharmacy Assistant", height=460, autoscroll=True)
            # Drug chips removed
            brief_box = gr.Textbox(
                label="💡 Key Points",
                placeholder="Ask a question then click Brief for a plain language summary",
                interactive=False,
                lines=2,
                visible=True
            )
            with gr.Row():
                msg    = gr.Textbox(
                    placeholder="Ask e.g. 'Do we have Amoxicillin?' or 'Good morning' for daily briefing",
                    label="",
                    scale=4
                )
                submit = gr.Button("Ask", variant="primary", scale=1)
                brief_btn = gr.Button("💡 Brief", variant="secondary", scale=1)
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Voice Input (click to record)",
                    visible=True
                )

            with gr.Row():
                export_btn   = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file  = gr.File(label="Download", scale=2, visible=False)

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
        [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])
    msg.submit(respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])

    def transcribe_audio(audio_path, chat_history, search_history):
        if not audio_path:
            return "", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), ""
        try:
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=f
                )
            transcribed = transcript.text
            result = respond(transcribed, chat_history, search_history)
            # result is (msg, chat, search, dropdown, history_md, brief)
            # Add audio reset at end
            return result[0], result[1], result[2], result[3], result[4], gr.update(value=None), result[5]
        except Exception as e:
            return "", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), ""

    audio_input.stop_recording(transcribe_audio,
        [audio_input, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, audio_input, brief_box])

    history_dropdown.change(reask_from_history,
        [history_dropdown, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])

    for btn, question in zip(quick_btns, QUICK_QUESTIONS):
        btn.click(click_quick_question,
            [gr.Textbox(value=question, visible=False), chatbot, search_history_state],
            [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])

    # Drug chips removed

    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])
    drug_lookup_btn.click(drug_summary_respond,
        [drug_dropdown, chatbot, search_history_state],
        [chatbot, search_history_state, history_dropdown, history_display, brief_box],
        scroll_to_output=True)

    def do_export(chat_history):
        f = export_chat(chat_history)
        return gr.update(value=f, visible=True) if f else gr.update(visible=False)



    def do_brief(chat_history):
        """Generate a proper natural language brief using GPT"""
        if not chat_history:
            return "No response yet — ask a question first."
        try:
            last = chat_history[-1]
            if isinstance(last, dict):
                last_response = last.get("content", "")
            elif isinstance(last, (list, tuple)):
                last_response = last[1] if len(last) > 1 else ""
            else:
                last_response = str(last)
            # Use GPT to generate a proper 2-3 sentence brief
            prompt = f"""Summarise the following pharmacy data response in 2-3 clear sentences suitable for a pharmacy manager. Focus on the most important numbers and actionable insights. Do not use bullet points or markdown.

Response:
{last_response[:1500]}

Brief summary:"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Could not generate brief: {str(e)}"

    export_btn.click(do_export, [chatbot], [export_file])
    brief_btn.click(do_brief, [chatbot], [brief_box])


demo.launch()