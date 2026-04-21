import os
import re
import sqlite3
import pandas as pd
import gradio as gr
from neo4j import GraphDatabase
from openai import OpenAI
from difflib import SequenceMatcher
from datetime import datetime
import json

# ── Credentials ──────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Build SQLite from CSVs on startup ─────────────────────────
DB_PATH = "sunrise_pharmacy.db"

def build_database():
    conn = sqlite3.connect(DB_PATH)
    tables = {
        "inventory":      "sunrise_pharmacy_inventory_pricing.csv",
        "batches":        "sunrise_pharmacy_batch_expiry.csv",
        "suppliers":      "sunrise_pharmacy_suppliers.csv",
        "drug_knowledge": "sunrise_pharmacy_drug_knowledge.csv",
        "interactions":   "sunrise_pharmacy_drug_interactions.csv",
        "transactions":   "sunrise_pharmacy_transactions_last_30_days.csv",
    }
    for table_name, filename in tables.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"✓ Loaded {len(df)} rows into {table_name}")
        else:
            print(f"⚠ File not found: {filename}")
    conn.commit()
    conn.close()
    print("Database ready ✓")

build_database()

# ── Thread-safe SQLite connection ─────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH)

# ── Load drug list ────────────────────────────────────────────
def get_all_drugs():
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT generic_name, brand_name, category FROM inventory ORDER BY generic_name",
            conn
        )
    return df

DRUGS_DF   = get_all_drugs()
DRUG_NAMES = DRUGS_DF["generic_name"].tolist()

# ── Fuzzy drug name matching ──────────────────────────────────
def fuzzy_match_drug(text, threshold=78):
    """
    Matches a word to the closest drug name using character similarity.
    Handles spelling errors like 'amoxicilin' -> 'Amoxicillin'.
    """
    text = text.lower().strip()
    # Remove apostrophes and punctuation
    text = re.sub(r"['\u2019\u2018`]", "", text)
    best_score = 0
    best_match = None
    for drug in DRUG_NAMES:
        drug_lower = drug.lower()
        # Fast path — direct substring
        if text in drug_lower or drug_lower in text:
            return drug
        score = SequenceMatcher(None, text, drug_lower).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = drug
    return best_match if best_score >= threshold else None

def fuzzy_correct_question(question):
    """
    Corrects drug name spelling errors in a question.
    Returns (corrected_question, correction_note).
    Short words (<4 chars) and common English words are skipped.
    """
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

# ── Intent classification — handles many phrasings ───────────
GREETINGS = {
    "hi", "hey", "hello", "good morning", "good afternoon", "good evening",
    "help", "what can you do", "how are you", "what do you do",
    "who are you", "what are you", "start", "begin"
}

THANKS = {"thank you", "thanks", "thank", "cheers", "appreciated", "great"}
FAREWELLS = {"bye", "goodbye", "see you", "see ya", "later", "exit", "quit"}

def classify_intent(question, conversation_history=None):
    """
    Classifies the intent of a question.
    Also detects conversational follow-ups using context from history.
    """
    q = question.lower().strip()
    q_clean = re.sub(r"['\u2019?!,.]", "", q)

    # Greetings / thanks / farewells
    if any(q_clean == g or q_clean.startswith(g) for g in GREETINGS):
        return "greeting"
    if any(g in q_clean for g in THANKS):
        return "thanks"
    if any(g in q_clean for g in FAREWELLS):
        return "farewell"

    # Follow-up detection — short questions referencing prior context
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

    # Stock, price, availability
    if any(w in q for w in [
        "stock", "have", "available", "availability", "quantity",
        "how many", "price", "cost", "how much", "in stock",
        "do we have", "do you have", "shelf"
    ]):
        return "stock_price"

    # Category browse
    category_keywords = [
        "antibiotics", "analgesics", "antihypertensives", "antidiabetics",
        "antimalarials", "vitamins", "antifungals", "gi medications",
        "respiratory", "antiretrovirals", "list all", "show all",
        "all drugs", "all medicines", "all medications"
    ]
    if any(w in q for w in category_keywords):
        return "category_browse"

    # Count / stats
    if any(w in q for w in [
        "how many drugs", "how many types", "how many medicines",
        "total drugs", "count", "drug types", "categories",
        "how many do we have", "inventory summary", "stock summary"
    ]):
        return "stats"

    # Expiry
    if any(w in q for w in [
        "expir", "expire", "expiry", "batch", "shelf life",
        "best before", "use by", "when does"
    ]):
        return "expiry"

    # Interactions
    if any(w in q for w in [
        "interact", "interaction", "together", "combine", "mix",
        "safe with", "take with", "used with", "combined with",
        "contraindic", "avoid with"
    ]):
        return "interaction"

    # Supplier
    if any(w in q for w in [
        "supplier", "order from", "who supply", "distributor",
        "vendor", "supplies", "supply", "who provides", "where do we order",
        "procurement", "purchase from", "buy from", "source"
    ]):
        return "supplier"

    # Sales
    if any(w in q for w in [
        "sold", "sales", "revenue", "dispensed", "transaction",
        "top selling", "best selling", "most popular", "most sold",
        "highest revenue", "performance", "turnover"
    ]):
        return "sales"

    # Alternatives / substitutes
    if any(w in q for w in [
        "alternative", "substitute", "instead of", "replace",
        "similar to", "other option", "other drug", "swap",
        "equivalent", "whats another", "what else can"
    ]):
        return "alternative"

    # Low stock / reorder alerts
    if any(w in q for w in [
        "low stock", "reorder", "running low", "almost out",
        "need to order", "below reorder", "critical stock", "alert"
    ]):
        return "low_stock"

    # Drug info (default)
    return "drug_info"

# ── Keyword extractor ─────────────────────────────────────────
STOPWORDS = {
    "what", "is", "the", "for", "do", "we", "have", "any", "of",
    "tell", "me", "about", "price", "cost", "stock", "interact",
    "with", "how", "much", "many", "does", "supplier", "supplies",
    "supply", "who", "interacts", "use", "used", "expiry", "expire",
    "when", "a", "an", "drug", "medicine", "medication", "our", "give",
    "alternative", "substitute", "instead", "similar", "whats",
    "give", "can", "you", "recommend", "please", "there", "get",
    "find", "show", "list", "all", "are", "in", "to", "from",
    "that", "this", "on", "at", "by", "or", "and", "also", "its",
    "which", "where", "would", "could", "should", "will", "was",
    "been", "being", "has", "had", "not", "no", "yes"
}

def extract_keywords(question):
    """Extract meaningful keywords from a question, removing stopwords."""
    clean = re.sub(r"['\u2019?!,.]", "", question.lower())
    words = clean.split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return keywords

def get_search_term(question):
    """Get primary search term — first meaningful keyword."""
    keywords = extract_keywords(question)
    return keywords[0] if keywords else question.lower()

# ── SQLite queries ────────────────────────────────────────────
def query_stock_price(question):
    keywords = extract_keywords(question)
    if not keywords:
        return []
    # Build parameterised conditions
    conditions = " OR ".join(
        ["LOWER(generic_name) LIKE ? OR LOWER(brand_name) LIKE ?" for _ in keywords]
    )
    params = []
    for k in keywords:
        params.extend([f"%{k}%", f"%{k}%"])
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category
        FROM inventory
        WHERE {conditions}
        LIMIT 5
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params).to_dict("records")

def query_category_browse(question):
    """Browse drugs by category."""
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
        "stomach":          "GI medications",
        "respiratory":      "Respiratory",
        "antiretroviral":   "Antiretrovirals",
        "hiv":              "Antiretrovirals",
        "arv":              "Antiretrovirals",
    }
    matched_category = None
    for keyword, category in categories.items():
        if keyword in q:
            matched_category = category
            break
    if not matched_category:
        return []
    sql = """
        SELECT generic_name, brand_name, quantity_in_stock,
               selling_price_usd, shelf_location, category
        FROM inventory
        WHERE category = ? AND quantity_in_stock > 0
        ORDER BY generic_name
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=(matched_category,)).to_dict("records")

def query_stats():
    """Inventory summary by category."""
    sql = """
        SELECT category,
               COUNT(*)              AS drug_count,
               SUM(quantity_in_stock) AS total_units,
               ROUND(AVG(selling_price_usd), 2) AS avg_price
        FROM inventory
        GROUP BY category
        ORDER BY category
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_low_stock():
    """Drugs at or below reorder level."""
    sql = """
        SELECT generic_name, brand_name, quantity_in_stock,
               reorder_level, category,
               (reorder_level - quantity_in_stock) AS units_below_reorder
        FROM inventory
        WHERE quantity_in_stock <= reorder_level
        ORDER BY units_below_reorder DESC
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_drug_summary(drug_name):
    """Full summary card: stock + price + nearest expiry."""
    sql = """
        SELECT i.generic_name, i.brand_name, i.formulation, i.strength,
               i.quantity_in_stock, i.reorder_level,
               i.selling_price_usd, i.shelf_location, i.category,
               MIN(b.expiry_date) AS nearest_expiry,
               CAST(MIN(julianday(b.expiry_date) - julianday('now')) AS INTEGER)
               AS days_to_expiry
        FROM inventory i
        LEFT JOIN batches b ON i.product_id = b.product_id
        WHERE LOWER(i.generic_name) LIKE LOWER(?)
        GROUP BY i.product_id
        LIMIT 1
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=(f"%{drug_name}%",)).to_dict("records")

def query_alternative(question):
    """Find drugs in the same category — fully parameterised."""
    keywords = extract_keywords(question)
    if not keywords:
        return []
    # Try each keyword until we find a matching drug
    search_param = None
    for k in keywords:
        sql_check = "SELECT generic_name FROM inventory WHERE LOWER(generic_name) LIKE ? LIMIT 1"
        with get_conn() as conn:
            result = pd.read_sql_query(sql_check, conn, params=(f"%{k}%",))
        if not result.empty:
            search_param = f"%{k}%"
            break
    if not search_param:
        return []
    sql = """
        SELECT a.generic_name, a.brand_name, a.quantity_in_stock,
               a.selling_price_usd, a.category
        FROM inventory a
        WHERE a.category = (
            SELECT category FROM inventory
            WHERE LOWER(generic_name) LIKE ? LIMIT 1
        )
        AND LOWER(a.generic_name) NOT LIKE ?
        AND a.quantity_in_stock > 0
        ORDER BY a.generic_name
        LIMIT 5
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=(search_param, search_param)).to_dict("records")

def query_expiry(question):
    q = question.lower()
    # General expiry alert — anything expiring soon
    if any(w in q for w in [
        "soon", "this month", "next month", "90 days", "expiring",
        "about to expire", "near expiry", "closest", "earliest"
    ]):
        sql = """
            SELECT i.generic_name, b.batch_number, b.expiry_date,
                   b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER)
                   AS days_remaining
            FROM batches b
            JOIN inventory i ON b.product_id = i.product_id
            WHERE julianday(b.expiry_date) - julianday('now') <= 90
            ORDER BY b.expiry_date ASC
        """
        with get_conn() as conn:
            return pd.read_sql_query(sql, conn).to_dict("records")
    # Specific drug expiry
    keywords = extract_keywords(question)
    if not keywords:
        return []
    conditions = " OR ".join(["LOWER(i.generic_name) LIKE ?" for _ in keywords])
    params = [f"%{k}%" for k in keywords]
    sql = f"""
        SELECT i.generic_name, b.batch_number, b.expiry_date,
               b.quantity_remaining,
               CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER)
               AS days_remaining
        FROM batches b
        JOIN inventory i ON b.product_id = i.product_id
        WHERE {conditions}
        ORDER BY b.expiry_date ASC
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params).to_dict("records")

def query_sales(question):
    sql = """
        SELECT i.generic_name,
               SUM(t.quantity_sold)              AS total_units,
               ROUND(SUM(t.total_amount), 2)     AS total_revenue,
               COUNT(*)                           AS num_transactions
        FROM transactions t
        JOIN inventory i ON t.product_id = i.product_id
        GROUP BY i.generic_name
        ORDER BY total_revenue DESC
        LIMIT 10
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

# ── Neo4j queries ─────────────────────────────────────────────
def query_neo4j_interaction(question):
    keywords = extract_keywords(question)
    search_term = keywords[0] if keywords else get_search_term(question)
    cypher = """
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) CONTAINS toLower($search)
           OR toLower(b.generic_name) CONTAINS toLower($search)
        RETURN a.generic_name AS drug_a,
               b.generic_name AS drug_b,
               r.severity     AS severity,
               r.description  AS description,
               r.recommendation AS recommendation
        ORDER BY
            CASE r.severity
                WHEN 'Major'    THEN 1
                WHEN 'Moderate' THEN 2
                WHEN 'Minor'    THEN 3
                ELSE 4
            END
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
        RETURN d.generic_name    AS name,
               d.drug_class      AS drug_class,
               d.indications     AS indications,
               d.contraindications AS contraindications,
               d.side_effects    AS side_effects,
               d.adult_dose      AS adult_dose,
               d.pediatric_dose  AS pediatric_dose,
               d.prescription    AS prescription,
               d.controlled      AS controlled,
               c.name            AS category
        LIMIT 3
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, search=search_term)]

def query_neo4j_supplier(question):
    keywords = extract_keywords(question)
    search_term = keywords[0] if keywords else get_search_term(question)
    cypher = """
        MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
        WHERE toLower(d.generic_name) CONTAINS toLower($search)
        RETURN d.generic_name  AS drug,
               s.name          AS supplier,
               s.contact       AS contact,
               s.phone         AS phone,
               s.city          AS city,
               s.lead_time     AS lead_time_days,
               s.payment_terms AS payment_terms
        LIMIT 5
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, search=search_term)]

# ── Main query router ─────────────────────────────────────────
def run_query(question, intent):
    if intent == "greeting":
        return "system", []
    elif intent in ("thanks", "farewell", "followup"):
        return "system", []
    elif intent == "stock_price":
        return "inventory database",               query_stock_price(question)
    elif intent == "category_browse":
        return "inventory database",               query_category_browse(question)
    elif intent == "stats":
        return "inventory database",               query_stats()
    elif intent == "low_stock":
        return "inventory database",               query_low_stock()
    elif intent == "expiry":
        return "batch records",                    query_expiry(question)
    elif intent == "interaction":
        return "drug interaction knowledge graph",  query_neo4j_interaction(question)
    elif intent == "supplier":
        return "supplier knowledge graph",         query_neo4j_supplier(question)
    elif intent == "sales":
        return "transaction records",              query_sales(question)
    elif intent == "alternative":
        return "inventory database",               query_alternative(question)
    else:
        return "drug knowledge graph",             query_neo4j_drug_info(question)

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional pharmacy assistant at Sunrise Pharmacy, Harare, Zimbabwe,
built by Netrisyl Insights. You assist pharmacy staff with drug information and inventory queries.

STRICT RULES:
1. Answer ONLY using the structured data provided — never use outside medical knowledge.
2. Stick strictly to what was asked — do not volunteer unrelated information.
3. If the data is insufficient, say: "I don't have enough data to answer that. Please consult a qualified pharmacist."
4. Never guess, infer, or hallucinate drug names, doses, interactions, or clinical facts.
5. For drug interactions, always state the severity level (Minor / Moderate / Major).
6. For stock questions, state the exact quantity and flag if at or below reorder level.
7. For expiry questions, flag anything expiring within 30 days as URGENT.
8. The transactions data covers the LAST 30 DAYS — never describe it as daily sales.
9. For alternatives, list all available options with stock levels and prices.
10. Keep answers to 3-5 sentences unless listing multiple items.
11. Always mention the data source at the end of your answer.
12. For symptom or diagnosis questions, respond: "Please consult a qualified pharmacist for clinical recommendations."
"""

GREETING_RESPONSE = """👋 Hello! I am the **Netrisyl Pharmacy Assistant** for Sunrise Pharmacy.

I can help pharmacy staff with:
- 📦 **Stock & Prices** — *"Do we have amoxicillin in stock?"* / *"What is the price of paracetamol?"*
- ⚠️ **Drug Interactions** — *"What interacts with metformin?"* / *"Is it safe to combine ibuprofen and aspirin?"*
- 📅 **Expiry Alerts** — *"Which batches are expiring soon?"* / *"When does our azithromycin expire?"*
- 🚚 **Suppliers** — *"Who supplies ciprofloxacin?"* / *"Where do we order losartan from?"*
- 💊 **Drug Information** — *"What is ibuprofen used for?"* / *"What is the adult dose of amoxicillin?"*
- 🔄 **Alternatives** — *"What is an alternative to amoxicillin?"* / *"What can replace ibuprofen?"*
- 💰 **Sales Summary** — *"What are the top selling drugs?"* / *"Show me sales performance"*
- 📊 **Inventory Browse** — *"List all antibiotics"* / *"How many drug categories do we have?"*
- 🔴 **Low Stock Alerts** — *"Which drugs are running low?"*

⚠️ For symptom-based or clinical recommendations, please consult a qualified pharmacist.

How can I help you today?"""

THANKS_RESPONSE = "You're welcome! Feel free to ask anytime. 😊"
FAREWELL_RESPONSE = "Goodbye! Come back anytime you need help. 👋"

# ── GPT-4o-mini answer generator ──────────────────────────────
def generate_answer(question, intent, source, data, conversation_history=None):
    """
    Generates a natural language answer using GPT-4o-mini.
    Passes conversation history for follow-up question handling.
    """
    # Handle system intents without API call
    if intent == "greeting":
        return GREETING_RESPONSE
    if intent == "thanks":
        return THANKS_RESPONSE
    if intent == "farewell":
        return FAREWELL_RESPONSE

    # Build message history for follow-up awareness
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            messages.append(turn)

    # Handle follow-up with no new data — rely on conversation context
    if intent == "followup":
        messages.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content

    # No data found
    if not data:
        return ("I could not find any information matching your question in our pharmacy database. "
                "Please try searching by the exact drug name "
                "(e.g. 'Do we have Amoxicillin?' or 'What interacts with Metformin?'). "
                "For clinical recommendations, please consult a qualified pharmacist.")

    user_prompt = f"""
Question: {question}
Intent: {intent}
Data source: {source}
Retrieved data:
{json.dumps(data, indent=2)}

Answer using ONLY the data provided above. Do not add any information from outside this data.
"""
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content

def generate_drug_summary_answer(drug_name, data):
    """Structured summary card for drug chip / lookup button."""
    if not data:
        return (f"I could not find **{drug_name}** in the inventory. "
                "Please check the spelling or use the search box to find the correct drug name.")
    d = data[0]
    stock_status = "⚠️ LOW STOCK — reorder needed" if d["quantity_in_stock"] <= d["reorder_level"] else "✅ In Stock"
    expiry_line = ""
    if d.get("days_to_expiry") is not None:
        if d["days_to_expiry"] <= 30:
            expiry_line = f"\n🚨 **URGENT:** Nearest batch expires in {d['days_to_expiry']} days ({d['nearest_expiry']})"
        elif d["days_to_expiry"] <= 90:
            expiry_line = f"\n⚠️ Nearest expiry: {d['nearest_expiry']} ({d['days_to_expiry']} days)"
        else:
            expiry_line = f"\n📅 Nearest expiry: {d['nearest_expiry']} ({d['days_to_expiry']} days)"
    return f"""**{d['generic_name']}** ({d['brand_name']}) — {d['formulation']} {d['strength']}

| Field | Value |
|---|---|
| **Stock** | {d['quantity_in_stock']} units — {stock_status} |
| **Reorder Level** | {d['reorder_level']} units |
| **Selling Price** | ${d['selling_price_usd']} |
| **Cost Price** | ${d['cost_price_usd']} |
| **Shelf Location** | {d['shelf_location']} |
| **Category** | {d['category']} |
{expiry_line}"""

# ── Export chat ───────────────────────────────────────────────
def export_chat(chat_history):
    if not chat_history:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pharmacy_chat_{timestamp}.txt"
    lines = [
        "Netrisyl Pharmacy Assistant — Chat Export",
        f"Sunrise Pharmacy | Harare, Zimbabwe",
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

    # Build conversation history for follow-up context
    conversation_history = []
    for turn in (chat_history or []):
        conversation_history.append({"role": turn["role"], "content": turn["content"]})

    try:
        # Step 1 — Fuzzy spell correction
        corrected_message, correction_note = fuzzy_correct_question(message)

        # Step 2 — Classify intent
        intent = classify_intent(corrected_message, conversation_history)

        # Step 3 — Query data
        source, data = run_query(corrected_message, intent)

        # Step 4 — Generate answer
        answer = generate_answer(
            corrected_message, intent, source, data, conversation_history
        )

        # Step 5 — Format response
        if intent in ("greeting", "thanks", "farewell"):
            full_answer = answer
        else:
            full_answer = answer
            if correction_note:
                full_answer = f"{correction_note}\n\n{answer}"
            full_answer = f"{full_answer}\n\n*Source: {source} | Intent: {intent}*"

    except Exception as e:
        full_answer = f"An error occurred: {str(e)}\nPlease try rephrasing your question."

    # Update chat history
    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})

    # Update search history
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

# ── Drug summary card ─────────────────────────────────────────
def drug_summary(drug_name, chat_history, search_history):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update()
    try:
        data   = query_drug_summary(drug_name)
        answer = generate_drug_summary_answer(drug_name, data)
    except Exception as e:
        answer = f"Error fetching summary: {str(e)}"
    label = f"Quick summary: {drug_name}"
    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": label})
    chat_history.append({"role": "assistant", "content": answer})
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
    "Which drugs are running low?"
]

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Netrisyl Pharmacy Assistant") as demo:

    # ── Header ────────────────────────────────────────────────
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

    # ── Shared state ──────────────────────────────────────────
    search_history_state = gr.State([])

    # ── Wire chat input ───────────────────────────────────────
    submit.click(
        respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display]
    )
    msg.submit(
        respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display]
    )

    # ── History dropdown — re-ask ─────────────────────────────
    history_dropdown.change(
        reask_from_history,
        [history_dropdown, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display]
    )

    # ── Quick question buttons ────────────────────────────────
    for btn, question in zip(quick_btns, QUICK_QUESTIONS):
        btn.click(
            click_quick_question,
            [gr.Textbox(value=question, visible=False),
             chatbot, search_history_state],
            [msg, chatbot, search_history_state, history_dropdown, history_display]
        )

    # ── Drug chip buttons ─────────────────────────────────────
    all_chips = list(zip(drug_chips, FEATURED_DRUGS[:5])) + \
                list(zip(drug_chips2, FEATURED_DRUGS[5:]))
    for chip, drug_name in all_chips:
        chip.click(
            drug_summary,
            [gr.Textbox(value=drug_name, visible=False),
             chatbot, search_history_state],
            [chatbot, search_history_state, history_dropdown, history_display]
        )

    # ── Drug search & lookup ──────────────────────────────────
    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])
    drug_lookup_btn.click(
        drug_summary,
        [drug_dropdown, chatbot, search_history_state],
        [chatbot, search_history_state, history_dropdown, history_display]
    )

    # ── Export ────────────────────────────────────────────────
    def do_export(chat_history):
        f = export_chat(chat_history)
        return gr.update(value=f, visible=True) if f else gr.update(visible=False)

    export_btn.click(do_export, [chatbot], [export_file])

demo.launch()