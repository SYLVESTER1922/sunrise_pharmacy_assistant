import os
import re
import sqlite3
import pandas as pd
import gradio as gr
from neo4j import GraphDatabase
from openai import OpenAI
import json

# ── Credentials ──────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Build SQLite from CSVs ────────────────────────────────────
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

def get_conn():
    return sqlite3.connect(DB_PATH)

# ── Intent classification ─────────────────────────────────────
def classify_intent(question):
    q = question.lower()

    # Greetings
    if any(w in q for w in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy']):
        if len(q.split()) <= 4:
            return "greeting"

    # Thanks
    if any(w in q for w in ['thank', 'thanks', 'cheers']):
        return "thanks"

    # Farewell
    if any(w in q for w in ['bye', 'goodbye', 'see you', 'later']):
        return "farewell"

    # Follow-up
    if any(w in q for w in ['tell me more', 'more about', 'what else', 'elaborate',
                             'expand', 'these', 'those', 'them', 'it', 'that',
                             'you mentioned', 'the same', 'summarize', 'summary']):
        return "followup"

    # Sales
    if any(w in q for w in ['sold', 'sales', 'revenue', 'dispensed', 'transaction',
                             'top selling', 'best selling', 'most sold']):
        return "sales"

    # Low stock
    if any(w in q for w in ['low stock', 'reorder', 'running low', 'almost out',
                             'need to order', 'below reorder', 'critical stock', 'alert']):
        return "low_stock"

    # Stock/Price
    if any(w in q for w in ['stock', 'have', 'available', 'quantity', 'how many',
                             'price', 'cost', 'how much', 'in stock']):
        return "stock_price"

    # Category browse
    if any(w in q for w in ['list all', 'show all', 'all antibiotics', 'all analgesics',
                             'category', 'categories', 'how many drugs', 'how many categories']):
        return "category_browse"

    # Stats
    if any(w in q for w in ['overview', 'statistics', 'summary', 'total drugs',
                             'total products', 'inventory summary']):
        return "stats"

    # Expiry
    if any(w in q for w in ['expir', 'expire', 'expiry', 'batch', 'shelf life']):
        return "expiry"

    # Interactions
    if any(w in q for w in ['interact', 'together', 'combine', 'mix', 'safe with',
                             'take with', 'combination']):
        return "interaction"

    # Supplier
    if any(w in q for w in ['supplier', 'order from', 'who supply', 'distributor',
                             'vendor', 'supplies', 'where to get']):
        return "supplier"

    # Alternative
    if any(w in q for w in ['alternative', 'substitute', 'instead of', 'replace',
                             'similar to', 'other option', 'other drug', 'swap',
                             'equivalent']):
        return "alternative"

    # Drug info (default)
    return "drug_info"

# ── Keyword extractor ─────────────────────────────────────────
STOPWORDS = {'what','is','the','for','do','we','have','any','of','tell','me','about',
             'price','cost','stock','interact','with','how','much','many','does',
             'supplier','supplies','supply','who','interacts','use','used','expiry',
             'expire','when','a','an','drug','medicine','medication','our','give',
             'alternative','substitute','instead','similar','whats','can','you',
             'recommend','please','there','get','find','show','list','all','are',
             'in','to','from','that','this','on','at','by','or','and','also','its',
             'which','where','would','could','should','will','was','been','being',
             'has','had','not','no','yes','quick','summary'}

def extract_keywords(question):
    clean = re.sub(r"['\u2019?!,.]", "", question.lower())
    words = clean.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def get_search_term(question):
    keywords = extract_keywords(question)
    return keywords[0] if keywords else question.lower()

# ── SQLite queries ────────────────────────────────────────────
def query_stock_price(question):
    keywords = extract_keywords(question)
    if not keywords:
        return []
    conditions = " OR ".join(
        ["LOWER(generic_name) LIKE ? OR LOWER(brand_name) LIKE ?" for _ in keywords])
    params = []
    for k in keywords:
        params.extend([f"%{k}%", f"%{k}%"])
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category
        FROM inventory WHERE {conditions} LIMIT 5
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params).to_dict("records")

def query_drug_summary(question):
    keywords = extract_keywords(question)
    if not keywords:
        return []
    conditions = " OR ".join(
        ["LOWER(generic_name) LIKE ? OR LOWER(brand_name) LIKE ?" for _ in keywords])
    params = []
    for k in keywords:
        params.extend([f"%{k}%", f"%{k}%"])
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category
        FROM inventory WHERE {conditions} LIMIT 1
    """
    with get_conn() as conn:
        inv = pd.read_sql_query(sql, conn, params=params).to_dict("records")
    if not inv:
        return []
    drug = inv[0]
    # Get nearest expiry
    exp_sql = """
        SELECT b.expiry_date,
               CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER) AS days_remaining
        FROM batches b
        JOIN inventory i ON b.product_id = i.product_id
        WHERE LOWER(i.generic_name) LIKE ?
        ORDER BY b.expiry_date ASC LIMIT 1
    """
    with get_conn() as conn:
        exp = pd.read_sql_query(exp_sql, conn, params=[f"%{keywords[0]}%"]).to_dict("records")
    if exp:
        drug['nearest_expiry'] = exp[0]['expiry_date']
        drug['days_to_expiry'] = exp[0]['days_remaining']
    return [drug]

def query_category_browse(question):
    q = question.lower()
    categories = {
        "antibiotic": "Antibiotics", "analgesic": "Analgesics",
        "antihypertensive": "Antihypertensives", "antidiabetic": "Antidiabetics",
        "antimalarial": "Antimalarials", "vitamin": "Vitamins/Supplements",
        "supplement": "Vitamins/Supplements", "antifungal": "Antifungals",
        "gi": "GI medications", "gastrointestinal": "GI medications",
        "respiratory": "Respiratory", "antiretroviral": "Antiretrovirals",
        "hiv": "Antiretrovirals", "arv": "Antiretrovirals",
    }
    matched = None
    for keyword, category in categories.items():
        if keyword in q:
            matched = category
            break
    if not matched:
        return []
    sql = """
        SELECT generic_name, brand_name, quantity_in_stock,
               selling_price_usd, shelf_location, category
        FROM inventory WHERE category = ? AND quantity_in_stock > 0
        ORDER BY generic_name
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=(matched,)).to_dict("records")

def query_stats():
    sql = """
        SELECT category, COUNT(*) AS drug_count,
               SUM(quantity_in_stock) AS total_units,
               ROUND(AVG(selling_price_usd), 2) AS avg_price
        FROM inventory GROUP BY category ORDER BY category
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_low_stock():
    sql = """
        SELECT generic_name, brand_name, quantity_in_stock,
               reorder_level, category,
               (reorder_level - quantity_in_stock) AS units_below_reorder
        FROM inventory WHERE quantity_in_stock <= reorder_level
        ORDER BY units_below_reorder DESC
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_expiry(question):
    if any(w in question.lower() for w in ['soon', 'this month', 'next month', '90 days', 'expiring']):
        sql = """
            SELECT i.generic_name, b.batch_number, b.expiry_date,
                   b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER) AS days_remaining
            FROM batches b JOIN inventory i ON b.product_id = i.product_id
            WHERE julianday(b.expiry_date) - julianday('now') <= 90
            ORDER BY b.expiry_date ASC
        """
        with get_conn() as conn:
            return pd.read_sql_query(sql, conn).to_dict("records")
    else:
        keywords = extract_keywords(question)
        if not keywords:
            return []
        conditions = " OR ".join([f"LOWER(i.generic_name) LIKE ?" for _ in keywords])
        params = [f"%{k}%" for k in keywords]
        sql = f"""
            SELECT i.generic_name, b.batch_number, b.expiry_date,
                   b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday('now') AS INTEGER) AS days_remaining
            FROM batches b JOIN inventory i ON b.product_id = i.product_id
            WHERE {conditions} ORDER BY b.expiry_date ASC
        """
        with get_conn() as conn:
            return pd.read_sql_query(sql, conn, params=params).to_dict("records")

def query_sales(question):
    sql = """
        SELECT i.generic_name, SUM(t.quantity_sold) AS total_units,
               SUM(t.total_amount) AS total_revenue,
               COUNT(*) AS num_transactions
        FROM transactions t JOIN inventory i ON t.product_id = i.product_id
        GROUP BY i.generic_name ORDER BY total_revenue DESC LIMIT 10
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_alternative(question):
    keywords = extract_keywords(question)
    if not keywords:
        return []
    # Find category of the drug
    conditions = " OR ".join(["LOWER(generic_name) LIKE ?" for _ in keywords])
    params = [f"%{k}%" for k in keywords]
    sql = f"SELECT category, generic_name FROM inventory WHERE {conditions} LIMIT 1"
    with get_conn() as conn:
        result = pd.read_sql_query(sql, conn, params=params).to_dict("records")
    if not result:
        return []
    category = result[0]['category']
    original = result[0]['generic_name']
    sql2 = """
        SELECT generic_name, brand_name, quantity_in_stock, selling_price_usd, category
        FROM inventory WHERE category = ? AND generic_name != ?
        ORDER BY quantity_in_stock DESC LIMIT 5
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql2, conn, params=(category, original)).to_dict("records")

# ── Neo4j queries ─────────────────────────────────────────────
def query_neo4j_interaction(question):
    search_term = get_search_term(question)
    cypher = """
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) CONTAINS toLower($search)
           OR toLower(b.generic_name) CONTAINS toLower($search)
        RETURN a.generic_name AS drug_a, b.generic_name AS drug_b,
               r.severity AS severity, r.description AS description,
               r.recommendation AS recommendation
        LIMIT 5
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, search=search_term)]

def query_neo4j_drug_info(question):
    search_term = get_search_term(question)
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
               c.name AS category
        LIMIT 3
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, search=search_term)]

def query_neo4j_supplier(question):
    search_term = get_search_term(question)
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
        return [dict(r) for r in session.run(cypher, search=search_term)]

# ── Main query router ─────────────────────────────────────────
def run_query(question, intent):
    if intent in ("greeting", "thanks", "farewell", "followup"):
        return "system", []
    elif intent == "stock_price":
        return "inventory database", query_stock_price(question)
    elif intent == "category_browse":
        return "inventory database", query_category_browse(question)
    elif intent == "stats":
        return "inventory database", query_stats()
    elif intent == "low_stock":
        return "inventory database", query_low_stock()
    elif intent == "expiry":
        return "batch records", query_expiry(question)
    elif intent == "interaction":
        return "drug interaction knowledge graph", query_neo4j_interaction(question)
    elif intent == "supplier":
        return "supplier knowledge graph", query_neo4j_supplier(question)
    elif intent == "sales":
        return "transaction records", query_sales(question)
    elif intent == "alternative":
        return "inventory database", query_alternative(question)
    else:
        # drug_info: try Neo4j first, fallback to inventory (handles brand names)
        neo4j_data = query_neo4j_drug_info(question)
        if neo4j_data:
            return "drug knowledge graph", neo4j_data
        inventory_data = query_stock_price(question)
        if inventory_data:
            return "inventory database", inventory_data
        return "drug knowledge graph", []

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

I can help with:
- 📦 **Stock & Prices** — "Do we have amoxicillin?"
- ⚠️ **Drug Interactions** — "What interacts with metformin?"
- 📅 **Expiry Alerts** — "Which batches are expiring soon?"
- 🚚 **Suppliers** — "Who supplies ciprofloxacin?"
- 💊 **Drug Information** — "What is ibuprofen used for?"
- 🔄 **Alternatives** — "What is an alternative to amoxicillin?"
- 💰 **Sales Summary** — "What are the top selling drugs?"
- 🔴 **Low Stock Alerts** — "Which drugs are running low?"

How can I help you today?"""

THANKS_RESPONSE = "You're welcome! Feel free to ask anytime. 😊"
FAREWELL_RESPONSE = "Goodbye! Come back anytime you need help. 👋"

# ── GPT-4o-mini answer generator ──────────────────────────────
def generate_answer(question, intent, source, data, conversation_history=None):
    if intent == "greeting":
        return GREETING_RESPONSE
    if intent == "thanks":
        return THANKS_RESPONSE
    if intent == "farewell":
        return FAREWELL_RESPONSE

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            messages.append(turn)

    if intent == "followup":
        messages.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=400)
        return response.choices[0].message.content

    # Low stock with no results = everything is fine
    if not data and intent == "low_stock":
        return "✅ All products are currently above reorder levels. No restocking needed at this time.\n\nData source: inventory database."

    if not data:
        return ("I could not find any information matching your question in our pharmacy database. "
                "Please try searching by the exact drug name (e.g. 'Do we have Amoxicillin?' or "
                "'What interacts with Metformin?'). For clinical recommendations, please consult a qualified pharmacist.")

    user_prompt = f"Question: {question}\nIntent: {intent}\nSource: {source}\nData: {json.dumps(data, indent=2)}\nAnswer using only the data provided."

    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=400)
    return response.choices[0].message.content

# ── Chat function ─────────────────────────────────────────────
def chat(question, conversation_history=None):
    intent = classify_intent(question)
    source, data = run_query(question, intent)
    answer = generate_answer(question, intent, source, data, conversation_history)
    return answer, intent, source

# ── Gradio interface ──────────────────────────────────────────
suggestions = [
    "Do we have amoxicillin in stock?",
    "What interacts with metformin?",
    "Which batches are expiring soon?",
    "Who supplies ciprofloxacin?",
    "What are the top selling drugs?",
    "Which drugs are running low?"
]

def respond(message, chat_history):
    if not message or message.strip() == "":
        return "", chat_history
    chat_history = chat_history or []
    conversation_history = []
    for h in chat_history:
        if isinstance(h, dict):
            conversation_history.append(h)
    try:
        answer, intent, source = chat(message, conversation_history)
        full_answer = f"{answer}\n*Source: {source} | Intent: {intent}*"
    except Exception as e:
        full_answer = f"Error: {str(e)}"
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})
    return "", chat_history

def click_suggestion(suggestion, chat_history):
    return respond(suggestion, chat_history)

with gr.Blocks(title="Netrisyl Pharmacy Assistant") as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #1a5276, #2e86c1);
                padding: 24px; border-radius: 10px;
                margin-bottom: 16px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 26px;">
            💊 Netrisyl Pharmacy Assistant
        </h1>
        <p style="color: #aed6f1; margin: 6px 0 0 0; font-size: 14px;">
            Powered by Neo4j Knowledge Graph + GPT-4o-mini | Sunrise Pharmacy, Harare
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Quick Questions")
            btns = [gr.Button(s, variant="secondary", size="sm") for s in suggestions]
            gr.Markdown("---")
            gr.Markdown("""
**Data Sources:**
- 📦 Inventory & Pricing
- 🧪 Drug Knowledge Graph
- ⚠️ Drug Interactions
- 📅 Batch & Expiry Records
- 🚚 Supplier Network
- 💰 Transaction History
            """)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Pharmacy Assistant", height=480)
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask about stock, prices, interactions, expiry...",
                                 label="", scale=5)
                submit = gr.Button("Ask", variant="primary", scale=1)

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    for btn, suggestion in zip(btns, suggestions):
        btn.click(
            fn=click_suggestion,
            inputs=[gr.Textbox(value=suggestion, visible=False), chatbot],
            outputs=[msg, chatbot])

    gr.HTML("""
    <div style="text-align: center; margin-top: 16px;
                color: #7f8c8d; font-size: 12px;">
        Netrisyl Insights · Harare, Zimbabwe · Powered by AI
    </div>
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)