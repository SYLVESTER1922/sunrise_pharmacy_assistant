
import os
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

# ── Clients ───────────────────────────────────────────────────
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Build SQLite from CSVs ────────────────────────────────────
# NOTE: Swap this section for Google Sheets or PostgreSQL in production

DB_PATH = "sunrise_pharmacy.db"

def build_database():
    """
    Builds the SQLite database from CSV files on startup.
    To upgrade to PostgreSQL: replace this function with a
    PostgreSQL connection and remove the CSV loading logic.
    """
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

# ── Thread-safe connection ────────────────────────────────────
def get_conn():
    """
    Returns a fresh SQLite connection per thread.
    To upgrade to PostgreSQL: replace with psycopg2.connect()
    """
    return sqlite3.connect(DB_PATH)

# ── Intent classification ─────────────────────────────────────
def classify_intent(question):
    q = question.lower()
    if any(w in q for w in ["stock", "have", "available", "quantity",
                              "how many", "price", "cost", "how much"]):
        return "stock_price"
    elif any(w in q for w in ["expir", "expire", "expiry", "batch"]):
        return "expiry"
    elif any(w in q for w in ["interact", "together", "combine",
                                "mix", "safe with"]):
        return "interaction"
    elif any(w in q for w in ["supplier", "order from", "who supply",
                                "distributor", "vendor", "supplies"]):
        return "supplier"
    elif any(w in q for w in ["sold", "sales", "revenue",
                                "dispensed", "transaction", "top selling"]):
        return "sales"
    else:
        return "drug_info"

# ── Keyword extractor ─────────────────────────────────────────
def extract_drug_name(question):
    stopwords = ["what", "is", "the", "for", "do", "we", "have", "any",
                 "of", "tell", "me", "about", "price", "cost", "stock",
                 "interact", "with", "how", "much", "many", "does",
                 "supplier", "supplies", "who", "interacts", "use",
                 "used", "expiry", "expire", "when", "a", "an", "drug",
                 "medicine", "medication", "our", "give"]
    words = question.lower().replace("?", "").split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(keywords)

# ── SQLite queries ────────────────────────────────────────────
def query_stock_price(question):
    keywords = extract_drug_name(question)
    parts = keywords.split()
    conditions = " OR ".join(
        [f"LOWER(generic_name) LIKE '%{p}%' OR LOWER(brand_name) LIKE '%{p}%'"
         for p in parts]
    )
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category
        FROM inventory WHERE {conditions} LIMIT 5
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_expiry(question):
    if any(w in question.lower() for w in
           ["soon", "this month", "next month", "90 days", "expiring"]):
        sql = """
            SELECT i.generic_name, b.batch_number, b.expiry_date,
                   b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday("now") AS INTEGER)
                   AS days_remaining
            FROM batches b
            JOIN inventory i ON b.product_id = i.product_id
            WHERE julianday(b.expiry_date) - julianday("now") <= 90
            ORDER BY b.expiry_date ASC
        """
        with get_conn() as conn:
            return pd.read_sql_query(sql, conn).to_dict("records")
    else:
        keywords = extract_drug_name(question)
        parts = keywords.split()
        conditions = " OR ".join(
            [f"LOWER(i.generic_name) LIKE '%{p}%'" for p in parts]
        )
        sql = f"""
            SELECT i.generic_name, b.batch_number, b.expiry_date,
                   b.quantity_remaining,
                   CAST(julianday(b.expiry_date) - julianday("now") AS INTEGER)
                   AS days_remaining
            FROM batches b
            JOIN inventory i ON b.product_id = i.product_id
            WHERE {conditions}
            ORDER BY b.expiry_date ASC
        """
        with get_conn() as conn:
            return pd.read_sql_query(sql, conn).to_dict("records")

def query_sales(question):
    sql = """
        SELECT i.generic_name,
               SUM(t.quantity_sold)  AS total_units,
               SUM(t.total_amount)   AS total_revenue,
               COUNT(*)              AS num_transactions
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
    keywords = extract_drug_name(question)
    parts = keywords.split()
    search_term = parts[0] if parts else keywords
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
    keywords = extract_drug_name(question)
    parts = keywords.split()
    search_term = parts[0] if parts else keywords
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
    keywords = extract_drug_name(question)
    parts = keywords.split()
    search_term = parts[0] if parts else keywords
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
def run_query(question):
    intent = classify_intent(question)
    if intent == "stock_price":
        return intent, "inventory database",        query_stock_price(question)
    elif intent == "expiry":
        return intent, "batch records",             query_expiry(question)
    elif intent == "interaction":
        return intent, "drug interaction knowledge graph", query_neo4j_interaction(question)
    elif intent == "supplier":
        return intent, "supplier knowledge graph",  query_neo4j_supplier(question)
    elif intent == "sales":
        return intent, "transaction records",       query_sales(question)
    else:
        return intent, "drug knowledge graph",      query_neo4j_drug_info(question)

# ── GPT-4o-mini answer generator ──────────────────────────────
def generate_answer(question, intent, source, data):
    if not data:
        return ("I could not find any information matching your question. "
                "Please check the drug name and try again.")
    system_prompt = """You are a helpful pharmacy assistant at Sunrise Pharmacy 
in Harare, Zimbabwe. You answer questions for pharmacy staff clearly and concisely.
Rules:
- Answer in 3-5 sentences maximum
- Always mention the data source
- For drug interactions, always state the severity level
- For stock questions, mention if stock is near reorder level
- For expiry questions, flag anything expiring within 30 days as URGENT
- Never make up information not in the data
- Use simple language suitable for pharmacy staff"""
    user_prompt = f"""
Question: {question}
Intent: {intent}
Source: {source}
Data: {json.dumps(data, indent=2)}
Answer using only the data provided.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content

# ── Gradio interface ──────────────────────────────────────────
suggestions = [
    "Do we have amoxicillin in stock?",
    "What interacts with metformin?",
    "Which batches are expiring soon?",
    "Who supplies ciprofloxacin?",
    "What is the price of paracetamol?",
    "What is ibuprofen used for?"
]

def respond(message, chat_history):
    if not message or message.strip() == "":
        return "", chat_history
    try:
        intent, source, data = run_query(message)
        answer = generate_answer(message, intent, source, data)
        full_answer = f"{answer}\n\n*Source: {source} | Intent: {intent}*"
    except Exception as e:
        full_answer = f"Error: {str(e)}"
    chat_history = chat_history or []
    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})
    return "", chat_history

def click_suggestion(suggestion, chat_history):
    return respond(suggestion, chat_history)

with gr.Blocks(theme=gr.themes.Soft(),
               title="Netrisyl Pharmacy Assistant") as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #1a5276, #2e86c1);
                padding: 24px; border-radius: 10px;
                margin-bottom: 16px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 26px;">
            💊 Netrisyl Pharmacy Assistant
        </h1>
        <p style="color: #aed6f1; margin: 6px 0 0 0; font-size: 14px;">
            Powered by Neo4j Knowledge Graph + GPT-4o-mini | Harare, Zimbabwe
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Quick Questions")
            gr.Markdown("Click any question to get an instant answer:")
            btns = [gr.Button(s, variant="secondary", size="sm")
                    for s in suggestions]
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
            chatbot = gr.Chatbot(
                label="Pharmacy Assistant",
                height=480,
                type="messages",
                avatar_images=(
                    None,
                    "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
                )
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about stock, prices, interactions, expiry...",
                    label="",
                    scale=5
                )
                submit = gr.Button("Ask", variant="primary", scale=1)

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond,  [msg, chatbot], [msg, chatbot])

    for btn, suggestion in zip(btns, suggestions):
        btn.click(
            fn=click_suggestion,
            inputs=[gr.Textbox(value=suggestion, visible=False), chatbot],
            outputs=[msg, chatbot]
        )

    gr.HTML("""
    <div style="text-align: center; margin-top: 16px;
                color: #7f8c8d; font-size: 12px;">
        Netrisyl Insights · Harare, Zimbabwe · Powered by AI
    </div>
    """)

demo.launch()
