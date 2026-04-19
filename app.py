import os
import sqlite3
import pandas as pd
import gradio as gr
from neo4j import GraphDatabase
from openai import OpenAI
import json
from datetime import datetime

# ── Credentials ──────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ── Clients ───────────────────────────────────────────────────
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

# ── Thread-safe connection ────────────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH)

# ── Load drug list for suggestions ───────────────────────────
def get_all_drugs():
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT generic_name, brand_name, category FROM inventory ORDER BY generic_name",
            conn
        )
    return df.to_dict("records")

ALL_DRUGS = get_all_drugs()
DRUG_NAMES = [d["generic_name"] for d in ALL_DRUGS]

# ── Intent classification ─────────────────────────────────────
GREETINGS = ["hi", "hey", "hello", "good morning", "good afternoon",
             "good evening", "help", "what can you do", "how are you"]

def classify_intent(question):
    q = question.lower().strip()
    if any(q == g or q.startswith(g) for g in GREETINGS):
        return "greeting"
    if any(w in q for w in ["stock", "have", "available", "quantity",
                              "how many", "price", "cost", "how much"]):
        return "stock_price"
    elif any(w in q for w in ["expir", "expire", "expiry", "batch"]):
        return "expiry"
    elif any(w in q for w in ["interact", "together", "combine",
                                "mix", "safe with"]):
        return "interaction"
    elif any(w in q for w in ["supplier", "order from", "who supply",
                                "distributor", "vendor", "supplies", "supply"]):
        return "supplier"
    elif any(w in q for w in ["sold", "sales", "revenue",
                                "dispensed", "transaction", "top selling"]):
        return "sales"
    elif any(w in q for w in ["alternative", "substitute", "instead of",
                                "replace", "similar"]):
        return "alternative"
    else:
        return "drug_info"

# ── Keyword extractor ─────────────────────────────────────────
def extract_drug_name(question):
    stopwords = ["what", "is", "the", "for", "do", "we", "have", "any",
                 "of", "tell", "me", "about", "price", "cost", "stock",
                 "interact", "with", "how", "much", "many", "does",
                 "supplier", "supplies", "supply", "who", "interacts",
                 "use", "used", "expiry", "expire", "when", "a", "an",
                 "drug", "medicine", "medication", "our", "give",
                 "alternative", "substitute", "instead", "similar"]
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

def query_drug_summary(drug_name):
    """Full summary card: stock + price + nearest expiry"""
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
        results = pd.read_sql_query(sql, conn, params=(f"%{drug_name}%",)).to_dict("records")
    return results

def query_alternative(question):
    """Find drugs in the same category"""
    keywords = extract_drug_name(question)
    parts = keywords.split()
    search = parts[0] if parts else keywords
    sql = f"""
        SELECT a.generic_name, a.brand_name, a.quantity_in_stock,
               a.selling_price_usd, a.category
        FROM inventory a
        WHERE a.category = (
            SELECT category FROM inventory
            WHERE LOWER(generic_name) LIKE '%{search}%' LIMIT 1
        )
        AND LOWER(a.generic_name) NOT LIKE '%{search}%'
        AND a.quantity_in_stock > 0
        LIMIT 5
    """
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn).to_dict("records")

def query_expiry(question):
    if any(w in question.lower() for w in
           ["soon", "this month", "next month", "90 days", "expiring"]):
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
    else:
        keywords = extract_drug_name(question)
        parts = keywords.split()
        conditions = " OR ".join(
            [f"LOWER(i.generic_name) LIKE '%{p}%'" for p in parts]
        )
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
    if intent == "greeting":
        return intent, "system", []
    elif intent == "stock_price":
        return intent, "inventory database",              query_stock_price(question)
    elif intent == "expiry":
        return intent, "batch records",                   query_expiry(question)
    elif intent == "interaction":
        return intent, "drug interaction knowledge graph", query_neo4j_interaction(question)
    elif intent == "supplier":
        return intent, "supplier knowledge graph",        query_neo4j_supplier(question)
    elif intent == "sales":
        return intent, "transaction records",             query_sales(question)
    elif intent == "alternative":
        return intent, "inventory database",              query_alternative(question)
    else:
        return intent, "drug knowledge graph",            query_neo4j_drug_info(question)

# ── GPT-4o-mini answer generator ──────────────────────────────
GREETING_RESPONSE = """👋 Hello! I'm the Netrisyl Pharmacy Assistant. I can help you with:

- 📦 **Stock & Prices** — *"Do we have amoxicillin in stock?"*
- ⚠️ **Drug Interactions** — *"What interacts with metformin?"*
- 📅 **Expiry Alerts** — *"Which batches are expiring soon?"*
- 🚚 **Suppliers** — *"Who supplies ciprofloxacin?"*
- 💊 **Drug Information** — *"What is ibuprofen used for?"*
- 🔄 **Alternatives** — *"What is an alternative to amoxicillin?"*
- 💰 **Sales Summary** — *"What are the top selling drugs?"*

You can also click a drug name below the chat box for a quick summary. How can I help you today?"""

def generate_answer(question, intent, source, data):
    if intent == "greeting":
        return GREETING_RESPONSE
    if not data:
        return ("I could not find any information matching your question. "
                "Please check the drug name and try again. "
                "You can also click a drug name button below for a quick lookup.")
    system_prompt = """You are a helpful pharmacy assistant at Sunrise Pharmacy
in Harare, Zimbabwe. You answer questions for pharmacy staff clearly and concisely.
Rules:
- Answer in 3-5 sentences maximum
- Always mention the data source
- For drug interactions, always state the severity level
- For stock questions, mention if stock is near reorder level
- For expiry questions, flag anything expiring within 30 days as URGENT
- For alternatives, list available options with stock levels
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

def generate_drug_summary_answer(drug_name, data):
    """Generate a summary card answer for drug quick-select"""
    if not data:
        return f"I could not find **{drug_name}** in the inventory."
    d = data[0]
    stock_status = "⚠️ LOW STOCK" if d["quantity_in_stock"] <= d["reorder_level"] else "✅ In Stock"
    expiry_alert = ""
    if d.get("days_to_expiry") and d["days_to_expiry"] <= 30:
        expiry_alert = f"\n🚨 **URGENT:** Nearest batch expires in {d['days_to_expiry']} days ({d['nearest_expiry']})"
    elif d.get("days_to_expiry"):
        expiry_alert = f"\n📅 Nearest expiry: {d['nearest_expiry']} ({d['days_to_expiry']} days)"
    return f"""**{d['generic_name']}** ({d['brand_name']}) — {d['formulation']} {d['strength']}

| | |
|---|---|
| **Stock** | {d['quantity_in_stock']} units {stock_status} |
| **Reorder Level** | {d['reorder_level']} units |
| **Selling Price** | ${d['selling_price_usd']} |
| **Shelf Location** | {d['shelf_location']} |
| **Category** | {d['category']} |
{expiry_alert}"""

# ── Export chat ───────────────────────────────────────────────
def export_chat(chat_history):
    if not chat_history:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"pharmacy_chat_{timestamp}.txt"
    lines = [f"Netrisyl Pharmacy Assistant — Chat Export",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 60, ""]
    for msg in chat_history:
        role = "Staff" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{msg['content']}\n")
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    return filename

# ── Drug search filter ────────────────────────────────────────
def filter_drugs(search_text):
    if not search_text or len(search_text) < 2:
        return gr.update(choices=DRUG_NAMES[:20])
    matches = [d for d in DRUG_NAMES
               if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches)

# ── Gradio interface ──────────────────────────────────────────
QUICK_QUESTIONS = [
    "Do we have amoxicillin in stock?",
    "What interacts with metformin?",
    "Which batches are expiring soon?",
    "Who supplies ciprofloxacin?",
    "What are the top selling drugs?",
    "What is ibuprofen used for?"
]

def respond(message, chat_history, search_history):
    if not message or message.strip() == "":
        return "", chat_history, search_history, gr.update(), gr.update()
    try:
        intent, source, data = run_query(message)
        answer = generate_answer(message, intent, source, data)
        if intent != "greeting":
            full_answer = f"{answer}\n\n*Source: {source} | Intent: {intent}*"
        else:
            full_answer = answer
    except Exception as e:
        full_answer = f"Error: {str(e)}"
    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})
    search_history = list(search_history or [])
    if message not in search_history:
        search_history.insert(0, message)
    search_history = search_history[:10]
    history_md = "\n".join([f"- {h}" for h in search_history])
    return "", chat_history, search_history, gr.update(choices=search_history, value=None), gr.update(value=history_md)

def drug_summary(drug_name, chat_history, search_history):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update()
    try:
        data = query_drug_summary(drug_name)
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
    search_history = search_history[:10]
    history_md = "\n".join([f"- {h}" for h in search_history])
    return chat_history, search_history, gr.update(choices=search_history, value=None), gr.update(value=history_md)

def click_quick_question(question, chat_history, search_history):
    return respond(question, chat_history, search_history)

# ── Featured drugs for quick chips ───────────────────────────
FEATURED_DRUGS = [
    "Amoxicillin", "Paracetamol", "Metformin", "Ibuprofen",
    "Ciprofloxacin", "Azithromycin", "Amlodipine", "Losartan",
    "Artemether/Lumefantrine", "Cotrimoxazole"
]

with gr.Blocks(title="Netrisyl Pharmacy Assistant") as demo:

    with gr.Row():

        # ── Left sidebar — Drug Lookup ────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Drug Lookup")
            drug_search = gr.Textbox(
                placeholder="Type drug name (e.g. amox)...",
                label="Search"
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
- 💰 Transaction History
            """)

        # ── Main chat ─────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Pharmacy Assistant",
                height=480
            )
            gr.Markdown("**💊 Quick Drug Lookup:**")
            with gr.Row():
                drug_chips = [gr.Button(d, variant="secondary", size="sm")
                              for d in FEATURED_DRUGS[:5]]
            with gr.Row():
                drug_chips2 = [gr.Button(d, variant="secondary", size="sm")
                               for d in FEATURED_DRUGS[5:]]
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about stock, prices, interactions, expiry...",
                    label="",
                    scale=5
                )
                submit = gr.Button("Ask", variant="primary", scale=1)
            with gr.Row():
                export_btn  = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file = gr.File(label="Download", scale=2, visible=False)

        # ── Right sidebar — Questions & History ───────────────
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Quick Questions")
            quick_btns = [gr.Button(q, variant="secondary", size="sm")
                          for q in QUICK_QUESTIONS]

            gr.Markdown("---")
            gr.Markdown("### 🕘 Search History")
            history_dropdown = gr.Dropdown(
                choices=[],
                label="Past questions — select to re-ask",
                interactive=True
            )
            history_display = gr.Markdown("*No searches yet*")

    with gr.Row():

        # ── Left sidebar ──────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Quick Questions")
            quick_btns = [gr.Button(q, variant="secondary", size="sm")
                          for q in QUICK_QUESTIONS]

            gr.Markdown("---")
            gr.Markdown("### 🔍 Drug Lookup")
            drug_search = gr.Textbox(
                placeholder="Type drug name (e.g. amox)...",
                label="Search"
            )
            drug_dropdown = gr.Dropdown(
                choices=DRUG_NAMES[:20],
                label="Select drug",
                interactive=True
            )
            drug_lookup_btn = gr.Button("📋 Get Summary", variant="primary", size="sm")

            gr.Markdown("---")
            gr.Markdown("### 🕘 Search History")
            history_display = gr.Markdown("*No searches yet*")

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

        # ── Main chat ─────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Pharmacy Assistant",
                height=480
            )

            # ── Drug chip buttons ─────────────────────────────
            gr.Markdown("**💊 Quick Drug Lookup:**")
            with gr.Row():
                drug_chips = [gr.Button(d, variant="secondary", size="sm")
                              for d in FEATURED_DRUGS[:5]]
            with gr.Row():
                drug_chips2 = [gr.Button(d, variant="secondary", size="sm")
                               for d in FEATURED_DRUGS[5:]]

            # ── Chat input ────────────────────────────────────
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about stock, prices, interactions, expiry, suppliers...",
                    label="",
                    scale=5
                )
                submit = gr.Button("Ask", variant="primary", scale=1)

            # ── Export ────────────────────────────────────────
            with gr.Row():
                export_btn  = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file = gr.File(label="Download", scale=2, visible=False)

    gr.HTML("""
    <div style="text-align: center; margin-top: 16px;
                color: #7f8c8d; font-size: 12px;">
        Netrisyl Insights · Harare, Zimbabwe · Powered by AI
    </div>
    """)

    # ── State ─────────────────────────────────────────────────
    search_history_state = gr.State([])

    # ── Wire chat input ───────────────────────────────────────
    submit.click(respond,
                 [msg, chatbot, search_history_state],
                 [msg, chatbot, search_history_state, history_dropdown, history_display])
    msg.submit(respond,
               [msg, chatbot, search_history_state],
               [msg, chatbot, search_history_state, history_dropdown, history_display])

    # ── Wire history dropdown — re-ask selected question ─────
    history_dropdown.change(
        fn=lambda q, h, s: respond(q, h, s) if q else ("", h, s, gr.update(), gr.update()),
        inputs=[history_dropdown, chatbot, search_history_state],
        outputs=[msg, chatbot, search_history_state, history_dropdown, history_display]
    )

    # ── Wire quick question buttons ───────────────────────────
    for btn, question in zip(quick_btns, QUICK_QUESTIONS):
        btn.click(
            fn=click_quick_question,
            inputs=[gr.Textbox(value=question, visible=False),
                    chatbot, search_history_state],
            outputs=[msg, chatbot, search_history_state, history_dropdown, history_display]
        )

    # ── Wire drug chip buttons ────────────────────────────────
    all_chips = list(zip(drug_chips, FEATURED_DRUGS[:5])) + \
                list(zip(drug_chips2, FEATURED_DRUGS[5:]))
    for chip, drug_name in all_chips:
        chip.click(
            fn=drug_summary,
            inputs=[gr.Textbox(value=drug_name, visible=False),
                    chatbot, search_history_state],
            outputs=[chatbot, search_history_state, history_dropdown, history_display]
        )

    # ── Wire drug search & lookup ─────────────────────────────
    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])
    drug_lookup_btn.click(
        fn=drug_summary,
        inputs=[drug_dropdown, chatbot, search_history_state],
        outputs=[chatbot, search_history_state, history_dropdown, history_display]
    )

    # ── Wire export ───────────────────────────────────────────
    def do_export(chat_history):
        f = export_chat(chat_history)
        if f:
            return gr.update(value=f, visible=True)
        return gr.update(visible=False)

    export_btn.click(do_export, [chatbot], [export_file])

demo.launch()
