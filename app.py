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
            "SELECT generic_name FROM inventory ORDER BY generic_name", conn
        )
    return df["generic_name"].tolist()

DRUG_NAMES = get_all_drugs()

# ── Fuzzy drug name matcher ───────────────────────────────────
from difflib import SequenceMatcher

def fuzzy_match_drug(text, threshold=75):
    text = text.lower().strip()
    best_score = 0
    best_match = None
    for drug in DRUG_NAMES:
        if text in drug.lower() or drug.lower() in text:
            return drug
        score = SequenceMatcher(None, text, drug.lower()).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = drug
    if best_score >= threshold:
        return best_match
    return None

def fuzzy_correct_question(question):
    words = question.replace("?", "").split()
    corrections = []
    corrected_words = list(words)
    for i, word in enumerate(words):
        if len(word) < 4:
            continue
        match = fuzzy_match_drug(word, threshold=78)
        if match and match.lower() != word.lower():
            corrected_words[i] = match
            corrections.append(f"'{word}' -> '{match}'")
    corrected = " ".join(corrected_words)
    note = f"*(Auto-corrected: {', '.join(corrections)})*" if corrections else ""
    return corrected, note

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
    elif any(w in q for w in ["flu", "fever", "pain", "cough", "malaria",
                                "diabetes", "hypertension", "infection",
                                "headache", "diarrhea", "stomach", "recommend",
                                "anything for", "what for", "treat"]):
        return "symptom"
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

def query_symptom(question):
    keywords = extract_drug_name(question)
    parts = [w for w in keywords.split() if len(w) > 3]
    if not parts:
        return []
    cypher = """
        MATCH (d:Drug)-[:IN_CATEGORY]->(c:Category)
        WHERE any(word IN $words WHERE toLower(d.indications) CONTAINS word)
        RETURN d.generic_name AS name,
               d.indications  AS indications,
               d.adult_dose   AS adult_dose,
               d.prescription AS prescription,
               c.name         AS category
        LIMIT 5
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, words=parts)]
        
def query_alternative(question):
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
        return intent, "inventory database",               query_stock_price(question)
    elif intent == "expiry":
        return intent, "batch records",                    query_expiry(question)
    elif intent == "interaction":
        return intent, "drug interaction knowledge graph",  query_neo4j_interaction(question)
    elif intent == "supplier":
        return intent, "supplier knowledge graph",         query_neo4j_supplier(question)
    elif intent == "sales":
        return intent, "transaction records",              query_sales(question)
    elif intent == "alternative":
        return intent, "inventory database",               query_alternative(question)
    elif intent == "symptom":
        return intent, "drug knowledge graph",             query_symptom(question)
    else:
        return intent, "drug knowledge graph",             query_neo4j_drug_info(question)

# ── GPT-4o-mini answer generator ──────────────────────────────
GREETING_RESPONSE = """👋 Hello! I'm the Netrisyl Pharmacy Assistant. I can help you with:

- 📦 **Stock & Prices** — *"Do we have amoxicillin in stock?"*
- ⚠️ **Drug Interactions** — *"What interacts with metformin?"*
- 📅 **Expiry Alerts** — *"Which batches are expiring soon?"*
- 🚚 **Suppliers** — *"Who supplies ciprofloxacin?"*
- 💊 **Drug Information** — *"What is ibuprofen used for?"*
- 🔄 **Alternatives** — *"What is an alternative to amoxicillin?"*
- 💰 **Sales Summary** — *"What are the top selling drugs?"*

Click a drug chip for a quick summary, or use the Drug Lookup on the left. How can I help?"""

def generate_answer(question, intent, source, data):
    if intent == "greeting":
        return GREETING_RESPONSE
    if not data:
        return ("I could not find any information matching your question. "
                "Please check the drug name and try again.")
    system_prompt = """You are a pharmacy assistant at Sunrise Pharmacy in Harare, Zimbabwe.
Answer ONLY using the structured data provided below. Never add facts from general knowledge.
Rules:
- Answer in 3-5 sentences maximum
- Stick strictly to the question asked — do not volunteer unrelated information
- Always mention the data source at the end
- For drug interactions, always state the severity level (Minor/Moderate/Major)
- For stock questions, state exact quantity and mention if at or below reorder level
- For expiry questions, flag anything expiring within 30 days as URGENT
- For alternatives, list available options with stock levels
- The transactions data covers the LAST 30 DAYS — never describe it as daily sales
- If the data does not contain enough information to answer, say exactly:
  "I don't have enough data to answer that. Please consult a pharmacist or reference guide."
- Never guess, infer, or hallucinate any drug names, doses, interactions, or facts
- Use simple professional language suitable for pharmacy staff"""
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
    lines = ["Netrisyl Pharmacy Assistant — Chat Export",
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
    matches = [d for d in DRUG_NAMES if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches)

# ── Core respond function ─────────────────────────────────────
def respond(message, chat_history, search_history):
    if not message or message.strip() == "":
        return "", chat_history, search_history, gr.update(), gr.update()
    try:
        # Apply fuzzy spelling correction
        corrected_message, correction_note = fuzzy_correct_question(message)
        intent, source, data = run_query(corrected_message)
        answer = generate_answer(corrected_message, intent, source, data)
        if intent == "greeting":
            full_answer = answer
        else:
            full_answer = answer
            if correction_note:
                full_answer = f"{correction_note}\n\n{answer}"
            full_answer = f"{full_answer}\n\n*Source: {source} | Intent: {intent}*"
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

# ── Drug summary (chip / lookup button) ──────────────────────
def drug_summary(drug_name, chat_history, search_history):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update()
    try:
        data  = query_drug_summary(drug_name)
        answer = generate_drug_summary_answer(drug_name, data)
    except Exception as e:
        answer = f"Error: {str(e)}"
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

def reask_from_history(selected_question, chat_history, search_history):
    if not selected_question:
        return "", chat_history, search_history, gr.update(), gr.update()
    return respond(selected_question, chat_history, search_history)

# ── Featured drugs ────────────────────────────────────────────
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
    "What is ibuprofen used for?"
]

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Netrisyl Pharmacy Assistant") as demo:

    # Header
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #0d1b2a, #1a3a5c);
                padding: 16px 24px; border-radius: 10px; margin-bottom: 16px;
                display: flex; align-items: center; justify-content: space-between;">
        <img src="https://huggingface.co/spaces/Sylvester1922/Netrisyl_pharmacy_assistant/resolve/main/NI_Logo.png"
             style="height: 70px; object-fit: contain;" alt="Netrisyl Insights"/>
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
- 💰 Transaction History
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
                    placeholder="Ask about stock, prices, interactions, expiry, suppliers...",
                    label="", scale=5
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
        Netrisyl Insights · Harare, Zimbabwe · Powered by AI
    </div>
    """)

    # ── Shared state ──────────────────────────────────────────
    search_history_state = gr.State([])

    # ── Chat input wiring ─────────────────────────────────────
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
            [gr.Textbox(value=question, visible=False), chatbot, search_history_state],
            [msg, chatbot, search_history_state, history_dropdown, history_display]
        )

    # ── Drug chip buttons ─────────────────────────────────────
    all_chips = list(zip(drug_chips, FEATURED_DRUGS[:5])) + \
                list(zip(drug_chips2, FEATURED_DRUGS[5:]))
    for chip, drug_name in all_chips:
        chip.click(
            drug_summary,
            [gr.Textbox(value=drug_name, visible=False), chatbot, search_history_state],
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