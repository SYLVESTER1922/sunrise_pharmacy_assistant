"""
Netrisyl Pharmacy Assistant — Single-File Demo App
==================================================

Purpose
-------
A Gradio-based pharmacy assistant for Sunrise Pharmacy that combines:
- PostgreSQL/Supabase for operational pharmacy data
- Neo4j for clinical, interaction, and supplier graph data
- OpenAI for strictly grounded clinical summaries

Demo Architecture
-----------------
This file is intentionally kept as one deployable `app.py` for Hugging Face Spaces.
The code is organized with clear sections so it remains easy to review, present,
and maintain during a live demo.

Design Principle
----------------
Operational answers are deterministic and database-driven.
Clinical answers are generated only from retrieved knowledge-graph data and include
pharmacist verification reminders.
"""

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

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION & EXTERNAL CLIENTS
# ═══════════════════════════════════════════════════════════════
# Environment variables are used for deployment safety on Hugging Face Spaces.
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

# ═══════════════════════════════════════════════════════════════
# 2. POSTGRESQL / SUPABASE CONNECTION LAYER
# ═══════════════════════════════════════════════════════════════
# Operational pharmacy data: inventory, batches, transactions, and sales.
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

# ═══════════════════════════════════════════════════════════════
# 3. DRUG MASTER DATA CACHE
# ═══════════════════════════════════════════════════════════════
# Used for fast fuzzy matching and dropdown search support.
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

# ═══════════════════════════════════════════════════════════════
# 4. FUZZY MATCHING & USER INPUT CLEANUP
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# 5. INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
# Rule-based routing keeps critical pharmacy workflows deterministic.
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

    # ── System intents ────────────────────────────────────────
    if any(q_clean == g or q_clean.startswith(g) for g in GREETINGS):
        return "greeting"
    if any(g in q_clean for g in THANKS):
        return "thanks"
    if any(g in q_clean for g in FAREWELLS):
        return "farewell"

    # ── SALES explicit — before followup and stock_price ─────
    if any(w in q for w in [
        "top selling", "least selling", "best selling", "worst selling",
        "bottom selling", "slow moving", "top drugs", "least drugs",
        "last day", "yesterday", "latest day", "most recent day",
        "last transaction", "recent sales",
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
        "last week", "this week", "past week",
        "best performing category", "top category",
        "by units sold", "by revenue", "by transactions",
        "bottom 5", "bottom 3", "bottom 10", "top 5", "top 3", "top 10",
        "most sold", "least sold", "best performer", "worst performer"
    ]):
        return "sales"

    # ── SUPPLIER explicit — before stock_price ────────────────
    if any(w in q for w in [
        "supplier", "order from", "who supply", "distributor",
        "vendor", "supplies", "supply", "who provides",
        "where do we order", "procurement", "purchase from",
        "buy from", "shortest lead", "fastest lead",
        "how many suppliers", "suppliers in", "lead time",
        "which supplier"
    ]):
        return "supplier"

    # ── BRIEFING — daily morning summary ─────────────────────
    if any(w in q for w in [
        "good morning", "morning briefing", "daily briefing",
        "daily summary", "morning summary", "start of day",
        "what do i need to know", "briefing"
    ]):
        return "briefing"

    # ── REORDER LIST ───────────────────────────────────────────
    if any(w in q for w in [
        "reorder list", "procurement list", "what to order",
        "what do we need to order", "order list", "action list",
        "what needs reordering", "reorder report"
    ]):
        return "reorder"

    # ── FORECAST ──────────────────────────────────────────────
    if any(w in q for w in [
        "forecast", "projection", "predict", "how long will stock last",
        "days of stock", "revenue forecast", "stock forecast",
        "how long until", "when will we run out"
    ]):
        return "forecast"

    # ── RECONCILIATION ────────────────────────────────────────
    if any(w in q for w in [
        "reconciliation", "reconcile", "discrepancy", "stock discrepancy",
        "missing stock", "stock variance", "stock loss", "shrinkage",
        "stock check", "audit", "check discrepancies", "stock audit",
        "investigate stock", "stock investigation", "losses"
    ]):
        return "reconciliation"

    # ── STATS explicit — before stock_price ──────────────────
    if any(w in q for w in [
        "how many drugs", "how many types", "how many medicines",
        "how many categories", "how many drug", "drug categories",
        "total drugs", "drug types", "categories we have",
        "how many do we have", "inventory summary", "stock summary",
        "inventory overview", "stock overview",
        "how many drug categories", "how many do we stock",
        "categories do we stock", "categories do we have"
    ]):
        return "stats"

    # ── CHEAPEST/EXPENSIVE — before drug_info ───────────────
    if any(w in q for w in [
        "cheapest", "most expensive", "lowest price", "highest price",
        "least expensive", "most costly"
    ]):
        return "stock_price"

    # ── ALTERNATIVES explicit — before drug_info ─────────────
    if any(w in q for w in [
        "alternative", "substitute", "instead of", "replace",
        "similar to", "other option", "other drug", "swap",
        "equivalent", "whats another", "what else can",
        "what can replace", "what can i use instead"
    ]):
        return "alternative"

    # ── INTERACTION explicit — before followup ────────────────
    if any(w in q for w in [
        "interact", "interaction", "safe with", "combine", "mix",
        "take with", "used with", "combined with",
        "contraindic", "avoid with"
    ]):
        return "interaction"

    # ── DRUG INFO explicit — before followup ──────────────────
    if any(w in q for w in [
        "dose", "dosage", "used for", "indication", "side effect",
        "contraindication", "what is", "tell me about", "drug class",
        "prescribed for", "treats", "what does", "what can"
    ]):
        return "drug_info"

    # ── Follow-up detection ───────────────────────────────────
    followup_refs = [
        "it", "these", "those", "them", "they",
        "the same", "above", "mentioned", "you said", "tell me more",
        "more about", "elaborate", "go on", "continue", "what else",
        "expand", "in detail", "and what about", "how about",
        "what about", "also", "another"
    ]
    if (conversation_history and len(q.split()) <= 8 and
            any(ref in q_clean for ref in followup_refs)):
        return "followup"

    # ── CATEGORY BROWSE ───────────────────────────────────────
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

    # ── LOW STOCK ─────────────────────────────────────────────
    if any(w in q for w in [
        "low stock", "running low", "almost out", "reorder",
        "need to order", "below reorder", "critical stock",
        "running low on stock", "low on stock", "need reorder",
        "stock alert", "drugs running", "which drugs are",
        "what drugs are", "what products are"
    ]):
        return "low_stock"

    # ── STOCK / PRICE ─────────────────────────────────────────
    if any(w in q for w in [
        "stock", "have", "available", "availability", "quantity",
        "price", "cost", "how much", "in stock",
        "do we have", "do you have", "shelf",
        "cheapest", "most expensive", "lowest price", "highest price"
    ]):
        return "stock_price"

    # ── EXPIRY ────────────────────────────────────────────────
    if any(w in q for w in [
        "expir", "expire", "expiry", "batch", "shelf life",
        "best before", "use by", "when does", "days until",
        "most urgent", "urgent batch", "needs action", "critical batch",
        "when will", "how long until", "how many days"
    ]):
        return "expiry"

    # ── SALES ─────────────────────────────────────────────────
    if any(w in q for w in [
        "sold", "sales", "revenue", "dispensed", "transaction",
        "highest revenue", "performance", "turnover",
        "customer type", "customer breakdown", "by customer",
        "breakdown", "split by", "prescription sales",
        "walk-in", "insurance sales"
    ]):
        return "sales"

    # ── Default: drug info ────────────────────────────────────
    return "drug_info"


# ═══════════════════════════════════════════════════════════════
# 6. KEYWORD EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════
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
# 7. OPERATIONAL DATA FORMATTERS
# These functions answer directly from structured data. No GPT is used here,
# which prevents hallucination for stock, price, sales, expiry, and reorder data.
# ═══════════════════════════════════════════════════════════════

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
        sql_c += " ORDER BY selling_price_usd ASC LIMIT 5"
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
            df_e = pd.read_sql_query("SELECT generic_name, brand_name, selling_price_usd, quantity_in_stock, shelf_location, category FROM inventory ORDER BY selling_price_usd DESC LIMIT 5", conn)
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
            ORDER BY s.lead_time ASC LIMIT 3
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
# 8. CLINICAL MODE — RETRIEVAL-GROUNDED GPT
# GPT is only used after data is retrieved from Neo4j. The prompt forbids
# unsupported clinical claims and requires pharmacist verification.
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
# 9. FEATURE: DAILY BRIEFING
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

    lines = [f"# 🌅 Good Morning! Daily Briefing — {today}\n"]

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
# 10. FEATURE: PROCUREMENT / REORDER ACTION LIST
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
# 11. FEATURE: REVENUE & STOCK FORECAST
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
            LIMIT 10
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
# 12. FEATURE: MULTI-DRUG INTERACTION CHECK
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
# 13. FEATURE: CONTROLLED SUBSTANCE CHECK
# ═══════════════════════════════════════════════════════════════
# This is handled in format_drug_summary and query_neo4j_drug_info
# The drug_knowledge table has a 'controlled' field — already displayed


# ═══════════════════════════════════════════════════════════════
# 14. FEATURE: SALES VS INVENTORY RECONCILIATION
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
# 15. MAIN ROUTER — OPERATIONAL VS CLINICAL DISPATCH
# Central routing point for the assistant. This keeps UI callbacks simple.
# ═══════════════════════════════════════════════════════════════

GREETING_RESPONSE = """👋 Good morning! I am the **Netrisyl Pharmacy Assistant** for Sunrise Pharmacy.

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

# ═══════════════════════════════════════════════════════════════
# 16. CHAT EXPORT UTILITY
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# 17. OPTIONAL TEXT-TO-SPEECH SUPPORT
# ═══════════════════════════════════════════════════════════════
import tempfile

# ═══════════════════════════════════════════════════════════════
# 18. UI HELPER: DRUG SEARCH FILTER
# ═══════════════════════════════════════════════════════════════
def filter_drugs(search_text):
    if not search_text or len(search_text) < 2:
        return gr.update(choices=DRUG_NAMES[:20])
    matches = [d for d in DRUG_NAMES if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches if matches else DRUG_NAMES[:20])

# ═══════════════════════════════════════════════════════════════
# 19. GRADIO CALLBACKS: CHAT RESPONSE HANDLERS
# ═══════════════════════════════════════════════════════════════
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
        print(f"Drug summary error: {type(e).__name__}: {e}")
        full_answer = "I couldn’t retrieve that drug summary safely. Please try selecting the drug again."
    label = f"Quick summary: {drug_name}"
    chat_history = list(chat_history or [])
    # Older Gradio Chatbot expects a list of (user_message, assistant_message) pairs.
    chat_history.append((label, full_answer))
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

# ═══════════════════════════════════════════════════════════════
# FINAL RELEASE OVERRIDES — conversational robustness layer
# These definitions intentionally override selected earlier helpers while
# preserving the single-file Hugging Face demo architecture.
# ═══════════════════════════════════════════════════════════════

def to_text(value):
    """Normalize any chat/state value into safe plain text before .lower(), regex, etc."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return to_text(value.get("content", ""))
    if isinstance(value, (list, tuple, set)):
        return " ".join(to_text(v) for v in value)
    return str(value)


def qnorm(value):
    return re.sub(r"\s+", " ", to_text(value).lower().strip())


def natural_welcome(user_text=""):
    q = qnorm(user_text)
    if "good morning" in q:
        opener = "Good morning"
    elif "good afternoon" in q:
        opener = "Good afternoon"
    elif "good evening" in q:
        opener = "Good evening"
    else:
        opener = "Hello"
    return f"""{opener} — I’m the **Sunrise Pharmacy Assistant**.

I can help with **stock checks, prices, low-stock alerts, expiry batches, sales, suppliers, forecasts, and medicine information**.

Try asking: *"Do we have amoxicillin?"*, *"Which batches expire soon?"*, or *"What interacts with metformin?"*"""

THANKS_RESPONSE = "You're welcome! Feel free to ask anytime. 😊"
FAREWELL_RESPONSE = "Goodbye! Come back anytime you need help. 👋"
OUT_OF_SCOPE_RESPONSE = (
    "I’m focused on Sunrise Pharmacy operations, so I’m best at helping with "
    "stock, prices, expiry batches, sales, suppliers, forecasts, and medicine information.\n\n"
    "Try asking: *\"Which drugs are running low?\"* or *\"What interacts with metformin?\"*"
)


def is_out_of_scope(q):
    q = qnorm(q)
    if q in {"???", "??", "asdf", "asdasdasd", "asdfghjkl"}:
        return True
    return any(p in q for p in [
        "joke", "weather", "world cup", "football", "soccer", "president",
        "dance", "sing", "movie", "song", "recipe", "restaurant", "bitcoin",
        "stock market", "lottery", "capital of"
    ])


def resolve_drug_name(text):
    """Prefer exact generic/brand matches before partial/fuzzy matches."""
    raw = to_text(text)
    clean = re.sub(r"[^a-zA-Z0-9/\-\s]", " ", raw).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    if not clean or DRUGS_DF.empty:
        return None
    df = DRUGS_DF.copy()
    df["_g"] = df["generic_name"].astype(str).str.lower()
    df["_b"] = df["brand_name"].astype(str).str.lower()
    stop = set(STOPWORDS) | {
        "ask", "about", "what", "check", "inventory", "stock", "available", "availability",
        "left", "need", "for", "does", "when", "expire", "expires", "expiry", "batch", "batches",
        "have", "has", "many", "next", "date", "tell", "more", "this", "that", "and", "how"
    }
    tokens = [t for t in clean.split() if t not in stop]
    candidates = [clean] + tokens
    for cand in candidates:
        exact = df[(df["_g"] == cand) | (df["_b"] == cand)]
        if not exact.empty:
            return exact.iloc[0]["generic_name"]
    for cand in candidates:
        if not cand:
            continue
        starts = df[df["_g"].str.startswith(cand) | df["_b"].str.startswith(cand)]
        if not starts.empty:
            starts = starts.assign(_len=starts["generic_name"].astype(str).str.len()).sort_values("_len")
            return starts.iloc[0]["generic_name"]
    for cand in candidates:
        if not cand:
            continue
        contains = df[df["_g"].str.contains(re.escape(cand), na=False) | df["_b"].str.contains(re.escape(cand), na=False)]
        if not contains.empty:
            contains = contains.assign(_len=contains["generic_name"].astype(str).str.len()).sort_values("_len")
            return contains.iloc[0]["generic_name"]
    best_score, best_name = 0, None
    for cand in candidates:
        if len(cand) < 4:
            continue
        for _, row in df.iterrows():
            for col in ["_g", "_b"]:
                score = SequenceMatcher(None, cand, row[col]).ratio() * 100
                if score > best_score:
                    best_score = score
                    best_name = row["generic_name"]
    return best_name if best_score >= 78 else None


def fuzzy_match_drug(text, threshold=78):
    return resolve_drug_name(text)


def fuzzy_correct_question(question):
    raw = to_text(question)
    words = re.sub(r"['\u2019?!,.]", "", raw).split()
    corrections = []
    corrected_words = list(words)
    skip_words = set(STOPWORDS) | {"ask", "what", "about", "tell", "more", "this", "that", "sales", "batches", "batch", "expiry", "expire", "supplier", "vendor", "stock", "price", "cost"}
    for i, word in enumerate(words):
        w = word.lower().strip()
        if len(w) < 4 or w in skip_words:
            continue
        exact = any(w == str(row["generic_name"]).lower() or w == str(row["brand_name"]).lower() for _, row in DRUGS_DF.iterrows())
        if exact:
            continue
        match = resolve_drug_name(w)
        if match and match.lower() != w and SequenceMatcher(None, w, match.lower()).ratio() * 100 >= 78:
            corrected_words[i] = match
            corrections.append(f"'{word}' → '{match}'")
    corrected = " ".join(corrected_words)
    note = f"*(Auto-corrected: {', '.join(corrections)})*" if corrections else ""
    return corrected, note


def extract_keywords(question):
    clean = re.sub(r"['\u2019?!,.]", "", qnorm(question))
    return [w for w in clean.split() if w not in STOPWORDS and len(w) > 2]


def get_search_term(question):
    drug = resolve_drug_name(question)
    if drug:
        return drug
    keywords = extract_keywords(question)
    return keywords[0] if keywords else qnorm(question)


def last_user_question(conversation_history):
    for turn in reversed(conversation_history or []):
        if isinstance(turn, dict) and turn.get("role") == "user":
            return to_text(turn.get("content"))
    return ""


def last_assistant_answer(conversation_history):
    for turn in reversed(conversation_history or []):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            return to_text(turn.get("content"))
    return ""


def infer_last_context(conversation_history):
    last_a = last_assistant_answer(conversation_history)
    last_u = last_user_question(conversation_history)
    text = qnorm(last_a + " " + last_u)
    ctx = {"intent": None, "drug": resolve_drug_name(last_u) or resolve_drug_name(last_a), "day": None}
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in text:
            ctx["day"] = day
            break
    if "batch records" in text or "expiry" in text or "expires" in text:
        ctx["intent"] = "expiry"
    elif "transaction records" in text or "selling drugs" in text or "sales" in text or ctx["day"]:
        ctx["intent"] = "sales"
    elif "supplier knowledge graph" in text or "supplier" in text or "lead time" in text:
        ctx["intent"] = "supplier"
    elif "drug interaction knowledge graph" in text or "interacts" in text or "interaction" in text:
        ctx["intent"] = "interaction"
    elif "inventory database" in text or "stock" in text or "shelf location" in text:
        ctx["intent"] = "stock_price"
    elif "drug knowledge graph" in text:
        ctx["intent"] = "drug_info"
    return ctx


def classify_intent(question, conversation_history=None):
    q = qnorm(question)
    q_clean = re.sub(r"[^a-z0-9\s]", "", q).strip()
    if not q_clean:
        return "unknown"
    if is_out_of_scope(q):
        return "out_of_scope"
    greeting_phrases = ["hi", "hello", "hey", "hey there", "hello there", "yo", "good morning", "good afternoon", "good evening", "morning", "afternoon", "evening", "can you help", "help", "what can you do", "who are you", "start"]
    if any(q_clean == g or q_clean.startswith(g + " ") for g in greeting_phrases):
        return "greeting"
    if any(x in q for x in ["thanks", "thank you", "thank", "appreciate", "much appreciated"]):
        return "thanks"
    if any(x in q for x in ["bye", "goodbye", "see you", "see ya", "later", "take care", "catch you"]):
        return "farewell"
    if q.startswith("quick summary:"):
        return "stock_price"
    followup_triggers = ["what about", "how about", "and ", "tell me more", "explain further", "more about", "this", "that", "these", "those", "of these", "same list", "only top"]
    if conversation_history and (len(q.split()) <= 7 or any(t in q for t in followup_triggers)):
        if any(t in q for t in followup_triggers):
            return "followup"
    if any(x in q for x in ["daily briefing", "morning briefing", "daily summary", "start of day", "briefing"]):
        return "briefing"
    if any(x in q for x in ["reconciliation", "reconcile", "discrepancy", "variance", "shrinkage", "audit"]):
        return "reconciliation"
    if any(x in q for x in ["forecast", "projection", "predict", "outlook", "how long will stock last", "days of stock", "run out"]):
        return "forecast"
    if any(x in q for x in ["supplier", "vendor", "who supplies", "where do we buy", "buy", "distributor", "lead time", "delivers fastest", "fastest vendor", "suppliers in"]):
        return "supplier"
    if any(x in q for x in ["alternative", "substitute", "instead of", "replace", "similar to", "equivalent"]):
        return "alternative"
    if any(x in q for x in ["interact", "interaction", "safe with", "taken together", "take together", "combine", "avoid with", "metformin and prednisolone"]):
        return "interaction"
    if any(x in q for x in ["used for", "used to treat", "what is", "side effect", "side effects", "dose", "dosage", "indication", "contraindication", "drug class", "what is ibuprofen for"]):
        return "drug_info"
    if any(x in q for x in ["expire", "expires", "expiry", "expiring", "batch", "batches", "best before", "nearing expiry"]):
        return "expiry"
    if any(x in q for x in ["best sellers", "worst sellers", "top selling", "least selling", "bottom", "top ", "by units", "quantity sold", "move fastest", "revenue", "sales", "sold", "yesterday", "last week", "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]):
        return "sales"
    if any(x in q for x in ["low stock", "almost finished", "stockout", "below minimum", "below reorder", "running low", "need restock", "need reordering", "replenishment", "reorder soon"]):
        return "low_stock"
    if any(x in q for x in ["reorder list", "procurement list", "what should we reorder", "what to order", "suggested order"]):
        return "reorder"
    if any(x in q for x in ["cheapest", "lowest priced", "lowest price", "most expensive", "cost the most", "highest price"]):
        return "stock_price"
    if any(x in q for x in ["inventory summary", "stock summary", "how many drugs", "how many categories", "categories"]):
        return "stats"
    if resolve_drug_name(q) or any(x in q for x in ["stock", "available", "availability", "do you have", "do we have", "how much", "left", "inventory"]):
        return "stock_price"
    return "unknown"


def format_stock_price(question):
    q = qnorm(question)
    limit_match = re.search(r"\b(\d+)\b", q)
    limit = max(1, min(int(limit_match.group(1)), 50)) if limit_match else 5
    categories = {"antibiotic": "Antibiotics", "analgesic": "Analgesics", "antihypertensive": "Antihypertensives", "antidiabetic": "Antidiabetics", "antimalarial": "Antimalarials", "antifungal": "Antifungals", "antiretroviral": "Antiretrovirals", "respiratory": "Respiratory", "vitamin": "Vitamins/Supplements", "supplement": "Vitamins/Supplements", "gi": "GI medications"}
    cat_match = next((cat for kw, cat in categories.items() if kw in q), None)
    if any(x in q for x in ["cheapest", "lowest priced", "lowest price", "least expensive"]):
        sql = "SELECT generic_name, brand_name, selling_price_usd, quantity_in_stock, shelf_location FROM inventory"
        params = []
        if cat_match:
            sql += " WHERE category = %s"
            params.append(cat_match)
        sql += " ORDER BY selling_price_usd ASC LIMIT %s"
        params.append(limit)
        conn = get_conn()
        try:
            df = pd.read_sql_query(sql, conn, params=tuple(params))
        finally:
            release_conn(conn)
        label = f"cheapest {cat_match}" if cat_match else "cheapest drugs"
        header = f"**Top {len(df)} {label}:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
        return header + "\n".join(f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | {r['quantity_in_stock']} | {r['shelf_location']} |" for _, r in df.iterrows())
    if any(x in q for x in ["most expensive", "cost the most", "highest price", "most costly"]):
        conn = get_conn()
        try:
            df = pd.read_sql_query("SELECT generic_name, brand_name, selling_price_usd, quantity_in_stock, shelf_location FROM inventory ORDER BY selling_price_usd DESC LIMIT %s", conn, params=(limit,))
        finally:
            release_conn(conn)
        header = f"**Top {len(df)} most expensive drugs:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
        return header + "\n".join(f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | {r['quantity_in_stock']} | {r['shelf_location']} |" for _, r in df.iterrows())
    drug = resolve_drug_name(question)
    if not drug:
        return "I couldn’t identify the medicine. Please mention the drug name directly, for example: *Check stock for amoxicillin*."
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT generic_name, brand_name, formulation, strength,
                   quantity_in_stock, reorder_level, selling_price_usd,
                   cost_price_usd, shelf_location, category
            FROM inventory
            WHERE LOWER(generic_name) = LOWER(%s)
               OR LOWER(brand_name) = LOWER(%s)
               OR LOWER(generic_name) LIKE LOWER(%s)
               OR LOWER(brand_name) LIKE LOWER(%s)
            ORDER BY CASE WHEN LOWER(generic_name) = LOWER(%s) THEN 0 WHEN LOWER(brand_name) = LOWER(%s) THEN 1 ELSE 2 END
            LIMIT 1
        """, conn, params=(drug, drug, f"%{drug}%", f"%{drug}%", drug, drug))
    finally:
        release_conn(conn)
    if df.empty:
        return f"I couldn’t find **{drug}** in the inventory database."
    r = df.iloc[0]
    stock_flag = "⚠️ LOW STOCK" if r['quantity_in_stock'] <= r['reorder_level'] else "✅ In Stock"
    return f"""**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}

| Field | Value |
|---|---|
| Stock | {r['quantity_in_stock']} units — {stock_flag} |
| Reorder Level | {r['reorder_level']} units |
| Selling Price | ${r['selling_price_usd']} |
| Cost Price | ${r['cost_price_usd']} |
| Shelf Location | {r['shelf_location']} |
| Category | {r['category']} |"""


def format_low_stock(question=""):
    nums = re.findall(r"\b(\d+)\b", qnorm(question))
    limit = max(1, min(int(nums[0]), 50)) if nums else 50
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT generic_name, brand_name, quantity_in_stock,
                   reorder_level, category,
                   ROUND((quantity_in_stock::numeric / NULLIF(reorder_level,0)) * 100, 0) AS stock_pct
            FROM inventory WHERE quantity_in_stock <= reorder_level
            ORDER BY stock_pct ASC LIMIT %s
        """, conn, params=(limit,))
    finally:
        release_conn(conn)
    if df.empty:
        return "✅ All products are currently above their reorder levels."
    header = f"⚠️ **{len(df)} drug(s) at or below reorder level:**\n\n| Drug | Brand | Stock | Reorder Level | % of Reorder | Category |\n|---|---|---|---|---|---|\n"
    rows = [f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['stock_pct']:.0f}% | {r['category']} |" for _, r in df.iterrows()]
    return header + "\n".join(rows)


def format_expiry(question):
    q = qnorm(question)
    nums = re.findall(r"\b(\d+)\b", q)
    limit = max(1, min(int(nums[0]), 50)) if nums else 10
    drug = resolve_drug_name(question)
    batch_count = any(x in q for x in ["how many batches", "batch count", "batches does", "batches of"])
    next_only = any(x in q for x in ["when does", "next expiry", "expiry date for"]) and drug
    global_first = any(x in q for x in ["expires first", "expire first", "first to expire", "earliest expiry", "which item expires first"])
    if drug:
        conn = get_conn()
        try:
            df = pd.read_sql_query("""
                SELECT b.batch_number, b.expiry_date, b.quantity_remaining,
                       (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days
                FROM batches b JOIN inventory i ON b.product_id = i.product_id
                WHERE LOWER(i.generic_name) = LOWER(%s)
                   OR LOWER(i.generic_name) LIKE LOWER(%s)
                   OR LOWER(i.brand_name) LIKE LOWER(%s)
                ORDER BY b.expiry_date ASC
            """, conn, params=(drug, f"%{drug}%", f"%{drug}%"))
        finally:
            release_conn(conn)
        if df.empty:
            return f"I couldn’t find batch records for **{drug}**."
        if next_only and not batch_count:
            r = df.iloc[0]
            d = int(r["days"])
            flag = "🚨 URGENT" if d <= 30 else ("⚠️ Warning" if d <= 90 else "✅ OK")
            return f"**Next expiry for {drug}**\n\nBatch **{r['batch_number']}** expires on **{str(r['expiry_date'])[:10]}** (**{d} days left**), quantity remaining: **{r['quantity_remaining']}**. {flag}"
        header = f"**{drug} has {len(df)} batch(es) on record.**\n\n| Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|\n"
        rows = []
        for _, r in df.iterrows():
            d = int(r["days"])
            flag = "🚨 URGENT" if d <= 30 else ("⚠️ Warning" if d <= 90 else "✅ OK")
            rows.append(f"| {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |")
        return header + "\n".join(rows)
    if global_first and not nums:
        limit = 1
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
                   b.quantity_remaining,
                   (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days_remaining
            FROM batches b JOIN inventory i ON b.product_id = i.product_id
            WHERE (b.expiry_date::date - CURRENT_DATE) <= 90
            ORDER BY b.expiry_date ASC LIMIT %s
        """, conn, params=(limit,))
    finally:
        release_conn(conn)
    if df.empty:
        return "✅ No batches expiring within the next 90 days."
    if global_first:
        r = df.iloc[0]
        d = int(r["days_remaining"])
        flag = "🚨 URGENT" if d <= 30 else ("⚠️ Warning" if d <= 90 else "✅ OK")
        return f"**Earliest expiry:** {r['generic_name']} ({r['brand_name']}), batch **{r['batch_number']}**, expires on **{str(r['expiry_date'])[:10]}** (**{d} days left**), quantity: **{r['quantity_remaining']}**. {flag}"
    header = f"**Batches expiring within 90 days** — {len(df)} found:\n\n| Drug | Brand | Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        d = int(r["days_remaining"])
        flag = "🚨 URGENT" if d <= 30 else ("⚠️ Warning" if d <= 90 else "📅 Monitor")
        rows.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |")
    return header + "\n".join(rows)


def format_supplier(question):
    q = qnorm(question)
    city = next((c.capitalize() for c in ["harare", "bulawayo", "mutare"] if c in q), None)
    if city:
        results = run_cypher("""
            MATCH (s:Supplier)
            WHERE toLower(s.city) = toLower($city)
            RETURN s.name AS supplier, s.lead_time AS lead_time_days, s.contact AS contact, s.phone AS phone
            ORDER BY s.lead_time ASC, s.name ASC LIMIT 20
        """, {"city": city})
        if not results:
            return f"I couldn’t find suppliers in {city}."
        header = f"**Suppliers in {city}:**\n\n| Supplier | Lead Time | Contact | Phone |\n|---|---|---|---|\n"
        return header + "\n".join(f"| {r['supplier']} | {r['lead_time_days']} days | {r['contact']} | {r.get('phone','')} |" for r in results)
    if any(x in q for x in ["fastest", "shortest", "quickest", "lead time", "delivers fastest", "fastest vendor"]):
        results = run_cypher("""
            MATCH (s:Supplier)
            RETURN s.name AS supplier, s.lead_time AS lead_time_days, s.city AS city, s.contact AS contact
            ORDER BY s.lead_time ASC LIMIT 5
        """)
        if not results:
            return "I couldn’t find supplier lead-time information."
        header = "**Suppliers by lead time:**\n\n| Supplier | Lead Time | City | Contact |\n|---|---|---|---|\n"
        return header + "\n".join(f"| {r['supplier']} | {r['lead_time_days']} days | {r['city']} | {r['contact']} |" for r in results)
    drug = resolve_drug_name(question)
    if not drug:
        return "Please mention the medicine name or city for supplier lookup."
    results = run_cypher("""
        MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
        WHERE toLower(d.generic_name) CONTAINS toLower($search)
        RETURN d.generic_name AS drug, s.name AS supplier, s.contact AS contact, s.phone AS phone,
               s.city AS city, s.lead_time AS lead_time_days, s.payment_terms AS payment_terms
        LIMIT 5
    """, {"search": drug})
    if not results:
        return f"I couldn’t find supplier information for **{drug}**."
    r = results[0]
    return f"""**Supplier for {r['drug']}:**

| Field | Value |
|---|---|
| Supplier | **{r['supplier']}** |
| Contact | {r['contact']} |
| Phone | {r['phone']} |
| City | {r['city']} |
| Lead Time | {r['lead_time_days']} days |
| Payment Terms | {r['payment_terms']} |"""


def format_revenue_forecast():
    conn = get_conn()
    try:
        df_daily = pd.read_sql_query("""
            SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily_revenue
            FROM (SELECT date, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date) t
        """, conn)
    finally:
        release_conn(conn)
    avg_daily_rev = float(df_daily.iloc[0]['avg_daily_revenue'])
    return f"""## 📈 Revenue Forecast

**Average Daily Revenue:** ${avg_daily_rev:,.2f}  
**30-Day Revenue Forecast:** ${avg_daily_rev * 30:,.2f}  
**90-Day Revenue Forecast:** ${avg_daily_rev * 90:,.2f}"""


def format_stock_depletion_forecast():
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, i.quantity_in_stock,
                   COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,2), 0) AS avg_daily,
                   i.selling_price_usd
            FROM inventory i LEFT JOIN transactions t ON i.product_id = t.product_id
            GROUP BY i.product_id, i.generic_name, i.brand_name, i.quantity_in_stock, i.selling_price_usd
            ORDER BY (i.quantity_in_stock * i.selling_price_usd) DESC LIMIT 10
        """, conn)
    finally:
        release_conn(conn)
    lines = ["## 📦 Stock Depletion Forecast\n", "| Drug | Brand | Stock | Avg Daily Sales | Days Remaining |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        if r['avg_daily'] and r['avg_daily'] > 0:
            days = round(r['quantity_in_stock'] / r['avg_daily'])
            flag = "🔴" if days < 30 else ("🟡" if days < 60 else "🟢")
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | {r['avg_daily']}/day | {flag} **{days} days** |")
        else:
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | No sales | ∞ |")
    return "\n".join(lines)


def handle_followup(question, conversation_history):
    q = qnorm(question)
    ctx = infer_last_context(conversation_history)
    drug = resolve_drug_name(question) or ctx.get("drug")
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in q:
            return format_sales(day), "transaction records", "operational"
    nums = re.findall(r"\b(\d+)\b", q)
    n = nums[0] if nums else "3"
    if any(x in q for x in ["of these", "of those", "same list", "only top", "top "]):
        if ctx.get("intent") == "expiry":
            return format_expiry(f"show next {n} expiries"), "batch records", "operational"
        if ctx.get("intent") == "sales":
            return format_sales(f"top {n} selling drugs"), "transaction records", "operational"
        if ctx.get("intent") == "stock_price":
            return format_low_stock(f"top {n}"), "inventory database", "operational"
    if any(x in q for x in ["what about", "how about"]):
        if not drug:
            return "I’m not sure which medicine you mean. Please name the drug directly.", "system", "system"
        if ctx.get("intent") == "expiry":
            return format_expiry(f"when does {drug} expire"), "batch records", "operational"
        if ctx.get("intent") == "supplier":
            return format_supplier(f"who supplies {drug}"), "supplier knowledge graph", "operational"
        if ctx.get("intent") in ["interaction", "drug_info"]:
            data = query_neo4j_drug_info(drug)
            answer = generate_clinical_answer(f"Tell me about {drug}", "drug_info", "drug knowledge graph", data, conversation_history)
            return answer, "drug knowledge graph", "clinical"
        return format_stock_price(f"check stock for {drug}"), "inventory database", "operational"
    if any(x in q for x in ["tell me more", "explain further", "more about", "elaborate"]):
        if ctx.get("intent") == "interaction" and drug:
            data = query_neo4j_interaction(drug)
            answer = generate_clinical_answer(f"Explain the interaction details for {drug}", "interaction", "drug interaction knowledge graph", data, conversation_history)
            return answer, "drug interaction knowledge graph", "clinical"
        if drug:
            data = query_neo4j_drug_info(drug)
            answer = generate_clinical_answer(f"Tell me more about {drug}", "drug_info", "drug knowledge graph", data, conversation_history)
            return answer, "drug knowledge graph", "clinical"
        return "I can explain further if you mention the medicine or topic you mean.", "system", "system"
    return "I’m not sure what you want me to continue from. Please rephrase or name the medicine directly.", "system", "system"


def route_and_respond(question, intent, conversation_history=None):
    question = to_text(question)
    q = qnorm(question)
    conversation_history = conversation_history or []
    if intent == "greeting":
        return natural_welcome(question), "system", "system"
    if intent == "thanks":
        return THANKS_RESPONSE, "system", "system"
    if intent == "farewell":
        return FAREWELL_RESPONSE, "system", "system"
    if intent == "out_of_scope":
        return OUT_OF_SCOPE_RESPONSE, "system", "system"
    if intent == "unknown":
        return "I’m not sure I understood that request. Please ask about stock, expiry dates, sales, suppliers, forecasts, or medicine information.", "system", "system"
    if intent == "followup":
        return handle_followup(question, conversation_history)
    if intent == "stock_price":
        return format_stock_price(question), "inventory database", "operational"
    if intent == "category_browse":
        return format_category_browse(question), "inventory database", "operational"
    if intent == "stats":
        return format_stats(), "inventory database", "operational"
    if intent == "low_stock":
        return format_low_stock(question), "inventory database", "operational"
    if intent == "expiry":
        return format_expiry(question), "batch records", "operational"
    if intent == "sales":
        return format_sales(question), "transaction records", "operational"
    if intent == "supplier":
        return format_supplier(question), "supplier knowledge graph", "operational"
    if intent == "alternative":
        return format_alternative(question), "inventory database", "operational"
    if intent == "briefing":
        return format_daily_briefing(), "inventory + batch + transaction records", "operational"
    if intent == "reorder":
        return format_reorder_list(), "inventory + transaction records", "operational"
    if intent == "forecast":
        if any(x in q for x in ["stock last", "days of stock", "run out", "how long will stock"]):
            return format_stock_depletion_forecast(), "inventory + transaction records", "operational"
        return format_revenue_forecast(), "inventory + transaction records", "operational"
    if intent == "reconciliation":
        return format_reconciliation(question), "inventory + batch + transaction records", "operational"
    if intent == "interaction":
        data = query_neo4j_interaction(question)
        answer = generate_clinical_answer(question, intent, "drug interaction knowledge graph", data, conversation_history)
        return answer, "drug interaction knowledge graph", "clinical"
    if intent == "drug_info":
        data = query_neo4j_drug_info(question)
        answer = generate_clinical_answer(question, intent, "drug knowledge graph", data, conversation_history)
        return answer, "drug knowledge graph", "clinical"
    return "I’m not sure I understood that request. Please rephrase it as a pharmacy question.", "system", "system"


def respond(message, chat_history, search_history):
    if not message or to_text(message).strip() == "":
        return "", chat_history, search_history, gr.update(), gr.update(), ""
    # Support older Hugging Face/Gradio Chatbot format: list of (user, assistant) tuples.
    # Also tolerate newer message dicts if the app is run locally on a newer Gradio.
    conversation_history = []
    for t in (chat_history or []):
        if isinstance(t, dict):
            conversation_history.append({"role": t.get("role", ""), "content": to_text(t.get("content", ""))})
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            conversation_history.append({"role": "user", "content": to_text(t[0])})
            conversation_history.append({"role": "assistant", "content": to_text(t[1])})
    try:
        corrected_message, correction_note = fuzzy_correct_question(message)
        intent = classify_intent(corrected_message, conversation_history)
        answer, source, mode = route_and_respond(corrected_message, intent, conversation_history)
        if mode == "system":
            full_answer = answer
        elif mode == "operational":
            header = f"*📦 Operational data — {source}*\n\n"
            full_answer = f"{correction_note}\n\n{header}{answer}" if correction_note else f"{header}{answer}"
        else:
            header = f"*🧪 Clinical data — {source}*\n\n"
            full_answer = f"{correction_note}\n\n{header}{answer}" if correction_note else f"{header}{answer}"
    except Exception as e:
        print(f"Assistant error: {type(e).__name__}: {e}")
        full_answer = "I’m sorry — I couldn’t process that request safely. Please rephrase it or mention the medicine name directly."
    chat_history = list(chat_history or [])
    # Older Gradio Chatbot expects a list of (user_message, assistant_message) pairs.
    chat_history.append((to_text(message), full_answer))
    search_history = list(search_history or [])
    if message not in search_history:
        search_history.insert(0, message)
    search_history = search_history[:15]
    history_md = "\n".join([f"- {h}" for h in search_history])
    return "", chat_history, search_history, gr.update(choices=search_history, value=None), gr.update(value=history_md), ""


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