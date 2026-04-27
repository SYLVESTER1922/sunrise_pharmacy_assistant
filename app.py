import os
import re
import psycopg2
import pandas as pd
from psycopg2 import pool
import gradio as gr
from neo4j import GraphDatabase
from openai import OpenAI
from difflib import SequenceMatcher
from datetime import datetime, date, timezone, timedelta
import json
import tempfile

# ══════════════════════════════════════════════════════════════════
# CONSTANTS — centralised phrase dictionaries (avoids scattered defs)
# ══════════════════════════════════════════════════════════════════

GREETING_TRIGGERS = {
    "hi", "hey", "hello", "morning", "afternoon", "evening",
    "howzit", "good morning", "good afternoon", "good evening",
    "good night", "how are you", "what's up", "whats up",
    "what can you do", "what do you do", "who are you",
    "what are you", "yo", "sup", "start", "help"
}

THANKS_TRIGGERS = {
    "thank you", "thanks", "thank", "cheers", "appreciated",
    "great", "ok", "okay", "cool", "perfect", "noted",
    "awesome", "brilliant", "nice", "wonderful",
    "excellent", "got it", "understood", "sure"
}

FAREWELL_TRIGGERS = {
    "bye", "goodbye", "good bye", "see you", "see ya",
    "later", "ciao", "take care", "exit", "quit",
    "talk later", "catch you later", "farewell"
}

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
    "anything", "something", "everything", "vendor", "distributor",
    "buy", "purchase", "procure"
}

# ── Credentials ───────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")


# ── Neo4j driver ──────────────────────────────────────────────────
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

driver = get_driver()

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


# ── OpenAI client ─────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)


# ── Supabase PostgreSQL connection pool ───────────────────────────
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(1, 10, SUPABASE_URL)
    return _pool

def get_conn():
    return get_pool().getconn()

def release_conn(conn):
    get_pool().putconn(conn)

# ── SQLAlchemy engine — used by pandas read_sql_query ─────────────
from sqlalchemy import create_engine as _sa_create_engine
_sa_engine = None

def get_engine():
    """Return a reusable SQLAlchemy engine for pandas queries."""
    global _sa_engine
    if _sa_engine is None:
        _sa_engine = _sa_create_engine(SUPABASE_URL)
    return _sa_engine

print("Supabase connection pool ready ✓")


# ── Load drug list at startup ─────────────────────────────────────
def get_all_drugs():
    df = pd.read_sql_query(
        "SELECT generic_name, brand_name, category FROM inventory ORDER BY generic_name",
        get_engine()
    )
    return df

DRUGS_DF    = get_all_drugs()
DRUG_NAMES  = DRUGS_DF["generic_name"].tolist()
BRAND_NAMES = DRUGS_DF["brand_name"].tolist()
# Map brand name → generic name for acronym/brand lookups (e.g. ORS → Oral Rehydration Salts)
BRAND_TO_GENERIC = dict(zip(
    DRUGS_DF["brand_name"].str.lower(),
    DRUGS_DF["generic_name"]
))


# ══════════════════════════════════════════════════════════════════
# FUZZY MATCHING — typo correction
# ══════════════════════════════════════════════════════════════════

def fuzzy_match_drug(text, threshold=78):
    text = re.sub(r"['\u2019\u2018`]", "", text.lower().strip())
    # 1. Exact generic name match
    for drug in DRUG_NAMES:
        if text == drug.lower():
            return drug
    # 2. Exact brand name match → return generic (handles ORS, TLD, Moxil etc)
    if text in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[text]
    # 3. Fuzzy on generic names
    best_score = 0
    best_match = None
    for drug in DRUG_NAMES:
        score = SequenceMatcher(None, text, drug.lower()).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = drug
    # 4. Fuzzy on brand names
    for brand, generic in BRAND_TO_GENERIC.items():
        score = SequenceMatcher(None, text, brand).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = generic
    return best_match if best_score >= threshold else None


def fuzzy_correct_question(question):
    """Correct misspelled drug names in a question. Returns (corrected_q, note)."""
    skip = SKIP_WORDS | {
        "soon", "please", "could", "would", "anything", "something",
        "find", "list", "show", "tell", "give", "have", "does", "there",
        "that", "this", "will", "about", "from"
    }
    words = re.sub(r"['\u2019?!,.]", "", question).split()
    corrections = []
    corrected_words = list(words)
    for i, word in enumerate(words):
        w = word.lower()
        # Strip possessive 's so "metformin's" matches "metformin"
        w_clean = re.sub(r"'s$|s'$", "", w)
        if len(w_clean) < 4 or w_clean in skip:
            continue
        match = fuzzy_match_drug(w_clean, threshold=78)
        if match and match.lower() != w:
            corrected_words[i] = match
            corrections.append(f"'{word}' → '{match}'")
    corrected = " ".join(corrected_words)
    note = f"*(Auto-corrected: {', '.join(corrections)})*" if corrections else ""
    return corrected, note


def extract_keywords(question: str) -> list:
    """Extract likely drug names and key terms from a question."""
    words = re.sub(r"[\'\u2019?!,.]", "", question.lower()).split()
    return [w for w in words if len(w) >= 4 and w not in SKIP_WORDS and not w.isdigit()]


def get_search_term(question: str) -> str:
    keywords = extract_keywords(question)
    return keywords[0] if keywords else question.lower()


# ══════════════════════════════════════════════════════════════════
# GPT TOOL-CALLING — intent + parameter extraction
# ══════════════════════════════════════════════════════════════════

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": (
                "Query drug inventory — stock levels, prices, categories, low stock alerts, profit margins. "
                "Do NOT use for batch or expiry questions — use query_expiry for those. "
                "Use filter=below_reorder for: 'running low', 'need restocking', 'critical stock', 'which drugs are low'. "
                "Use sort_by=margin_desc for: 'highest profit margin', 'most profitable drug', 'best margin'. "
                "Use sort_by=margin_asc for: 'lowest margin', 'least profitable'. "
                "Use filter=cheapest + sort_by=price_asc for: 'cheapest drugs'. "
                "Use filter=most_expensive + sort_by=price_desc for: 'most expensive drugs'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name":   {"type": "string",  "description": "Specific drug name to look up, or null for all"},
                    "category":    {"type": "string",  "description": "Drug category e.g. Antibiotics, Analgesics, or null"},
                    "filter":      {
                        "type": "string",
                        "enum": ["below_reorder", "above_reorder", "all", "cheapest", "most_expensive"],
                        "description": "Stock filter"
                    },
                    "sort_by":     {
                        "type": "string",
                        "enum": ["stock_pct", "price_asc", "price_desc", "name", "margin_desc", "margin_asc"],
                        "description": "Sort order"
                    },
                    "limit":       {"type": "integer", "description": "Number of results, default 10"}
                },
                "required": ["filter"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sales",
            "description": (
                "Query sales transactions — top/bottom sellers, revenue, day analysis, total units sold, total transactions. "
                "Use direction=bottom and sort_by=units for: 'least selling', 'worst sellers', 'lowest units sold'. "
                "Use sort_by=units for 'top by units sold'. "
                "Use period=all_time for: 'how many units did we sell', 'total units sold', 'how many transactions', 'total sales'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction":   {"type": "string",  "enum": ["top", "bottom"], "description": "Top or bottom performers"},
                    "sort_by":     {"type": "string",  "enum": ["revenue", "units", "transactions"], "description": "Metric to rank by"},
                    "period":      {
                        "type": "string",
                        "enum": ["all_time", "last_day", "last_week", "current_month", "day_of_week", "best_day", "customer_type", "total_summary"],
                        "description": (
                            "Time period / query mode. "
                            "Use all_time for general top/bottom drug rankings. "
                            "Use last_day for: yesterday revenue, yesterday sales, how many transactions yesterday. "
                            "Use last_week for: last week sales, last 7 days. "
                            "Use current_month for: this month revenue, how much this month, monthly total. "
                            "Use best_day for: which day of the week has highest sales/revenue, busiest day, best day. "
                            "Use total_summary for: how many units did we sell in total, total transactions, total revenue summary, average daily revenue. "
                            "Use day_of_week + day_name for: sales on Friday, what happened on Saturday. "
                            "Use customer_type for: sales by customer type, walk-in vs prescription."
                        )
                    },
                    "day_name":    {"type": "string",  "description": "Day name e.g. Saturday, only for day_of_week period"},
                    "limit":       {"type": "integer", "description": "Number of results, default 10"}
                },
                "required": ["period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_expiry",
            "description": (
                "Query batch expiry records — upcoming expiries, specific drug batches, batch count, month filter. "
                "Use for: 'how many batches of X', 'when does X expire', 'batches expiring soon', "
                "'what expires first', 'nearest expiry', 'expiring in May', 'which drugs have more than N batches'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name":    {"type": "string",  "description": "Specific drug name, or null for all"},
                    "within_days":  {"type": "integer", "description": "Show batches expiring within N days, default 90"},
                    "limit":        {"type": "integer", "description": "Number of results, default 10"},
                    "top_only":     {"type": "boolean", "description": "If true, return only the single soonest-expiring batch"},
                    "month_name":   {"type": "string",  "description": "Month name e.g. May, June — for named month expiry filter"},
                    "count_only":   {"type": "boolean", "description": "If true, return batch count per drug"},
                    "min_batches":  {"type": "integer", "description": "Minimum batch count filter when count_only=true"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_supplier",
            "description": (
                "Query supplier information — who supplies a drug, lead times, city breakdown, supplier count, payment terms. "
                "Triggered by: supplier, vendor, distributor, buy from, order from, who supplies, source, "
                "'how many suppliers', 'fastest vendor', 'fastest supplier', 'quickest lead time', "
                "'best payment terms', 'payment terms', 'credit terms', 'who gives best terms'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name":   {"type": "string",  "description": "Drug name to find supplier for, or null"},
                    "city":        {"type": "string",  "description": "Filter suppliers by city, or null"},
                    "sort_by":     {"type": "string",  "enum": ["lead_time", "name"], "description": "Sort order"},
                    "limit":       {"type": "integer", "description": "Number of results, default 5"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_stats",
            "description": "Query inventory statistics — category summary, total products, inventory value",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_clinical",
            "description": (
                "Look up clinical drug information — interactions, dosage, side effects, indications. "
                "ONLY use for clearly clinical questions about pharmacology. "
                "Do NOT use for out-of-scope questions like weather, news, personal queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name":   {"type": "string",  "description": "Primary drug name"},
                    "drug_name_2": {"type": "string",  "description": "Second drug for interaction check, or null"},
                    "query_type":  {
                        "type": "string",
                        "enum": ["interaction", "drug_info"],
                        "description": "Type of clinical query"
                    }
                },
                "required": ["drug_name", "query_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_briefing",
            "description": "Generate a FULL daily briefing — low stock alerts + urgent expiry + yesterday revenue combined. ONLY trigger for: 'daily briefing', 'morning briefing', 'daily summary', 'start of day', 'briefing', 'anything I should know', 'stock situation'. Do NOT use for 'yesterday revenue' or 'yesterday sales' alone — use query_sales with period=last_day for those.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_combined_risk",
            "description": "Find drugs that are BOTH low on stock AND expiring soon. Use for: 'low stock and expiring', 'running out and expiring', 'critical drugs', 'what needs urgent attention'.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_reorder",
            "description": "Generate a procurement action list with suggested order quantities. Only use when user explicitly asks for order quantities, suggested amounts, or procurement plan. NOT for simply listing low stock drugs.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_forecast",
            "description": "Revenue and stock depletion forecast based on current sales rate. Triggered by: forecast, projection, predict, how long will stock last.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_reconciliation",
            "description": "Stock reconciliation — find discrepancies between received, sold, and current stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Specific drug to check, or null for all"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_alternatives",
            "description": "Find alternative drugs in the same therapeutic category",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Drug to find alternatives for"}
                },
                "required": ["drug_name"]
            }
        }
    }
]


def _is_phrase_match(q_clean: str, trigger_set: set) -> bool:
    """Check if q_clean exactly matches or starts with any trigger."""
    if q_clean in trigger_set:
        return True
    for t in trigger_set:
        if q_clean.startswith(t + " ") or q_clean == t:
            return True
    return False


def classify_intent_with_tools(question: str, conversation_history=None) -> dict:
    """
    Use GPT function calling to extract intent AND structured parameters.
    GPT decides WHAT to query. Python executes SQL — zero hallucination on figures.
    Returns: {"tool": str, "params": dict, "confidence": float}
    """
    q       = question.lower().strip()
    q_clean = re.sub(r"[?!.,'\u2019 ]+$", "", q).strip()

    # ── Fast-path: greetings / thanks / farewells — no GPT call needed ──
    if _is_phrase_match(q_clean, GREETING_TRIGGERS):
        return {"tool": "greeting", "params": {}, "confidence": 1.0}
    if _is_phrase_match(q_clean, THANKS_TRIGGERS):
        return {"tool": "thanks", "params": {}, "confidence": 1.0}
    if _is_phrase_match(q_clean, FAREWELL_TRIGGERS):
        return {"tool": "farewell", "params": {}, "confidence": 1.0}

    # ── drug_summary shortcut ──
    if q.startswith("quick summary:"):
        drug = question.replace("quick summary:", "").replace("Quick summary:", "").strip()
        return {"tool": "drug_summary", "params": {"drug_name": drug}, "confidence": 1.0}

    # ── sales total/summary fast-paths (GPT misroutes these) ──
    total_units_phrases = [
        "how many units", "total units sold", "units sold in total",
        "how much have we sold", "total sales units", "units have we sold"
    ]
    avg_revenue_phrases = [
        "average daily revenue", "avg daily revenue", "average revenue per day",
        "daily revenue average", "what is our average"
    ]
    this_month_phrases = [
        "this month", "how much this month", "revenue this month",
        "this months revenue", "monthly revenue", "made this month"
    ]
    best_day_phrases = [
        "which day", "what day", "busiest day", "best day of the week",
        "highest revenue day", "most revenue day", "day of the week has"
    ]
    if any(p in q_clean for p in total_units_phrases):
        return {"tool": "query_sales", "params": {"period": "total_summary"}, "confidence": 1.0}
    if any(p in q_clean for p in avg_revenue_phrases):
        return {"tool": "query_sales", "params": {"period": "total_summary"}, "confidence": 1.0}
    if any(p in q_clean for p in this_month_phrases):
        return {"tool": "query_sales", "params": {"period": "current_month"}, "confidence": 1.0}
    if any(p in q_clean for p in best_day_phrases):
        return {"tool": "query_sales", "params": {"period": "best_day"}, "confidence": 1.0}

    # ── batch count fast-path ──
    import re as _re
    batch_count_match = _re.search(r"(more than|at least|over|above|[0-9]+)\s+([0-9]+)?\s*batch", q_clean)
    batch_plain = any(p in q_clean for p in ["how many batches", "number of batches", "batch count"])
    if batch_count_match or batch_plain:
        # Only count_only when NO specific drug mentioned
        _kws = extract_keywords(q_clean)
        _has_drug = any(fuzzy_match_drug(k, threshold=85) for k in _kws)
        if not _has_drug:
            nums = _re.findall(r"[0-9]+", q_clean)
            min_b = int(nums[0]) + 1 if batch_count_match and nums else (int(nums[0]) if nums else 2)
            return {"tool": "query_expiry", "params": {"count_only": True, "min_batches": min_b}, "confidence": 1.0}
    # "what about N batches" follow-up
    about_batch = _re.search(r"what about\s+([0-9]+)\s+batch", q_clean)
    if about_batch:
        _kws2 = extract_keywords(q_clean)
        _has_drug2 = any(fuzzy_match_drug(k, threshold=85) for k in _kws2)
        if not _has_drug2:
            min_b = int(about_batch.group(1))
            return {"tool": "query_expiry", "params": {"count_only": True, "min_batches": min_b}, "confidence": 1.0}

    # ── Build context note for follow-up detection ──
    context = ""
    if conversation_history:
        last_asst = next(
            (m for m in reversed(conversation_history)
             if isinstance(m, dict) and m.get("role") == "assistant"),
            None
        )
        if last_asst:
            prev = last_asst.get("content", "")
            if isinstance(prev, list):
                prev = " ".join(c.get("text","") if isinstance(c,dict) else str(c) for c in prev)
            prev = str(prev)[:300]
            sort_hint = ""
            if "by revenue" in prev.lower():
                sort_hint = " Previous query used sort_by=revenue — preserve that metric."
            elif "by units" in prev.lower():
                sort_hint = " Previous query used sort_by=units — preserve that metric."
            elif "by transactions" in prev.lower():
                sort_hint = " Previous query used sort_by=transactions — preserve that metric."
            if prev:
                context = f"\n\nPrevious assistant response:\n{prev}{sort_hint}"

    system = (
        "You are a pharmacy operations assistant. "
        "Given a staff question, call the appropriate tool with precise parameters. "
        "Extract numbers from the question for limit parameters. "
        "For 'least selling' or 'worst sellers', use query_sales with direction=bottom. "
        "For 'top N by units sold', use query_sales with sort_by=units. "
        "For follow-up questions referencing 'it', 'them', 'those', 'tell me more', "
        "'what about X', 'and Y' — use the same tool as the previous response context. "
        "For 'what about [city]?' or 'and [city]?' or '[city]?' after a supplier query — use query_supplier with city=[city]. "
        "Short location names like 'Harare', 'Bulawayo', 'Mutare' after a supplier question are city filters. "
        "For 'fastest vendor', 'quickest supplier', 'shortest lead time', 'who is fastest' — ALWAYS use query_supplier with sort_by=lead_time and drug_name=null. NEVER inherit drug context for these. "
        "For 'slowest supplier', 'longest lead time', 'who is slowest' — use query_supplier with sort_by=lead_time, direction=desc, drug_name=null. NEVER inherit drug context. "
        "For 'their lead time', 'what is their lead time' — use query_supplier and extract the drug name from the previous assistant response context. "
        "For 'best payment terms', 'who has best payment terms', 'credit terms' — use query_supplier with sort_by=payment_terms. "
        "For 'now show bottom N', 'now show top N', 'flip to bottom' — keep the same sort_by metric from context, only change direction. "
        "For 'how much this month', 'revenue this month', 'monthly revenue' — use query_sales with period=last_week (best available approximation). "
        "For clearly clinical questions about a named drug (molecular weight, mechanism of action, pharmacokinetics) — use query_clinical even if the data may not be in the knowledge base. "
        "For questions completely unrelated to pharmacy (weather, sports, politics, personal) "
        "do NOT call any tool — respond with no tool_call."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": question + context}
            ],
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=150,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            params = json.loads(tc.function.arguments)
            return {"tool": tc.function.name, "params": params, "confidence": 0.95}
        # GPT chose not to call a tool — out-of-scope
        return {"tool": "out_of_scope", "params": {}, "confidence": 0.9}
    except Exception as e:
        print(f"Tool calling failed: {e}")
        return _keyword_fallback_tool(q)


def _keyword_fallback_tool(q: str) -> dict:
    """Keyword fallback when GPT tool calling fails."""
    if any(w in q for w in ["low", "running low", "reorder", "below reorder", "critical"]):
        return {"tool": "query_inventory", "params": {"filter": "below_reorder", "limit": 10}, "confidence": 0.5}
    if any(w in q for w in ["expir", "batch", "batches", "days until"]):
        return {"tool": "query_expiry", "params": {"within_days": 90, "limit": 10}, "confidence": 0.5}
    if any(w in q for w in ["sold", "sales", "revenue", "top", "bottom", "selling"]):
        return {"tool": "query_sales", "params": {"period": "all_time", "direction": "top", "sort_by": "revenue", "limit": 10}, "confidence": 0.5}
    if any(w in q for w in ["supplier", "vendor", "distributor", "order from", "lead time", "who supplies", "buy from"]):
        return {"tool": "query_supplier", "params": {}, "confidence": 0.5}
    if any(w in q for w in ["interact", "safe with", "combine", "avoid"]):
        kws = extract_keywords(q)
        drug = kws[0] if kws else ""
        return {"tool": "query_clinical", "params": {"query_type": "interaction", "drug_name": drug}, "confidence": 0.5}
    if any(w in q for w in ["alternative", "substitute", "instead of", "replace"]):
        kws = extract_keywords(q)
        return {"tool": "query_alternatives", "params": {"drug_name": kws[0] if kws else ""}, "confidence": 0.5}
    if any(w in q for w in ["categories", "how many categories", "inventory summary"]):
        return {"tool": "query_stats", "params": {}, "confidence": 0.5}
    kws = extract_keywords(q)
    drug = kws[0] if kws else q
    return {"tool": "query_clinical", "params": {"query_type": "drug_info", "drug_name": drug}, "confidence": 0.3}


# ══════════════════════════════════════════════════════════════════
# SQL EXECUTORS — GPT provides params, Python runs safe SQL
# ══════════════════════════════════════════════════════════════════

def execute_query_inventory(params: dict) -> str:
    """Execute inventory query with GPT-provided parameters."""
    filt      = params.get("filter", "all")
    drug_name = params.get("drug_name")
    category  = params.get("category")
    limit     = max(1, min(params.get("limit", 10), 50))
    sort_by   = params.get("sort_by", "name")

    sort_map = {
        "stock_pct":   "stock_pct ASC",
        "price_asc":   "selling_price_usd ASC",
        "price_desc":  "selling_price_usd DESC",
        "name":        "generic_name ASC",
        "margin_desc": "margin DESC",
        "margin_asc":  "margin ASC",
    }
    order = sort_map.get(sort_by, "generic_name ASC")

    where_clauses = []
    sql_params: list = []

    if filt == "below_reorder":
        where_clauses.append("quantity_in_stock <= reorder_level")
        order = "stock_pct ASC"
    elif filt == "cheapest":
        order = "selling_price_usd ASC"
    elif filt == "most_expensive":
        order = "selling_price_usd DESC"

    if drug_name:
        where_clauses.append("(LOWER(generic_name) LIKE %s OR LOWER(brand_name) LIKE %s)")
        sql_params.append(f"%{drug_name.lower()}%")
        sql_params.append(f"%{drug_name.lower()}%")

    if category:
        cat_map = {
            "antibiotic": "Antibiotics", "analgesic": "Analgesics",
            "antihypertensive": "Antihypertensives", "antidiabetic": "Antidiabetics",
            "antimalarial": "Antimalarials", "antifungal": "Antifungals",
            "antiretroviral": "Antiretrovirals", "respiratory": "Respiratory",
            "vitamin": "Vitamins/Supplements", "gi": "GI medications",
        }
        matched_cat = next((v for k, v in cat_map.items() if k in category.lower()), category)
        where_clauses.append("category = %s")
        sql_params.append(matched_cat)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category,
               ROUND((quantity_in_stock::numeric/NULLIF(reorder_level,0))*100,0) AS stock_pct,
               ROUND(((selling_price_usd - cost_price_usd)/NULLIF(selling_price_usd,0)*100)::numeric,1) AS margin
        FROM inventory
        {where}
        ORDER BY {order}
        LIMIT %s
    """
    sql_params.append(limit)

    df = pd.read_sql_query(sql, get_engine(), params=tuple(sql_params))

    if df.empty:
        if filt == "below_reorder":
            return "✅ **Good news** — all products are currently above their reorder levels."
        return "❌ No drugs found matching that criteria."

    if filt == "below_reorder":
        header = f"⚠️ **{len(df)} drug(s) at or below reorder level:**\n\n"
        header += "| Drug | Brand | Stock | Reorder | % | Category |\n|---|---|---|---|---|---|\n"
        rows = [
            f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | "
            f"{r['reorder_level']} | {r['stock_pct']:.0f}% | {r['category']} |"
            for _, r in df.iterrows()
        ]
        return header + "\n".join(rows)

    if drug_name:
        # Prefer exact name match to avoid e.g. "paracetamol" → "Codeine/Paracetamol"
        exact = df[df['generic_name'].str.lower() == drug_name.lower()]
        if not exact.empty:
            df = exact.reset_index(drop=True)
    if drug_name and len(df) == 1:
        r = df.iloc[0]
        flag = "⚠️ LOW STOCK" if r['quantity_in_stock'] <= r['reorder_level'] else "✅ In Stock"
        return (
            f"**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}\n"
            "| Field | Value |\n|---|---|\n"
            f"| Stock | {r['quantity_in_stock']} units — {flag} |\n"
            f"| Reorder Level | {r['reorder_level']} units |\n"
            f"| Selling Price | ${r['selling_price_usd']} |\n"
            f"| Cost Price | ${r['cost_price_usd']} |\n"
            f"| Shelf Location | {r['shelf_location']} |\n"
            f"| Category | {r['category']} |\n"
        )

    if category and not drug_name:
        cat_name = df.iloc[0]['category']
        header = f"**{cat_name}** — {len(df)} drugs\n\n"
        header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n|---|---|---|---|---|---|---|\n"
        rows = [
            f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
            f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |"
            for _, r in df.iterrows()
        ]
        return header + "\n".join(rows)

    # Ranking (cheapest / most expensive / margin / search results)
    if filt == "cheapest":
        label = f"Cheapest {limit} drugs in stock"
    elif filt == "most_expensive":
        label = f"Most expensive {limit} drugs"
    elif sort_by in ("margin_desc", "margin_asc"):
        label = f"{'Highest' if sort_by == 'margin_desc' else 'Lowest'} margin drugs"
        header = f"**{label}:**\n\n| Drug | Brand | Sell Price | Cost Price | Margin% | Stock |\n|---|---|---|---|---|---|\n"
        rows = [
            f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | "
            f"${r['cost_price_usd']} | {r['margin']}% | {r['quantity_in_stock']} |"
            for _, r in df.iterrows()
        ]
        return header + "\n".join(rows)
    elif drug_name:
        label = f"Drugs matching '{drug_name}'"
    else:
        label = f"Top {limit} drugs"
    header = f"**{label}:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | {r['quantity_in_stock']} | {r['shelf_location']} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows)


def execute_query_sales(params: dict) -> str:
    """Execute sales query with GPT-provided parameters."""
    period    = params.get("period", "all_time")
    direction = params.get("direction", "top")
    sort_by   = params.get("sort_by", "revenue")
    day_name  = params.get("day_name", "")
    limit     = max(1, min(params.get("limit", 10), 50))

    if period == "customer_type":
        return _sales_customer_type()

    if period == "last_day":
        return _sales_last_day()

    if period == "last_week":
        return _sales_last_week()

    if period == "current_month":
        return _sales_current_month()

    if period == "best_day":
        return _sales_best_day()

    if period == "total_summary":
        return _sales_total_summary()

    if period == "day_of_week" and day_name:
        return _sales_day_of_week(day_name)

    # Top/bottom sellers — all_time
    order     = "DESC" if direction == "top" else "ASC"
    label     = f"{'Top' if direction == 'top' else 'Bottom'} {limit}"
    col_map   = {"revenue": "total_revenue", "units": "total_units", "transactions": "num_transactions"}
    sort_col  = col_map.get(sort_by, "total_revenue")
    sort_label = f"by {sort_by}"

    df = pd.read_sql_query(f"""
        SELECT i.brand_name, i.generic_name,
               SUM(t.quantity_sold)                    AS total_units,
               ROUND(SUM(t.total_amount)::numeric, 2)  AS total_revenue,
               COUNT(*)                                 AS num_transactions
        FROM transactions t
        JOIN inventory i ON t.product_id = i.product_id
        GROUP BY i.brand_name, i.generic_name
        ORDER BY {sort_col} {order}
        LIMIT %s
    """, get_engine(), params=(limit,))

    header  = f"**{label} Selling Drugs** {sort_label} (Last 30 days)\n\n"
    header += "| Rank | Brand | Generic | Units | Revenue | Transactions |\n|---|---|---|---|---|---|\n"
    rows = [
        f"| {i+1} | {r['brand_name']} | {r['generic_name']} | "
        f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['num_transactions']} |"
        for i, (_, r) in enumerate(df.iterrows())
    ]
    return header + "\n".join(rows)


def _sales_best_day() -> str:
    df = pd.read_sql_query(
        "SELECT TRIM(TO_CHAR(date::date, 'Day')) AS day_name,"
        " EXTRACT(DOW FROM date::date)::INTEGER AS dow,"
        " COUNT(DISTINCT date::date) AS occurrences,"
        " ROUND(SUM(total_amount)::numeric,2) AS total_revenue,"
        " ROUND(AVG(daily_rev)::numeric,2) AS avg_revenue,"
        " SUM(quantity_sold) AS total_units,"
        " COUNT(*) AS total_transactions"
        " FROM transactions t"
        " JOIN (SELECT date::date AS d, SUM(total_amount) AS daily_rev"
        "       FROM transactions GROUP BY date::date) dr ON t.date::date = dr.d"
        " GROUP BY day_name, dow ORDER BY avg_revenue DESC",
        get_engine()
    )
    if df.empty:
        return "No sales data available."
    best_name = str(df.iloc[0]["day_name"]).strip()
    rows = [
        "| Day | Avg Revenue | Total Revenue | Occurrences | Transactions | Units |",
        "|---|---|---|---|---|---|"
    ]
    for _, r in df.iterrows():
        dn = str(r["day_name"]).strip()
        star = " ⭐" if dn == best_name else ""
        rows.append(
            f"| {dn}{star} | **${r['avg_revenue']:,.2f}** | "
            f"${r['total_revenue']:,.2f} | {r['occurrences']} | "
            f"{r['total_transactions']} | {r['total_units']} |"
        )
    header = "**Revenue by Day of Week** \u2014 best day is **" + best_name + "**\n\n"
    return header + "\n".join(rows)


def _sales_current_month() -> str:
    df_total = pd.read_sql_query(
        "SELECT TO_CHAR(DATE_TRUNC('month', CURRENT_DATE), 'Month YYYY') AS month_label,"
        " COUNT(*) AS transactions, SUM(quantity_sold) AS total_units,"
        " ROUND(SUM(total_amount)::numeric,2) AS total_revenue"
        " FROM transactions WHERE date::date >= DATE_TRUNC('month', CURRENT_DATE)",
        get_engine()
    )
    df_daily = pd.read_sql_query(
        "SELECT date::date AS day, COUNT(*) AS txns, SUM(quantity_sold) AS units,"
        " ROUND(SUM(total_amount)::numeric,2) AS revenue"
        " FROM transactions WHERE date::date >= DATE_TRUNC('month', CURRENT_DATE)"
        " GROUP BY date::date ORDER BY date::date DESC",
        get_engine()
    )
    r = df_total.iloc[0]
    month = str(r["month_label"]).strip() if r["month_label"] else "This month"
    if not r["total_revenue"]:
        return "No transactions recorded for " + month + " yet."
    out = (
        "**" + month + " Revenue**\n\n"
        + f"Total Revenue: **${r['total_revenue']:,.2f}** | "
        + f"Transactions: **{r['transactions']}** | Units: **{r['total_units']}**\n\n"
    )
    if not df_daily.empty:
        days = len(df_daily)
        avg  = round(float(r["total_revenue"]) / days, 2)
        out += f"Days recorded: **{days}** | Daily average: **${avg:,.2f}**\n\n"
        out += "| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
        out += "\n".join(
            f"| {str(row['day'])[:10]} | {row['txns']} | {row['units']} | ${row['revenue']:,.2f} |"
            for _, row in df_daily.iterrows()
        )
    return out


def _sales_total_summary() -> str:
    df = pd.read_sql_query(
        "SELECT COUNT(*) AS total_transactions, SUM(quantity_sold) AS total_units,"
        " ROUND(SUM(total_amount)::numeric,2) AS total_revenue,"
        " COUNT(DISTINCT date::date) AS trading_days,"
        " ROUND(AVG(daily_rev)::numeric,2) AS avg_daily_revenue,"
        " MIN(date::date) AS first_date, MAX(date::date) AS last_date"
        " FROM transactions"
        " JOIN (SELECT date::date AS d, SUM(total_amount) AS daily_rev"
        "       FROM transactions GROUP BY date::date) dr"
        " ON transactions.date::date = dr.d",
        get_engine()
    )
    r = df.iloc[0]
    first = str(r["first_date"])[:10]
    last  = str(r["last_date"])[:10]
    return (
        "**Overall Sales Summary**\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Total Revenue | **${r['total_revenue']:,.2f}** |\n"
        f"| Total Units Sold | **{r['total_units']}** |\n"
        f"| Total Transactions | **{r['total_transactions']}** |\n"
        f"| Trading Days | {r['trading_days']} |\n"
        f"| Avg Daily Revenue | **${r['avg_daily_revenue']:,.2f}** |\n"
        f"| Date Range | {first} \u2192 {last} |\n"
    )


def _sales_customer_type() -> str:
    df = pd.read_sql_query("""
        SELECT customer_type,
               COUNT(*)                                  AS num_transactions,
               SUM(quantity_sold)                        AS total_units,
               ROUND(SUM(total_amount)::numeric, 2)      AS total_revenue,
               ROUND((SUM(total_amount) * 100.0 /
                   (SELECT SUM(total_amount) FROM transactions))::numeric, 1)
               AS revenue_pct
        FROM transactions
        GROUP BY customer_type
        ORDER BY total_revenue DESC
    """, get_engine())
    header  = "**Sales by Customer Type** (Last 30 days)\n\n"
    header += "| Customer Type | Transactions | Units Sold | Revenue | % of Total |\n|---|---|---|---|---|\n"
    rows = [
        f"| {r['customer_type']} | {r['num_transactions']} | "
        f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['revenue_pct']}% |"
        for _, r in df.iterrows()
    ]
    total = df['total_revenue'].sum()
    return header + "\n".join(rows) + f"\n\n**Total Revenue: ${total:,.2f}**"


def _sales_last_day() -> str:
    df_d  = pd.read_sql_query("""
        SELECT date, COUNT(*) AS num_transactions,
               SUM(quantity_sold) AS total_units,
               ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
        FROM transactions
        WHERE date = (SELECT MAX(date) FROM transactions)
        GROUP BY date
    """, get_engine())
    df_dr = pd.read_sql_query("""
        SELECT i.brand_name, i.generic_name,
               SUM(t.quantity_sold) AS units,
               ROUND(SUM(t.total_amount)::numeric, 2) AS revenue
        FROM transactions t JOIN inventory i ON t.product_id = i.product_id
        WHERE t.date = (SELECT MAX(date) FROM transactions)
        GROUP BY i.brand_name, i.generic_name
        ORDER BY revenue DESC
    """, get_engine())
    df_ct = pd.read_sql_query("""
        SELECT customer_type, ROUND(SUM(total_amount)::numeric, 2) AS revenue
        FROM transactions
        WHERE date = (SELECT MAX(date) FROM transactions)
        GROUP BY customer_type ORDER BY revenue DESC
    """, get_engine())
    if df_d.empty:
        return "No transactions found."
    r = df_d.iloc[0]
    out  = f"**Sales for {str(r['date'])[:10]}** (Last recorded day)\n\n"
    out += f"Transactions: **{r['num_transactions']}** | Units: **{r['total_units']}** | Revenue: **${r['total_revenue']:,.2f}**\n\n"
    out += "**By Drug:**\n\n| Brand | Generic | Units | Revenue |\n|---|---|---|---|\n"
    out += "\n".join(
        f"| {row['brand_name']} | {row['generic_name']} | {row['units']} | ${row['revenue']:,.2f} |"
        for _, row in df_dr.iterrows()
    )
    out += "\n\n**By Customer Type:** " + " | ".join(
        f"{row['customer_type']}: ${row['revenue']:,.2f}" for _, row in df_ct.iterrows()
    )
    return out


def _sales_last_week() -> str:
    df = pd.read_sql_query("""
        SELECT date, COUNT(*) AS num_transactions,
               SUM(quantity_sold) AS total_units,
               ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
        FROM transactions
        WHERE date::date >= (SELECT MAX(date::date) - 7 FROM transactions)
        GROUP BY date ORDER BY date DESC
    """, get_engine())
    if df.empty:
        return "No transactions found for last week."
    header = "**Last Week Sales**\n\n| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
    rows   = [
        f"| {str(r['date'])[:10]} | {r['num_transactions']} | {r['total_units']} | ${r['total_revenue']:,.2f} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows) + f"\n\n**Total: ${df['total_revenue'].sum():,.2f}**"


def _sales_day_of_week(day_name: str) -> str:
    day_map = {"monday":1,"tuesday":2,"wednesday":3,"thursday":4,"friday":5,"saturday":6,"sunday":0}
    day_num = day_map.get(day_name.lower(), 6)
    # Summary totals across all recorded Fridays/Saturdays/etc.
    df_summary = pd.read_sql_query("""
        SELECT
            COUNT(DISTINCT t.date::date)             AS num_days,
            COUNT(*)                                  AS total_transactions,
            SUM(t.quantity_sold)                      AS total_units,
            ROUND(SUM(t.total_amount)::numeric, 2)    AS total_revenue,
            ROUND(AVG(daily_rev.rev)::numeric, 2)     AS avg_daily_revenue
        FROM transactions t
        JOIN (
            SELECT date::date AS d, SUM(total_amount) AS rev
            FROM transactions
            WHERE EXTRACT(DOW FROM date::date) = %(dow)s
            GROUP BY date::date
        ) daily_rev ON t.date::date = daily_rev.d
        WHERE EXTRACT(DOW FROM t.date::date) = %(dow)s
    """, get_engine(), params={"dow": day_num})
    df_drugs = pd.read_sql_query("""
        SELECT i.brand_name, i.generic_name,
               SUM(t.quantity_sold)                   AS total_units,
               ROUND(SUM(t.total_amount)::numeric, 2) AS total_revenue,
               COUNT(*)                               AS transactions
        FROM transactions t JOIN inventory i ON t.product_id = i.product_id
        WHERE EXTRACT(DOW FROM t.date::date) = %(dow)s
        GROUP BY i.brand_name, i.generic_name
        ORDER BY total_units DESC
        LIMIT 10
    """, get_engine(), params={"dow": day_num})
    if df_drugs.empty:
        return f"No sales data found for {day_name.capitalize()}s."
    s = df_summary.iloc[0]
    out  = f"**{day_name.capitalize()} Sales Summary** (across {s['num_days']} recorded {day_name.capitalize()}s)\n\n"
    out += f"Total Transactions: **{s['total_transactions']}** | "
    out += f"Total Units: **{s['total_units']}** | "
    out += f"Total Revenue: **${s['total_revenue']:,.2f}** | "
    out += f"Avg per {day_name.capitalize()}: **${s['avg_daily_revenue']:,.2f}**\n\n"
    out += f"**Drug Breakdown:**\n\n"
    out += "| Rank | Brand | Generic | Units | Revenue | Transactions |\n|---|---|---|---|---|---|\n"
    out += "\n".join(
        f"| {i+1} | {r['brand_name']} | {r['generic_name']} | {r['total_units']} | ${r['total_revenue']:,.2f} | {r['transactions']} |"
        for i, (_, r) in enumerate(df_drugs.iterrows())
    )
    return out


MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

def execute_query_expiry(params: dict) -> str:
    """Execute expiry query with GPT-provided parameters."""
    drug_name   = params.get("drug_name")
    within_days = params.get("within_days", 90)
    limit       = max(1, min(params.get("limit", 10), 50))
    top_only    = params.get("top_only", False)
    month_name  = params.get("month_name", "")
    count_only  = params.get("count_only", False)

    # Named month filter — "expiring in May"
    if month_name:
        month_num = MONTH_MAP.get(month_name.lower())
        if month_num:
            df = pd.read_sql_query("""
                SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
                       b.quantity_remaining,
                       (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days_remaining
                FROM batches b JOIN inventory i ON b.product_id = i.product_id
                WHERE EXTRACT(MONTH FROM b.expiry_date::date) = %s
                  AND EXTRACT(YEAR  FROM b.expiry_date::date) >= EXTRACT(YEAR FROM CURRENT_DATE)
                ORDER BY b.expiry_date ASC
            """, get_engine(), params=(month_num,))
            if df.empty:
                return f"✅ No batches expiring in {month_name.capitalize()}."
            header  = f"**Batches expiring in {month_name.capitalize()}** — {len(df)} found:\n\n"
            header += "| Drug | Brand | Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|---|---|\n"
            rows = []
            for _, r in df.iterrows():
                d = r['days_remaining']
                flag = "🚨 URGENT" if d < 30 else ("⚠️ Warning" if d < 90 else "📅 Monitor")
                rows.append(
                    f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | "
                    f"{str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |"
                )
            return header + "\n".join(rows)

    # Batch count per drug — "which drugs have more than 3 batches"
    if count_only:
        min_batches = params.get("min_batches", 2)
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, COUNT(b.batch_id) AS batch_count,
                   MIN(b.expiry_date) AS nearest_expiry,
                   SUM(b.quantity_remaining) AS total_qty
            FROM inventory i JOIN batches b ON i.product_id = b.product_id
            GROUP BY i.product_id, i.generic_name, i.brand_name
            HAVING COUNT(b.batch_id) >= %s
            ORDER BY batch_count DESC
        """, get_engine(), params=(min_batches,))
        if df.empty:
            return f"No drugs found with {min_batches} or more batches."
        header  = f"**Drugs with {min_batches}+ batches:**\n\n"
        header += "| Drug | Brand | Batches | Nearest Expiry | Total Qty |\n|---|---|---|---|---|\n"
        rows = [
            f"| {r['generic_name']} | {r['brand_name']} | **{r['batch_count']}** | "
            f"{str(r['nearest_expiry'])[:10]} | {r['total_qty']} |"
            for _, r in df.iterrows()
        ]
        return header + "\n".join(rows)

    if drug_name:
        df = pd.read_sql_query("""
            SELECT b.batch_number, b.expiry_date, b.quantity_remaining,
                   (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days
            FROM batches b JOIN inventory i ON b.product_id = i.product_id
            WHERE LOWER(i.generic_name) LIKE %s
            ORDER BY b.expiry_date ASC
        """, get_engine(), params=(f"%{drug_name.lower()}%",))
        if df.empty:
            return f"❌ No batch records found for {drug_name}."
        header = f"**{drug_name} — {len(df)} batch(es):**\n\n| Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|\n"
        rows = []
        for _, r in df.iterrows():
            d    = r['days']
            flag = "🚨 URGENT" if d < 30 else ("⚠️ Warning" if d < 90 else "✅ OK")
            rows.append(f"| {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |")
        return header + "\n".join(rows)

    # General expiry alert — with optional top_only (first-to-expire)
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
               b.quantity_remaining,
               (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days_remaining
        FROM batches b JOIN inventory i ON b.product_id = i.product_id
        WHERE (b.expiry_date::date - CURRENT_DATE) <= %s
        ORDER BY b.expiry_date ASC
        LIMIT %s
    """, get_engine(), params=(within_days, 1 if top_only else limit))

    if df.empty:
        return f"✅ No batches expiring within {within_days} days."

    if top_only:
        r = df.iloc[0]
        d = r['days_remaining']
        flag = "🚨 URGENT" if d < 30 else ("⚠️ Warning" if d < 90 else "📅 Monitor")
        return (
            f"**First to expire:** {r['generic_name']} ({r['brand_name']})\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Batch | {r['batch_number']} |\n"
            f"| Expiry Date | {str(r['expiry_date'])[:10]} |\n"
            f"| Days Remaining | **{d}** — {flag} |\n"
            f"| Qty Remaining | {r['quantity_remaining']} |\n"
        )

    header  = f"**Batches expiring within {within_days} days** — {len(df)} found:\n\n"
    header += "| Drug | Brand | Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        d    = r['days_remaining']
        flag = "🚨 URGENT" if d < 30 else ("⚠️ Warning" if d < 90 else "📅 Monitor")
        rows.append(
            f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | "
            f"{str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |"
        )
    return header + "\n".join(rows)


def execute_query_supplier(params: dict) -> str:
    """Execute supplier query with GPT-provided parameters."""
    drug_name = params.get("drug_name")
    city      = params.get("city")
    sort_by   = params.get("sort_by", "name")
    limit     = max(1, min(params.get("limit", 5), 20))

    # Payment terms ranking
    if sort_by == "payment_terms" and not drug_name:
        results = run_cypher("""
            MATCH (s:Supplier)
            RETURN DISTINCT s.name AS supplier, s.payment_terms AS payment_terms,
                   s.lead_time AS lead_time_days, s.city AS city, s.contact AS contact
            ORDER BY s.payment_terms DESC
            LIMIT 10
        """)
        if not results:
            return "❌ No supplier payment terms found."
        header = ("**Suppliers by Payment Terms:**\n\n"
                  "| Supplier | Payment Terms | Lead Time | City | Contact |\n"
                  "|---|---|---|---|---|\n")
        return header + "\n".join(
            f"| {r['supplier']} | **{r['payment_terms']}** | {r['lead_time_days']} days | {r['city']} | {r['contact']} |"
            for r in results
        )
    # Lead time ranking
    direction = params.get("direction", "asc")
    if sort_by == "lead_time" and not drug_name:
        order = "ASC" if direction != "desc" else "DESC"
        label = "fastest" if order == "ASC" else "slowest"
        results = run_cypher(f"""
            MATCH (s:Supplier)
            RETURN s.name AS supplier, s.lead_time AS lead_time_days,
                   s.city AS city, s.contact AS contact
            ORDER BY s.lead_time {order} LIMIT 5
        """)
        if not results:
            return "❌ No supplier information found."
        header = f"**Suppliers by lead time ({label} first):**\n\n| Supplier | Lead Time | City | Contact |\n|---|---|---|---|\n"
        return header + "\n".join(
            f"| {r['supplier']} | {r['lead_time_days']} days | {r['city']} | {r['contact']} |"
            for r in results
        )

    # City filter
    if city:
        results = run_cypher("""
            MATCH (s:Supplier)
            WHERE toLower(s.city) CONTAINS toLower($city)
            RETURN s.city AS city, count(s) AS supplier_count, collect(s.name) AS suppliers
            ORDER BY supplier_count DESC
        """, {"city": city})
        if not results:
            return f"❌ No suppliers found in {city}."
        header = f"**Suppliers in {city.title()}:**\n\n| City | Count | Suppliers |\n|---|---|---|\n"
        return header + "\n".join(
            f"| {r['city']} | {r['supplier_count']} | {', '.join(r['suppliers'])} |"
            for r in results
        )

    # Drug-specific or category-specific supplier
    if drug_name:
        # First try exact drug match
        results = run_cypher("""
            MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
            WHERE toLower(d.generic_name) CONTAINS toLower($search)
            RETURN DISTINCT d.generic_name AS drug, s.name AS supplier,
                   s.contact AS contact, s.phone AS phone,
                   s.city AS city, s.lead_time AS lead_time_days,
                   s.payment_terms AS payment_terms
            LIMIT 5
        """, {"search": drug_name})
        # If not found as drug, try as category (e.g. "antibiotics")
        if not results:
            df_cat = pd.read_sql_query("""
                    SELECT DISTINCT i.generic_name
                    FROM inventory i
                    WHERE LOWER(i.category) LIKE %s
                    LIMIT 5
                """, get_engine(), params=(f"%{drug_name.lower()}%",))
            if not df_cat.empty:
                cat_drugs = df_cat["generic_name"].tolist()
                results = run_cypher("""
                    MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
                    WHERE d.generic_name IN $drugs
                    RETURN d.generic_name AS drug, s.name AS supplier,
                           s.contact AS contact, s.phone AS phone,
                           s.city AS city, s.lead_time AS lead_time_days,
                           s.payment_terms AS payment_terms
                    ORDER BY s.lead_time ASC LIMIT 5
                """, {"drugs": cat_drugs})
        if not results:
            return f"❌ No supplier found for {drug_name}."
        if len(results) == 1:
            r = results[0]
            return (
                f"**Supplier for {r['drug']}:**\n\n"
                "| Field | Value |\n|---|---|\n"
                f"| Supplier | **{r['supplier']}** |\n"
                f"| Contact | {r['contact']} |\n"
                f"| Phone | {r['phone']} |\n"
                f"| City | {r['city']} |\n"
                f"| Lead Time | {r['lead_time_days']} days |\n"
                f"| Payment Terms | {r['payment_terms']} |\n"
            )
        # Multiple results (category query)
        header = f"**Suppliers for {drug_name}:**\n\n| Drug | Supplier | City | Lead Time | Contact |\n|---|---|---|---|---|\n"
        return header + "\n".join(
            f"| {r['drug']} | {r['supplier']} | {r['city']} | {r['lead_time_days']} days | {r['contact']} |"
            for r in results
        )

    # Default — total count + city breakdown
    results = run_cypher("""
        MATCH (s:Supplier)
        RETURN s.city AS city, count(s) AS supplier_count, collect(s.name) AS suppliers
        ORDER BY supplier_count DESC
    """)
    total = sum(r['supplier_count'] for r in results)
    header = f"**{total} suppliers** across {len(results)} cities:\n\n| City | Count | Suppliers |\n|---|---|---|\n"
    return header + "\n".join(
        f"| {r['city']} | {r['supplier_count']} | {', '.join(r['suppliers'])} |"
        for r in results
    )


def format_stats() -> str:
    df = pd.read_sql_query("""
        SELECT category,
               COUNT(*)               AS drug_count,
               SUM(quantity_in_stock) AS total_units,
               ROUND(AVG(selling_price_usd)::numeric, 2) AS avg_price,
               ROUND(SUM(quantity_in_stock * cost_price_usd)::numeric, 2) AS inventory_value
        FROM inventory
        GROUP BY category ORDER BY inventory_value DESC
    """, get_engine())
    total_drugs = df['drug_count'].sum()
    total_value = df['inventory_value'].sum()
    header  = f"**Inventory Summary** — {total_drugs} products across {len(df)} categories\n\n"
    header += f"Total inventory value: **${total_value:,.2f}**\n\n"
    header += "| Category | Drugs | Total Units | Avg Price | Inv. Value |\n|---|---|---|---|---|\n"
    rows = [
        f"| {r['category']} | {r['drug_count']} | {r['total_units']} | "
        f"${r['avg_price']} | ${r['inventory_value']:,.2f} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows)


def format_alternative(drug_name: str) -> str:
    """Find drugs in the same therapeutic category."""
    if not drug_name:
        return "❌ Please specify a drug name."

    result = pd.read_sql_query(
        "SELECT generic_name, category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1",
        get_engine(), params=(f"%{drug_name.lower()}%",)
    )

    if result.empty:
        return f"❌ **{drug_name}** not found in inventory."

    found_name = result.iloc[0]["generic_name"]
    category   = result.iloc[0]["category"]
    search_pct = f"%{drug_name.lower()}%"

    df = pd.read_sql_query("""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, selling_price_usd, shelf_location
        FROM inventory
        WHERE category = %s
          AND LOWER(generic_name) NOT LIKE %s
          AND quantity_in_stock > 0
        ORDER BY generic_name
    """, get_engine(), params=(category, search_pct))

    if df.empty:
        return f"❌ No in-stock alternatives found for **{found_name}** in category {category}."

    header  = f"**Alternatives to {found_name}** (category: {category})\n\n"
    header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n|---|---|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
        f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows) + "\n\n⚠️ **Clinical Note:** Therapeutic substitution requires pharmacist approval."


def format_drug_summary(drug_name: str) -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.formulation, i.strength,
               i.quantity_in_stock, i.reorder_level,
               i.selling_price_usd, i.cost_price_usd,
               i.shelf_location, i.category,
               MIN(b.expiry_date) AS nearest_expiry,
               (MIN(b.expiry_date::date) - CURRENT_DATE)::INTEGER AS days_to_expiry
        FROM inventory i
        LEFT JOIN batches b ON i.product_id = b.product_id
        WHERE LOWER(i.generic_name) LIKE LOWER(%s)
        GROUP BY i.product_id, i.generic_name, i.brand_name, i.formulation,
                 i.strength, i.quantity_in_stock, i.reorder_level,
                 i.selling_price_usd, i.cost_price_usd, i.shelf_location, i.category
        LIMIT 1
    """, get_engine(), params=(f"%{drug_name}%",))

    if df.empty:
        return f"❌ **{drug_name}** not found in inventory."

    r = df.iloc[0]
    stock_status = "⚠️ LOW STOCK — reorder needed" if r['quantity_in_stock'] <= r['reorder_level'] else "✅ In Stock"
    expiry_line  = ""
    if r.get("days_to_expiry") is not None:
        d = int(r['days_to_expiry'])
        exp_date = str(r['nearest_expiry'])[:10]
        if d <= 30:
            expiry_line = f"\n🚨 **URGENT:** Nearest batch expires in {d} days ({exp_date})"
        elif d <= 90:
            expiry_line = f"\n⚠️ Nearest expiry: {exp_date} ({d} days)"
        else:
            expiry_line = f"\n📅 Nearest expiry: {exp_date} ({d} days)"

    return (
        f"**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}\n\n"
        "| Field | Value |\n|---|---|\n"
        f"| **Stock** | {r['quantity_in_stock']} units — {stock_status} |\n"
        f"| **Reorder Level** | {r['reorder_level']} units |\n"
        f"| **Selling Price** | ${r['selling_price_usd']} |\n"
        f"| **Cost Price** | ${r['cost_price_usd']} |\n"
        f"| **Shelf Location** | {r['shelf_location']} |\n"
        f"| **Category** | {r['category']} |"
        f"{expiry_line}"
    )


# ══════════════════════════════════════════════════════════════════
# DAILY BRIEFING
# ══════════════════════════════════════════════════════════════════

def format_daily_briefing() -> str:
    today = date.today().strftime("%A, %d %B %Y")

    df_stock = pd.read_sql_query("""
        SELECT generic_name, brand_name, quantity_in_stock, reorder_level,
               ROUND((quantity_in_stock::numeric/NULLIF(reorder_level,0))*100,0) AS pct
        FROM inventory WHERE quantity_in_stock <= reorder_level
        ORDER BY pct ASC LIMIT 5
    """, get_engine())
    df_exp = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, b.batch_number,
               (b.expiry_date::date - CURRENT_DATE)::INTEGER AS days_left,
               b.quantity_remaining
        FROM batches b JOIN inventory i ON b.product_id = i.product_id
        WHERE (b.expiry_date::date - CURRENT_DATE) <= 30
        ORDER BY days_left ASC LIMIT 5
    """, get_engine())
    df_rev = pd.read_sql_query("""
        SELECT ROUND(SUM(total_amount)::numeric,2) AS revenue,
               COUNT(*) AS txns, SUM(quantity_sold) AS units
        FROM transactions
        WHERE date = (SELECT MAX(date) FROM transactions)
    """, get_engine())
    df_avg = pd.read_sql_query("""
        SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily
        FROM (SELECT date, SUM(total_amount) AS daily_rev
              FROM transactions GROUP BY date) t
    """, get_engine())

    cat_tz = timezone(timedelta(hours=2))
    hour   = datetime.now(tz=cat_tz).hour
    tod    = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 17 else "Good Evening")

    lines = [f"# 🌅 {tod}! Daily Briefing — {today}\n"]

    rev = df_rev.iloc[0]
    avg = df_avg.iloc[0]['avg_daily']
    trend = "📈 above" if rev['revenue'] > avg else "📉 below"
    lines.append("## 💰 Yesterday's Revenue")
    lines.append(f"**${rev['revenue']:,.2f}** ({rev['txns']} transactions, {rev['units']} units)")
    lines.append(f"30-day avg: **${avg:,.2f}** — Yesterday was {trend} average\n")

    if df_stock.empty:
        lines.append("## ✅ Stock Levels\nAll products above reorder level.\n")
    else:
        lines.append(f"## 🔴 Low Stock — {len(df_stock)} drug(s) need reordering")
        lines.append("| Drug | Brand | Stock | Reorder | % |\n|---|---|---|---|---|")
        for _, r in df_stock.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['pct']:.0f}% |")
        lines.append("")

    if df_exp.empty:
        lines.append("## ✅ Expiry Status\nNo batches expiring within 30 days.\n")
    else:
        lines.append(f"## 🚨 Urgent Expiry — {len(df_exp)} batch(es) expiring within 30 days")
        lines.append("| Drug | Brand | Batch | Days Left | Qty |\n|---|---|---|---|---|")
        for _, r in df_exp.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | **{r['days_left']}** | {r['quantity_remaining']} |")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# REORDER ACTION LIST
# ══════════════════════════════════════════════════════════════════

def format_reorder_list() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock,
               i.reorder_level, i.category,
               COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,1), 0) AS avg_daily_sales,
               (i.reorder_level * 2 - i.quantity_in_stock) AS suggested_order
        FROM inventory i
        LEFT JOIN transactions t ON i.product_id = t.product_id
        WHERE i.quantity_in_stock <= i.reorder_level
        GROUP BY i.product_id, i.generic_name, i.brand_name,
                 i.quantity_in_stock, i.reorder_level, i.category
        ORDER BY (i.quantity_in_stock::float/NULLIF(i.reorder_level,1)) ASC
    """, get_engine())

    if df.empty:
        return "✅ All products are above reorder level. No procurement action needed."

    header  = f"## 📋 Procurement Action List — {len(df)} drug(s) to reorder\n\n"
    header += "| Drug | Brand | Current Stock | Reorder Level | Avg Daily Sales | Suggested Order | Category |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | "
        f"{r['reorder_level']} | {r['avg_daily_sales']} units/day | "
        f"**{max(int(r['suggested_order']),1)}** units | {r['category']} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows) + "\n\n*Suggested order = 2× reorder level minus current stock.*"


# ══════════════════════════════════════════════════════════════════
# REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════

def format_revenue_forecast() -> str:
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
    """, get_engine())
    df_daily = pd.read_sql_query("""
        SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily_revenue
        FROM (SELECT date, SUM(total_amount) AS daily_rev
              FROM transactions GROUP BY date) t
    """, get_engine())

    avg_daily_rev = float(df_daily.iloc[0]['avg_daily_revenue'])
    forecast_30   = round(avg_daily_rev * 30, 2)
    forecast_90   = round(avg_daily_rev * 90, 2)

    lines = ["## 📈 Revenue & Stock Forecast\n"]
    lines.append(f"**Average Daily Revenue:** ${avg_daily_rev:,.2f}")
    lines.append(f"**30-Day Revenue Forecast:** ${forecast_30:,.2f}")
    lines.append(f"**90-Day Revenue Forecast:** ${forecast_90:,.2f}\n")
    lines.append("**Days of Stock Remaining (Top 15 by value):**\n")
    lines.append("| Drug | Brand | Stock | Avg Daily Sales | Days Remaining |\n|---|---|---|---|---|")
    for _, r in df.iterrows():
        if r['avg_daily'] > 0:
            days = round(r['quantity_in_stock'] / r['avg_daily'])
            flag = "🔴" if days < 30 else ("🟡" if days < 60 else "🟢")
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | {r['avg_daily']}/day | {flag} **{days} days** |")
        else:
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | No sales | ∞ |")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# STOCK RECONCILIATION
# ══════════════════════════════════════════════════════════════════

def format_reconciliation(drug_name=None) -> str:
    drug_filter = ""
    params = []
    if drug_name:
        drug_filter = "WHERE LOWER(i.generic_name) LIKE %s"
        params = [f"%{drug_name.lower()}%"]
    df = pd.read_sql_query(f"""
        SELECT i.generic_name, i.brand_name,
               SUM(b.quantity_received) AS total_received,
               SUM(t.quantity_sold)     AS total_sold,
               i.quantity_in_stock      AS current_stock,
               (SUM(b.quantity_received) - COALESCE(SUM(t.quantity_sold),0) - i.quantity_in_stock)
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
    """, get_engine(), params=params if params else None)

    if df.empty:
        return "✅ Stock reconciliation is clean — no significant discrepancies found."

    header  = "## ⚠️ Stock Reconciliation — Discrepancies Found\n\n"
    header += "| Drug | Brand | Received | Sold | Current Stock | Discrepancy |\n|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        flag = "🔴" if abs(r['discrepancy']) > 20 else "🟡"
        rows.append(
            f"| {r['generic_name']} | {r['brand_name']} | {r['total_received']:.0f} | "
            f"{r['total_sold']:.0f} | {r['current_stock']} | {flag} **{r['discrepancy']:.0f}** |"
        )
    return header + "\n".join(rows) + "\n\n*Discrepancy = Received − Sold − Current Stock. Non-zero may indicate theft, wastage or data entry errors.*"


# ══════════════════════════════════════════════════════════════════
# MULTI-DRUG INTERACTION CHECK
# ══════════════════════════════════════════════════════════════════

def format_multi_interaction(question: str):
    keywords = extract_keywords(question)
    if len(keywords) < 2:
        return None
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
    return run_cypher(cypher, {"drugs": [k.lower() for k in keywords]})


# ══════════════════════════════════════════════════════════════════
# NEO4J CLINICAL QUERIES
# ══════════════════════════════════════════════════════════════════

def query_neo4j_interaction(question: str):
    keywords = extract_keywords(question)
    search_term = keywords[0] if keywords else get_search_term(question)
    cypher = """
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) CONTAINS toLower($search)
           OR toLower(b.generic_name) CONTAINS toLower($search)
        RETURN a.generic_name AS drug_a, b.generic_name AS drug_b,
               r.severity AS severity, r.description AS description,
               r.recommendation AS recommendation
        ORDER BY CASE r.severity WHEN 'Major' THEN 1
                 WHEN 'Moderate' THEN 2 WHEN 'Minor' THEN 3 ELSE 4 END
        LIMIT 5
    """
    return run_cypher(cypher, {"search": search_term})


def query_neo4j_drug_info(question: str):
    keywords    = extract_keywords(question)
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


# ══════════════════════════════════════════════════════════════════
# CLINICAL ANSWER GENERATION — GPT only for clinical queries
# ══════════════════════════════════════════════════════════════════

CLINICAL_DISCLAIMER = (
    "\n\n---\n⚠️ **Clinical Disclaimer:** This information is sourced from the pharmacy "
    "knowledge base. Always verify drug interactions, dosages and contraindications "
    "with a qualified pharmacist before dispensing."
)

CLINICAL_SYSTEM_PROMPT = """You are a pharmacy data assistant at Sunrise Pharmacy, Harare, Zimbabwe.
You are given STRUCTURED DATA retrieved from the pharmacy knowledge graph.
Your ONLY job is to summarise that data clearly for pharmacy staff.

ABSOLUTE RULES:
1. Use ONLY the data provided below. Never add information from your training knowledge.
2. If the data does not contain the answer, say exactly: "This information is not available in our knowledge base."
3. Never invent, guess or infer drug names, doses, quantities, interactions or clinical facts.
4. Never add interactions, contraindications or side effects not explicitly in the data.
5. Keep the answer to 3-5 sentences. Be precise and factual.
6. For interactions, always state the exact severity level from the data (Minor/Moderate/Major).
7. End with: "Source: drug knowledge graph" or "Source: drug interaction knowledge graph".
"""


def generate_clinical_answer(question, intent, source, data, conversation_history=None):
    """GPT is ONLY called for clinical queries. All operational queries use SQL formatters."""
    if not data:
        if intent == "interaction":
            return (
                "No recorded interaction found between these drugs in our knowledge base. "
                "This does not confirm safety — always verify with a clinical pharmacist "
                "or a current drug interaction reference before dispensing."
                + CLINICAL_DISCLAIMER
            )
        return (
            "No information found for that drug in our knowledge base. "
            "Please check the drug name." + CLINICAL_DISCLAIMER
        )

    messages = [{"role": "system", "content": CLINICAL_SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = (
        f"RETRIEVED DATA FROM KNOWLEDGE BASE:\n{json.dumps(data, indent=2)}\n\n"
        f"QUESTION FROM PHARMACY STAFF: {question}\n\n"
        "Summarise the above data to answer the question. Use ONLY what is in the data above."
    )
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=400
    )
    result = response.choices[0].message.content
    # Avoid double disclaimer if GPT already wrote one
    if "Clinical Disclaimer" in result:
        return result
    return result + CLINICAL_DISCLAIMER


# ══════════════════════════════════════════════════════════════════
# GREETING & SYSTEM RESPONSES
# ══════════════════════════════════════════════════════════════════

def _cat_hour() -> int:
    """Current hour in Central Africa Time (UTC+2)."""
    return datetime.now(tz=timezone(timedelta(hours=2))).hour


def get_greeting_response(question: str = "") -> str:
    q = question.lower().strip().rstrip("!.,?")
    # Map common greetings to a natural echo
    echo_map = {
        "good morning": "Good morning",
        "morning":      "Good morning",
        "good afternoon": "Good afternoon",
        "afternoon":    "Good afternoon",
        "good evening": "Good evening",
        "evening":      "Good evening",
        "good night":   "Good night",
        "hi":  "Hi",
        "hey": "Hey",
        "hello": "Hello",
        "howzit": "Howzit",
        "yo":  "Hey",
        "sup": "Hey",
    }
    opener = next((v for k, v in echo_map.items() if q.startswith(k)), "Hello")
    return (
        f"{opener}! I'm your Sunrise Pharmacy Assistant. "
        "Ask me about stock levels, expiry dates, sales, suppliers, or drug interactions — "
        "whatever you need. How can I help?"
    )


THANKS_RESPONSE   = "You're welcome! Feel free to ask anytime. 😊"
FAREWELL_RESPONSE = "Goodbye! Come back anytime you need help. 👋"
OUT_OF_SCOPE_RESPONSES = [
    "I'm here to help with pharmacy operations — stock, sales, expiry, suppliers and clinical queries. Could you rephrase with a pharmacy-related question?",
    "That's outside what I can help with. I focus on pharmacy data — inventory, transactions, drug information and supplier details.",
    "I specialise in pharmacy operations. For that question you may need a different tool. Anything pharmacy-related I can help with?",
]
_oos_idx = 0

def _out_of_scope_response() -> str:
    global _oos_idx
    r = OUT_OF_SCOPE_RESPONSES[_oos_idx % len(OUT_OF_SCOPE_RESPONSES)]
    _oos_idx += 1
    return r


# ══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════

def _combined_low_expiry() -> str:
    """Drugs that are BOTH low on stock AND have batches expiring within 90 days."""
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock, i.reorder_level,
               ROUND((i.quantity_in_stock::numeric/NULLIF(i.reorder_level,0))*100,0) AS stock_pct,
               MIN(b.expiry_date) AS nearest_expiry,
               (MIN(b.expiry_date::date) - CURRENT_DATE)::INTEGER AS days_to_expiry
        FROM inventory i
        JOIN batches b ON i.product_id = b.product_id
        WHERE i.quantity_in_stock <= i.reorder_level
          AND (b.expiry_date::date - CURRENT_DATE) <= 90
        GROUP BY i.product_id, i.generic_name, i.brand_name,
                 i.quantity_in_stock, i.reorder_level
        ORDER BY days_to_expiry ASC, stock_pct ASC
    """, get_engine())
    if df.empty:
        return "✅ No drugs are currently both low on stock AND expiring within 90 days."
    header  = f"**⚠️ {len(df)} drug(s) — LOW STOCK + EXPIRING SOON:**\n\n"
    header += "| Drug | Brand | Stock | Reorder | Stock% | Nearest Expiry | Days Left |\n|---|---|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | "
        f"{r['reorder_level']} | {r['stock_pct']:.0f}% | {str(r['nearest_expiry'])[:10]} | **{r['days_to_expiry']}** |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows)


def route_and_respond(question: str, conversation_history=None):
    """
    Returns (answer: str, source: str, mode: str)
    mode ∈ {"system", "operational", "clinical"}
    """
    corrected_q, correction_note = fuzzy_correct_question(question)

    result = classify_intent_with_tools(corrected_q, conversation_history)
    tool   = result["tool"]
    params = result["params"]

    def _wrap(answer):
        # Remove duplicate disclaimer if correction note + answer both have it
        if correction_note and "Clinical Disclaimer" in answer:
            return f"{correction_note}\n\n{answer}"
        return f"{correction_note}\n\n{answer}" if correction_note else answer

    # ── System responses ────────────────────────────────────────────
    if tool == "greeting":
        return get_greeting_response(question), "", "system"
    if tool == "thanks":
        return THANKS_RESPONSE, "", "system"
    if tool == "farewell":
        return FAREWELL_RESPONSE, "", "system"
    if tool == "out_of_scope":
        return _out_of_scope_response(), "", "system"
    if tool == "drug_summary":
        return _wrap(format_drug_summary(params.get("drug_name", ""))), "inventory + batch records", "operational"

    # ── Operational executors ───────────────────────────────────────
    if tool == "query_inventory":
        return _wrap(execute_query_inventory(params)), "inventory database", "operational"
    if tool == "query_sales":
        return _wrap(execute_query_sales(params)), "transaction records", "operational"
    if tool == "query_expiry":
        return _wrap(execute_query_expiry(params)), "batch records", "operational"
    if tool == "query_supplier":
        return _wrap(execute_query_supplier(params)), "supplier knowledge graph", "operational"
    if tool == "query_stats":
        return _wrap(format_stats()), "inventory database", "operational"
    if tool == "query_briefing":
        return format_daily_briefing(), "inventory + batch + transaction records", "operational"
    if tool == "query_combined_risk":
        return _combined_low_expiry(), "inventory + batch records", "operational"
    if tool == "query_reorder":
        return format_reorder_list(), "inventory + transaction records", "operational"
    if tool == "query_forecast":
        return format_revenue_forecast(), "inventory + transaction records", "operational"
    if tool == "query_reconciliation":
        drug = params.get("drug_name")
        return _wrap(format_reconciliation(drug)), "inventory + batch + transaction records", "operational"
    if tool == "query_alternatives":
        return _wrap(format_alternative(params.get("drug_name", ""))), "inventory database", "operational"

    # ── Clinical (Neo4j + GPT) ──────────────────────────────────────
    if tool == "query_clinical":
        drug_name  = params.get("drug_name", "")
        drug_name2 = params.get("drug_name_2")
        query_type = params.get("query_type", "drug_info")

        if query_type == "interaction":
            if drug_name2:
                data = format_multi_interaction(f"{drug_name} {drug_name2}")
                if not data:
                    data = query_neo4j_interaction(drug_name)
            else:
                data = query_neo4j_interaction(drug_name)
            answer = generate_clinical_answer(
                corrected_q, "interaction", "drug interaction knowledge graph", data, conversation_history
            )
            return _wrap(answer), "drug interaction knowledge graph", "clinical"
        else:
            data   = query_neo4j_drug_info(drug_name)
            answer = generate_clinical_answer(
                corrected_q, "drug_info", "drug knowledge graph", data, conversation_history
            )
            return _wrap(answer), "drug knowledge graph", "clinical"

    # ── Fallback clinical ───────────────────────────────────────────
    data   = query_neo4j_drug_info(corrected_q)
    answer = generate_clinical_answer(
        corrected_q, "drug_info", "drug knowledge graph", data, conversation_history
    )
    return _wrap(answer), "drug knowledge graph", "clinical"


# ══════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════

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
    for m in chat_history:
        role = "Staff" if m["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{m['content']}\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return filename


# ══════════════════════════════════════════════════════════════════
# DRUG SEARCH FILTER (left sidebar)
# ══════════════════════════════════════════════════════════════════

def filter_drugs(search_text):
    if not search_text or len(search_text) < 2:
        return gr.update(choices=DRUG_NAMES[:20])
    matches = [d for d in DRUG_NAMES if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches if matches else DRUG_NAMES[:20])


# ══════════════════════════════════════════════════════════════════
# RESPOND — core callback
# ══════════════════════════════════════════════════════════════════

def respond(message, chat_history, search_history):
    """
    chat_history is a list of {"role": ..., "content": ...} dicts.
    Returns 6-tuple: (cleared_msg, chat_history, search_history,
                      dropdown_update, history_md_update, brief_clear)
    """
    if not message or not message.strip():
        return "", chat_history, search_history, gr.update(), gr.update(), ""

    # Convert chat_history to plain conversation list for context
    conversation_history = [
        {"role": t["role"], "content": t["content"]}
        for t in (chat_history or [])
    ]

    try:
        answer, source, mode = route_and_respond(message, conversation_history)

        if mode == "system":
            full_answer = answer
        elif mode == "operational":
            header      = f"*📦 Operational data — {source}*\n\n" if source else ""
            full_answer = f"{header}{answer}"
        else:  # clinical
            header      = f"*🧪 Clinical data — {source}*\n\n" if source else ""
            full_answer = f"{header}{answer}"

    except Exception as e:
        full_answer = f"⚠️ Something went wrong. Please try rephrasing your question.\n\n*Details: {str(e)}*"

    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})

    search_history = list(search_history or [])
    if message not in search_history:
        search_history.insert(0, message)
    search_history = search_history[:15]
    history_md = "\n".join(f"- {h}" for h in search_history)

    return (
        "",
        chat_history,
        search_history,
        gr.update(choices=search_history, value=None),
        gr.update(value=history_md),
        ""   # clear brief_box on new question
    )


def drug_summary_respond(drug_name, chat_history, search_history):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update(), ""
    try:
        answer      = format_drug_summary(drug_name)
        full_answer = "*📦 Operational data — inventory + batch records*\n\n" + answer
    except Exception as e:
        full_answer = f"⚠️ Error: {str(e)}"

    label = f"Quick summary: {drug_name}"
    chat_history   = list(chat_history or [])
    chat_history.append({"role": "user",      "content": label})
    chat_history.append({"role": "assistant", "content": full_answer})

    search_history = list(search_history or [])
    if label not in search_history:
        search_history.insert(0, label)
    search_history = search_history[:15]
    history_md = "\n".join(f"- {h}" for h in search_history)

    return (
        chat_history,
        search_history,
        gr.update(choices=search_history, value=None),
        gr.update(value=history_md),
        ""
    )


def click_quick_question(question, chat_history, search_history):
    return respond(question, chat_history, search_history)


def reask_from_history(selected_question, chat_history, search_history):
    if not selected_question:
        return "", chat_history, search_history, gr.update(), gr.update(), ""
    return respond(selected_question, chat_history, search_history)


# ══════════════════════════════════════════════════════════════════
# QUICK QUESTIONS
# ══════════════════════════════════════════════════════════════════

QUICK_QUESTIONS = [
    "Good morning",
    "Which drugs are running low on stock?",
    "Which batches are expiring soon?",
    "What is the reorder list?",
    "What are the top selling drugs?",
    "Revenue forecast",
    "Do we have amoxicillin in stock?",
    "What interacts with metformin?"
]


# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════

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

        # ── LEFT sidebar — Drug Lookup ────────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🔍 Drug Lookup")
            drug_search = gr.Textbox(placeholder="Type e.g. amox...", label="Search drug name")
            drug_dropdown = gr.Dropdown(choices=DRUG_NAMES[:20], label="Select drug", interactive=True)
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

        # ── CENTRE — Chat ─────────────────────────────────────────
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(
                label="Pharmacy Assistant",
                height=460,
                autoscroll=True
            )
            brief_box = gr.Textbox(
                label="💡 Key Points",
                placeholder="Ask a question then click Brief for a plain-language summary",
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
                submit    = gr.Button("Ask", variant="primary", scale=1)
                brief_btn = gr.Button("💡 Brief", variant="secondary", scale=1)
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Voice Input (click to record)",
                    visible=True
                )
            with gr.Row():
                export_btn  = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file = gr.File(label="Download", scale=2, visible=False)

        # ── RIGHT sidebar — Quick Questions & History ─────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 💡 Quick Questions")
            quick_btns = [gr.Button(q, variant="secondary", size="sm") for q in QUICK_QUESTIONS]
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

    # ── State ────────────────────────────────────────────────────────
    search_history_state = gr.State([])

    # ── Text input wiring ────────────────────────────────────────────
    submit.click(respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])
    msg.submit(respond,
        [msg, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])

    # ── Voice input ──────────────────────────────────────────────────
    def transcribe_audio(audio_path, chat_history, search_history):
        if not audio_path:
            return "", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), ""
        try:
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            result = respond(transcript.text, chat_history, search_history)
            # result is 6-tuple; add audio reset as 7th
            return result[0], result[1], result[2], result[3], result[4], gr.update(value=None), result[5]
        except Exception:
            return "", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), ""

    audio_input.stop_recording(transcribe_audio,
        [audio_input, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, audio_input, brief_box])

    # ── History re-ask ───────────────────────────────────────────────
    history_dropdown.change(reask_from_history,
        [history_dropdown, chatbot, search_history_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])

    # ── Quick question buttons ───────────────────────────────────────
    for btn, question in zip(quick_btns, QUICK_QUESTIONS):
        btn.click(click_quick_question,
            [gr.Textbox(value=question, visible=False), chatbot, search_history_state],
            [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box])

    # ── Drug lookup sidebar ──────────────────────────────────────────
    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])
    drug_lookup_btn.click(drug_summary_respond,
        [drug_dropdown, chatbot, search_history_state],
        [chatbot, search_history_state, history_dropdown, history_display, brief_box],
        scroll_to_output=True)

    # ── Export ───────────────────────────────────────────────────────
    def do_export(chat_history):
        f = export_chat(chat_history)
        return gr.update(value=f, visible=True) if f else gr.update(visible=False)

    export_btn.click(do_export, [chatbot], [export_file])

    # ── Brief ────────────────────────────────────────────────────────
    def do_brief(chat_history):
        if not chat_history:
            return "No response yet — ask a question first."
        try:
            # Find the last assistant message regardless of format
            last_response = ""
            for entry in reversed(chat_history):
                if isinstance(entry, dict) and entry.get("role") == "assistant":
                    content = entry.get("content", "")
                    # Gradio can store content as a list of blocks — flatten to str
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    last_response = str(content)
                    break
                elif isinstance(entry, (list, tuple)) and len(entry) > 1:
                    last_response = str(entry[1] or "")
                    break
            if not last_response:
                return "No response yet — ask a question first."
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarise the following pharmacy data response in 2-3 clear sentences "
                        "for a pharmacy manager. Focus on the most important numbers and actionable insights. "
                        "Do not use bullet points or markdown.\n\nResponse:\n"
                        + last_response[:1500]
                        + "\n\nBrief summary:"
                    )
                }],
                temperature=0.0,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Could not generate brief: {str(e)}"

    brief_btn.click(do_brief, [chatbot], [brief_box])


demo.launch()