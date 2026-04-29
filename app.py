"""
Netrisyl Pharmacy Assistant — app.py
=====================================
Architecture (5 layers):
  1. Deterministic router  — handles ~80% of questions with zero GPT calls
  2. Entity extractor      — drug names, numbers, days, cities, months
  3. GPT router            — ambiguous questions only (~15%)
  4. SQL / Cypher executors — Python always runs the data queries
  5. Safe fallback         — clear message when nothing matches

Languages: English (en) | French (fr)
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════
import os, re, json
import pandas as pd
import gradio as gr
from psycopg2 import pool
from neo4j import GraphDatabase
from openai import OpenAI
from difflib import SequenceMatcher
from datetime import datetime, date, timezone, timedelta
from sqlalchemy import create_engine as _sa_create_engine


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & PHRASE DICTIONARIES
# ═══════════════════════════════════════════════════════════════════

# ── Trigger sets (EN + FR) ─────────────────────────────────────────
GREETING_TRIGGERS = {
    # EN
    "hi","hey","hello","morning","afternoon","evening","howzit",
    "good morning","good afternoon","good evening","good night",
    "how are you","what's up","whats up","what can you do",
    "who are you","what are you","yo","sup","start","help",
    # FR
    "bonjour","salut","bonsoir","bon matin","allô","comment allez-vous",
    "comment vas-tu","quoi de neuf","qui êtes-vous","aide",
}
THANKS_TRIGGERS = {
    # EN
    "thank you","thanks","thank","cheers","appreciated","great","ok",
    "okay","cool","perfect","noted","awesome","brilliant","nice",
    "wonderful","excellent","got it","understood","sure",
    # FR
    "merci","merci beaucoup","super","parfait","noté","compris",
    "d'accord","bien","excellent","génial",
}
FAREWELL_TRIGGERS = {
    # EN
    "bye","goodbye","good bye","see you","see ya","later","ciao",
    "take care","exit","quit","talk later","catch you later","farewell",
    # FR
    "au revoir","à bientôt","bonne journée","salut","à plus","bonne nuit",
}
SKIP_WORDS = {
    # EN
    "what","which","who","where","when","how","why","is","are","was",
    "were","do","does","did","have","has","had","will","can","could",
    "should","would","the","a","an","in","on","at","for","of","to",
    "and","or","but","with","from","about","we","our","us","i","my",
    "me","stock","drug","drugs","medicine","medicines","pharmacy",
    "pharmacist","please","tell","show","give","find","get","check",
    "supplier","supply","supplies","order","source","batch","expiry",
    "soon","selling","sales","name","information","info","details",
    "need","want","medication","medications","tablet","capsule",
    "injection","anything","something","everything","vendor",
    "distributor","buy","purchase","procure",
    # FR
    "quel","quelle","quels","quelles","qui","où","quand","comment",
    "pourquoi","est","sont","avons","avez","ont","nous","vous","ils",
    "elles","un","une","des","les","le","la","en","sur","pour","avec",
    "par","sans","médicament","médicaments","pharmacie","pharmacien",
    "vente","ventes","fournisseur","lot","péremption",
}

# ── Day name mapping (EN + FR) → canonical English ────────────────
DAY_MAP = {
    "monday":"monday","tuesday":"tuesday","wednesday":"wednesday",
    "thursday":"thursday","friday":"friday","saturday":"saturday","sunday":"sunday",
    "lundi":"monday","mardi":"tuesday","mercredi":"wednesday",
    "jeudi":"thursday","vendredi":"friday","samedi":"saturday","dimanche":"sunday",
}

# ── Month name mapping (EN + FR) → month number ───────────────────
MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "janvier":1,"février":2,"mars":3,"avril":4,"mai":5,"juin":6,
    "juillet":7,"août":8,"septembre":9,"octobre":10,"novembre":11,"décembre":12,
}

# ── Category synonyms → canonical DB category ─────────────────────
CATEGORY_MAP = {
    "antibiotic":"Antibiotics","antibiotique":"Antibiotics","antibiotics":"Antibiotics",
    "analgesic":"Analgesics","analgesics":"Analgesics","painkiller":"Analgesics",
    "analgesique":"Analgesics","antidouleur":"Analgesics",
    "antihypertensive":"Antihypertensives","antihypertensives":"Antihypertensives",
    "antihypertenseur":"Antihypertensives",
    "antifungal":"Antifungals","antifungals":"Antifungals","antifongique":"Antifungals",
    "antidiabetic":"Antidiabetics","antidiabetics":"Antidiabetics",
    "antidiabétique":"Antidiabetics",
    "antimalarial":"Antimalarials","antimalarials":"Antimalarials",
    "antipaludique":"Antimalarials","antipaludéen":"Antimalarials",
    "antiretroviral":"Antiretrovirals","antiretrovirals":"Antiretrovirals",
    "antirétroviral":"Antiretrovirals","arv":"Antiretrovirals",
    "respiratory":"Respiratory","respiratoire":"Respiratory",
    "vitamin":"Vitamins/Supplements","vitamins":"Vitamins/Supplements",
    "vitamine":"Vitamins/Supplements","supplement":"Vitamins/Supplements",
    "gi":"GI medications","digestif":"GI medications","gastrointestinal":"GI medications",
}

# ── System responses (bilingual) ──────────────────────────────────
THANKS_RESPONSE = {
    "en": "You're welcome! Feel free to ask anytime. 😊",
    "fr": "De rien! N'hésitez pas à demander. 😊",
}
FAREWELL_RESPONSE = {
    "en": "Goodbye! Come back anytime you need help. 👋",
    "fr": "Au revoir! Revenez quand vous avez besoin d'aide. 👋",
}
OUT_OF_SCOPE_RESPONSES = {
    "en": [
        "I'm here to help with pharmacy operations — stock, sales, expiry, suppliers and clinical queries. Could you rephrase?",
        "That's outside what I can help with. I focus on pharmacy data — inventory, transactions, drug information and supplier details.",
        "I specialise in pharmacy operations. Anything pharmacy-related I can help with?",
    ],
    "fr": [
        "Je suis ici pour les opérations pharmaceutiques — stock, ventes, expiration, fournisseurs et clinique. Pouvez-vous reformuler?",
        "Cela dépasse ce que je peux aider. Je me concentre sur les données pharmaceutiques.",
        "Je me spécialise dans les opérations de pharmacie. Quelque chose de lié à la pharmacie?",
    ],
}
CLINICAL_DISCLAIMER = {
    "en": (
        "\n\n---\n⚠️ **Clinical Disclaimer:** This information is sourced from the pharmacy "
        "knowledge base. Always verify with a qualified pharmacist before dispensing."
    ),
    "fr": (
        "\n\n---\n⚠️ **Avertissement Clinique:** Ces informations proviennent de la base de "
        "données de la pharmacie. Vérifiez toujours avec un pharmacien qualifié avant de dispenser."
    ),
}
_oos_idx = 0
def _out_of_scope(lang: str = "en") -> str:
    global _oos_idx
    r = OUT_OF_SCOPE_RESPONSES[lang][_oos_idx % len(OUT_OF_SCOPE_RESPONSES[lang])]
    _oos_idx += 1
    return r


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — CONNECTIONS
# ═══════════════════════════════════════════════════════════════════

NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")

# ── Neo4j ─────────────────────────────────────────────────────────
def _make_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

_driver = _make_driver()
try:
    with _driver.session() as _s: _s.run("RETURN 1")
    print("Neo4j pre-warmed ✓")
except Exception as _e:
    print(f"Neo4j pre-warm failed: {_e}")

def run_cypher(cypher: str, params: dict = None) -> list:
    global _driver
    for attempt in range(2):
        try:
            with _driver.session() as s:
                return [dict(r) for r in s.run(cypher, **(params or {}))]
        except Exception:
            if attempt == 0: _driver = _make_driver()
            else: raise

# ── OpenAI ────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Supabase (psycopg2 pool + SQLAlchemy for pandas) ─────────────
_pool = None
def _get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(1, 10, SUPABASE_URL)
    return _pool
def get_conn():    return _get_pool().getconn()
def release_conn(c): _get_pool().putconn(c)

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = _sa_create_engine(SUPABASE_URL)
    return _engine

print("Supabase ready ✓")

# ── Drug catalogue ────────────────────────────────────────────────
_drugs_df = pd.read_sql_query(
    "SELECT generic_name, brand_name, category FROM inventory ORDER BY generic_name",
    get_engine()
)
DRUG_NAMES       = _drugs_df["generic_name"].tolist()
BRAND_TO_GENERIC = dict(zip(_drugs_df["brand_name"].str.lower(), _drugs_df["generic_name"]))


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — ENTITY EXTRACTOR  (Layer 2)
# Runs before routing. Extracts all useful slots from the question.
# ═══════════════════════════════════════════════════════════════════

class Entities:
    """Structured slots extracted from a question."""
    __slots__ = ("drug","number","day","category","city","month","keywords")
    def __init__(self):
        self.drug     = None   # matched generic drug name
        self.number   = 10     # numeric quantity, default 10
        self.day      = None   # canonical English day name
        self.category = None   # canonical DB category string
        self.city     = None   # city name (title-cased)
        self.month    = None   # month name (lowercase)
        self.keywords = []     # non-skip words from the question

    def __repr__(self):
        return (f"Entities(drug={self.drug}, number={self.number}, day={self.day}, "
                f"category={self.category}, city={self.city}, month={self.month})")


def _fuzzy_match_drug(text: str, threshold: int = 78) -> str | None:
    """Return the best-matching drug generic name, or None."""
    text = re.sub(r"['\u2019\u2018`]", "", text.lower().strip())
    # Exact generic
    for d in DRUG_NAMES:
        if text == d.lower(): return d
    # Exact brand → generic
    if text in BRAND_TO_GENERIC: return BRAND_TO_GENERIC[text]
    # Fuzzy generic
    best_s, best_m = 0, None
    for d in DRUG_NAMES:
        s = SequenceMatcher(None, text, d.lower()).ratio() * 100
        if s > best_s: best_s, best_m = s, d
    # Fuzzy brand → generic
    for brand, generic in BRAND_TO_GENERIC.items():
        s = SequenceMatcher(None, text, brand).ratio() * 100
        if s > best_s: best_s, best_m = s, generic
    return best_m if best_s >= threshold else None


def extract_entities(question: str) -> Entities:
    """Parse a question into structured entities."""
    e   = Entities()
    q   = re.sub(r"['\u2019?!,.]", "", question.lower()).strip()
    words = q.split()

    # Keywords (non-skip, length ≥ 4)
    e.keywords = [w for w in words if len(w) >= 4 and w not in SKIP_WORDS and not w.isdigit()]

    # Number
    nums = re.findall(r"\b\d+\b", q)
    if nums: e.number = int(nums[0])

    # Day
    for w in words:
        if w in DAY_MAP: e.day = DAY_MAP[w]; break

    # Month
    for w in words:
        if w in MONTH_MAP: e.month = w; break

    # Category
    for k, v in CATEGORY_MAP.items():
        if k in q: e.category = v; break

    # City (simple known-city list for Zimbabwe)
    for city in ["harare","bulawayo","mutare","kwekwe","gweru","masvingo",
                 "chinhoyi","bindura","marondera","chitungwiza"]:
        if city in q: e.city = city.title(); break

    # Drug — try each keyword, take first match
    for kw in e.keywords:
        # Strip possessive trailing s
        kw_clean = kw[:-1] if kw.endswith("s") and _fuzzy_match_drug(kw[:-1], 90) else kw
        m = _fuzzy_match_drug(kw_clean, threshold=82)
        if m: e.drug = m; break

    return e


def fuzzy_correct_question(question: str):
    """Auto-correct misspelled drug names. Returns (corrected_q, note)."""
    skip_extra = SKIP_WORDS | {
        "soon","please","could","would","anything","something",
        "find","list","show","tell","give","have","does","there",
        "that","this","will","about","from",
    }
    words = re.sub(r"['\u2019?!,.]", "", question).split()
    corrections, corrected = [], list(words)
    for i, word in enumerate(words):
        w = word.lower()
        w_clean = w[:-1] if w.endswith("s") and _fuzzy_match_drug(w[:-1], 90) else w
        w_clean = re.sub(r"[\u2019']s$", "", w_clean)
        if len(w_clean) < 4 or w_clean in skip_extra: continue
        m = _fuzzy_match_drug(w_clean, threshold=78)
        if m and m.lower() != w and m.lower() != w_clean:
            corrected[i] = m
            corrections.append(f"'{word}' → '{m}'")
    note = f"*(Auto-corrected: {', '.join(corrections)})*" if corrections else ""
    return " ".join(corrected), note


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — DETERMINISTIC ROUTER  (Layer 1)
# Handles ~80% of questions with zero GPT calls.
# Each rule is: (match_fn, intent_id, confidence)
# match_fn receives (q_lower: str, e: Entities) → bool
# ═══════════════════════════════════════════════════════════════════

def _any(q: str, *phrases) -> bool:
    """True if any phrase appears in q."""
    return any(p in q for p in phrases)

# Rules are evaluated in order — first match wins
DETERMINISTIC_RULES = [
    # ── Greetings / social ─────────────────────────────────────────
    (lambda q,e: any(q == t or q.startswith(t+" ") for t in GREETING_TRIGGERS),
     "greeting", 1.0),
    (lambda q,e: any(q == t or q.startswith(t+" ") for t in THANKS_TRIGGERS),
     "thanks", 1.0),
    (lambda q,e: any(q == t or q.startswith(t+" ") for t in FAREWELL_TRIGGERS),
     "farewell", 1.0),

    # ── Drug summary shortcut ──────────────────────────────────────
    (lambda q,e: q.startswith("quick summary:") or q.startswith("résumé rapide:"),
     "drug_summary", 1.0),

    # ── Inventory ─────────────────────────────────────────────────
    (lambda q,e: e.drug and _any(q,"in stock","do we have","avons-nous","en stock",
                                    "stock level","niveau de stock","check stock",
                                    "check inventory","vérifier le stock"),
     "stock_check", 1.0),
    (lambda q,e: _any(q,"running low","below reorder","low on stock","low stock",
                         "critical stock","need restock","reorder level","stock faible",
                         "réapprovisionner","rupture de stock","niveau bas"),
     "low_stock", 1.0),
    (lambda q,e: _any(q,"how many products","total inventory","inventory summary",
                         "inventory value","total inventory value","how many drugs",
                         "combien de produits","valeur du stock","résumé du stock"),
     "inventory_summary", 1.0),
    (lambda q,e: e.category and _any(q,"show me all","all drugs","list all",
                                        "drugs we carry","montrer tous","liste de",
                                        "tous les médicaments"),
     "category_browse", 1.0),
    (lambda q,e: _any(q,"cheapest","lowest price","most affordable","prix le plus bas",
                         "moins cher","abordable"),
     "cheapest_drugs", 1.0),
    (lambda q,e: _any(q,"most expensive","highest price","prix le plus élevé","plus cher"),
     "expensive_drugs", 1.0),
    (lambda q,e: _any(q,"highest margin","most profitable","best margin","highest profit",
                         "marge la plus élevée","plus rentable","meilleure marge"),
     "highest_margin", 1.0),
    (lambda q,e: e.drug and _any(q,"alternative","substitute","instead of","replace",
                                    "alternatives pour","substitut","remplacer"),
     "drug_alternatives", 1.0),

    # ── Sales ──────────────────────────────────────────────────────
    (lambda q,e: _any(q,"top selling","best selling","most sold","best sellers",
                         "meilleures ventes","plus vendu","meilleurs ventes"),
     "top_sellers", 1.0),
    (lambda q,e: _any(q,"least selling","worst selling","worst sellers","bottom sellers",
                         "lowest sales","moins vendu","pires ventes","ventes les plus faibles"),
     "worst_sellers", 1.0),
    (lambda q,e: _any(q,"yesterday","last day","hier","d'hier","ventes d'hier",
                         "chiffre d'hier","revenu d'hier"),
     "yesterday_sales", 1.0),
    (lambda q,e: _any(q,"this month","monthly revenue","month to date","revenue this month",
                         "ce mois","revenu du mois","chiffre du mois","ce mois-ci"),
     "this_month_sales", 1.0),
    (lambda q,e: _any(q,"how many units","total units sold","units sold in total",
                         "average daily revenue","avg daily revenue","overall sales summary",
                         "combien d'unités","unités vendues","revenu journalier moyen",
                         "résumé global des ventes"),
     "total_summary", 1.0),
    (lambda q,e: _any(q,"which day","best day","busiest day","highest revenue day",
                         "day of the week","quel jour","jour le plus","meilleur jour"),
     "best_day", 1.0),
    (lambda q,e: e.day is not None and _any(q,"sales on","revenue on","what happened on",
                                               "sales","ventes du","chiffre du",
                                               "comment avons-nous performé"),
     "day_sales", 1.0),
    (lambda q,e: _any(q,"customer type","by customer","walk-in","prescription","insurance",
                         "par type de client","type de clientèle"),
     "customer_type_sales", 1.0),

    # ── Expiry ─────────────────────────────────────────────────────
    (lambda q,e: _any(q,"expiring soon","expire soon","expiry alert","batches expiring",
                         "expire bientôt","périme bientôt","alerte expiration"),
     "expiry_soon", 1.0),
    (lambda q,e: e.drug and _any(q,"when does","when do","expire","expiry date",
                                    "expiration","quand expire","date d'expiration",
                                    "date de péremption","péremption"),
     "expiry_drug", 1.0),
    (lambda q,e: _any(q,"expires first","first to expire","nearest expiry",
                         "expire en premier","premier à expirer","expiration la plus proche"),
     "first_expiry", 1.0),
    (lambda q,e: e.month and _any(q,"expiring in","expire in","expires in","expiry in",
                                     "expiration en","périme en"),
     "expiry_month", 1.0),
    (lambda q,e: e.drug and _any(q,"how many batches","batches does","number of batches",
                                    "batch count","combien de lots","nombre de lots"),
     "batch_count_drug", 1.0),
    (lambda q,e: _any(q,"how many batches","number of batches","batch count",
                         "combien de lots","nombre de lots") and not e.drug,
     "batch_count_all", 1.0),
    (lambda q,e: _any(q,"more than","at least","over","plus de","au moins") and
                 _any(q,"batch","lot","lots","batches"),
     "multi_batch", 1.0),

    # ── Suppliers ──────────────────────────────────────────────────
    (lambda q,e: e.drug and _any(q,"who supplies","supplier for","who provides",
                                    "fournisseur de","qui fournit","qui approvisionne",
                                    "lead time for","délai pour"),
     "supplier_drug", 1.0),
    (lambda q,e: _any(q,"fastest supplier","fastest vendor","quickest supplier",
                         "shortest lead time","fournisseur le plus rapide",
                         "délai le plus court","livraison la plus rapide"),
     "fastest_supplier", 1.0),
    (lambda q,e: _any(q,"slowest supplier","longest lead time","slowest vendor",
                         "fournisseur le plus lent","délai le plus long"),
     "slowest_supplier", 1.0),
    (lambda q,e: _any(q,"how many suppliers","number of suppliers",
                         "combien de fournisseurs","nombre de fournisseurs"),
     "supplier_count", 1.0),
    (lambda q,e: e.city and _any(q,"suppliers in","vendors in","fournisseurs à",
                                    "fournisseurs de","vendeurs à"),
     "supplier_city", 1.0),
    (lambda q,e: _any(q,"payment terms","best payment","credit terms","who gives best terms",
                         "meilleures conditions de paiement","délai de paiement",
                         "conditions de crédit"),
     "payment_terms", 1.0),

    # ── Clinical ───────────────────────────────────────────────────
    (lambda q,e: e.drug and _any(q,"interacts with","drug interactions","what interacts",
                                    "interactions de","interactions avec","interagit avec"),
     "drug_interactions", 1.0),
    (lambda q,e: e.drug and _any(q,"safe with","safe to take","can it be taken",
                                    "sûr avec","peut-on prendre","compatible avec"),
     "drug_safety", 1.0),
    (lambda q,e: e.drug and _any(q,"side effects","adverse effects","effets secondaires",
                                    "effets indésirables"),
     "side_effects", 1.0),
    (lambda q,e: e.drug and _any(q,"dosage","dose","how much to give","posologie",
                                    "dose recommandée","quelle dose"),
     "dosage", 1.0),
    (lambda q,e: e.drug and _any(q,"contraindicated","contraindications","contre-indiqué",
                                    "contre-indications"),
     "contraindications", 1.0),
    (lambda q,e: e.drug and _any(q,"what is","used for","what are","tell me about",
                                    "information on","à quoi sert","parle moi de",
                                    "fiche médicale","qu'est-ce que"),
     "drug_info", 1.0),

    # ── Operational alerts ────────────────────────────────────────
    (lambda q,e: _any(q,"daily briefing","morning briefing","daily summary",
                         "anything i should know","start of day","what should i know",
                         "résumé quotidien","rapport du jour","ce que je dois savoir",
                         "début de journée","rapport journalier"),
     "daily_briefing", 1.0),
    (lambda q,e: _any(q,"reorder list","procurement list","what to order","order list",
                         "liste de réapprovisionnement","que commander","liste de commande",
                         "liste d'achats"),
     "reorder_list", 1.0),
    (lambda q,e: _any(q,"low and expiring","expiring and low","critical drugs",
                         "urgent attention","faible et expirant","médicaments critiques",
                         "attention urgente"),
     "combined_risk", 1.0),
    (lambda q,e: _any(q,"reconciliation","discrepancy","stock mismatch","reconcile",
                         "réconciliation","écart de stock","discordance"),
     "stock_reconciliation", 1.0),
    (lambda q,e: _any(q,"forecast","projection","predict","how long will stock last",
                         "prévision","projection de revenus","combien de temps durera"),
     "revenue_forecast", 1.0),
    (lambda q,e: e.drug and _any(q,"full summary","drug profile","résumé de",
                                    "fiche de","profil du médicament"),
     "drug_summary", 1.0),
]


def deterministic_route(q_lower: str, entities: Entities):
    """
    Layer 1: try every rule in order, return first match.
    Returns (intent_id, confidence) or (None, 0).
    """
    for match_fn, intent_id, conf in DETERMINISTIC_RULES:
        try:
            if match_fn(q_lower, entities):
                return intent_id, conf
        except Exception:
            continue
    return None, 0.0


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — GPT ROUTER  (Layer 3)
# Only called when deterministic routing returns None.
# GPT classifies intent + extracts params — Python runs the query.
# ═══════════════════════════════════════════════════════════════════

GPT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": (
                "Drug inventory — stock levels, prices, categories, profit margins. "
                "filter=below_reorder for low stock. sort_by=margin_desc for best margin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "category":  {"type": "string"},
                    "filter":    {"type": "string",
                                  "enum": ["below_reorder","all","cheapest","most_expensive"]},
                    "sort_by":   {"type": "string",
                                  "enum": ["stock_pct","price_asc","price_desc","name",
                                           "margin_desc","margin_asc"]},
                    "limit":     {"type": "integer"},
                },
                "required": ["filter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sales",
            "description": (
                "Sales transactions. period: all_time|last_day|last_week|current_month|"
                "best_day|total_summary|day_of_week|customer_type. "
                "direction: top|bottom. sort_by: revenue|units|transactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period":    {"type": "string",
                                  "enum": ["all_time","last_day","last_week","current_month",
                                           "best_day","total_summary","day_of_week","customer_type"]},
                    "direction": {"type": "string", "enum": ["top","bottom"]},
                    "sort_by":   {"type": "string", "enum": ["revenue","units","transactions"]},
                    "day_name":  {"type": "string"},
                    "limit":     {"type": "integer"},
                },
                "required": ["period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_expiry",
            "description": "Batch expiry. drug_name for specific drug. within_days for window. top_only for nearest. month_name for named month. count_only+min_batches for batch counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name":   {"type": "string"},
                    "within_days": {"type": "integer"},
                    "limit":       {"type": "integer"},
                    "top_only":    {"type": "boolean"},
                    "month_name":  {"type": "string"},
                    "count_only":  {"type": "boolean"},
                    "min_batches": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_supplier",
            "description": "Supplier info. sort_by=lead_time for speed ranking. sort_by=payment_terms for credit. city for city filter. drug_name for drug-specific supplier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "city":      {"type": "string"},
                    "sort_by":   {"type": "string", "enum": ["lead_time","name","payment_terms"]},
                    "direction": {"type": "string", "enum": ["asc","desc"]},
                    "limit":     {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_clinical",
            "description": "Clinical drug info — interactions, dosage, side effects, indications. query_type: interaction|drug_info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name":   {"type": "string"},
                    "drug_name_2": {"type": "string"},
                    "query_type":  {"type": "string", "enum": ["interaction","drug_info"]},
                },
                "required": ["drug_name","query_type"],
            },
        },
    },
]

GPT_SYSTEM_PROMPT = (
    "You are a pharmacy operations assistant. "
    "Given a staff question, call the MOST appropriate tool with precise parameters. "
    "For 'fastest vendor' or 'slowest supplier' — use query_supplier with sort_by=lead_time, drug_name=null. "
    "For 'is X safe with Y' — use query_clinical with query_type=interaction, both drug names. "
    "For questions unrelated to pharmacy — respond with NO tool call."
)

def gpt_route(question: str, conversation_history: list = None) -> dict:
    """
    Layer 3: GPT tool-calling for ambiguous questions.
    Returns {"intent": str, "params": dict} or {"intent": "out_of_scope", "params": {}}.
    """
    messages = [{"role": "system", "content": GPT_SYSTEM_PROMPT}]
    if conversation_history:
        last = next((m for m in reversed(conversation_history)
                     if isinstance(m, dict) and m.get("role") == "assistant"), None)
        if last:
            prev = str(last.get("content", ""))[:250]
            sort_hint = ""
            if "by revenue" in prev.lower(): sort_hint = " Previous sort: revenue."
            elif "by units" in prev.lower(): sort_hint = " Previous sort: units."
            messages.append({"role": "user",
                              "content": f"[Context from previous answer: {prev}{sort_hint}]"})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=GPT_TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=150,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return {
                "intent": tc.function.name,   # e.g. "query_inventory"
                "params": json.loads(tc.function.arguments),
            }
        return {"intent": "out_of_scope", "params": {}}
    except Exception as e:
        print(f"GPT route error: {e}")
        return {"intent": "out_of_scope", "params": {}}


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — SQL / CYPHER EXECUTORS  (Layer 4)
# Python always runs the data queries. GPT never touches numbers.
# ═══════════════════════════════════════════════════════════════════

def exec_stock_check(e: Entities) -> str:
    return _exec_inventory({"filter": "all", "drug_name": e.drug, "limit": 5})

def exec_low_stock(e: Entities) -> str:
    return _exec_inventory({"filter": "below_reorder", "limit": 20})

def exec_inventory_summary(e: Entities) -> str:
    df = pd.read_sql_query("""
        SELECT category, COUNT(*) AS drug_count,
               SUM(quantity_in_stock) AS total_units,
               ROUND(AVG(selling_price_usd)::numeric,2) AS avg_price,
               ROUND(SUM(quantity_in_stock*cost_price_usd)::numeric,2) AS inventory_value
        FROM inventory GROUP BY category ORDER BY inventory_value DESC
    """, get_engine())
    total_drugs = df["drug_count"].sum()
    total_value = df["inventory_value"].sum()
    out  = f"**Inventory Summary** — {total_drugs} products across {len(df)} categories\n\n"
    out += f"Total inventory value: **${total_value:,.2f}**\n\n"
    out += "| Category | Drugs | Total Units | Avg Price | Inv. Value |\n|---|---|---|---|---|\n"
    out += "\n".join(
        f"| {r['category']} | {r['drug_count']} | {r['total_units']} | "
        f"${r['avg_price']} | ${r['inventory_value']:,.2f} |"
        for _, r in df.iterrows()
    )
    return out

def exec_category_browse(e: Entities) -> str:
    return _exec_inventory({"filter": "all", "category": e.category or "", "limit": 50})

def exec_cheapest(e: Entities) -> str:
    return _exec_inventory({"filter": "cheapest", "sort_by": "price_asc", "limit": e.number})

def exec_expensive(e: Entities) -> str:
    return _exec_inventory({"filter": "most_expensive", "sort_by": "price_desc", "limit": e.number})

def exec_highest_margin(e: Entities) -> str:
    return _exec_inventory({"filter": "all", "sort_by": "margin_desc", "limit": e.number})

def exec_drug_alternatives(e: Entities) -> str:
    if not e.drug: return "❌ Please specify a drug name."
    result = pd.read_sql_query(
        "SELECT generic_name, category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1",
        get_engine(), params=(f"%{e.drug.lower()}%",)
    )
    if result.empty: return f"❌ **{e.drug}** not found in inventory."
    cat = result.iloc[0]["category"]
    df = pd.read_sql_query("""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, selling_price_usd, shelf_location
        FROM inventory
        WHERE category=%s AND LOWER(generic_name) NOT LIKE %s AND quantity_in_stock>0
        ORDER BY generic_name
    """, get_engine(), params=(cat, f"%{e.drug.lower()}%"))
    if df.empty: return f"❌ No in-stock alternatives for **{e.drug}** in {cat}."
    out  = f"**Alternatives to {e.drug}** (category: {cat})\n\n"
    out += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n|---|---|---|---|---|---|---|\n"
    out += "\n".join(
        f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
        f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |"
        for _, r in df.iterrows()
    )
    return out + "\n\n⚠️ **Clinical Note:** Therapeutic substitution requires pharmacist approval."

def exec_drug_summary(e: Entities) -> str:
    drug = e.drug or ""
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.formulation, i.strength,
               i.quantity_in_stock, i.reorder_level, i.selling_price_usd, i.cost_price_usd,
               i.shelf_location, i.category,
               MIN(b.expiry_date) AS nearest_expiry,
               (MIN(b.expiry_date::date) - CURRENT_DATE)::INTEGER AS days_to_expiry
        FROM inventory i LEFT JOIN batches b ON i.product_id = b.product_id
        WHERE LOWER(i.generic_name) LIKE LOWER(%s)
        GROUP BY i.product_id, i.generic_name, i.brand_name, i.formulation, i.strength,
                 i.quantity_in_stock, i.reorder_level, i.selling_price_usd, i.cost_price_usd,
                 i.shelf_location, i.category
        LIMIT 1
    """, get_engine(), params=(f"%{drug}%",))
    if df.empty: return f"❌ **{drug}** not found in inventory."
    r = df.iloc[0]
    status = "⚠️ LOW STOCK — reorder needed" if r["quantity_in_stock"] <= r["reorder_level"] else "✅ In Stock"
    exp_line = ""
    if r.get("days_to_expiry") is not None:
        d = int(r["days_to_expiry"])
        dt = str(r["nearest_expiry"])[:10]
        exp_line = (f"\n🚨 **URGENT:** Expires in {d} days ({dt})" if d <= 30
                    else f"\n⚠️ Nearest expiry: {dt} ({d} days)" if d <= 90
                    else f"\n📅 Nearest expiry: {dt} ({d} days)")
    return (
        f"**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}\n\n"
        "| Field | Value |\n|---|---|\n"
        f"| **Stock** | {r['quantity_in_stock']} units — {status} |\n"
        f"| **Reorder Level** | {r['reorder_level']} units |\n"
        f"| **Selling Price** | ${r['selling_price_usd']} |\n"
        f"| **Cost Price** | ${r['cost_price_usd']} |\n"
        f"| **Shelf Location** | {r['shelf_location']} |\n"
        f"| **Category** | {r['category']} |{exp_line}"
    )

# ── Sales executors ────────────────────────────────────────────────
def exec_top_sellers(e: Entities) -> str:
    return _exec_sales({"period":"all_time","direction":"top","sort_by":"units","limit":e.number})
def exec_worst_sellers(e: Entities) -> str:
    return _exec_sales({"period":"all_time","direction":"bottom","sort_by":"units","limit":e.number})
def exec_yesterday_sales(e: Entities) -> str:  return _exec_sales({"period":"last_day"})
def exec_this_month(e: Entities) -> str:        return _exec_sales({"period":"current_month"})
def exec_total_summary(e: Entities) -> str:     return _exec_sales({"period":"total_summary"})
def exec_best_day(e: Entities) -> str:          return _exec_sales({"period":"best_day"})
def exec_day_sales(e: Entities) -> str:
    return _exec_sales({"period":"day_of_week","day_name": e.day or "saturday"})
def exec_customer_type(e: Entities) -> str:     return _exec_sales({"period":"customer_type"})

# ── Expiry executors ───────────────────────────────────────────────
def exec_expiry_soon(e: Entities) -> str:
    return _exec_expiry({"within_days": 90, "limit": 20})
def exec_expiry_drug(e: Entities) -> str:
    return _exec_expiry({"drug_name": e.drug})
def exec_first_expiry(e: Entities) -> str:
    return _exec_expiry({"top_only": True, "within_days": 365})
def exec_expiry_month(e: Entities) -> str:
    return _exec_expiry({"month_name": e.month or ""})
def exec_batch_count_drug(e: Entities) -> str:
    return _exec_expiry({"drug_name": e.drug})
def exec_batch_count_all(e: Entities) -> str:
    return _exec_expiry({"count_only": True, "min_batches": 2})
def exec_multi_batch(e: Entities) -> str:
    return _exec_expiry({"count_only": True, "min_batches": e.number + 1})

# ── Supplier executors ─────────────────────────────────────────────
def exec_supplier_drug(e: Entities) -> str:
    return _exec_supplier({"drug_name": e.drug})
def exec_fastest_supplier(e: Entities) -> str:
    return _exec_supplier({"sort_by": "lead_time", "direction": "asc"})
def exec_slowest_supplier(e: Entities) -> str:
    return _exec_supplier({"sort_by": "lead_time", "direction": "desc"})
def exec_supplier_count(e: Entities) -> str:
    return _exec_supplier({})
def exec_supplier_city(e: Entities) -> str:
    return _exec_supplier({"city": e.city or ""})
def exec_payment_terms(e: Entities) -> str:
    return _exec_supplier({"sort_by": "payment_terms"})

# ── Clinical executors ─────────────────────────────────────────────
def exec_drug_interactions(e: Entities, lang: str = "en") -> str:
    data = _neo4j_interaction(e.drug or "")
    return _clinical_answer(f"interactions with {e.drug}", "interaction",
                             "drug interaction knowledge graph", data, lang=lang)
def exec_drug_safety(e: Entities, lang: str = "en") -> str:
    data = _neo4j_interaction(e.drug or "")
    return _clinical_answer(f"safe with {e.drug}", "interaction",
                             "drug interaction knowledge graph", data, lang=lang)
def exec_drug_info(e: Entities, lang: str = "en") -> str:
    data = _neo4j_drug_info(e.drug or "")
    return _clinical_answer(f"info about {e.drug}", "drug_info",
                             "drug knowledge graph", data, lang=lang)
def exec_side_effects(e: Entities, lang: str = "en") -> str:
    data = _neo4j_drug_info(e.drug or "")
    return _clinical_answer(f"side effects of {e.drug}", "drug_info",
                             "drug knowledge graph", data, lang=lang)
def exec_dosage(e: Entities, lang: str = "en") -> str:
    data = _neo4j_drug_info(e.drug or "")
    return _clinical_answer(f"dosage of {e.drug}", "drug_info",
                             "drug knowledge graph", data, lang=lang)
def exec_contraindications(e: Entities, lang: str = "en") -> str:
    data = _neo4j_drug_info(e.drug or "")
    return _clinical_answer(f"contraindications of {e.drug}", "drug_info",
                             "drug knowledge graph", data, lang=lang)

# ── Operational executors ──────────────────────────────────────────
def exec_daily_briefing(e: Entities, lang: str = "en") -> str:
    return _exec_daily_briefing(lang)
def exec_reorder_list(e: Entities) -> str:
    return _exec_reorder()
def exec_combined_risk(e: Entities) -> str:
    return _exec_combined_risk()
def exec_stock_reconciliation(e: Entities) -> str:
    return _exec_reconciliation()
def exec_revenue_forecast(e: Entities) -> str:
    return _exec_forecast()

# ── Intent → executor map ──────────────────────────────────────────
# NOTE: clinical and briefing executors need `lang` — handled in dispatch
INTENT_EXECUTOR_MAP = {
    "stock_check":          exec_stock_check,
    "low_stock":            exec_low_stock,
    "inventory_summary":    exec_inventory_summary,
    "category_browse":      exec_category_browse,
    "cheapest_drugs":       exec_cheapest,
    "expensive_drugs":      exec_expensive,
    "highest_margin":       exec_highest_margin,
    "drug_alternatives":    exec_drug_alternatives,
    "drug_summary":         exec_drug_summary,
    "top_sellers":          exec_top_sellers,
    "worst_sellers":        exec_worst_sellers,
    "yesterday_sales":      exec_yesterday_sales,
    "this_month_sales":     exec_this_month,
    "total_summary":        exec_total_summary,
    "best_day":             exec_best_day,
    "day_sales":            exec_day_sales,
    "customer_type_sales":  exec_customer_type,
    "expiry_soon":          exec_expiry_soon,
    "expiry_drug":          exec_expiry_drug,
    "first_expiry":         exec_first_expiry,
    "expiry_month":         exec_expiry_month,
    "batch_count_drug":     exec_batch_count_drug,
    "batch_count_all":      exec_batch_count_all,
    "multi_batch":          exec_multi_batch,
    "supplier_drug":        exec_supplier_drug,
    "fastest_supplier":     exec_fastest_supplier,
    "slowest_supplier":     exec_slowest_supplier,
    "supplier_count":       exec_supplier_count,
    "supplier_city":        exec_supplier_city,
    "payment_terms":        exec_payment_terms,
    "drug_interactions":    exec_drug_interactions,
    "drug_safety":          exec_drug_safety,
    "drug_info":            exec_drug_info,
    "side_effects":         exec_side_effects,
    "dosage":               exec_dosage,
    "contraindications":    exec_contraindications,
    "daily_briefing":       exec_daily_briefing,
    "reorder_list":         exec_reorder_list,
    "combined_risk":        exec_combined_risk,
    "stock_reconciliation": exec_stock_reconciliation,
    "revenue_forecast":     exec_revenue_forecast,
}

CLINICAL_INTENTS = {
    "drug_interactions","drug_safety","drug_info",
    "side_effects","dosage","contraindications",
}
LANG_INTENTS = CLINICAL_INTENTS | {"daily_briefing"}


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — INTERNAL SQL / CYPHER HELPERS
# These are the actual query implementations called by executors above.
# ═══════════════════════════════════════════════════════════════════

def _exec_inventory(params: dict) -> str:
    filt      = params.get("filter", "all")
    drug_name = params.get("drug_name")
    category  = params.get("category")
    limit     = max(1, min(params.get("limit", 10), 50))
    sort_by   = params.get("sort_by", "name")
    sort_map  = {
        "stock_pct":"stock_pct ASC","price_asc":"selling_price_usd ASC",
        "price_desc":"selling_price_usd DESC","name":"generic_name ASC",
        "margin_desc":"margin DESC","margin_asc":"margin ASC",
    }
    order  = sort_map.get(sort_by, "generic_name ASC")
    where, sql_params = [], []

    if filt == "below_reorder": where.append("quantity_in_stock <= reorder_level"); order = "stock_pct ASC"
    elif filt == "cheapest":    order = "selling_price_usd ASC"
    elif filt == "most_expensive": order = "selling_price_usd DESC"

    if drug_name:
        where.append("(LOWER(generic_name) LIKE %s OR LOWER(brand_name) LIKE %s)")
        sql_params += [f"%{drug_name.lower()}%", f"%{drug_name.lower()}%"]
    if category:
        matched = next((v for k,v in CATEGORY_MAP.items() if k in category.lower()), category)
        where.append("category = %s"); sql_params.append(matched)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd, cost_price_usd,
               shelf_location, category,
               ROUND((quantity_in_stock::numeric/NULLIF(reorder_level,0))*100,0) AS stock_pct,
               ROUND(((selling_price_usd-cost_price_usd)/NULLIF(selling_price_usd,0)*100)::numeric,1) AS margin
        FROM inventory {where_sql} ORDER BY {order} LIMIT %s
    """
    sql_params.append(limit)
    df = pd.read_sql_query(sql, get_engine(), params=tuple(sql_params))
    if df.empty:
        return "✅ All products above reorder level." if filt == "below_reorder" else "❌ No drugs found matching that criteria."

    if filt == "below_reorder":
        out  = f"⚠️ **{len(df)} drug(s) at or below reorder level:**\n\n"
        out += "| Drug | Brand | Stock | Reorder | % | Category |\n|---|---|---|---|---|---|\n"
        return out + "\n".join(
            f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | "
            f"{r['reorder_level']} | {r['stock_pct']:.0f}% | {r['category']} |"
            for _, r in df.iterrows()
        )

    if drug_name:
        exact = df[df["generic_name"].str.lower() == drug_name.lower()]
        if not exact.empty: df = exact.reset_index(drop=True)
    if drug_name and len(df) == 1:
        r = df.iloc[0]
        flag = "⚠️ LOW STOCK" if r["quantity_in_stock"] <= r["reorder_level"] else "✅ In Stock"
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
        cat_name = df.iloc[0]["category"]
        out  = f"**{cat_name}** — {len(df)} drugs\n\n"
        out += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n|---|---|---|---|---|---|---|\n"
        return out + "\n".join(
            f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
            f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |"
            for _, r in df.iterrows()
        )
    if sort_by in ("margin_desc","margin_asc"):
        label = "Highest" if sort_by == "margin_desc" else "Lowest"
        out  = f"**{label} margin drugs:**\n\n"
        out += "| Drug | Brand | Sell Price | Cost Price | Margin% | Stock |\n|---|---|---|---|---|---|\n"
        return out + "\n".join(
            f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | "
            f"${r['cost_price_usd']} | {r['margin']}% | {r['quantity_in_stock']} |"
            for _, r in df.iterrows()
        )
    label = ("Cheapest" if filt=="cheapest" else
             "Most expensive" if filt=="most_expensive" else
             f"Drugs matching '{drug_name}'" if drug_name else f"Top {limit} drugs")
    out  = f"**{label}:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
    return out + "\n".join(
        f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | "
        f"{r['quantity_in_stock']} | {r['shelf_location']} |"
        for _, r in df.iterrows()
    )


def _exec_sales(params: dict) -> str:
    period    = params.get("period","all_time")
    direction = params.get("direction","top")
    sort_by   = params.get("sort_by","units")
    day_name  = params.get("day_name","")
    limit     = max(1, min(params.get("limit", 10), 50))

    if period == "customer_type": return _sales_customer_type()
    if period == "last_day":      return _sales_last_day()
    if period == "last_week":     return _sales_last_week()
    if period == "current_month": return _sales_current_month()
    if period == "best_day":      return _sales_best_day()
    if period == "total_summary": return _sales_total_summary()
    if period == "day_of_week" and day_name: return _sales_day_of_week(day_name)

    order    = "DESC" if direction == "top" else "ASC"
    label    = f"{'Top' if direction=='top' else 'Bottom'} {limit}"
    col_map  = {"revenue":"total_revenue","units":"total_units","transactions":"num_transactions"}
    sort_col = col_map.get(sort_by,"total_units")
    df = pd.read_sql_query(f"""
        SELECT i.brand_name, i.generic_name,
               SUM(t.quantity_sold) AS total_units,
               ROUND(SUM(t.total_amount)::numeric,2) AS total_revenue,
               COUNT(*) AS num_transactions
        FROM transactions t JOIN inventory i ON t.product_id=i.product_id
        GROUP BY i.brand_name, i.generic_name
        ORDER BY {sort_col} {order} LIMIT %s
    """, get_engine(), params=(limit,))
    out  = f"**{label} Selling Drugs** by {sort_by}\n\n"
    out += "| Rank | Brand | Generic | Units | Revenue | Transactions |\n|---|---|---|---|---|---|\n"
    return out + "\n".join(
        f"| {i+1} | {r['brand_name']} | {r['generic_name']} | "
        f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['num_transactions']} |"
        for i,(_, r) in enumerate(df.iterrows())
    )

def _sales_best_day() -> str:
    df = pd.read_sql_query(
        "SELECT TRIM(TO_CHAR(date::date,'Day')) AS day_name, EXTRACT(DOW FROM date::date)::INTEGER AS dow,"
        " COUNT(DISTINCT date::date) AS occ, ROUND(SUM(total_amount)::numeric,2) AS total_rev,"
        " ROUND(AVG(daily_rev)::numeric,2) AS avg_rev, SUM(quantity_sold) AS units, COUNT(*) AS txns"
        " FROM transactions t"
        " JOIN (SELECT date::date AS d, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date::date) dr"
        " ON t.date::date=dr.d GROUP BY day_name,dow ORDER BY avg_rev DESC", get_engine())
    if df.empty: return "No sales data available."
    best = str(df.iloc[0]["day_name"]).strip()
    rows = ["| Day | Avg Revenue | Total Revenue | Occurrences | Transactions | Units |","|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        dn = str(r["day_name"]).strip()
        rows.append(f"| {dn}{'⭐' if dn==best else ''} | **${r['avg_rev']:,.2f}** | "
                    f"${r['total_rev']:,.2f} | {r['occ']} | {r['txns']} | {r['units']} |")
    return f"**Revenue by Day of Week** — best day is **{best}**\n\n" + "\n".join(rows)

def _sales_current_month() -> str:
    df_t = pd.read_sql_query(
        "SELECT TO_CHAR(DATE_TRUNC('month',CURRENT_DATE),'Month YYYY') AS month_label,"
        " COUNT(*) AS transactions, SUM(quantity_sold) AS total_units,"
        " ROUND(SUM(total_amount)::numeric,2) AS total_revenue"
        " FROM transactions WHERE date::date>=DATE_TRUNC('month',CURRENT_DATE)", get_engine())
    df_d = pd.read_sql_query(
        "SELECT date::date AS day, COUNT(*) AS txns, SUM(quantity_sold) AS units,"
        " ROUND(SUM(total_amount)::numeric,2) AS revenue"
        " FROM transactions WHERE date::date>=DATE_TRUNC('month',CURRENT_DATE)"
        " GROUP BY date::date ORDER BY date::date DESC", get_engine())
    r = df_t.iloc[0]
    month = str(r["month_label"]).strip() if r["month_label"] else "This month"
    if not r["total_revenue"]: return f"No transactions recorded for {month} yet."
    out  = f"**{month} Revenue**\n\nTotal: **${r['total_revenue']:,.2f}** | Transactions: **{r['transactions']}** | Units: **{r['total_units']}**\n\n"
    if not df_d.empty:
        days = len(df_d); avg = round(float(r["total_revenue"])/days,2)
        out += f"Days recorded: **{days}** | Daily avg: **${avg:,.2f}**\n\n"
        out += "| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
        out += "\n".join(f"| {str(row['day'])[:10]} | {row['txns']} | {row['units']} | ${row['revenue']:,.2f} |"
                         for _, row in df_d.iterrows())
    return out

def _sales_total_summary() -> str:
    df = pd.read_sql_query(
        "SELECT COUNT(*) AS total_transactions, SUM(quantity_sold) AS total_units,"
        " ROUND(SUM(total_amount)::numeric,2) AS total_revenue,"
        " COUNT(DISTINCT date::date) AS trading_days,"
        " ROUND(AVG(daily_rev)::numeric,2) AS avg_daily_revenue,"
        " MIN(date::date) AS first_date, MAX(date::date) AS last_date"
        " FROM transactions"
        " JOIN (SELECT date::date AS d, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date::date) dr"
        " ON transactions.date::date=dr.d", get_engine())
    r = df.iloc[0]
    return (
        "**Overall Sales Summary**\n\n| Metric | Value |\n|---|---|\n"
        f"| Total Revenue | **${r['total_revenue']:,.2f}** |\n"
        f"| Total Units Sold | **{r['total_units']}** |\n"
        f"| Total Transactions | **{r['total_transactions']}** |\n"
        f"| Trading Days | {r['trading_days']} |\n"
        f"| Avg Daily Revenue | **${r['avg_daily_revenue']:,.2f}** |\n"
        f"| Date Range | {str(r['first_date'])[:10]} → {str(r['last_date'])[:10]} |\n"
    )

def _sales_customer_type() -> str:
    df = pd.read_sql_query("""
        SELECT customer_type, COUNT(*) AS num_transactions, SUM(quantity_sold) AS total_units,
               ROUND(SUM(total_amount)::numeric,2) AS total_revenue,
               ROUND((SUM(total_amount)*100.0/(SELECT SUM(total_amount) FROM transactions))::numeric,1) AS pct
        FROM transactions GROUP BY customer_type ORDER BY total_revenue DESC
    """, get_engine())
    out  = "**Sales by Customer Type**\n\n| Type | Transactions | Units | Revenue | % |\n|---|---|---|---|---|\n"
    out += "\n".join(f"| {r['customer_type']} | {r['num_transactions']} | {r['total_units']} | ${r['total_revenue']:,.2f} | {r['pct']}% |"
                     for _, r in df.iterrows())
    return out + f"\n\n**Total: ${df['total_revenue'].sum():,.2f}**"

def _sales_last_day() -> str:
    df_d  = pd.read_sql_query("SELECT date, COUNT(*) AS txns, SUM(quantity_sold) AS units, ROUND(SUM(total_amount)::numeric,2) AS revenue FROM transactions WHERE date=(SELECT MAX(date) FROM transactions) GROUP BY date", get_engine())
    df_dr = pd.read_sql_query("SELECT i.brand_name, i.generic_name, SUM(t.quantity_sold) AS units, ROUND(SUM(t.total_amount)::numeric,2) AS revenue FROM transactions t JOIN inventory i ON t.product_id=i.product_id WHERE t.date=(SELECT MAX(date) FROM transactions) GROUP BY i.brand_name,i.generic_name ORDER BY revenue DESC", get_engine())
    df_ct = pd.read_sql_query("SELECT customer_type, ROUND(SUM(total_amount)::numeric,2) AS revenue FROM transactions WHERE date=(SELECT MAX(date) FROM transactions) GROUP BY customer_type ORDER BY revenue DESC", get_engine())
    if df_d.empty: return "No transactions found."
    r = df_d.iloc[0]
    out  = f"**Sales for {str(r['date'])[:10]}** (last recorded day)\n\n"
    out += f"Transactions: **{r['txns']}** | Units: **{r['units']}** | Revenue: **${r['revenue']:,.2f}**\n\n"
    out += "**By Drug:**\n\n| Brand | Generic | Units | Revenue |\n|---|---|---|---|\n"
    out += "\n".join(f"| {row['brand_name']} | {row['generic_name']} | {row['units']} | ${row['revenue']:,.2f} |"
                     for _, row in df_dr.iterrows())
    out += "\n\n**By Customer Type:** " + " | ".join(f"{row['customer_type']}: ${row['revenue']:,.2f}"
                                                      for _, row in df_ct.iterrows())
    return out

def _sales_last_week() -> str:
    df = pd.read_sql_query("SELECT date, COUNT(*) AS txns, SUM(quantity_sold) AS units, ROUND(SUM(total_amount)::numeric,2) AS revenue FROM transactions WHERE date::date>=(SELECT MAX(date::date)-7 FROM transactions) GROUP BY date ORDER BY date DESC", get_engine())
    if df.empty: return "No transactions found for last week."
    out  = "**Last Week Sales**\n\n| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
    out += "\n".join(f"| {str(r['date'])[:10]} | {r['txns']} | {r['units']} | ${r['revenue']:,.2f} |"
                     for _, r in df.iterrows())
    return out + f"\n\n**Total: ${df['revenue'].sum():,.2f}**"

def _sales_day_of_week(day_name: str) -> str:
    day_num_map = {"monday":1,"tuesday":2,"wednesday":3,"thursday":4,"friday":5,"saturday":6,"sunday":0}
    dow = day_num_map.get(day_name.lower(), 6)
    df_s = pd.read_sql_query("""
        SELECT COUNT(DISTINCT t.date::date) AS num_days, COUNT(*) AS txns,
               SUM(t.quantity_sold) AS units, ROUND(SUM(t.total_amount)::numeric,2) AS revenue,
               ROUND(AVG(dr.rev)::numeric,2) AS avg_rev
        FROM transactions t
        JOIN (SELECT date::date AS d, SUM(total_amount) AS rev FROM transactions GROUP BY date::date) dr
        ON t.date::date=dr.d WHERE EXTRACT(DOW FROM t.date::date)=%(dow)s
    """, get_engine(), params={"dow": dow})
    df_d = pd.read_sql_query("""
        SELECT i.brand_name, i.generic_name, SUM(t.quantity_sold) AS units,
               ROUND(SUM(t.total_amount)::numeric,2) AS revenue, COUNT(*) AS txns
        FROM transactions t JOIN inventory i ON t.product_id=i.product_id
        WHERE EXTRACT(DOW FROM t.date::date)=%(dow)s
        GROUP BY i.brand_name,i.generic_name ORDER BY units DESC LIMIT 10
    """, get_engine(), params={"dow": dow})
    if df_d.empty: return f"No sales data found for {day_name.capitalize()}s."
    s = df_s.iloc[0]
    out  = f"**{day_name.capitalize()} Sales Summary** (across {s['num_days']} recorded {day_name.capitalize()}s)\n\n"
    out += f"Transactions: **{s['txns']}** | Units: **{s['units']}** | Revenue: **${s['revenue']:,.2f}** | Avg/day: **${s['avg_rev']:,.2f}**\n\n"
    out += "| Rank | Brand | Generic | Units | Revenue | Transactions |\n|---|---|---|---|---|---|\n"
    out += "\n".join(f"| {i+1} | {r['brand_name']} | {r['generic_name']} | {r['units']} | ${r['revenue']:,.2f} | {r['txns']} |"
                     for i,(_, r) in enumerate(df_d.iterrows()))
    return out


def _exec_expiry(params: dict) -> str:
    drug_name   = params.get("drug_name")
    within_days = params.get("within_days", 90)
    limit       = max(1, min(params.get("limit", 10), 50))
    top_only    = params.get("top_only", False)
    month_name  = params.get("month_name", "")
    count_only  = params.get("count_only", False)
    min_batches = params.get("min_batches", 2)

    if month_name:
        mn = MONTH_MAP.get(month_name.lower())
        if mn:
            df = pd.read_sql_query("""
                SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
                       b.quantity_remaining, (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days
                FROM batches b JOIN inventory i ON b.product_id=i.product_id
                WHERE EXTRACT(MONTH FROM b.expiry_date::date)=%s
                  AND EXTRACT(YEAR FROM b.expiry_date::date)>=EXTRACT(YEAR FROM CURRENT_DATE)
                ORDER BY b.expiry_date ASC
            """, get_engine(), params=(mn,))
            if df.empty: return f"✅ No batches expiring in {month_name.capitalize()}."
            out  = f"**Batches expiring in {month_name.capitalize()}** — {len(df)} found:\n\n"
            out += "| Drug | Brand | Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|---|---|\n"
            for _, r in df.iterrows():
                d = r["days"]
                flag = "🚨 URGENT" if d<30 else ("⚠️ Warning" if d<90 else "📅 Monitor")
                out += f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |\n"
            return out

    if count_only:
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, COUNT(b.batch_id) AS batch_count,
                   MIN(b.expiry_date) AS nearest_expiry, SUM(b.quantity_remaining) AS total_qty
            FROM inventory i JOIN batches b ON i.product_id=b.product_id
            GROUP BY i.product_id,i.generic_name,i.brand_name
            HAVING COUNT(b.batch_id)>=%s ORDER BY batch_count DESC
        """, get_engine(), params=(min_batches,))
        if df.empty: return f"No drugs found with {min_batches} or more batches."
        out  = f"**Drugs with {min_batches}+ batches:**\n\n| Drug | Brand | Batches | Nearest Expiry | Total Qty |\n|---|---|---|---|---|\n"
        return out + "\n".join(
            f"| {r['generic_name']} | {r['brand_name']} | **{r['batch_count']}** | {str(r['nearest_expiry'])[:10]} | {r['total_qty']} |"
            for _, r in df.iterrows()
        )

    if drug_name:
        df = pd.read_sql_query("""
            SELECT b.batch_number, b.expiry_date, b.quantity_remaining,
                   (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days
            FROM batches b JOIN inventory i ON b.product_id=i.product_id
            WHERE LOWER(i.generic_name) LIKE %s ORDER BY b.expiry_date ASC
        """, get_engine(), params=(f"%{drug_name.lower()}%",))
        if df.empty: return f"❌ No batch records found for {drug_name}."
        out  = f"**{drug_name} — {len(df)} batch(es):**\n\n| Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|\n"
        for _, r in df.iterrows():
            d = r["days"]; flag = "🚨 URGENT" if d<30 else ("⚠️ Warning" if d<90 else "✅ OK")
            out += f"| {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |\n"
        return out

    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
               b.quantity_remaining, (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days_remaining
        FROM batches b JOIN inventory i ON b.product_id=i.product_id
        WHERE (b.expiry_date::date-CURRENT_DATE)<=%s ORDER BY b.expiry_date ASC LIMIT %s
    """, get_engine(), params=(within_days, 1 if top_only else limit))
    if df.empty: return f"✅ No batches expiring within {within_days} days."
    if top_only:
        r = df.iloc[0]; d = r["days_remaining"]
        flag = "🚨 URGENT" if d<30 else ("⚠️ Warning" if d<90 else "📅 Monitor")
        return (
            f"**First to expire:** {r['generic_name']} ({r['brand_name']})\n\n"
            "| Field | Value |\n|---|---|\n"
            f"| Batch | {r['batch_number']} |\n"
            f"| Expiry Date | {str(r['expiry_date'])[:10]} |\n"
            f"| Days Remaining | **{d}** — {flag} |\n"
            f"| Qty Remaining | {r['quantity_remaining']} |\n"
        )
    out  = f"**Batches expiring within {within_days} days** — {len(df)} found:\n\n"
    out += "| Drug | Brand | Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        d = r["days_remaining"]; flag = "🚨 URGENT" if d<30 else ("⚠️ Warning" if d<90 else "📅 Monitor")
        out += f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |\n"
    return out


def _exec_supplier(params: dict) -> str:
    drug_name = params.get("drug_name")
    city      = params.get("city")
    sort_by   = params.get("sort_by","name")
    direction = params.get("direction","asc")

    if sort_by == "payment_terms" and not drug_name:
        results = run_cypher("MATCH (s:Supplier) RETURN DISTINCT s.name AS supplier, s.payment_terms AS payment_terms, s.lead_time AS lead_time_days, s.city AS city, s.contact AS contact ORDER BY s.payment_terms DESC LIMIT 10")
        if not results: return "❌ No supplier payment terms found."
        out  = "**Suppliers by Payment Terms:**\n\n| Supplier | Payment Terms | Lead Time | City | Contact |\n|---|---|---|---|---|\n"
        return out + "\n".join(f"| {r['supplier']} | **{r['payment_terms']}** | {r['lead_time_days']} days | {r['city']} | {r['contact']} |" for r in results)

    if sort_by == "lead_time" and not drug_name:
        order = "ASC" if direction != "desc" else "DESC"
        label = "fastest" if order=="ASC" else "slowest"
        results = run_cypher(f"MATCH (s:Supplier) RETURN s.name AS supplier, s.lead_time AS lead_time_days, s.city AS city, s.contact AS contact ORDER BY s.lead_time {order} LIMIT 5")
        if not results: return "❌ No supplier information found."
        out  = f"**Suppliers by lead time ({label} first):**\n\n| Supplier | Lead Time | City | Contact |\n|---|---|---|---|\n"
        return out + "\n".join(f"| {r['supplier']} | {r['lead_time_days']} days | {r['city']} | {r['contact']} |" for r in results)

    if city:
        results = run_cypher("MATCH (s:Supplier) WHERE toLower(s.city) CONTAINS toLower($city) RETURN s.city AS city, count(s) AS cnt, collect(s.name) AS suppliers ORDER BY cnt DESC", {"city": city})
        if not results: return f"❌ No suppliers found in {city}."
        out  = f"**Suppliers in {city.title()}:**\n\n| City | Count | Suppliers |\n|---|---|---|\n"
        return out + "\n".join(f"| {r['city']} | {r['cnt']} | {', '.join(r['suppliers'])} |" for r in results)

    if drug_name:
        results = run_cypher("""
            MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
            WHERE toLower(d.generic_name) CONTAINS toLower($search)
            RETURN DISTINCT d.generic_name AS drug, s.name AS supplier,
                   s.contact AS contact, s.phone AS phone, s.city AS city,
                   s.lead_time AS lead_time_days, s.payment_terms AS payment_terms
            LIMIT 5
        """, {"search": drug_name})
        if not results:
            df_cat = pd.read_sql_query("SELECT DISTINCT generic_name FROM inventory WHERE LOWER(category) LIKE %s LIMIT 5", get_engine(), params=(f"%{drug_name.lower()}%",))
            if not df_cat.empty:
                results = run_cypher("MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier) WHERE d.generic_name IN $drugs RETURN DISTINCT d.generic_name AS drug, s.name AS supplier, s.contact AS contact, s.phone AS phone, s.city AS city, s.lead_time AS lead_time_days, s.payment_terms AS payment_terms ORDER BY s.lead_time ASC LIMIT 5", {"drugs": df_cat["generic_name"].tolist()})
        if not results: return f"❌ No supplier found for {drug_name}."
        if len(results) == 1:
            r = results[0]
            return (f"**Supplier for {r['drug']}:**\n\n| Field | Value |\n|---|---|\n"
                    f"| Supplier | **{r['supplier']}** |\n| Contact | {r['contact']} |\n"
                    f"| Phone | {r['phone']} |\n| City | {r['city']} |\n"
                    f"| Lead Time | {r['lead_time_days']} days |\n| Payment Terms | {r['payment_terms']} |\n")
        out  = f"**Suppliers for {drug_name}:**\n\n| Drug | Supplier | City | Lead Time | Contact |\n|---|---|---|---|---|\n"
        return out + "\n".join(f"| {r['drug']} | {r['supplier']} | {r['city']} | {r['lead_time_days']} days | {r['contact']} |" for r in results)

    results = run_cypher("MATCH (s:Supplier) RETURN s.city AS city, count(s) AS cnt, collect(s.name) AS suppliers ORDER BY cnt DESC")
    total   = sum(r["cnt"] for r in results)
    out     = f"**{total} suppliers** across {len(results)} cities:\n\n| City | Count | Suppliers |\n|---|---|---|\n"
    return out + "\n".join(f"| {r['city']} | {r['cnt']} | {', '.join(r['suppliers'])} |" for r in results)


def _neo4j_interaction(drug: str) -> list:
    return run_cypher("""
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) CONTAINS toLower($s) OR toLower(b.generic_name) CONTAINS toLower($s)
        RETURN a.generic_name AS drug_a, b.generic_name AS drug_b,
               r.severity AS severity, r.description AS description, r.recommendation AS recommendation
        ORDER BY CASE r.severity WHEN 'Major' THEN 1 WHEN 'Moderate' THEN 2 ELSE 3 END LIMIT 5
    """, {"s": drug.lower()})

def _neo4j_drug_info(drug: str) -> list:
    return run_cypher("""
        MATCH (d:Drug)-[:IN_CATEGORY]->(c:Category)
        WHERE toLower(d.generic_name) CONTAINS toLower($s)
        RETURN d.generic_name AS name, d.drug_class AS drug_class,
               d.indications AS indications, d.contraindications AS contraindications,
               d.side_effects AS side_effects, d.adult_dose AS adult_dose,
               d.pediatric_dose AS pediatric_dose, d.prescription AS prescription,
               d.controlled AS controlled, c.name AS category LIMIT 3
    """, {"s": drug.lower()})

_CLINICAL_SYSTEM = """You are a pharmacy data assistant. You have been given STRUCTURED DATA from the knowledge graph.
Summarise it clearly for pharmacy staff in 3-5 sentences. Rules:
1. Use ONLY the data provided. Never invent facts.
2. Empty data: for interactions say "No recorded interaction found — does not confirm safety, verify with pharmacist."
   For other queries say "Not available in our knowledge base."
3. State exact severity levels (Minor/Moderate/Major) for interactions.
4. End with: "Source: drug knowledge graph" or "Source: drug interaction knowledge graph".
5. Respond in the same language as the question."""

def _clinical_answer(question: str, intent: str, source: str,
                      data: list, history: list = None, lang: str = "en") -> str:
    disclaimer = CLINICAL_DISCLAIMER[lang]
    if not data:
        if intent == "interaction":
            msg = ("No recorded interaction found in our knowledge base. "
                   "This does not confirm safety — always verify with a pharmacist." if lang=="en" else
                   "Aucune interaction enregistrée dans notre base. "
                   "Cela ne confirme pas la sécurité — vérifiez toujours avec un pharmacien.")
        else:
            msg = ("This information is not available in our knowledge base." if lang=="en" else
                   "Cette information n'est pas disponible dans notre base de données.")
        return msg + disclaimer

    messages = [{"role":"system","content":_CLINICAL_SYSTEM}]
    if history:
        for m in history[-4:]:
            messages.append({"role":m["role"],"content":m["content"]})
    messages.append({"role":"user","content":
        f"DATA:\n{json.dumps(data,indent=2)}\n\nQUESTION: {question}\n\nSummarise the data only."})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.0, max_tokens=400)
        result = resp.choices[0].message.content
        if "Clinical Disclaimer" in result or "Avertissement" in result: return result
        return result + disclaimer
    except Exception as e:
        return f"Clinical query failed: {e}{disclaimer}"


def _exec_daily_briefing(lang: str = "en") -> str:
    today = date.today().strftime("%A, %d %B %Y")
    df_stock = pd.read_sql_query("SELECT generic_name, brand_name, quantity_in_stock, reorder_level, ROUND((quantity_in_stock::numeric/NULLIF(reorder_level,0))*100,0) AS pct FROM inventory WHERE quantity_in_stock<=reorder_level ORDER BY pct ASC LIMIT 5", get_engine())
    df_exp   = pd.read_sql_query("SELECT i.generic_name, i.brand_name, b.batch_number, (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days_left, b.quantity_remaining FROM batches b JOIN inventory i ON b.product_id=i.product_id WHERE (b.expiry_date::date-CURRENT_DATE)<=30 ORDER BY days_left ASC LIMIT 5", get_engine())
    df_rev   = pd.read_sql_query("SELECT ROUND(SUM(total_amount)::numeric,2) AS revenue, COUNT(*) AS txns, SUM(quantity_sold) AS units FROM transactions WHERE date=(SELECT MAX(date) FROM transactions)", get_engine())
    df_avg   = pd.read_sql_query("SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily FROM (SELECT date, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date) t", get_engine())

    hour = datetime.now(tz=timezone(timedelta(hours=2))).hour
    if lang == "fr":
        tod = "Bonjour" if hour<12 else ("Bon après-midi" if hour<17 else "Bonsoir")
    else:
        tod = "Good morning" if hour<12 else ("Good afternoon" if hour<17 else "Good evening")

    lines = [f"# 🌅 {tod}! {'Rapport Quotidien' if lang=='fr' else 'Daily Briefing'} — {today}\n"]
    rev = df_rev.iloc[0]; avg = df_avg.iloc[0]["avg_daily"]
    trend = "📈" if rev["revenue"] > avg else "📉"
    lines += [f"## 💰 {'Revenu d\'Hier' if lang=='fr' else 'Yesterday\'s Revenue'}\n"
              f"**${rev['revenue']:,.2f}** ({rev['txns']} {'transactions'}, {rev['units']} {'unités' if lang=='fr' else 'units'})\n"
              f"Avg 30j: **${avg:,.2f}** {trend}\n"]

    if df_stock.empty:
        lines.append(f"## ✅ {'Stock' if lang=='fr' else 'Stock Levels'}\n{'Tous les produits au-dessus du niveau de réapprovisionnement.' if lang=='fr' else 'All products above reorder level.'}\n")
    else:
        lines += [f"## 🔴 {'Stock Faible' if lang=='fr' else 'Low Stock'} — {len(df_stock)} {'médicament(s)' if lang=='fr' else 'drug(s)'}",
                  "| Drug | Brand | Stock | Reorder | % |\n|---|---|---|---|---|"]
        for _, r in df_stock.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['pct']:.0f}% |")
        lines.append("")

    if df_exp.empty:
        lines.append(f"## ✅ {'Expiration' if lang=='fr' else 'Expiry Status'}\n{'Aucun lot expirant dans 30 jours.' if lang=='fr' else 'No batches expiring within 30 days.'}\n")
    else:
        lines += [f"## 🚨 {'Expiration Urgente' if lang=='fr' else 'Urgent Expiry'} — {len(df_exp)} {'lot(s)' if lang=='fr' else 'batch(es)'}",
                  "| Drug | Brand | Batch | Days Left | Qty |\n|---|---|---|---|---|"]
        for _, r in df_exp.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | **{r['days_left']}** | {r['quantity_remaining']} |")
    return "\n".join(lines)

def _exec_reorder() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock, i.reorder_level, i.category,
               COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,1),0) AS avg_daily,
               (i.reorder_level*2-i.quantity_in_stock) AS suggested_order
        FROM inventory i LEFT JOIN transactions t ON i.product_id=t.product_id
        WHERE i.quantity_in_stock<=i.reorder_level
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock,i.reorder_level,i.category
        ORDER BY (i.quantity_in_stock::float/NULLIF(i.reorder_level,1)) ASC
    """, get_engine())
    if df.empty: return "✅ All products above reorder level."
    out  = f"## 📋 Procurement Action List — {len(df)} drug(s)\n\n"
    out += "| Drug | Brand | Stock | Reorder | Avg Daily | Suggested Order | Category |\n|---|---|---|---|---|---|---|\n"
    out += "\n".join(
        f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | "
        f"{r['avg_daily']}/day | **{max(int(r['suggested_order']),1)}** | {r['category']} |"
        for _, r in df.iterrows()
    )
    return out + "\n\n*Suggested = 2× reorder level − current stock.*"

def _exec_combined_risk() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock, i.reorder_level,
               ROUND((i.quantity_in_stock::numeric/NULLIF(i.reorder_level,0))*100,0) AS stock_pct,
               MIN(b.expiry_date) AS nearest_expiry,
               (MIN(b.expiry_date::date)-CURRENT_DATE)::INTEGER AS days_to_expiry
        FROM inventory i JOIN batches b ON i.product_id=b.product_id
        WHERE i.quantity_in_stock<=i.reorder_level AND (b.expiry_date::date-CURRENT_DATE)<=90
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock,i.reorder_level
        ORDER BY days_to_expiry ASC, stock_pct ASC
    """, get_engine())
    if df.empty: return "✅ No drugs currently both low on stock AND expiring within 90 days."
    out  = f"**⚠️ {len(df)} drug(s) — LOW STOCK + EXPIRING SOON:**\n\n"
    out += "| Drug | Brand | Stock | Reorder | % | Nearest Expiry | Days Left |\n|---|---|---|---|---|---|---|\n"
    out += "\n".join(
        f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | "
        f"{r['stock_pct']:.0f}% | {str(r['nearest_expiry'])[:10]} | **{r['days_to_expiry']}** |"
        for _, r in df.iterrows()
    )
    return out

def _exec_reconciliation() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, SUM(b.quantity_received) AS received,
               SUM(t.quantity_sold) AS sold, i.quantity_in_stock,
               (SUM(b.quantity_received)-COALESCE(SUM(t.quantity_sold),0)-i.quantity_in_stock) AS discrepancy
        FROM inventory i LEFT JOIN batches b ON i.product_id=b.product_id
        LEFT JOIN transactions t ON i.product_id=t.product_id
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock
        HAVING ABS(SUM(b.quantity_received)-COALESCE(SUM(t.quantity_sold),0)-i.quantity_in_stock)>5
        ORDER BY ABS(SUM(b.quantity_received)-COALESCE(SUM(t.quantity_sold),0)-i.quantity_in_stock) DESC LIMIT 10
    """, get_engine())
    if df.empty: return "✅ No significant stock discrepancies found."
    out  = "## ⚠️ Stock Reconciliation Discrepancies\n\n"
    out += "| Drug | Brand | Received | Sold | Current | Discrepancy |\n|---|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        flag = "🔴" if abs(r["discrepancy"])>20 else "🟡"
        out += f"| {r['generic_name']} | {r['brand_name']} | {r['received']:.0f} | {r['sold']:.0f} | {r['quantity_in_stock']} | {flag} **{r['discrepancy']:.0f}** |\n"
    return out + "\n*Discrepancy = Received − Sold − Current Stock.*"

def _exec_forecast() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock,
               COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,2),0) AS avg_daily
        FROM inventory i LEFT JOIN transactions t ON i.product_id=t.product_id
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock,i.selling_price_usd
        ORDER BY (i.quantity_in_stock*i.selling_price_usd) DESC LIMIT 15
    """, get_engine())
    df_d = pd.read_sql_query("SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg FROM (SELECT date, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date) t", get_engine())
    avg = float(df_d.iloc[0]["avg"])
    out  = f"## 📈 Revenue & Stock Forecast\n\nAvg Daily Revenue: **${avg:,.2f}** | 30-Day: **${avg*30:,.2f}** | 90-Day: **${avg*90:,.2f}**\n\n"
    out += "| Drug | Brand | Stock | Avg Daily Sales | Days Remaining |\n|---|---|---|---|---|\n"
    for _, r in df.iterrows():
        if r["avg_daily"] > 0:
            days = round(r["quantity_in_stock"]/r["avg_daily"])
            flag = "🔴" if days<30 else ("🟡" if days<60 else "🟢")
            out += f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | {r['avg_daily']}/day | {flag} **{days} days** |\n"
        else:
            out += f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | No sales | ∞ |\n"
    return out


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN DISPATCH  (orchestrates all 5 layers)
# ═══════════════════════════════════════════════════════════════════

def get_greeting_response(question: str = "", lang: str = "en") -> str:
    q = question.lower().strip().rstrip("!.,?")
    echo_map = {
        "good morning":"Good morning","morning":"Good morning",
        "good afternoon":"Good afternoon","afternoon":"Good afternoon",
        "good evening":"Good evening","evening":"Good evening",
        "good night":"Good night","hi":"Hi","hey":"Hey","hello":"Hello",
        "howzit":"Howzit","yo":"Hey","sup":"Hey",
        "bonjour":"Bonjour","salut":"Salut","bonsoir":"Bonsoir","allô":"Allô",
    }
    opener = next((v for k,v in echo_map.items() if q.startswith(k)),
                  "Hello" if lang=="en" else "Bonjour")
    if lang == "fr":
        return (f"{opener}! Je suis votre Assistant Pharmacie Sunrise. "
                "Posez-moi des questions sur les stocks, les dates d'expiration, "
                "les ventes, les fournisseurs ou les interactions médicamenteuses. "
                "Comment puis-je vous aider?")
    return (f"{opener}! I'm your Sunrise Pharmacy Assistant. "
            "Ask me about stock levels, expiry dates, sales, suppliers, or drug interactions — "
            "whatever you need. How can I help?")


def dispatch(question: str, lang: str = "en",
             conversation_history: list = None) -> tuple[str, str]:
    """
    Master dispatcher — runs all 5 layers.
    Returns (answer: str, mode: str)
    mode ∈ {"system", "operational", "clinical", "caution"}
    """
    # ── Pre-process ─────────────────────────────────────────────────
    corrected_q, correction_note = fuzzy_correct_question(question)
    q_lower  = corrected_q.lower().strip()
    q_clean  = re.sub(r"[?!.,'\u2019 ]+$", "", q_lower).strip()
    entities = extract_entities(q_lower)

    def _wrap(answer: str) -> str:
        if correction_note and "Clinical Disclaimer" not in answer:
            return f"{correction_note}\n\n{answer}"
        return f"{correction_note}\n\n{answer}" if correction_note else answer

    # ── System responses (fastest path) ────────────────────────────
    if any(q_clean == t or q_clean.startswith(t+" ") for t in GREETING_TRIGGERS):
        return get_greeting_response(question, lang), "system"
    if any(q_clean == t or q_clean.startswith(t+" ") for t in THANKS_TRIGGERS):
        return THANKS_RESPONSE[lang], "system"
    if any(q_clean == t or q_clean.startswith(t+" ") for t in FAREWELL_TRIGGERS):
        return FAREWELL_RESPONSE[lang], "system"

    # ── Drug summary shortcut ───────────────────────────────────────
    if q_lower.startswith("quick summary:") or q_lower.startswith("résumé rapide:"):
        drug = re.sub(r"^(quick summary:|résumé rapide:)\s*", "", corrected_q, flags=re.I).strip()
        entities.drug = drug
        return _wrap(exec_drug_summary(entities)), "operational"

    # ── Layer 1: Deterministic routing ─────────────────────────────
    intent, conf = deterministic_route(q_lower, entities)

    # ── Layer 3: GPT routing (if Layer 1 missed) ───────────────────
    if not intent:
        gpt_result = gpt_route(corrected_q, conversation_history)
        gpt_intent = gpt_result["intent"]
        gpt_params = gpt_result["params"]

        if gpt_intent == "out_of_scope":
            return _out_of_scope(lang), "system"

        # Map GPT tool names to our internal intent IDs
        GPT_TOOL_TO_INTENT = {
            "query_inventory":  _gpt_inventory_intent(gpt_params),
            "query_sales":      _gpt_sales_intent(gpt_params),
            "query_expiry":     _gpt_expiry_intent(gpt_params),
            "query_supplier":   _gpt_supplier_intent(gpt_params),
            "query_clinical":   _gpt_clinical_intent(gpt_params),
            "query_briefing":   "daily_briefing",
            "query_combined_risk": "combined_risk",
            "query_reorder":    "reorder_list",
            "query_forecast":   "revenue_forecast",
            "query_reconciliation": "stock_reconciliation",
            "query_alternatives": "drug_alternatives",
            "query_stats":      "inventory_summary",
        }
        intent = GPT_TOOL_TO_INTENT.get(gpt_intent)

        # Merge GPT-extracted entities into our entities object
        if gpt_params.get("drug_name") and not entities.drug:
            entities.drug = gpt_params["drug_name"]
        if gpt_params.get("limit"):
            entities.number = gpt_params["limit"]
        if gpt_params.get("day_name") and not entities.day:
            entities.day = gpt_params["day_name"].lower()
        if gpt_params.get("month_name") and not entities.month:
            entities.month = gpt_params["month_name"].lower()
        if gpt_params.get("city") and not entities.city:
            entities.city = gpt_params["city"]

        if not intent:
            caution = ("⚠️ **Proceed with caution** — I'm not fully sure about this query.\n\n"
                       if lang == "en" else
                       "⚠️ **Procéder avec prudence** — Je ne suis pas sûr de cette requête.\n\n")
            return caution + _out_of_scope(lang), "caution"

    # ── Layer 4: Execute ────────────────────────────────────────────
    executor = INTENT_EXECUTOR_MAP.get(intent)
    if not executor:
        return _out_of_scope(lang), "system"

    try:
        if intent in LANG_INTENTS:
            answer = executor(entities, lang)
        else:
            answer = executor(entities)
        mode = "clinical" if intent in CLINICAL_INTENTS else "operational"
        return _wrap(answer), mode
    except Exception as ex:
        return (f"⚠️ Something went wrong. Please try rephrasing.\n\n*Details: {ex}*",
                "caution")


# ── GPT param → intent ID helpers ─────────────────────────────────
def _gpt_inventory_intent(p: dict) -> str:
    f = p.get("filter","all"); s = p.get("sort_by","")
    if f == "below_reorder": return "low_stock"
    if s == "margin_desc":   return "highest_margin"
    if f == "cheapest":      return "cheapest_drugs"
    if f == "most_expensive": return "expensive_drugs"
    if p.get("drug_name"):   return "stock_check"
    if p.get("category"):    return "category_browse"
    return "inventory_summary"

def _gpt_sales_intent(p: dict) -> str:
    period = p.get("period","all_time")
    if period == "last_day":      return "yesterday_sales"
    if period == "last_week":     return "yesterday_sales"
    if period == "current_month": return "this_month_sales"
    if period == "best_day":      return "best_day"
    if period == "total_summary": return "total_summary"
    if period == "day_of_week":   return "day_sales"
    if period == "customer_type": return "customer_type_sales"
    d = p.get("direction","top")
    return "top_sellers" if d == "top" else "worst_sellers"

def _gpt_expiry_intent(p: dict) -> str:
    if p.get("top_only"):    return "first_expiry"
    if p.get("month_name"):  return "expiry_month"
    if p.get("count_only"):  return "multi_batch"
    if p.get("drug_name"):   return "expiry_drug"
    return "expiry_soon"

def _gpt_supplier_intent(p: dict) -> str:
    s = p.get("sort_by",""); d = p.get("direction","asc")
    if s == "lead_time" and d == "asc":  return "fastest_supplier"
    if s == "lead_time" and d == "desc": return "slowest_supplier"
    if s == "payment_terms": return "payment_terms"
    if p.get("drug_name"):   return "supplier_drug"
    if p.get("city"):        return "supplier_city"
    return "supplier_count"

def _gpt_clinical_intent(p: dict) -> str:
    qt = p.get("query_type","drug_info")
    return "drug_interactions" if qt == "interaction" else "drug_info"


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — GRADIO CALLBACKS
# ═══════════════════════════════════════════════════════════════════

QUICK_QUESTIONS = {
    "en": [
        "Good morning",
        "Which drugs are running low?",
        "Which batches are expiring soon?",
        "What is the reorder list?",
        "What are the top selling drugs?",
        "Revenue forecast",
        "Do we have Amoxicillin in stock?",
        "What interacts with Metformin?",
    ],
    "fr": [
        "Bonjour",
        "Quels médicaments ont un stock faible?",
        "Quels lots expirent bientôt?",
        "Quelle est la liste de réapprovisionnement?",
        "Quels sont les médicaments les plus vendus?",
        "Prévision des revenus",
        "Avons-nous de l'Amoxicilline en stock?",
        "Qu'est-ce qui interagit avec la Metformine?",
    ],
}

def filter_drugs(search_text: str):
    if not search_text or len(search_text) < 2:
        return gr.update(choices=DRUG_NAMES[:20])
    matches = [d for d in DRUG_NAMES if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches if matches else DRUG_NAMES[:20])


def respond(message: str, chat_history: list, search_history: list, lang: str = "en"):
    if not message or not message.strip():
        return "", chat_history, search_history, gr.update(), gr.update(), ""

    conversation_history = [
        {"role": t["role"], "content": t["content"]}
        for t in (chat_history or [])
    ]

    answer, mode = dispatch(message, lang, conversation_history)

    if mode == "system":
        full_answer = answer
    elif mode == "clinical":
        header = f"*🧪 Clinical data — drug knowledge graph*\n\n"
        full_answer = header + answer if not answer.lstrip().startswith("*🧪") else answer
    elif mode == "caution":
        full_answer = answer
    else:
        full_answer = f"*📦 Operational data*\n\n{answer}"

    chat_history   = list(chat_history or [])
    search_history = list(search_history or [])
    chat_history.append({"role": "user",      "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})
    if message not in search_history:
        search_history.insert(0, message)
    search_history = search_history[:15]
    history_md = "\n".join(f"- {h}" for h in search_history)

    return ("", chat_history, search_history,
            gr.update(choices=search_history, value=None),
            gr.update(value=history_md), "")


def drug_summary_respond(drug_name: str, chat_history: list,
                          search_history: list, lang: str = "en"):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update(), ""
    try:
        e = Entities(); e.drug = drug_name
        answer      = exec_drug_summary(e)
        full_answer = "*📦 Operational data — inventory + batch records*\n\n" + answer
    except Exception as ex:
        full_answer = f"⚠️ Error: {ex}"
    label = f"Quick summary: {drug_name}"
    chat_history   = list(chat_history or [])
    search_history = list(search_history or [])
    chat_history.append({"role": "user",      "content": label})
    chat_history.append({"role": "assistant", "content": full_answer})
    if label not in search_history:
        search_history.insert(0, label)
    search_history = search_history[:15]
    history_md = "\n".join(f"- {h}" for h in search_history)
    return (chat_history, search_history,
            gr.update(choices=search_history, value=None),
            gr.update(value=history_md), "")


def export_chat(chat_history: list):
    if not chat_history: return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"pharmacy_chat_{ts}.txt"
    lines = ["Netrisyl Pharmacy Assistant — Chat Export",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*60, ""]
    for m in chat_history:
        role = "Staff" if m["role"]=="user" else "Assistant"
        lines.append(f"[{role}]\n{m['content']}\n")
    with open(fname,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    return fname


# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — GRADIO UI
# ═══════════════════════════════════════════════════════════════════

with gr.Blocks(title="Netrisyl Pharmacy Assistant") as demo:

    gr.HTML("""<script>
    function scrollChat() {
        document.querySelectorAll('.chatbot,[class*="chatbot"],.message-wrap,.messages')
            .forEach(el => { el.scrollTop = el.scrollHeight; });
    }
    const obs = new MutationObserver(scrollChat);
    document.addEventListener('DOMContentLoaded', () => {
        const t = document.querySelector('.gradio-container');
        if (t) obs.observe(t, {childList:true, subtree:true});
        setInterval(scrollChat, 500);
    });
    </script>""")

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
                padding:16px 24px;border-radius:10px;margin-bottom:16px;
                display:flex;align-items:center;justify-content:space-between;">
        <img src="https://huggingface.co/spaces/Sylvester1922/Netrisyl_pharmacy_assistant/resolve/main/NI_Logo.png"
             style="height:70px;object-fit:contain;" onerror="this.style.display='none'" alt=""/>
        <div style="text-align:center;flex:1;">
            <h1 style="color:white;margin:0;font-size:24px;">💊 Pharmacy Assistant</h1>
            <p style="color:#aed6f1;margin:4px 0 0 0;font-size:13px;">
                Powered by Neo4j + GPT-4o-mini | Harare, Zimbabwe
            </p>
        </div>
        <div style="width:180px;"></div>
    </div>""")

    # ── State ──────────────────────────────────────────────────────
    lang_state           = gr.State("en")
    search_history_state = gr.State([])

    with gr.Row():

        # ── LEFT sidebar ───────────────────────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🌐 Language / Langue")
            with gr.Row():
                btn_en = gr.Button("🇬🇧 English",  variant="primary",   size="sm")
                btn_fr = gr.Button("🇫🇷 Français", variant="secondary", size="sm")
            gr.Markdown("---")
            gr.Markdown("### 🔍 Drug Lookup / Recherche")
            drug_search   = gr.Textbox(
                placeholder="Type e.g. amox... / Tapez ex. amox...",
                label="Search / Rechercher")
            drug_dropdown = gr.Dropdown(
                choices=DRUG_NAMES[:20],
                label="Select drug / Sélectionner",
                interactive=True)
            drug_lookup_btn = gr.Button(
                "📋 Get Summary / Voir Fiche", variant="primary", size="sm")
            gr.Markdown("---")
            gr.Markdown("""
**Data Sources / Sources:**
- 📦 Inventory & Pricing
- 🧪 Drug Knowledge Graph
- ⚠️ Drug Interactions
- 📅 Batch & Expiry Records
- 🚚 Supplier Network
- 💰 30-Day Transactions
            """)
            gr.Markdown("---")
            gr.Markdown("""
**Response modes:**
- 📦 *Operational* — direct SQL, no AI
- 🧪 *Clinical* — AI summary + disclaimer
- ⚠️ *Caution* — GPT fallback
            """)

        # ── CENTRE — Chat ──────────────────────────────────────────
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(label="Pharmacy Assistant", height=460, autoscroll=True)
            brief_box = gr.Textbox(
                label="💡 Key Points / Points Clés",
                placeholder="Ask a question then click Brief / Posez une question puis cliquez Résumé",
                interactive=False, lines=2)
            with gr.Row():
                msg       = gr.Textbox(
                    placeholder="Ask e.g. 'Do we have Amoxicillin?' / Ex. 'Avons-nous de l\\'Amoxicilline?'",
                    label="", scale=4)
                submit    = gr.Button("Ask / Demander",    variant="primary",   scale=1)
                brief_btn = gr.Button("💡 Brief / Résumé", variant="secondary", scale=1)
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone"], type="filepath",
                                       label="🎤 Voice / Voix")
            with gr.Row():
                export_btn  = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file = gr.File(label="Download", scale=2, visible=False)

        # ── RIGHT sidebar ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 💡 Quick Questions / Questions Rapides")
            quick_btns_en = [gr.Button(q, variant="secondary", size="sm", visible=True)
                             for q in QUICK_QUESTIONS["en"]]
            quick_btns_fr = [gr.Button(q, variant="secondary", size="sm", visible=False)
                             for q in QUICK_QUESTIONS["fr"]]
            gr.Markdown("---")
            gr.Markdown("### 🕘 Search History / Historique")
            history_dropdown = gr.Dropdown(choices=[], label="Re-ask / Re-poser", interactive=True)
            history_display  = gr.Markdown("*No searches yet / Aucune recherche*")

    gr.HTML("""<div style="text-align:center;margin-top:16px;color:#7f8c8d;font-size:12px;">
        Netrisyl Insights · Harare, Zimbabwe · Data. Analytics. Intelligence.
    </div>""")

    # ── Language toggle ────────────────────────────────────────────
    def set_lang_en():
        return ("en",
                gr.update(variant="primary"),   gr.update(variant="secondary"),
                *[gr.update(visible=True)  for _ in QUICK_QUESTIONS["en"]],
                *[gr.update(visible=False) for _ in QUICK_QUESTIONS["fr"]])
    def set_lang_fr():
        return ("fr",
                gr.update(variant="secondary"), gr.update(variant="primary"),
                *[gr.update(visible=False) for _ in QUICK_QUESTIONS["en"]],
                *[gr.update(visible=True)  for _ in QUICK_QUESTIONS["fr"]])

    lang_outputs = [lang_state, btn_en, btn_fr] + quick_btns_en + quick_btns_fr
    btn_en.click(set_lang_en, [], lang_outputs)
    btn_fr.click(set_lang_fr, [], lang_outputs)

    # ── Main submit wiring ─────────────────────────────────────────
    IO_IN  = [msg, chatbot, search_history_state, lang_state]
    IO_OUT = [msg, chatbot, search_history_state, history_dropdown, history_display, brief_box]

    submit.click(respond, IO_IN, IO_OUT)
    msg.submit(respond,   IO_IN, IO_OUT)

    # ── Voice ──────────────────────────────────────────────────────
    def transcribe(audio_path, chat_history, search_history, lang):
        if not audio_path:
            return "", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), ""
        try:
            with open(audio_path,"rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            r = respond(transcript.text, chat_history, search_history, lang)
            return r[0], r[1], r[2], r[3], r[4], gr.update(value=None), r[5]
        except Exception:
            return "", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), ""

    audio_input.stop_recording(transcribe,
        [audio_input, chatbot, search_history_state, lang_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, audio_input, brief_box])

    # ── History re-ask ─────────────────────────────────────────────
    def reask(selected, chat_history, search_history, lang):
        if not selected: return "", chat_history, search_history, gr.update(), gr.update(), ""
        return respond(selected, chat_history, search_history, lang)

    history_dropdown.change(reask,
        [history_dropdown, chatbot, search_history_state, lang_state], IO_OUT)

    # ── Quick question buttons ─────────────────────────────────────
    for btn, q in zip(quick_btns_en, QUICK_QUESTIONS["en"]):
        btn.click(lambda ch, sh, l, _q=q: respond(_q, ch, sh, l),
                  [chatbot, search_history_state, lang_state], IO_OUT)
    for btn, q in zip(quick_btns_fr, QUICK_QUESTIONS["fr"]):
        btn.click(lambda ch, sh, l, _q=q: respond(_q, ch, sh, l),
                  [chatbot, search_history_state, lang_state], IO_OUT)

    # ── Drug lookup ────────────────────────────────────────────────
    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])
    drug_lookup_btn.click(drug_summary_respond,
        [drug_dropdown, chatbot, search_history_state, lang_state],
        [chatbot, search_history_state, history_dropdown, history_display, brief_box])

    # ── Export ─────────────────────────────────────────────────────
    export_btn.click(
        lambda ch: gr.update(value=export_chat(ch), visible=True) if ch else gr.update(visible=False),
        [chatbot], [export_file])

    # ── Brief ──────────────────────────────────────────────────────
    def do_brief(chat_history, lang="en"):
        empty_msg = "No response yet." if lang=="en" else "Pas encore de réponse."
        if not chat_history: return empty_msg
        try:
            last = ""
            for entry in reversed(chat_history):
                if isinstance(entry, dict) and entry.get("role") == "assistant":
                    c = entry.get("content","")
                    if isinstance(c, list):
                        c = " ".join(x.get("text","") if isinstance(x,dict) else str(x) for x in c)
                    last = str(c); break
                elif isinstance(entry,(list,tuple)) and len(entry)>1:
                    last = str(entry[1] or ""); break
            if not last: return empty_msg
            lang_instr = "Respond in French." if lang=="fr" else "Respond in English."
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":
                    f"Summarise this pharmacy data in 2-3 clear sentences for a manager. "
                    f"Key numbers and actionable insights only. No bullets or markdown. {lang_instr}\n\n{last[:1500]}\n\nSummary:"}],
                temperature=0.0, max_tokens=150)
            return r.choices[0].message.content.strip()
        except Exception as ex:
            return f"Could not generate brief: {ex}"

    brief_btn.click(do_brief, [chatbot, lang_state], [brief_box])


demo.launch()
