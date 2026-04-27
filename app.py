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

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

GREETING_TRIGGERS = {
    "hi","hey","hello","morning","afternoon","evening","howzit",
    "good morning","good afternoon","good evening","good night",
    "how are you","what's up","whats up","what can you do",
    "what do you do","who are you","what are you","yo","sup","start","help",
    "bonjour","salut","bonsoir","bon matin","comment allez-vous","comment vas-tu",
    "quoi de neuf","qu'est-ce que vous faites","qui êtes-vous","allô"
}

THANKS_TRIGGERS = {
    "thank you","thanks","thank","cheers","appreciated","great","ok","okay",
    "cool","perfect","noted","awesome","brilliant","nice","wonderful","excellent",
    "got it","understood","sure",
    "merci","merci beaucoup","super","parfait","noté","compris","d'accord","bien"
}

FAREWELL_TRIGGERS = {
    "bye","goodbye","good bye","see you","see ya","later","ciao","take care",
    "exit","quit","talk later","catch you later","farewell",
    "au revoir","à bientôt","bonne journée","salut","à plus"
}

SKIP_WORDS = {
    "what","which","who","where","when","how","why","is","are","was","were",
    "do","does","did","have","has","had","will","can","could","should","would",
    "the","a","an","in","on","at","for","of","to","and","or","but","with","from",
    "about","we","our","us","i","my","me","stock","drug","drugs","medicine",
    "medicines","pharmacy","pharmacist","please","tell","show","give","find",
    "get","check","supplier","supply","supplies","order","source","batch",
    "expiry","soon","selling","sales","name","information","info","details",
    "need","want","medication","medications","tablet","capsule","injection",
    "anything","something","everything","vendor","distributor","buy","purchase",
    "procure","quel","quelle","quels","quelles","qui","où","quand","comment",
    "pourquoi","est","sont","avons","avez","ont","nous","vous","ils","elles",
    "un","une","des","les","le","la","en","sur","pour","avec","par","sans",
    "médicament","médicaments","pharmacie","pharmacien","stock","vente","ventes"
}

# ── Credentials ────────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")

# ── Neo4j driver ───────────────────────────────────────────────────
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

driver = get_driver()
try:
    with driver.session() as _s:
        _s.run("RETURN 1")
    print("Neo4j connection pre-warmed ✓")
except Exception as _e:
    print(f"Neo4j pre-warm failed: {_e}")

def run_cypher(cypher, params=None):
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

# ── OpenAI client ──────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Supabase pool + SQLAlchemy engine ─────────────────────────────
_pool = None
def get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(1, 10, SUPABASE_URL)
    return _pool
def get_conn():  return get_pool().getconn()
def release_conn(conn): get_pool().putconn(conn)

from sqlalchemy import create_engine as _sa_create_engine
_sa_engine = None
def get_engine():
    global _sa_engine
    if _sa_engine is None:
        _sa_engine = _sa_create_engine(SUPABASE_URL)
    return _sa_engine

print("Supabase connection pool ready ✓")

# ── Drug list at startup ───────────────────────────────────────────
def get_all_drugs():
    return pd.read_sql_query(
        "SELECT generic_name, brand_name, category FROM inventory ORDER BY generic_name",
        get_engine()
    )

DRUGS_DF         = get_all_drugs()
DRUG_NAMES       = DRUGS_DF["generic_name"].tolist()
BRAND_NAMES      = DRUGS_DF["brand_name"].tolist()
BRAND_TO_GENERIC = dict(zip(
    DRUGS_DF["brand_name"].str.lower(),
    DRUGS_DF["generic_name"]
))


# ══════════════════════════════════════════════════════════════════
# FUZZY MATCHING
# ══════════════════════════════════════════════════════════════════

def fuzzy_match_drug(text, threshold=78):
    text = re.sub(r"['\u2019\u2018`]", "", text.lower().strip())
    for drug in DRUG_NAMES:
        if text == drug.lower():
            return drug
    if text in BRAND_TO_GENERIC:
        return BRAND_TO_GENERIC[text]
    best_score, best_match = 0, None
    for drug in DRUG_NAMES:
        score = SequenceMatcher(None, text, drug.lower()).ratio() * 100
        if score > best_score:
            best_score, best_match = score, drug
    for brand, generic in BRAND_TO_GENERIC.items():
        score = SequenceMatcher(None, text, brand).ratio() * 100
        if score > best_score:
            best_score, best_match = score, generic
    return best_match if best_score >= threshold else None


def fuzzy_correct_question(question):
    skip = SKIP_WORDS | {
        "soon","please","could","would","anything","something",
        "find","list","show","tell","give","have","does","there",
        "that","this","will","about","from"
    }
    words = re.sub(r"['\u2019?!,.]", "", question).split()
    corrections, corrected_words = [], list(words)
    for i, word in enumerate(words):
        w = word.lower()
        w_clean = w[:-1] if w.endswith('s') and fuzzy_match_drug(w[:-1], threshold=90) else w
        w_clean = re.sub(r"[\u2019']s$", "", w_clean)
        if len(w_clean) < 4 or w_clean in skip:
            continue
        match = fuzzy_match_drug(w_clean, threshold=78)
        if match and match.lower() != w and match.lower() != w_clean:
            corrected_words[i] = match
            corrections.append(f"'{word}' → '{match}'")
    corrected = " ".join(corrected_words)
    note = f"*(Auto-corrected: {', '.join(corrections)})*" if corrections else ""
    return corrected, note


def extract_keywords(question: str) -> list:
    words = re.sub(r"[\'\u2019?!,.]", "", question.lower()).split()
    return [w for w in words if len(w) >= 4 and w not in SKIP_WORDS and not w.isdigit()]

def get_search_term(question: str) -> str:
    keywords = extract_keywords(question)
    return keywords[0] if keywords else question.lower()


# ══════════════════════════════════════════════════════════════════
# HARDCODED INTENT ENGINE
# Each intent has: keywords (EN+FR), a SQL executor, and a
# question template in both languages.
# ══════════════════════════════════════════════════════════════════

def _detect_drug(q: str):
    """Extract drug name from a question. Returns matched drug or None."""
    kws = extract_keywords(q)
    for k in kws:
        m = fuzzy_match_drug(k, threshold=82)
        if m:
            return m
    return None

def _detect_number(q: str, default=10):
    nums = re.findall(r'\d+', q)
    return int(nums[0]) if nums else default

def _detect_day(q: str):
    days = {"monday":"monday","tuesday":"tuesday","wednesday":"wednesday",
            "thursday":"thursday","friday":"friday","saturday":"saturday","sunday":"sunday",
            "lundi":"monday","mardi":"tuesday","mercredi":"wednesday",
            "jeudi":"thursday","vendredi":"friday","samedi":"saturday","dimanche":"sunday"}
    for word in q.lower().split():
        if word in days:
            return days[word]
    return None

def _detect_category(q: str):
    cats = {
        "antibiotic":"Antibiotics","antibiotique":"Antibiotics","antibiotics":"Antibiotics",
        "analgesic":"Analgesics","analgesique":"Analgesics","analgesics":"Analgesics","painkiller":"Analgesics",
        "antihypertensive":"Antihypertensives","antifungal":"Antifungals","antifongique":"Antifungals",
        "antidiabetic":"Antidiabetics","antidiabétique":"Antidiabetics",
        "antimalarial":"Antimalarials","antipaludique":"Antimalarials",
        "antiretroviral":"Antiretrovirals","antirétroviral":"Antiretrovirals",
        "respiratory":"Respiratory","respiratoire":"Respiratory",
        "vitamin":"Vitamins/Supplements","vitamine":"Vitamins/Supplements",
        "gi":"GI medications","digestif":"GI medications","gastrointestinal":"GI medications"
    }
    q_lower = q.lower()
    for k, v in cats.items():
        if k in q_lower:
            return v
    return None

def _detect_city(q: str):
    cities = ["harare","bulawayo","mutare","kwekwe","gweru","masvingo",
              "chinhoyi","bindura","marondera","chitungwiza"]
    for city in cities:
        if city in q.lower():
            return city.title()
    return None

def _detect_month(q: str):
    months = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
        "janvier":1,"février":2,"mars":3,"avril":4,"mai":5,"juin":6,
        "juillet":7,"août":8,"septembre":9,"octobre":10,"novembre":11,"décembre":12
    }
    for word in q.lower().split():
        if word in months:
            return word
    return None


# ── Intent definitions ─────────────────────────────────────────────
# Each entry: (intent_id, en_keywords, fr_keywords, needs_drug, needs_day, needs_category)
INTENT_PATTERNS = [
    # Stock / Inventory
    ("stock_check",
     ["in stock","do we have","stock level","how much stock","check stock","check inventory","avons-nous","est-ce qu'on a","en stock","niveau de stock"],
     True, False, False),
    ("low_stock",
     ["running low","below reorder","low on stock","low stock","critical stock","need restock","reorder level","stock faible","réapprovisionner","rupture","niveau bas"],
     False, False, False),
    ("inventory_summary",
     ["how many products","total inventory","inventory summary","inventory value","how many drugs","combien de produits","valeur du stock","résumé du stock"],
     False, False, False),
    ("category_browse",
     ["show me all","all drugs in","list all","drugs we carry","show all","montre tous","tous les médicaments","liste de"],
     False, False, True),
    ("drug_summary",
     ["full summary","quick summary","summary of","summarise","résumé de","fiche de"],
     True, False, False),
    ("cheapest_drugs",
     ["cheapest","lowest price","most affordable","prix le plus bas","moins cher","abordable"],
     False, False, False),
    ("expensive_drugs",
     ["most expensive","highest price","prix le plus élevé","plus cher"],
     False, False, False),
    ("highest_margin",
     ["highest margin","most profitable","best margin","highest profit","marge la plus élevée","plus rentable","meilleure marge"],
     False, False, False),
    ("drug_alternatives",
     ["alternative","substitute","instead of","replace","alternatives pour","substitut de","remplacer"],
     True, False, False),
    # Sales
    ("top_sellers",
     ["top selling","best selling","most sold","highest revenue drug","top sellers","meilleures ventes","médicaments les plus vendus","plus vendu"],
     False, False, False),
    ("worst_sellers",
     ["least selling","worst selling","worst sellers","bottom sellers","lowest sales","moins vendu","pires ventes","ventes les plus faibles"],
     False, False, False),
    ("yesterday_sales",
     ["yesterday","last day","yesterday revenue","yesterday sales","hier","ventes d'hier","chiffre d'hier","revenu d'hier"],
     False, False, False),
    ("this_month_sales",
     ["this month","monthly revenue","month to date","revenue this month","ce mois","revenu du mois","chiffre du mois"],
     False, False, False),
    ("total_summary",
     ["total revenue","how many units","total units","total sales","average daily","avg daily","total transactions","revenu total","unités vendues","ventes totales","moyenne journalière"],
     False, False, False),
    ("best_day",
     ["best day","busiest day","highest revenue day","which day","what day of the week","jour le plus chargé","meilleur jour","quel jour"],
     False, False, False),
    ("day_sales",
     ["sales on","revenue on","what happened on","ventes du","chiffre du","lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche","monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
     False, True, False),
    ("customer_type_sales",
     ["customer type","by customer","walk-in","prescription","insurance","par type de client","type de clientèle"],
     False, False, False),
    # Expiry
    ("expiry_soon",
     ["expiring soon","expire soon","expiry alert","batches expiring","nearly expired","expire bientôt","périme bientôt","alerte expiration"],
     False, False, False),
    ("expiry_drug",
     ["when does","when do","expire","expiry date","expiration","quand expire","date d'expiration","date de péremption"],
     True, False, False),
    ("first_expiry",
     ["expires first","first to expire","nearest expiry","expire en premier","premier à expirer","expiration la plus proche"],
     False, False, False),
    ("expiry_month",
     ["expiring in","expire in","expires in","expiry in","expiration en","périme en"],
     False, False, False),
    ("batch_count",
     ["how many batches","batches does","number of batches","batch count","combien de lots","nombre de lots"],
     True, False, False),
    ("multi_batch",
     ["more than","at least","over","drugs with","batches","lots de","plus de","au moins"],
     False, False, False),
    ("expiry_quantity",
     ["stock expiring","quantity expiring","how much expires","quantité qui expire","stock qui périme"],
     False, False, False),
    # Suppliers
    ("supplier_drug",
     ["who supplies","who is the supplier","supplier for","who provides","fournisseur de","qui fournit","qui approvisionne"],
     True, False, False),
    ("fastest_supplier",
     ["fastest supplier","fastest vendor","quickest supplier","shortest lead time","fournisseur le plus rapide","délai le plus court","livraison la plus rapide"],
     False, False, False),
    ("slowest_supplier",
     ["slowest supplier","longest lead time","slowest vendor","fournisseur le plus lent","délai le plus long"],
     False, False, False),
    ("supplier_count",
     ["how many suppliers","number of suppliers","combien de fournisseurs","nombre de fournisseurs"],
     False, False, False),
    ("supplier_city",
     ["suppliers in","vendors in","fournisseurs à","fournisseurs de","vendeurs à"],
     False, False, False),
    ("payment_terms",
     ["payment terms","best payment","credit terms","who gives best terms","meilleures conditions de paiement","délai de paiement","conditions de crédit"],
     False, False, False),
    ("lead_time",
     ["lead time for","delivery time for","délai pour","délai de livraison pour"],
     True, False, False),
    # Clinical
    ("drug_interactions",
     ["interacts with","drug interactions","what interacts","interactions de","interactions avec","contre-indications avec"],
     True, False, False),
    ("drug_safety",
     ["safe with","safe to take with","can it be taken with","sûr avec","peut-on prendre avec","compatible avec"],
     True, False, False),
    ("drug_info",
     ["what is","what are the","tell me about","drug information","information sur","qu'est-ce que","parle moi de","fiche médicale"],
     True, False, False),
    ("side_effects",
     ["side effects","adverse effects","effets secondaires","effets indésirables"],
     True, False, False),
    ("dosage",
     ["dosage","dose","how much to give","posologie","dose recommandée","quelle dose"],
     True, False, False),
    ("contraindications",
     ["contraindicated","contraindications","contre-indiqué","contre-indications"],
     True, False, False),
    # Operational alerts
    ("daily_briefing",
     ["daily briefing","morning briefing","daily summary","anything i should know","start of day","briefing","daily report","résumé quotidien","rapport du jour","ce que je dois savoir","début de journée"],
     False, False, False),
    ("reorder_list",
     ["reorder list","procurement list","what to order","order list","liste de réapprovisionnement","que commander","liste de commande"],
     False, False, False),
    ("combined_risk",
     ["low and expiring","expiring and low","critical drugs","urgent attention","faible et expirant","médicaments critiques","attention urgente"],
     False, False, False),
    ("stock_reconciliation",
     ["reconciliation","discrepancy","stock mismatch","reconcile","réconciliation","écart de stock","discordance"],
     False, False, False),
    ("revenue_forecast",
     ["forecast","projection","predict","how long will stock last","prévision","projection de revenus","combien de temps durera"],
     False, False, False),
]

# ── Question templates per intent ──────────────────────────────────
# {drug}, {n}, {day}, {category}, {city}, {month} are slot placeholders
QUESTION_TEMPLATES = {
    "stock_check": {
        "en": ["Do we have {drug} in stock?", "What is the stock level for {drug}?", "Check inventory for {drug}"],
        "fr": ["Avons-nous {drug} en stock?", "Quel est le niveau de stock de {drug}?", "Vérifier le stock de {drug}"]
    },
    "low_stock": {
        "en": ["Which drugs are running low on stock?", "Which drugs are below reorder level?", "What needs restocking?"],
        "fr": ["Quels médicaments ont un stock faible?", "Quels médicaments sont sous le niveau de réapprovisionnement?", "Que faut-il réapprovisionner?"]
    },
    "inventory_summary": {
        "en": ["How many products do we carry?", "What is our total inventory value?", "Show me the inventory summary"],
        "fr": ["Combien de produits avons-nous?", "Quelle est la valeur totale de notre stock?", "Montrer le résumé du stock"]
    },
    "category_browse": {
        "en": ["Show me all {category} we carry", "List all drugs in {category}", "What {category} do we have?"],
        "fr": ["Montrer tous les {category} disponibles", "Lister tous les médicaments de {category}", "Quels {category} avons-nous?"]
    },
    "drug_summary": {
        "en": ["Give me a full summary of {drug}", "Show me the {drug} drug profile", "Quick summary: {drug}"],
        "fr": ["Donner un résumé complet de {drug}", "Montrer la fiche de {drug}", "Résumé rapide: {drug}"]
    },
    "cheapest_drugs": {
        "en": ["What are the cheapest drugs we carry?", "Show me the lowest priced drugs", "Which drugs cost the least?"],
        "fr": ["Quels sont les médicaments les moins chers?", "Montrer les médicaments au prix le plus bas", "Quels médicaments coûtent le moins?"]
    },
    "expensive_drugs": {
        "en": ["What are our most expensive drugs?", "Which drugs have the highest price?"],
        "fr": ["Quels sont nos médicaments les plus chers?", "Quels médicaments ont le prix le plus élevé?"]
    },
    "highest_margin": {
        "en": ["Which drug gives us the highest profit margin?", "What is our most profitable drug?", "Show me drugs by profit margin"],
        "fr": ["Quel médicament nous donne la meilleure marge?", "Quel est notre médicament le plus rentable?", "Montrer les médicaments par marge bénéficiaire"]
    },
    "drug_alternatives": {
        "en": ["What are the alternatives to {drug}?", "What can replace {drug}?", "Show me substitutes for {drug}"],
        "fr": ["Quelles sont les alternatives à {drug}?", "Que peut remplacer {drug}?", "Montrer les substituts de {drug}"]
    },
    "top_sellers": {
        "en": ["What are the top {n} selling drugs?", "What are our best sellers?", "Which drugs sell the most?"],
        "fr": ["Quels sont les {n} médicaments les plus vendus?", "Quelles sont nos meilleures ventes?", "Quels médicaments se vendent le plus?"]
    },
    "worst_sellers": {
        "en": ["What are the {n} worst selling drugs?", "Which drugs sell the least?", "Show me our slowest moving drugs"],
        "fr": ["Quels sont les {n} médicaments les moins vendus?", "Quels médicaments se vendent le moins?", "Montrer nos médicaments à rotation lente"]
    },
    "yesterday_sales": {
        "en": ["What was yesterday's revenue?", "How many transactions did we have yesterday?", "Show me yesterday's sales"],
        "fr": ["Quel était le chiffre d'affaires d'hier?", "Combien de transactions avons-nous eu hier?", "Montrer les ventes d'hier"]
    },
    "this_month_sales": {
        "en": ["How much did we make this month?", "What is our revenue so far this month?", "Show me this month's sales"],
        "fr": ["Combien avons-nous fait ce mois?", "Quel est notre chiffre d'affaires ce mois?", "Montrer les ventes du mois"]
    },
    "total_summary": {
        "en": ["How many units have we sold in total?", "What is our average daily revenue?", "Show me the overall sales summary"],
        "fr": ["Combien d'unités avons-nous vendu au total?", "Quelle est notre recette journalière moyenne?", "Montrer le résumé global des ventes"]
    },
    "best_day": {
        "en": ["Which day of the week has the highest revenue?", "What is our busiest day?", "Which day do we sell the most?"],
        "fr": ["Quel jour de la semaine a le plus grand chiffre?", "Quel est notre jour le plus chargé?", "Quel jour vendons-nous le plus?"]
    },
    "day_sales": {
        "en": ["What were the sales on {day}?", "Show me {day} revenue", "How did we perform on {day}?"],
        "fr": ["Quelles étaient les ventes le {day}?", "Montrer le chiffre du {day}", "Comment avons-nous performé le {day}?"]
    },
    "customer_type_sales": {
        "en": ["Show me sales by customer type", "What is the revenue split by customer type?", "Walk-in vs prescription vs insurance sales"],
        "fr": ["Montrer les ventes par type de client", "Quelle est la répartition du chiffre par type de client?", "Ventes clients directs vs ordonnances vs assurances"]
    },
    "expiry_soon": {
        "en": ["Which batches are expiring soon?", "What is expiring within 30 days?", "Show me urgent expiry alerts"],
        "fr": ["Quels lots expirent bientôt?", "Qu'est-ce qui expire dans 30 jours?", "Montrer les alertes d'expiration urgentes"]
    },
    "expiry_drug": {
        "en": ["When does {drug} expire?", "What is the expiry date for {drug}?", "Show me all batches of {drug}"],
        "fr": ["Quand expire {drug}?", "Quelle est la date de péremption de {drug}?", "Montrer tous les lots de {drug}"]
    },
    "first_expiry": {
        "en": ["Which item expires first?", "What is the nearest expiry date?", "Which drug expires soonest?"],
        "fr": ["Quel article expire en premier?", "Quelle est la date de péremption la plus proche?", "Quel médicament expire le plus tôt?"]
    },
    "expiry_month": {
        "en": ["Show me everything expiring in {month}", "What expires in {month}?", "Batches expiring in {month}"],
        "fr": ["Montrer tout ce qui expire en {month}", "Qu'est-ce qui expire en {month}?", "Lots expirant en {month}"]
    },
    "batch_count": {
        "en": ["How many batches does {drug} have?", "How many lots of {drug} do we carry?"],
        "fr": ["Combien de lots de {drug} avons-nous?", "Combien de lots de {drug} stockons-nous?"]
    },
    "multi_batch": {
        "en": ["Which drugs have more than {n} batches?", "Show drugs with more than {n} lots"],
        "fr": ["Quels médicaments ont plus de {n} lots?", "Montrer les médicaments avec plus de {n} lots"]
    },
    "expiry_quantity": {
        "en": ["How much stock is expiring within 90 days?", "What is the total quantity expiring soon?"],
        "fr": ["Quelle quantité expire dans 90 jours?", "Quelle est la quantité totale qui expire bientôt?"]
    },
    "supplier_drug": {
        "en": ["Who supplies {drug}?", "Which supplier provides {drug}?", "Where do we get {drug} from?"],
        "fr": ["Qui fournit {drug}?", "Quel fournisseur approvisionne {drug}?", "D'où venons-nous {drug}?"]
    },
    "fastest_supplier": {
        "en": ["Who is our fastest supplier?", "Which vendor has the shortest lead time?", "Who delivers first?"],
        "fr": ["Qui est notre fournisseur le plus rapide?", "Quel fournisseur a le délai le plus court?", "Qui livre en premier?"]
    },
    "slowest_supplier": {
        "en": ["Who is our slowest supplier?", "Which vendor has the longest lead time?"],
        "fr": ["Qui est notre fournisseur le plus lent?", "Quel fournisseur a le délai le plus long?"]
    },
    "supplier_count": {
        "en": ["How many suppliers do we have?", "How many vendors do we work with?"],
        "fr": ["Combien de fournisseurs avons-nous?", "Avec combien de fournisseurs travaillons-nous?"]
    },
    "supplier_city": {
        "en": ["Which suppliers are in {city}?", "Show me vendors based in {city}"],
        "fr": ["Quels fournisseurs sont à {city}?", "Montrer les fournisseurs basés à {city}"]
    },
    "payment_terms": {
        "en": ["Which supplier has the best payment terms?", "Show me supplier payment terms", "Who gives us the most credit?"],
        "fr": ["Quel fournisseur a les meilleures conditions de paiement?", "Montrer les conditions de paiement des fournisseurs", "Qui nous accorde le plus de crédit?"]
    },
    "lead_time": {
        "en": ["What is the lead time for {drug}?", "How long to receive {drug}?"],
        "fr": ["Quel est le délai de livraison pour {drug}?", "Combien de temps pour recevoir {drug}?"]
    },
    "drug_interactions": {
        "en": ["What interacts with {drug}?", "What are the drug interactions for {drug}?", "What should not be combined with {drug}?"],
        "fr": ["Qu'est-ce qui interagit avec {drug}?", "Quelles sont les interactions de {drug}?", "Que ne doit-on pas combiner avec {drug}?"]
    },
    "drug_safety": {
        "en": ["Is {drug} safe to take together?", "Can {drug} be combined with other drugs?"],
        "fr": ["Est-il sûr de prendre {drug} ensemble?", "Peut-on combiner {drug} avec d'autres médicaments?"]
    },
    "drug_info": {
        "en": ["What is {drug} used for?", "Tell me about {drug}", "Show me information on {drug}"],
        "fr": ["À quoi sert {drug}?", "Parle-moi de {drug}", "Montrer des informations sur {drug}"]
    },
    "side_effects": {
        "en": ["What are the side effects of {drug}?", "What adverse effects does {drug} have?"],
        "fr": ["Quels sont les effets secondaires de {drug}?", "Quels effets indésirables a {drug}?"]
    },
    "dosage": {
        "en": ["What is the dosage for {drug}?", "How much {drug} should be given?"],
        "fr": ["Quelle est la posologie de {drug}?", "Combien de {drug} faut-il donner?"]
    },
    "contraindications": {
        "en": ["What are the contraindications for {drug}?", "When should {drug} not be used?"],
        "fr": ["Quelles sont les contre-indications de {drug}?", "Quand ne doit-on pas utiliser {drug}?"]
    },
    "daily_briefing": {
        "en": ["Give me the daily briefing", "What should I know today?", "Show me the morning summary"],
        "fr": ["Donner le rapport quotidien", "Que dois-je savoir aujourd'hui?", "Montrer le résumé du matin"]
    },
    "reorder_list": {
        "en": ["What is the reorder list?", "Show me the procurement action list", "What do we need to order?"],
        "fr": ["Quelle est la liste de réapprovisionnement?", "Montrer la liste d'achats", "Que devons-nous commander?"]
    },
    "combined_risk": {
        "en": ["Which drugs are low on stock AND expiring soon?", "What needs urgent attention?", "Show me critical stock alerts"],
        "fr": ["Quels médicaments sont faibles ET expirent bientôt?", "Que nécessite une attention urgente?", "Montrer les alertes de stock critiques"]
    },
    "stock_reconciliation": {
        "en": ["Show me the stock reconciliation", "Are there any stock discrepancies?"],
        "fr": ["Montrer la réconciliation du stock", "Y a-t-il des écarts de stock?"]
    },
    "revenue_forecast": {
        "en": ["Give me the revenue forecast", "How long will our stock last?", "Show me the sales projection"],
        "fr": ["Donner la prévision de revenus", "Combien de temps durera notre stock?", "Montrer la projection des ventes"]
    },
}

# ── Intent → executor mapping ──────────────────────────────────────
# These are the GUARANTEED routes — no GPT involved

# ══════════════════════════════════════════════════════════════════
# FOLLOW-UP TEMPLATES
# Keyed by last intent — defines what to suggest next.
# Slots inherited from previous turn: {drug}, {n}, {day}, {category}
# ══════════════════════════════════════════════════════════════════

FOLLOW_UP_MAP = {
    "stock_check": [
        ("When does {drug} expire?",                "expiry_drug"),
        ("Who supplies {drug}?",                    "supplier_drug"),
        ("What interacts with {drug}?",             "drug_interactions"),
        ("What are the side effects of {drug}?",    "side_effects"),
        ("What are alternatives to {drug}?",        "drug_alternatives"),
    ],
    "low_stock": [
        ("Show me the full reorder list",           "reorder_list"),
        ("Which drugs are low AND expiring soon?",  "combined_risk"),
        ("What is our inventory summary?",          "inventory_summary"),
    ],
    "top_sellers": [
        ("Now show the bottom {n}",                 "worst_sellers"),
        ("Top {n} by revenue",                      "top_sellers_revenue"),
        ("What was yesterday's revenue?",           "yesterday_sales"),
        ("Which day has the highest sales?",        "best_day"),
    ],
    "worst_sellers": [
        ("Now show the top {n}",                    "top_sellers"),
        ("What was yesterday's revenue?",           "yesterday_sales"),
        ("Show me sales by customer type",          "customer_type_sales"),
    ],
    "yesterday_sales": [
        ("How much did we make this month?",        "this_month_sales"),
        ("Show me the overall sales summary",       "total_summary"),
        ("Which day has the highest revenue?",      "best_day"),
        ("What are our top sellers?",               "top_sellers"),
    ],
    "this_month_sales": [
        ("Show me the overall sales summary",       "total_summary"),
        ("What was yesterday's revenue?",           "yesterday_sales"),
        ("What is our revenue forecast?",           "revenue_forecast"),
    ],
    "total_summary": [
        ("Which day has the highest revenue?",      "best_day"),
        ("Show me sales by customer type",          "customer_type_sales"),
        ("What are our top sellers?",               "top_sellers"),
    ],
    "best_day": [
        ("What were sales on {day}?",               "day_sales"),
        ("What are our top sellers?",               "top_sellers"),
        ("What was yesterday's revenue?",           "yesterday_sales"),
    ],
    "day_sales": [
        ("What about Monday?",                      "day_sales_monday"),
        ("What about Friday?",                      "day_sales_friday"),
        ("What about Saturday?",                    "day_sales_saturday"),
        ("What about Sunday?",                      "day_sales_sunday"),
        ("Which day has the highest revenue?",      "best_day"),
    ],
    "expiry_drug": [
        ("Do we have {drug} in stock?",             "stock_check"),
        ("Who supplies {drug}?",                    "supplier_drug"),
        ("What interacts with {drug}?",             "drug_interactions"),
        ("Show full summary of {drug}",             "drug_summary"),
    ],
    "expiry_soon": [
        ("Which item expires first?",               "first_expiry"),
        ("Which drugs are low AND expiring soon?",  "combined_risk"),
        ("Show me the reorder list",                "reorder_list"),
    ],
    "first_expiry": [
        ("Show all batches expiring within 30 days","expiry_soon"),
        ("Which drugs are low AND expiring soon?",  "combined_risk"),
    ],
    "batch_count": [
        ("Do we have {drug} in stock?",             "stock_check"),
        ("When does {drug} expire?",                "expiry_drug"),
        ("Who supplies {drug}?",                    "supplier_drug"),
    ],
    "supplier_drug": [
        ("What is the lead time for {drug}?",       "lead_time"),
        ("Do we have {drug} in stock?",             "stock_check"),
        ("When does {drug} expire?",                "expiry_drug"),
        ("What interacts with {drug}?",             "drug_interactions"),
    ],
    "fastest_supplier": [
        ("Who is our slowest supplier?",            "slowest_supplier"),
        ("Which supplier has the best payment terms?","payment_terms"),
        ("How many suppliers do we have?",          "supplier_count"),
    ],
    "slowest_supplier": [
        ("Who is our fastest supplier?",            "fastest_supplier"),
        ("Which supplier has the best payment terms?","payment_terms"),
    ],
    "supplier_count": [
        ("Who is our fastest supplier?",            "fastest_supplier"),
        ("Which supplier has the best payment terms?","payment_terms"),
    ],
    "payment_terms": [
        ("Who is our fastest supplier?",            "fastest_supplier"),
        ("How many suppliers do we have?",          "supplier_count"),
    ],
    "drug_interactions": [
        ("What are the side effects of {drug}?",    "side_effects"),
        ("What is the dosage for {drug}?",          "dosage"),
        ("What are the contraindications for {drug}?","contraindications"),
        ("Do we have {drug} in stock?",             "stock_check"),
    ],
    "drug_info": [
        ("What interacts with {drug}?",             "drug_interactions"),
        ("What are the side effects of {drug}?",    "side_effects"),
        ("What is the dosage for {drug}?",          "dosage"),
        ("Do we have {drug} in stock?",             "stock_check"),
    ],
    "side_effects": [
        ("What interacts with {drug}?",             "drug_interactions"),
        ("What are the contraindications for {drug}?","contraindications"),
        ("What is the dosage for {drug}?",          "dosage"),
    ],
    "dosage": [
        ("What are the side effects of {drug}?",    "side_effects"),
        ("What interacts with {drug}?",             "drug_interactions"),
        ("What are the contraindications for {drug}?","contraindications"),
    ],
    "contraindications": [
        ("What interacts with {drug}?",             "drug_interactions"),
        ("What are the side effects of {drug}?",    "side_effects"),
    ],
    "drug_summary": [
        ("What interacts with {drug}?",             "drug_interactions"),
        ("When does {drug} expire?",                "expiry_drug"),
        ("Who supplies {drug}?",                    "supplier_drug"),
        ("What are alternatives to {drug}?",        "drug_alternatives"),
    ],
    "drug_alternatives": [
        ("Do we have {drug} in stock?",             "stock_check"),
        ("What interacts with {drug}?",             "drug_interactions"),
    ],
    "daily_briefing": [
        ("Show me the full reorder list",           "reorder_list"),
        ("Which batches are expiring soon?",        "expiry_soon"),
        ("What are our top sellers?",               "top_sellers"),
        ("Give me the revenue forecast",            "revenue_forecast"),
    ],
    "reorder_list": [
        ("Which drugs are low AND expiring soon?",  "combined_risk"),
        ("Which drugs are below reorder level?",    "low_stock"),
        ("Give me the revenue forecast",            "revenue_forecast"),
    ],
    "combined_risk": [
        ("Show me the reorder list",                "reorder_list"),
        ("Which batches are expiring soon?",        "expiry_soon"),
        ("Which drugs are running low?",            "low_stock"),
    ],
    "revenue_forecast": [
        ("What is our average daily revenue?",      "total_summary"),
        ("What are our top sellers?",               "top_sellers"),
        ("Show me this month's sales",             "this_month_sales"),
    ],
    "inventory_summary": [
        ("Which drugs are running low?",            "low_stock"),
        ("Show me the reorder list",                "reorder_list"),
        ("What is our highest margin drug?",        "highest_margin"),
    ],
    "highest_margin": [
        ("What are our most expensive drugs?",      "expensive_drugs"),
        ("What are the cheapest drugs?",            "cheapest_drugs"),
        ("What are our top sellers?",               "top_sellers"),
    ],
    "customer_type_sales": [
        ("What are our top sellers?",               "top_sellers"),
        ("What was yesterday's revenue?",          "yesterday_sales"),
        ("Which day has the highest revenue?",      "best_day"),
    ],
}

# Variant follow-up intents for day-specific follow-ups
DAY_FOLLOWUP_INTENTS = {
    "day_sales_monday":   ("day_sales", "monday"),
    "day_sales_tuesday":  ("day_sales", "tuesday"),
    "day_sales_wednesday":("day_sales", "wednesday"),
    "day_sales_thursday": ("day_sales", "thursday"),
    "day_sales_friday":   ("day_sales", "friday"),
    "day_sales_saturday": ("day_sales", "saturday"),
    "day_sales_sunday":   ("day_sales", "sunday"),
    "top_sellers_revenue":("top_sellers_rev", None),
}

# FR follow-up templates (parallel to EN in FOLLOW_UP_MAP)
FOLLOW_UP_MAP_FR = {
    "stock_check": [
        ("Quand expire {drug}?",                        "expiry_drug"),
        ("Qui fournit {drug}?",                         "supplier_drug"),
        ("Qu'est-ce qui interagit avec {drug}?",       "drug_interactions"),
        ("Quels sont les effets secondaires de {drug}?","side_effects"),
        ("Quelles sont les alternatives à {drug}?",     "drug_alternatives"),
    ],
    "low_stock": [
        ("Montrer la liste complète de réapprovisionnement","reorder_list"),
        ("Quels médicaments sont faibles ET expirent bientôt?","combined_risk"),
        ("Quel est notre résumé de stock?",             "inventory_summary"),
    ],
    "top_sellers": [
        ("Maintenant montrer les {n} pires",            "worst_sellers"),
        ("Top {n} par chiffre d'affaires",             "top_sellers"),
        ("Quel était le chiffre d'hier?",              "yesterday_sales"),
        ("Quel jour a le plus grand chiffre?",          "best_day"),
    ],
    "worst_sellers": [
        ("Maintenant montrer les {n} meilleurs",        "top_sellers"),
        ("Quel était le chiffre d'hier?",              "yesterday_sales"),
        ("Montrer les ventes par type de client",       "customer_type_sales"),
    ],
    "yesterday_sales": [
        ("Combien avons-nous fait ce mois?",            "this_month_sales"),
        ("Montrer le résumé global des ventes",         "total_summary"),
        ("Quel jour a le plus grand chiffre?",          "best_day"),
    ],
    "expiry_drug": [
        ("Avons-nous {drug} en stock?",                 "stock_check"),
        ("Qui fournit {drug}?",                         "supplier_drug"),
        ("Qu'est-ce qui interagit avec {drug}?",       "drug_interactions"),
    ],
    "supplier_drug": [
        ("Quel est le délai pour {drug}?",              "lead_time"),
        ("Avons-nous {drug} en stock?",                 "stock_check"),
        ("Quand expire {drug}?",                        "expiry_drug"),
    ],
    "drug_interactions": [
        ("Quels sont les effets secondaires de {drug}?","side_effects"),
        ("Quelle est la posologie de {drug}?",          "dosage"),
        ("Avons-nous {drug} en stock?",                 "stock_check"),
    ],
    "drug_summary": [
        ("Qu'est-ce qui interagit avec {drug}?",       "drug_interactions"),
        ("Quand expire {drug}?",                        "expiry_drug"),
        ("Qui fournit {drug}?",                         "supplier_drug"),
    ],
    "daily_briefing": [
        ("Montrer la liste de réapprovisionnement",     "reorder_list"),
        ("Quels lots expirent bientôt?",                "expiry_soon"),
        ("Quels sont nos meilleures ventes?",           "top_sellers"),
    ],
}


# ── Pronoun / vague follow-up triggers ────────────────────────────
FOLLOWUP_TRIGGERS_EN = [
    "tell me more","more details","more info","what else","and","also",
    "what about","how about","show more","more about","elaborate",
    "give me more","anything else","what next","continue","go on",
    "now show","show the","and the","flip","reverse","opposite",
    "top","bottom","same for","what about it","and it","and that",
]
FOLLOWUP_TRIGGERS_FR = [
    "dis m'en plus","plus de détails","plus d'informations","et aussi",
    "qu'en est-il","comment","montrer plus","plus sur","élaborer",
    "donner plus","autre chose","quoi d'autre","continuer","et",
    "maintenant","montrer le","et le","et les","inverser","opposé",
    "haut","bas","pareil pour","et ça","et cela",
]


def is_followup_phrase(q: str, lang: str = "en") -> bool:
    """Return True if the question looks like a follow-up / continuation."""
    q_lower = q.lower().strip()
    triggers = FOLLOWUP_TRIGGERS_EN if lang == "en" else FOLLOWUP_TRIGGERS_FR
    # Short queries are almost always follow-ups
    if len(q_lower.split()) <= 4:
        return True
    return any(q_lower.startswith(t) or t in q_lower for t in triggers)


def get_followup_suggestions(last_intent: str, last_drug: str, last_n: int,
                              last_day: str, last_category: str,
                              lang: str = "en", top_n: int = 5) -> list:
    """
    Given the last executed intent + its slots, return follow-up question suggestions.
    Returns list of (question_text, intent_id, drug, n, day, category, city, month).
    """
    fmap = FOLLOW_UP_MAP_FR if lang == "fr" else FOLLOW_UP_MAP
    templates = fmap.get(last_intent, FOLLOW_UP_MAP.get(last_intent, []))
    if not templates:
        return []

    results = []
    for tpl_text, followup_intent_id in templates[:top_n]:
        # Resolve day-specific variants
        actual_intent = followup_intent_id
        override_day  = last_day
        if followup_intent_id in DAY_FOLLOWUP_INTENTS:
            actual_intent, override_day = DAY_FOLLOWUP_INTENTS[followup_intent_id]
            if override_day is None:
                actual_intent = "top_sellers"

        # Fill slots — inherit from previous turn
        drug     = last_drug or ""
        n_val    = last_n or 10
        day_disp = (override_day or last_day or "Saturday").capitalize()

        question_text = tpl_text
        if "{drug}" in question_text:
            if not drug:
                continue
            question_text = question_text.replace("{drug}", drug)
        question_text = question_text.replace("{n}", str(n_val))
        question_text = question_text.replace("{day}", day_disp)

        results.append((question_text, actual_intent, drug, n_val,
                         override_day or last_day, last_category, None, None))
        if len(results) >= top_n:
            break

    return results


def get_combined_suggestions(user_input: str, last_intent_ctx: dict,
                              lang: str = "en", top_n: int = 5) -> list:
    """
    Combine fresh suggestions + follow-up suggestions.
    If the input looks like a follow-up AND we have a last intent,
    prioritise follow-ups. Otherwise prioritise fresh matches.
    """
    last_intent  = last_intent_ctx.get("intent_id", "")
    last_drug    = last_intent_ctx.get("drug", "")
    last_n       = last_intent_ctx.get("n", 10)
    last_day     = last_intent_ctx.get("day", "")
    last_category= last_intent_ctx.get("category", "")

    fresh = get_suggestions(user_input, lang, top_n=top_n)

    # If user typed a follow-up phrase, mix in follow-ups
    if last_intent and is_followup_phrase(user_input, lang):
        followups = get_followup_suggestions(
            last_intent, last_drug, last_n, last_day, last_category, lang, top_n
        )
        # Deduplicate by intent_id
        seen = set()
        combined = []
        for s in followups + fresh:
            if s[1] not in seen:
                combined.append(s)
                seen.add(s[1])
        return combined[:top_n]

    return fresh[:top_n]



def execute_hardcoded_intent(intent_id: str, drug: str, n: int, day: str,
                              category: str, city: str, month: str, lang: str) -> str:
    """Execute a hardcoded intent directly — zero GPT routing."""

    # Stock / Inventory
    if intent_id == "stock_check":
        return execute_query_inventory({"filter": "all", "drug_name": drug, "limit": 5})
    if intent_id == "low_stock":
        return execute_query_inventory({"filter": "below_reorder", "limit": 20})
    if intent_id == "inventory_summary":
        return format_stats()
    if intent_id == "category_browse":
        return execute_query_inventory({"filter": "all", "category": category or "", "limit": 30})
    if intent_id == "drug_summary":
        return format_drug_summary(drug or "")
    if intent_id == "cheapest_drugs":
        return execute_query_inventory({"filter": "cheapest", "sort_by": "price_asc", "limit": n})
    if intent_id == "expensive_drugs":
        return execute_query_inventory({"filter": "most_expensive", "sort_by": "price_desc", "limit": n})
    if intent_id == "highest_margin":
        return execute_query_inventory({"filter": "all", "sort_by": "margin_desc", "limit": n})
    if intent_id == "drug_alternatives":
        return format_alternative(drug or "")

    # Sales
    if intent_id == "top_sellers":
        return execute_query_sales({"period": "all_time", "direction": "top", "sort_by": "units", "limit": n})
    if intent_id == "worst_sellers":
        return execute_query_sales({"period": "all_time", "direction": "bottom", "sort_by": "units", "limit": n})
    if intent_id == "yesterday_sales":
        return execute_query_sales({"period": "last_day"})
    if intent_id == "this_month_sales":
        return execute_query_sales({"period": "current_month"})
    if intent_id == "total_summary":
        return execute_query_sales({"period": "total_summary"})
    if intent_id == "best_day":
        return execute_query_sales({"period": "best_day"})
    if intent_id == "day_sales":
        return execute_query_sales({"period": "day_of_week", "day_name": day or "saturday"})
    if intent_id == "customer_type_sales":
        return execute_query_sales({"period": "customer_type"})

    # Expiry
    if intent_id == "expiry_soon":
        return execute_query_expiry({"within_days": 90, "limit": 20})
    if intent_id == "expiry_drug":
        return execute_query_expiry({"drug_name": drug})
    if intent_id == "first_expiry":
        return execute_query_expiry({"top_only": True, "within_days": 365})
    if intent_id == "expiry_month":
        return execute_query_expiry({"month_name": month or ""})
    if intent_id == "batch_count":
        return execute_query_expiry({"drug_name": drug})
    if intent_id == "multi_batch":
        return execute_query_expiry({"count_only": True, "min_batches": n})
    if intent_id == "expiry_quantity":
        return _expiry_quantity_summary()

    # Suppliers
    if intent_id == "supplier_drug":
        return execute_query_supplier({"drug_name": drug})
    if intent_id == "fastest_supplier":
        return execute_query_supplier({"sort_by": "lead_time", "direction": "asc"})
    if intent_id == "slowest_supplier":
        return execute_query_supplier({"sort_by": "lead_time", "direction": "desc"})
    if intent_id == "supplier_count":
        return execute_query_supplier({})
    if intent_id == "supplier_city":
        return execute_query_supplier({"city": city or ""})
    if intent_id == "payment_terms":
        return execute_query_supplier({"sort_by": "payment_terms"})
    if intent_id == "lead_time":
        return execute_query_supplier({"drug_name": drug})

    # Clinical
    if intent_id == "drug_interactions":
        data = query_neo4j_interaction(drug or "")
        return generate_clinical_answer(drug or "", "interaction", "drug interaction knowledge graph", data)
    if intent_id == "drug_safety":
        data = query_neo4j_interaction(drug or "")
        return generate_clinical_answer(f"Is {drug} safe?", "interaction", "drug interaction knowledge graph", data)
    if intent_id in ("drug_info", "side_effects", "dosage", "contraindications"):
        data = query_neo4j_drug_info(drug or "")
        return generate_clinical_answer(f"{intent_id} {drug}", "drug_info", "drug knowledge graph", data)

    # Operational
    if intent_id == "daily_briefing":
        return format_daily_briefing()
    if intent_id == "reorder_list":
        return format_reorder_list()
    if intent_id == "combined_risk":
        return _combined_low_expiry()
    if intent_id == "stock_reconciliation":
        return format_reconciliation()
    if intent_id == "revenue_forecast":
        return format_revenue_forecast()

    return "❌ Intent not implemented yet."


def _expiry_quantity_summary() -> str:
    df = pd.read_sql_query("""
        SELECT COUNT(*) AS batch_count,
               SUM(b.quantity_remaining) AS total_qty,
               ROUND(SUM(b.quantity_remaining * i.cost_price_usd)::numeric,2) AS est_value
        FROM batches b JOIN inventory i ON b.product_id = i.product_id
        WHERE (b.expiry_date::date - CURRENT_DATE) <= 90
    """, get_engine())
    r = df.iloc[0]
    return (
        f"**Stock Expiring Within 90 Days**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Batches affected | **{r['batch_count']}** |\n"
        f"| Total Units | **{r['total_qty']}** |\n"
        f"| Estimated Value | **${r['est_value']:,.2f}** |\n"
    )


# ══════════════════════════════════════════════════════════════════
# SUGGESTION ENGINE
# Scores a user's typed text against all intents and returns
# the top matched question templates with their intent IDs
# ══════════════════════════════════════════════════════════════════

def score_intent(q_lower: str, keywords: list) -> float:
    """Score how well a question matches an intent's keywords."""
    score = 0.0
    for kw in keywords:
        if kw in q_lower:
            score += len(kw.split()) * 2.0  # longer phrase = higher score
        else:
            # partial word match
            for word in kw.split():
                if len(word) > 3 and word in q_lower:
                    score += 0.5
    return score


def get_suggestions(user_input: str, lang: str = "en", top_n: int = 5):
    """
    Given a user's typed text, return top matching hardcoded questions.
    Returns list of (question_text, intent_id, slots) tuples.
    """
    if not user_input or len(user_input.strip()) < 2:
        return []

    q_lower = user_input.lower().strip()

    # Extract slots
    drug     = _detect_drug(q_lower)
    n        = _detect_number(q_lower, default=10)
    day      = _detect_day(q_lower)
    category = _detect_category(q_lower)
    city     = _detect_city(q_lower)
    month    = _detect_month(q_lower)

    scored = []
    for entry in INTENT_PATTERNS:
        intent_id    = entry[0]
        all_keywords = entry[1]
        needs_drug   = entry[2]
        needs_day    = entry[3]
        needs_cat    = entry[4]

        # Skip if requires drug but none detected (for drug-specific intents)
        if needs_drug and not drug and intent_id not in (
            "drug_info","side_effects","dosage","contraindications",
            "drug_interactions","drug_safety","drug_alternatives"
        ):
            continue
        if needs_day and not day:
            continue

        s = score_intent(q_lower, all_keywords)
        if s > 0:
            scored.append((s, intent_id, drug, n, day, category, city, month))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    seen_intents = set()
    results = []

    for s, intent_id, drug, n, day, category, city, month in scored[:top_n * 2]:
        if intent_id in seen_intents:
            continue
        if intent_id not in QUESTION_TEMPLATES:
            continue

        templates = QUESTION_TEMPLATES[intent_id][lang]
        # Pick best template (first one — they're ordered by natural frequency)
        tpl = templates[0]

        # Fill slots
        day_display = (day or "Saturday").capitalize()
        cat_display = category or "Antibiotics"
        city_display = city or "Harare"
        month_display = (month or "May").capitalize()
        drug_display = drug or ""

        question_text = tpl
        if "{drug}" in question_text:
            if not drug_display:
                continue  # skip if drug needed but not found
            question_text = question_text.replace("{drug}", drug_display)
        question_text = question_text.replace("{n}", str(n))
        question_text = question_text.replace("{day}", day_display)
        question_text = question_text.replace("{category}", cat_display)
        question_text = question_text.replace("{city}", city_display)
        question_text = question_text.replace("{month}", month_display)

        results.append((question_text, intent_id, drug, n, day, category, city, month))
        seen_intents.add(intent_id)

        if len(results) >= top_n:
            break

    return results


def is_hardcoded_match(user_input: str, lang: str = "en") -> bool:
    """Returns True if the question matches at least one hardcoded intent."""
    suggestions = get_suggestions(user_input, lang, top_n=1)
    return len(suggestions) > 0


# ══════════════════════════════════════════════════════════════════
# SQL EXECUTORS (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════

def execute_query_inventory(params: dict) -> str:
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
    where_clauses, sql_params = [], []

    if filt == "below_reorder":
        where_clauses.append("quantity_in_stock <= reorder_level")
        order = "stock_pct ASC"
    elif filt == "cheapest":
        order = "selling_price_usd ASC"
    elif filt == "most_expensive":
        order = "selling_price_usd DESC"

    if drug_name:
        where_clauses.append("(LOWER(generic_name) LIKE %s OR LOWER(brand_name) LIKE %s)")
        sql_params += [f"%{drug_name.lower()}%", f"%{drug_name.lower()}%"]

    if category:
        cat_map = {
            "antibiotic":"Antibiotics","analgesic":"Analgesics",
            "antihypertensive":"Antihypertensives","antidiabetic":"Antidiabetics",
            "antimalarial":"Antimalarials","antifungal":"Antifungals",
            "antiretroviral":"Antiretrovirals","respiratory":"Respiratory",
            "vitamin":"Vitamins/Supplements","gi":"GI medications",
        }
        matched_cat = next((v for k,v in cat_map.items() if k in category.lower()), category)
        where_clauses.append("category = %s")
        sql_params.append(matched_cat)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"""
        SELECT generic_name, brand_name, formulation, strength,
               quantity_in_stock, reorder_level, selling_price_usd,
               cost_price_usd, shelf_location, category,
               ROUND((quantity_in_stock::numeric/NULLIF(reorder_level,0))*100,0) AS stock_pct,
               ROUND(((selling_price_usd - cost_price_usd)/NULLIF(selling_price_usd,0)*100)::numeric,1) AS margin
        FROM inventory {where}
        ORDER BY {order}
        LIMIT %s
    """
    sql_params.append(limit)
    df = pd.read_sql_query(sql, get_engine(), params=tuple(sql_params))

    if df.empty:
        if filt == "below_reorder":
            return "✅ All products are currently above their reorder levels."
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
        header  = f"**{cat_name}** — {len(df)} drugs\n\n"
        header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n|---|---|---|---|---|---|---|\n"
        rows = [
            f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | "
            f"{r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |"
            for _, r in df.iterrows()
        ]
        return header + "\n".join(rows)

    if sort_by in ("margin_desc", "margin_asc"):
        label = "Highest" if sort_by == "margin_desc" else "Lowest"
        header = f"**{label} margin drugs:**\n\n| Drug | Brand | Sell Price | Cost Price | Margin% | Stock |\n|---|---|---|---|---|---|\n"
        rows = [
            f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | "
            f"${r['cost_price_usd']} | {r['margin']}% | {r['quantity_in_stock']} |"
            for _, r in df.iterrows()
        ]
        return header + "\n".join(rows)

    if filt == "cheapest":      label = f"Cheapest {limit} drugs in stock"
    elif filt == "most_expensive": label = f"Most expensive {limit} drugs"
    elif drug_name:             label = f"Drugs matching '{drug_name}'"
    else:                       label = f"Top {limit} drugs"
    header = f"**{label}:**\n\n| Drug | Brand | Price | Stock | Shelf |\n|---|---|---|---|---|\n"
    rows = [
        f"| {r['generic_name']} | {r['brand_name']} | ${r['selling_price_usd']} | {r['quantity_in_stock']} | {r['shelf_location']} |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows)


def execute_query_sales(params: dict) -> str:
    period    = params.get("period", "all_time")
    direction = params.get("direction", "top")
    sort_by   = params.get("sort_by", "revenue")
    day_name  = params.get("day_name", "")
    limit     = max(1, min(params.get("limit", 10), 50))

    if period == "customer_type":  return _sales_customer_type()
    if period == "last_day":       return _sales_last_day()
    if period == "last_week":      return _sales_last_week()
    if period == "current_month":  return _sales_current_month()
    if period == "best_day":       return _sales_best_day()
    if period == "total_summary":  return _sales_total_summary()
    if period == "day_of_week" and day_name: return _sales_day_of_week(day_name)

    order    = "DESC" if direction == "top" else "ASC"
    label    = f"{'Top' if direction == 'top' else 'Bottom'} {limit}"
    col_map  = {"revenue": "total_revenue", "units": "total_units", "transactions": "num_transactions"}
    sort_col = col_map.get(sort_by, "total_revenue")
    sort_lbl = f"by {sort_by}"
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
    header  = f"**{label} Selling Drugs** {sort_lbl} (Last 30 days)\n\n"
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
        dn   = str(r["day_name"]).strip()
        star = " ⭐" if dn == best_name else ""
        rows.append(
            f"| {dn}{star} | **${r['avg_revenue']:,.2f}** | "
            f"${r['total_revenue']:,.2f} | {r['occurrences']} | "
            f"{r['total_transactions']} | {r['total_units']} |"
        )
    return "**Revenue by Day of Week** — best day is **" + best_name + "**\n\n" + "\n".join(rows)


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
    r     = df_total.iloc[0]
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
    r     = df.iloc[0]
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
        f"| Date Range | {first} → {last} |\n"
    )


def _sales_customer_type() -> str:
    df = pd.read_sql_query("""
        SELECT customer_type, COUNT(*) AS num_transactions,
               SUM(quantity_sold) AS total_units,
               ROUND(SUM(total_amount)::numeric, 2) AS total_revenue,
               ROUND((SUM(total_amount)*100.0/(SELECT SUM(total_amount) FROM transactions))::numeric,1) AS revenue_pct
        FROM transactions GROUP BY customer_type ORDER BY total_revenue DESC
    """, get_engine())
    header  = "**Sales by Customer Type**\n\n"
    header += "| Customer Type | Transactions | Units | Revenue | % of Total |\n|---|---|---|---|---|\n"
    rows = [
        f"| {r['customer_type']} | {r['num_transactions']} | "
        f"{r['total_units']} | ${r['total_revenue']:,.2f} | {r['revenue_pct']}% |"
        for _, r in df.iterrows()
    ]
    return header + "\n".join(rows) + f"\n\n**Total: ${df['total_revenue'].sum():,.2f}**"


def _sales_last_day() -> str:
    df_d  = pd.read_sql_query("SELECT date, COUNT(*) AS num_transactions, SUM(quantity_sold) AS total_units, ROUND(SUM(total_amount)::numeric,2) AS total_revenue FROM transactions WHERE date=(SELECT MAX(date) FROM transactions) GROUP BY date", get_engine())
    df_dr = pd.read_sql_query("SELECT i.brand_name, i.generic_name, SUM(t.quantity_sold) AS units, ROUND(SUM(t.total_amount)::numeric,2) AS revenue FROM transactions t JOIN inventory i ON t.product_id=i.product_id WHERE t.date=(SELECT MAX(date) FROM transactions) GROUP BY i.brand_name,i.generic_name ORDER BY revenue DESC", get_engine())
    df_ct = pd.read_sql_query("SELECT customer_type, ROUND(SUM(total_amount)::numeric,2) AS revenue FROM transactions WHERE date=(SELECT MAX(date) FROM transactions) GROUP BY customer_type ORDER BY revenue DESC", get_engine())
    if df_d.empty:
        return "No transactions found."
    r   = df_d.iloc[0]
    out = f"**Sales for {str(r['date'])[:10]}** (Last recorded day)\n\n"
    out += f"Transactions: **{r['num_transactions']}** | Units: **{r['total_units']}** | Revenue: **${r['total_revenue']:,.2f}**\n\n"
    out += "**By Drug:**\n\n| Brand | Generic | Units | Revenue |\n|---|---|---|---|\n"
    out += "\n".join(f"| {row['brand_name']} | {row['generic_name']} | {row['units']} | ${row['revenue']:,.2f} |" for _, row in df_dr.iterrows())
    out += "\n\n**By Customer Type:** " + " | ".join(f"{row['customer_type']}: ${row['revenue']:,.2f}" for _, row in df_ct.iterrows())
    return out


def _sales_last_week() -> str:
    df = pd.read_sql_query("SELECT date, COUNT(*) AS num_transactions, SUM(quantity_sold) AS total_units, ROUND(SUM(total_amount)::numeric,2) AS total_revenue FROM transactions WHERE date::date>=(SELECT MAX(date::date)-7 FROM transactions) GROUP BY date ORDER BY date DESC", get_engine())
    if df.empty:
        return "No transactions found for last week."
    header = "**Last Week Sales**\n\n| Date | Transactions | Units | Revenue |\n|---|---|---|---|\n"
    rows   = [f"| {str(r['date'])[:10]} | {r['num_transactions']} | {r['total_units']} | ${r['total_revenue']:,.2f} |" for _, r in df.iterrows()]
    return header + "\n".join(rows) + f"\n\n**Total: ${df['total_revenue'].sum():,.2f}**"


def _sales_day_of_week(day_name: str) -> str:
    day_map = {"monday":1,"tuesday":2,"wednesday":3,"thursday":4,"friday":5,"saturday":6,"sunday":0}
    day_num = day_map.get(day_name.lower(), 6)
    df_summary = pd.read_sql_query("""
        SELECT COUNT(DISTINCT t.date::date) AS num_days, COUNT(*) AS total_transactions,
               SUM(t.quantity_sold) AS total_units, ROUND(SUM(t.total_amount)::numeric,2) AS total_revenue,
               ROUND(AVG(daily_rev.rev)::numeric,2) AS avg_daily_revenue
        FROM transactions t
        JOIN (SELECT date::date AS d, SUM(total_amount) AS rev FROM transactions GROUP BY date::date) daily_rev
        ON t.date::date = daily_rev.d
        WHERE EXTRACT(DOW FROM t.date::date) = %(dow)s
    """, get_engine(), params={"dow": day_num})
    df_drugs = pd.read_sql_query("""
        SELECT i.brand_name, i.generic_name,
               SUM(t.quantity_sold) AS total_units,
               ROUND(SUM(t.total_amount)::numeric,2) AS total_revenue, COUNT(*) AS transactions
        FROM transactions t JOIN inventory i ON t.product_id=i.product_id
        WHERE EXTRACT(DOW FROM t.date::date) = %(dow)s
        GROUP BY i.brand_name, i.generic_name ORDER BY total_units DESC LIMIT 10
    """, get_engine(), params={"dow": day_num})
    if df_drugs.empty:
        return f"No sales data found for {day_name.capitalize()}s."
    s   = df_summary.iloc[0]
    out = f"**{day_name.capitalize()} Sales Summary** (across {s['num_days']} recorded {day_name.capitalize()}s)\n\n"
    out += f"Transactions: **{s['total_transactions']}** | Units: **{s['total_units']}** | Revenue: **${s['total_revenue']:,.2f}** | Avg: **${s['avg_daily_revenue']:,.2f}**\n\n"
    out += "**Drug Breakdown:**\n\n| Rank | Brand | Generic | Units | Revenue | Transactions |\n|---|---|---|---|---|---|\n"
    out += "\n".join(
        f"| {i+1} | {r['brand_name']} | {r['generic_name']} | {r['total_units']} | ${r['total_revenue']:,.2f} | {r['transactions']} |"
        for i, (_, r) in enumerate(df_drugs.iterrows())
    )
    return out


MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "janvier":1,"février":2,"mars":3,"avril":4,"mai":5,"juin":6,
    "juillet":7,"août":8,"septembre":9,"octobre":10,"novembre":11,"décembre":12
}

def execute_query_expiry(params: dict) -> str:
    drug_name   = params.get("drug_name")
    within_days = params.get("within_days", 90)
    limit       = max(1, min(params.get("limit", 10), 50))
    top_only    = params.get("top_only", False)
    month_name  = params.get("month_name", "")
    count_only  = params.get("count_only", False)
    min_batches = params.get("min_batches", 2)

    if month_name:
        month_num = MONTH_MAP.get(month_name.lower())
        if month_num:
            df = pd.read_sql_query("""
                SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
                       b.quantity_remaining, (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days_remaining
                FROM batches b JOIN inventory i ON b.product_id=i.product_id
                WHERE EXTRACT(MONTH FROM b.expiry_date::date)=%s
                  AND EXTRACT(YEAR FROM b.expiry_date::date)>=EXTRACT(YEAR FROM CURRENT_DATE)
                ORDER BY b.expiry_date ASC
            """, get_engine(), params=(month_num,))
            if df.empty:
                return f"✅ No batches expiring in {month_name.capitalize()}."
            header  = f"**Batches expiring in {month_name.capitalize()}** — {len(df)} found:\n\n"
            header += "| Drug | Brand | Batch | Expiry | Days Left | Qty | Status |\n|---|---|---|---|---|---|---|\n"
            rows = []
            for _, r in df.iterrows():
                d    = r['days_remaining']
                flag = "🚨 URGENT" if d < 30 else ("⚠️ Warning" if d < 90 else "📅 Monitor")
                rows.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |")
            return header + "\n".join(rows)

    if count_only:
        df = pd.read_sql_query("""
            SELECT i.generic_name, i.brand_name, COUNT(b.batch_id) AS batch_count,
                   MIN(b.expiry_date) AS nearest_expiry, SUM(b.quantity_remaining) AS total_qty
            FROM inventory i JOIN batches b ON i.product_id=b.product_id
            GROUP BY i.product_id, i.generic_name, i.brand_name
            HAVING COUNT(b.batch_id) >= %s ORDER BY batch_count DESC
        """, get_engine(), params=(min_batches,))
        if df.empty:
            return f"No drugs found with {min_batches} or more batches."
        header  = f"**Drugs with {min_batches}+ batches:**\n\n"
        header += "| Drug | Brand | Batches | Nearest Expiry | Total Qty |\n|---|---|---|---|---|\n"
        rows = [f"| {r['generic_name']} | {r['brand_name']} | **{r['batch_count']}** | {str(r['nearest_expiry'])[:10]} | {r['total_qty']} |" for _, r in df.iterrows()]
        return header + "\n".join(rows)

    if drug_name:
        df = pd.read_sql_query("""
            SELECT b.batch_number, b.expiry_date, b.quantity_remaining,
                   (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days
            FROM batches b JOIN inventory i ON b.product_id=i.product_id
            WHERE LOWER(i.generic_name) LIKE %s ORDER BY b.expiry_date ASC
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

    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, b.batch_number, b.expiry_date,
               b.quantity_remaining, (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days_remaining
        FROM batches b JOIN inventory i ON b.product_id=i.product_id
        WHERE (b.expiry_date::date-CURRENT_DATE) <= %s
        ORDER BY b.expiry_date ASC LIMIT %s
    """, get_engine(), params=(within_days, 1 if top_only else limit))

    if df.empty:
        return f"✅ No batches expiring within {within_days} days."

    if top_only:
        r    = df.iloc[0]
        d    = r['days_remaining']
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
        rows.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | {str(r['expiry_date'])[:10]} | **{d}** | {r['quantity_remaining']} | {flag} |")
    return header + "\n".join(rows)


def execute_query_supplier(params: dict) -> str:
    drug_name = params.get("drug_name")
    city      = params.get("city")
    sort_by   = params.get("sort_by", "name")
    direction = params.get("direction", "asc")

    if sort_by == "payment_terms" and not drug_name:
        results = run_cypher("MATCH (s:Supplier) RETURN DISTINCT s.name AS supplier, s.payment_terms AS payment_terms, s.lead_time AS lead_time_days, s.city AS city, s.contact AS contact ORDER BY s.payment_terms DESC LIMIT 10")
        if not results:
            return "❌ No supplier payment terms found."
        header = "**Suppliers by Payment Terms:**\n\n| Supplier | Payment Terms | Lead Time | City | Contact |\n|---|---|---|---|---|\n"
        return header + "\n".join(f"| {r['supplier']} | **{r['payment_terms']}** | {r['lead_time_days']} days | {r['city']} | {r['contact']} |" for r in results)

    if sort_by == "lead_time" and not drug_name:
        order = "ASC" if direction != "desc" else "DESC"
        label = "fastest" if order == "ASC" else "slowest"
        results = run_cypher(f"MATCH (s:Supplier) RETURN s.name AS supplier, s.lead_time AS lead_time_days, s.city AS city, s.contact AS contact ORDER BY s.lead_time {order} LIMIT 5")
        if not results:
            return "❌ No supplier information found."
        header = f"**Suppliers by lead time ({label} first):**\n\n| Supplier | Lead Time | City | Contact |\n|---|---|---|---|\n"
        return header + "\n".join(f"| {r['supplier']} | {r['lead_time_days']} days | {r['city']} | {r['contact']} |" for r in results)

    if city:
        results = run_cypher("MATCH (s:Supplier) WHERE toLower(s.city) CONTAINS toLower($city) RETURN s.city AS city, count(s) AS supplier_count, collect(s.name) AS suppliers ORDER BY supplier_count DESC", {"city": city})
        if not results:
            return f"❌ No suppliers found in {city}."
        header = f"**Suppliers in {city.title()}:**\n\n| City | Count | Suppliers |\n|---|---|---|\n"
        return header + "\n".join(f"| {r['city']} | {r['supplier_count']} | {', '.join(r['suppliers'])} |" for r in results)

    if drug_name:
        results = run_cypher("""
            MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier)
            WHERE toLower(d.generic_name) CONTAINS toLower($search)
            RETURN DISTINCT d.generic_name AS drug, s.name AS supplier,
                   s.contact AS contact, s.phone AS phone,
                   s.city AS city, s.lead_time AS lead_time_days, s.payment_terms AS payment_terms
            LIMIT 5
        """, {"search": drug_name})
        if not results:
            df_cat = pd.read_sql_query("SELECT DISTINCT generic_name FROM inventory WHERE LOWER(category) LIKE %s LIMIT 5", get_engine(), params=(f"%{drug_name.lower()}%",))
            if not df_cat.empty:
                cat_drugs = df_cat["generic_name"].tolist()
                results = run_cypher("MATCH (d:Drug)-[:SUPPLIED_BY]->(s:Supplier) WHERE d.generic_name IN $drugs RETURN DISTINCT d.generic_name AS drug, s.name AS supplier, s.contact AS contact, s.phone AS phone, s.city AS city, s.lead_time AS lead_time_days, s.payment_terms AS payment_terms ORDER BY s.lead_time ASC LIMIT 5", {"drugs": cat_drugs})
        if not results:
            return f"❌ No supplier found for {drug_name}."
        if len(results) == 1:
            r = results[0]
            return (
                f"**Supplier for {r['drug']}:**\n\n| Field | Value |\n|---|---|\n"
                f"| Supplier | **{r['supplier']}** |\n| Contact | {r['contact']} |\n"
                f"| Phone | {r['phone']} |\n| City | {r['city']} |\n"
                f"| Lead Time | {r['lead_time_days']} days |\n| Payment Terms | {r['payment_terms']} |\n"
            )
        header = f"**Suppliers for {drug_name}:**\n\n| Drug | Supplier | City | Lead Time | Contact |\n|---|---|---|---|---|\n"
        return header + "\n".join(f"| {r['drug']} | {r['supplier']} | {r['city']} | {r['lead_time_days']} days | {r['contact']} |" for r in results)

    results = run_cypher("MATCH (s:Supplier) RETURN s.city AS city, count(s) AS supplier_count, collect(s.name) AS suppliers ORDER BY supplier_count DESC")
    total   = sum(r['supplier_count'] for r in results)
    header  = f"**{total} suppliers** across {len(results)} cities:\n\n| City | Count | Suppliers |\n|---|---|---|\n"
    return header + "\n".join(f"| {r['city']} | {r['supplier_count']} | {', '.join(r['suppliers'])} |" for r in results)


def format_stats() -> str:
    df = pd.read_sql_query("""
        SELECT category, COUNT(*) AS drug_count, SUM(quantity_in_stock) AS total_units,
               ROUND(AVG(selling_price_usd)::numeric,2) AS avg_price,
               ROUND(SUM(quantity_in_stock*cost_price_usd)::numeric,2) AS inventory_value
        FROM inventory GROUP BY category ORDER BY inventory_value DESC
    """, get_engine())
    total_drugs = df['drug_count'].sum()
    total_value = df['inventory_value'].sum()
    header  = f"**Inventory Summary** — {total_drugs} products across {len(df)} categories\n\nTotal inventory value: **${total_value:,.2f}**\n\n"
    header += "| Category | Drugs | Total Units | Avg Price | Inv. Value |\n|---|---|---|---|---|\n"
    rows = [f"| {r['category']} | {r['drug_count']} | {r['total_units']} | ${r['avg_price']} | ${r['inventory_value']:,.2f} |" for _, r in df.iterrows()]
    return header + "\n".join(rows)


def format_alternative(drug_name: str) -> str:
    if not drug_name:
        return "❌ Please specify a drug name."
    result = pd.read_sql_query("SELECT generic_name, category FROM inventory WHERE LOWER(generic_name) LIKE %s LIMIT 1", get_engine(), params=(f"%{drug_name.lower()}%",))
    if result.empty:
        return f"❌ **{drug_name}** not found in inventory."
    found_name = result.iloc[0]["generic_name"]
    category   = result.iloc[0]["category"]
    df = pd.read_sql_query("SELECT generic_name, brand_name, formulation, strength, quantity_in_stock, selling_price_usd, shelf_location FROM inventory WHERE category=%s AND LOWER(generic_name) NOT LIKE %s AND quantity_in_stock>0 ORDER BY generic_name", get_engine(), params=(category, f"%{drug_name.lower()}%"))
    if df.empty:
        return f"❌ No in-stock alternatives for **{found_name}** in {category}."
    header  = f"**Alternatives to {found_name}** (category: {category})\n\n"
    header += "| Drug | Brand | Form | Strength | Stock | Price | Shelf |\n|---|---|---|---|---|---|---|\n"
    rows = [f"| {r['generic_name']} | {r['brand_name']} | {r['formulation']} | {r['strength']} | {r['quantity_in_stock']} | ${r['selling_price_usd']} | {r['shelf_location']} |" for _, r in df.iterrows()]
    return header + "\n".join(rows) + "\n\n⚠️ **Clinical Note:** Therapeutic substitution requires pharmacist approval."


def format_drug_summary(drug_name: str) -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.formulation, i.strength,
               i.quantity_in_stock, i.reorder_level, i.selling_price_usd, i.cost_price_usd,
               i.shelf_location, i.category,
               MIN(b.expiry_date) AS nearest_expiry,
               (MIN(b.expiry_date::date)-CURRENT_DATE)::INTEGER AS days_to_expiry
        FROM inventory i LEFT JOIN batches b ON i.product_id=b.product_id
        WHERE LOWER(i.generic_name) LIKE LOWER(%s)
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.formulation,i.strength,
                 i.quantity_in_stock,i.reorder_level,i.selling_price_usd,i.cost_price_usd,
                 i.shelf_location,i.category
        LIMIT 1
    """, get_engine(), params=(f"%{drug_name}%",))
    if df.empty:
        return f"❌ **{drug_name}** not found in inventory."
    r = df.iloc[0]
    status = "⚠️ LOW STOCK — reorder needed" if r['quantity_in_stock'] <= r['reorder_level'] else "✅ In Stock"
    expiry_line = ""
    if r.get("days_to_expiry") is not None:
        d        = int(r['days_to_expiry'])
        exp_date = str(r['nearest_expiry'])[:10]
        expiry_line = f"\n🚨 **URGENT:** Nearest batch expires in {d} days ({exp_date})" if d <= 30 else (f"\n⚠️ Nearest expiry: {exp_date} ({d} days)" if d <= 90 else f"\n📅 Nearest expiry: {exp_date} ({d} days)")
    return (
        f"**{r['generic_name']}** ({r['brand_name']}) — {r['formulation']} {r['strength']}\n\n"
        "| Field | Value |\n|---|---|\n"
        f"| **Stock** | {r['quantity_in_stock']} units — {status} |\n"
        f"| **Reorder Level** | {r['reorder_level']} units |\n"
        f"| **Selling Price** | ${r['selling_price_usd']} |\n"
        f"| **Cost Price** | ${r['cost_price_usd']} |\n"
        f"| **Shelf Location** | {r['shelf_location']} |\n"
        f"| **Category** | {r['category']} |"
        f"{expiry_line}"
    )


def format_daily_briefing() -> str:
    today  = date.today().strftime("%A, %d %B %Y")
    df_stock = pd.read_sql_query("SELECT generic_name, brand_name, quantity_in_stock, reorder_level, ROUND((quantity_in_stock::numeric/NULLIF(reorder_level,0))*100,0) AS pct FROM inventory WHERE quantity_in_stock<=reorder_level ORDER BY pct ASC LIMIT 5", get_engine())
    df_exp   = pd.read_sql_query("SELECT i.generic_name, i.brand_name, b.batch_number, (b.expiry_date::date-CURRENT_DATE)::INTEGER AS days_left, b.quantity_remaining FROM batches b JOIN inventory i ON b.product_id=i.product_id WHERE (b.expiry_date::date-CURRENT_DATE)<=30 ORDER BY days_left ASC LIMIT 5", get_engine())
    df_rev   = pd.read_sql_query("SELECT ROUND(SUM(total_amount)::numeric,2) AS revenue, COUNT(*) AS txns, SUM(quantity_sold) AS units FROM transactions WHERE date=(SELECT MAX(date) FROM transactions)", get_engine())
    df_avg   = pd.read_sql_query("SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily FROM (SELECT date, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date) t", get_engine())

    cat_tz = timezone(timedelta(hours=2))
    hour   = datetime.now(tz=cat_tz).hour
    tod    = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 17 else "Good Evening")
    lines  = [f"# 🌅 {tod}! Daily Briefing — {today}\n"]

    rev = df_rev.iloc[0]
    avg = df_avg.iloc[0]['avg_daily']
    trend = "📈 above" if rev['revenue'] > avg else "📉 below"
    lines += [f"## 💰 Yesterday's Revenue\n**${rev['revenue']:,.2f}** ({rev['txns']} transactions, {rev['units']} units)\n30-day avg: **${avg:,.2f}** — Yesterday was {trend} average\n"]

    if df_stock.empty:
        lines.append("## ✅ Stock Levels\nAll products above reorder level.\n")
    else:
        lines += [f"## 🔴 Low Stock — {len(df_stock)} drug(s) need reordering", "| Drug | Brand | Stock | Reorder | % |\n|---|---|---|---|---|"]
        for _, r in df_stock.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['pct']:.0f}% |")
        lines.append("")

    if df_exp.empty:
        lines.append("## ✅ Expiry Status\nNo batches expiring within 30 days.\n")
    else:
        lines += [f"## 🚨 Urgent Expiry — {len(df_exp)} batch(es) expiring within 30 days", "| Drug | Brand | Batch | Days Left | Qty |\n|---|---|---|---|---|"]
        for _, r in df_exp.iterrows():
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['batch_number']} | **{r['days_left']}** | {r['quantity_remaining']} |")
    return "\n".join(lines)


def format_reorder_list() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock, i.reorder_level, i.category,
               COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,1),0) AS avg_daily_sales,
               (i.reorder_level*2-i.quantity_in_stock) AS suggested_order
        FROM inventory i LEFT JOIN transactions t ON i.product_id=t.product_id
        WHERE i.quantity_in_stock<=i.reorder_level
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock,i.reorder_level,i.category
        ORDER BY (i.quantity_in_stock::float/NULLIF(i.reorder_level,1)) ASC
    """, get_engine())
    if df.empty:
        return "✅ All products are above reorder level. No procurement action needed."
    header  = f"## 📋 Procurement Action List — {len(df)} drug(s) to reorder\n\n"
    header += "| Drug | Brand | Current Stock | Reorder Level | Avg Daily Sales | Suggested Order | Category |\n|---|---|---|---|---|---|---|\n"
    rows = [f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['avg_daily_sales']} units/day | **{max(int(r['suggested_order']),1)}** units | {r['category']} |" for _, r in df.iterrows()]
    return header + "\n".join(rows) + "\n\n*Suggested order = 2× reorder level minus current stock.*"


def format_revenue_forecast() -> str:
    df = pd.read_sql_query("""
        SELECT i.generic_name, i.brand_name, i.quantity_in_stock,
               COALESCE(ROUND(SUM(t.quantity_sold)::numeric/30,2),0) AS avg_daily, i.selling_price_usd
        FROM inventory i LEFT JOIN transactions t ON i.product_id=t.product_id
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock,i.selling_price_usd
        ORDER BY (i.quantity_in_stock*i.selling_price_usd) DESC LIMIT 15
    """, get_engine())
    df_daily = pd.read_sql_query("SELECT ROUND(AVG(daily_rev)::numeric,2) AS avg_daily_revenue FROM (SELECT date, SUM(total_amount) AS daily_rev FROM transactions GROUP BY date) t", get_engine())
    avg_daily_rev = float(df_daily.iloc[0]['avg_daily_revenue'])
    lines = [
        "## 📈 Revenue & Stock Forecast\n",
        f"**Average Daily Revenue:** ${avg_daily_rev:,.2f}",
        f"**30-Day Revenue Forecast:** ${avg_daily_rev*30:,.2f}",
        f"**90-Day Revenue Forecast:** ${avg_daily_rev*90:,.2f}\n",
        "**Days of Stock Remaining (Top 15 by value):**\n",
        "| Drug | Brand | Stock | Avg Daily Sales | Days Remaining |\n|---|---|---|---|---|"
    ]
    for _, r in df.iterrows():
        if r['avg_daily'] > 0:
            days = round(r['quantity_in_stock'] / r['avg_daily'])
            flag = "🔴" if days < 30 else ("🟡" if days < 60 else "🟢")
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | {r['avg_daily']}/day | {flag} **{days} days** |")
        else:
            lines.append(f"| {r['generic_name']} | {r['brand_name']} | {r['quantity_in_stock']} | No sales | ∞ |")
    return "\n".join(lines)


def format_reconciliation(drug_name=None) -> str:
    drug_filter = ""
    params = []
    if drug_name:
        drug_filter = "WHERE LOWER(i.generic_name) LIKE %s"
        params = [f"%{drug_name.lower()}%"]
    df = pd.read_sql_query(f"""
        SELECT i.generic_name, i.brand_name, SUM(b.quantity_received) AS total_received,
               SUM(t.quantity_sold) AS total_sold, i.quantity_in_stock,
               (SUM(b.quantity_received)-COALESCE(SUM(t.quantity_sold),0)-i.quantity_in_stock) AS discrepancy
        FROM inventory i LEFT JOIN batches b ON i.product_id=b.product_id
        LEFT JOIN transactions t ON i.product_id=t.product_id
        {drug_filter}
        GROUP BY i.product_id,i.generic_name,i.brand_name,i.quantity_in_stock
        HAVING ABS(SUM(b.quantity_received)-COALESCE(SUM(t.quantity_sold),0)-i.quantity_in_stock)>5
        ORDER BY ABS(SUM(b.quantity_received)-COALESCE(SUM(t.quantity_sold),0)-i.quantity_in_stock) DESC LIMIT 10
    """, get_engine(), params=params if params else None)
    if df.empty:
        return "✅ Stock reconciliation is clean — no significant discrepancies found."
    header  = "## ⚠️ Stock Reconciliation — Discrepancies Found\n\n"
    header += "| Drug | Brand | Received | Sold | Current Stock | Discrepancy |\n|---|---|---|---|---|---|\n"
    rows = []
    for _, r in df.iterrows():
        flag = "🔴" if abs(r['discrepancy']) > 20 else "🟡"
        rows.append(f"| {r['generic_name']} | {r['brand_name']} | {r['total_received']:.0f} | {r['total_sold']:.0f} | {r['current_stock']} | {flag} **{r['discrepancy']:.0f}** |")
    return header + "\n".join(rows) + "\n\n*Discrepancy = Received − Sold − Current Stock.*"


def _combined_low_expiry() -> str:
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
    if df.empty:
        return "✅ No drugs are currently both low on stock AND expiring within 90 days."
    header  = f"**⚠️ {len(df)} drug(s) — LOW STOCK + EXPIRING SOON:**\n\n"
    header += "| Drug | Brand | Stock | Reorder | Stock% | Nearest Expiry | Days Left |\n|---|---|---|---|---|---|---|\n"
    rows = [f"| {r['generic_name']} | {r['brand_name']} | **{r['quantity_in_stock']}** | {r['reorder_level']} | {r['stock_pct']:.0f}% | {str(r['nearest_expiry'])[:10]} | **{r['days_to_expiry']}** |" for _, r in df.iterrows()]
    return header + "\n".join(rows)


# ── Neo4j clinical queries ─────────────────────────────────────────
def query_neo4j_interaction(question: str):
    keywords    = extract_keywords(question)
    search_term = keywords[0] if keywords else question.lower()
    return run_cypher("""
        MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
        WHERE toLower(a.generic_name) CONTAINS toLower($search) OR toLower(b.generic_name) CONTAINS toLower($search)
        RETURN a.generic_name AS drug_a, b.generic_name AS drug_b,
               r.severity AS severity, r.description AS description, r.recommendation AS recommendation
        ORDER BY CASE r.severity WHEN 'Major' THEN 1 WHEN 'Moderate' THEN 2 WHEN 'Minor' THEN 3 ELSE 4 END LIMIT 5
    """, {"search": search_term})


def query_neo4j_drug_info(question: str):
    keywords    = extract_keywords(question)
    search_term = keywords[0] if keywords else question.lower()
    return run_cypher("""
        MATCH (d:Drug)-[:IN_CATEGORY]->(c:Category)
        WHERE toLower(d.generic_name) CONTAINS toLower($search)
        RETURN d.generic_name AS name, d.drug_class AS drug_class, d.indications AS indications,
               d.contraindications AS contraindications, d.side_effects AS side_effects,
               d.adult_dose AS adult_dose, d.pediatric_dose AS pediatric_dose,
               d.prescription AS prescription, d.controlled AS controlled, c.name AS category
        LIMIT 3
    """, {"search": search_term})


CLINICAL_DISCLAIMER = (
    "\n\n---\n⚠️ **Clinical Disclaimer:** This information is sourced from the pharmacy "
    "knowledge base. Always verify with a qualified pharmacist before dispensing."
)

CLINICAL_SYSTEM_PROMPT = """You are a pharmacy data assistant at Sunrise Pharmacy, Harare, Zimbabwe.
You are given STRUCTURED DATA retrieved from the pharmacy knowledge graph.
Your ONLY job is to summarise that data clearly for pharmacy staff.

ABSOLUTE RULES:
1. Use ONLY the data provided below. Never add information from your training knowledge.
2. If the data is empty or does not contain the answer: for interactions say "No recorded interaction found in our knowledge base. This does not confirm safety — verify with a pharmacist or drug reference." For other queries say "This information is not available in our knowledge base."
3. Never invent drug names, doses, quantities, interactions or clinical facts.
4. Keep the answer to 3-5 sentences. Be precise and factual.
5. For interactions, always state the exact severity level (Minor/Moderate/Major).
6. End with: "Source: drug knowledge graph" or "Source: drug interaction knowledge graph".
"""

def generate_clinical_answer(question, intent, source, data, conversation_history=None):
    if not data:
        if intent == "interaction":
            return (
                "No recorded interaction found between these drugs in our knowledge base. "
                "This does not confirm safety — always verify with a clinical pharmacist "
                "or a current drug interaction reference before dispensing."
                + CLINICAL_DISCLAIMER
            )
        return "No information found for that drug in our knowledge base. Please check the drug name." + CLINICAL_DISCLAIMER

    messages = [{"role": "system", "content": CLINICAL_SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"RETRIEVED DATA:\n{json.dumps(data, indent=2)}\n\nQUESTION: {question}\n\nSummarise the above data only."})

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0, max_tokens=400)
    result   = response.choices[0].message.content
    if "Clinical Disclaimer" in result:
        return result
    return result + CLINICAL_DISCLAIMER


# ══════════════════════════════════════════════════════════════════
# GREETING / SYSTEM RESPONSES
# ══════════════════════════════════════════════════════════════════

def _cat_hour() -> int:
    return datetime.now(tz=timezone(timedelta(hours=2))).hour

def get_greeting_response(question: str = "", lang: str = "en") -> str:
    q = question.lower().strip().rstrip("!.,?")
    echo_map = {
        "good morning":"Good morning","morning":"Good morning",
        "good afternoon":"Good afternoon","afternoon":"Good afternoon",
        "good evening":"Good evening","evening":"Good evening",
        "good night":"Good night","hi":"Hi","hey":"Hey","hello":"Hello",
        "howzit":"Howzit","yo":"Hey","sup":"Hey",
        "bonjour":"Bonjour","salut":"Salut","bonsoir":"Bonsoir",
        "bon matin":"Bonjour","allô":"Allô"
    }
    opener = next((v for k,v in echo_map.items() if q.startswith(k)), "Hello")
    if lang == "fr":
        return (f"{opener}! Je suis votre Assistant Pharmacie Sunrise. "
                "Posez-moi des questions sur les niveaux de stock, les dates d'expiration, "
                "les ventes, les fournisseurs ou les interactions médicamenteuses. Comment puis-je vous aider?")
    return (f"{opener}! I'm your Sunrise Pharmacy Assistant. "
            "Ask me about stock levels, expiry dates, sales, suppliers, or drug interactions — "
            "whatever you need. How can I help?")

THANKS_RESPONSES = {
    "en": "You're welcome! Feel free to ask anytime. 😊",
    "fr": "De rien! N'hésitez pas à demander. 😊"
}
FAREWELL_RESPONSES = {
    "en": "Goodbye! Come back anytime you need help. 👋",
    "fr": "Au revoir! Revenez quand vous avez besoin d'aide. 👋"
}
CAUTION_RESPONSES = {
    "en": "⚠️ **Proceed with caution** — this question isn't in my hardcoded list, so the answer may be less reliable. I'll try my best with AI routing.",
    "fr": "⚠️ **Procéder avec prudence** — cette question n'est pas dans ma liste prédéfinie, la réponse peut être moins fiable. Je vais essayer avec le routage IA."
}
OUT_OF_SCOPE = {
    "en": [
        "I specialise in pharmacy operations. Could you rephrase with a pharmacy-related question?",
        "That's outside what I can help with — I focus on inventory, sales, expiry, suppliers and clinical queries.",
        "I'm here for pharmacy operations questions. Anything pharmacy-related I can help with?"
    ],
    "fr": [
        "Je me spécialise dans les opérations pharmaceutiques. Pouvez-vous reformuler avec une question liée à la pharmacie?",
        "Cela dépasse ce que je peux aider — je me concentre sur le stock, les ventes, l'expiration, les fournisseurs et les questions cliniques.",
        "Je suis ici pour les questions opérationnelles de pharmacie. Quelque chose de pharmaceutique à vous aider?"
    ]
}
_oos_idx = 0
def _out_of_scope(lang="en") -> str:
    global _oos_idx
    r = OUT_OF_SCOPE[lang][_oos_idx % len(OUT_OF_SCOPE[lang])]
    _oos_idx += 1
    return r


# ══════════════════════════════════════════════════════════════════
# CAUTION-PATH: GPT routing for unmatched free-type questions
# ══════════════════════════════════════════════════════════════════

def gpt_route_and_respond(question: str, conversation_history=None, lang: str = "en") -> str:
    """GPT fallback for questions that don't match any hardcoded intent."""
    q       = question.lower().strip()
    q_clean = re.sub(r"[?!.,'\u2019 ]+$", "", q).strip()

    # Fast-path checks
    if _is_phrase_match(q_clean, GREETING_TRIGGERS):
        return get_greeting_response(question, lang)
    if _is_phrase_match(q_clean, THANKS_TRIGGERS):
        return THANKS_RESPONSES[lang]
    if _is_phrase_match(q_clean, FAREWELL_TRIGGERS):
        return FAREWELL_RESPONSES[lang]

    # Build context
    context = ""
    if conversation_history:
        last_asst = next((m for m in reversed(conversation_history) if isinstance(m, dict) and m.get("role") == "assistant"), None)
        if last_asst:
            prev = last_asst.get("content", "")
            if isinstance(prev, list):
                prev = " ".join(c.get("text","") if isinstance(c,dict) else str(c) for c in prev)
            prev = str(prev)[:300]
            sort_hint = ""
            if "by revenue" in prev.lower(): sort_hint = " Previous query used sort_by=revenue."
            elif "by units" in prev.lower(): sort_hint = " Previous query used sort_by=units."
            if prev:
                context = f"\n\nPrevious assistant response:\n{prev}{sort_hint}"

    lang_instruction = "Respond in French." if lang == "fr" else "Respond in English."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    f"You are a pharmacy operations assistant. {lang_instruction} "
                    "For questions completely unrelated to pharmacy, respond: 'OUT_OF_SCOPE'"
                )},
                {"role": "user", "content": question + context}
            ],
            temperature=0.0, max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        if result == "OUT_OF_SCOPE":
            return _out_of_scope(lang)
        return result
    except Exception as e:
        return f"⚠️ Something went wrong. Please try rephrasing. *(Error: {str(e)})*"


def _is_phrase_match(q_clean: str, trigger_set: set) -> bool:
    if q_clean in trigger_set:
        return True
    for t in trigger_set:
        if q_clean.startswith(t + " ") or q_clean == t:
            return True
    return False


# ══════════════════════════════════════════════════════════════════
# MAIN RESPOND FUNCTION
# ══════════════════════════════════════════════════════════════════

def respond_hardcoded(intent_id: str, drug: str, n: int, day: str,
                      category: str, city: str, month: str,
                      lang: str, chat_history, search_history):
    """Execute a hardcoded intent from a suggestion button click."""
    try:
        answer  = execute_hardcoded_intent(intent_id, drug, n, day, category, city, month, lang)
        # Determine source label
        clinical_intents = {"drug_interactions","drug_safety","drug_info","side_effects","dosage","contraindications"}
        if intent_id in clinical_intents:
            header = "*🧪 Clinical data — drug knowledge graph*\n\n" if not answer.lstrip().startswith("*🧪") else ""
        else:
            header = "*📦 Operational data*\n\n"
        full_answer = header + answer
    except Exception as e:
        full_answer = f"⚠️ Something went wrong. Please try rephrasing.\n\n*Details: {str(e)}*"

    chat_history   = list(chat_history or [])
    search_history = list(search_history or [])
    chat_history.append({"role": "user",      "content": f"[{intent_id}]"})
    chat_history.append({"role": "assistant", "content": full_answer})
    if intent_id not in search_history:
        search_history.insert(0, intent_id)
    search_history = search_history[:15]
    history_md = "\n".join(f"- {h}" for h in search_history)
    return (
        chat_history, search_history,
        gr.update(choices=search_history, value=None),
        gr.update(value=history_md),
        ""  # clear brief
    )


def respond(message, chat_history, search_history, lang, last_intent_ctx=None):
    """Main respond — routes to hardcoded or GPT caution path."""
    if last_intent_ctx is None:
        last_intent_ctx = {}
    if not message or not message.strip():
        return "", chat_history, search_history, gr.update(), gr.update(), "", [], last_intent_ctx

    corrected_q, correction_note = fuzzy_correct_question(message)
    q_lower = corrected_q.lower().strip()

    # ── System fast-paths ──
    q_clean = re.sub(r"[?!.,'\u2019 ]+$", "", q_lower).strip()
    if _is_phrase_match(q_clean, GREETING_TRIGGERS):
        answer = get_greeting_response(message, lang)
        full_answer = answer
        suggestions = []
    elif _is_phrase_match(q_clean, THANKS_TRIGGERS):
        full_answer = THANKS_RESPONSES[lang]
        suggestions = []
    elif _is_phrase_match(q_clean, FAREWELL_TRIGGERS):
        full_answer = FAREWELL_RESPONSES[lang]
        suggestions = []
    else:
        # ── Try hardcoded + follow-up matching ──
        suggestions = get_combined_suggestions(corrected_q, last_intent_ctx, lang, top_n=5)

        if suggestions:
            # Has matches — show suggestions, don't execute yet
            # Return without adding to chat — suggestions shown in UI
            suggestion_texts = [s[0] for s in suggestions]
            if correction_note:
                return ("", chat_history, search_history,
                        gr.update(), gr.update(), "", suggestion_texts)
            return ("", chat_history, search_history,
                    gr.update(), gr.update(), "", suggestion_texts, last_intent_ctx)
        else:
            # No hardcoded match — caution path via GPT
            conversation_history = [{"role": t["role"], "content": t["content"]} for t in (chat_history or [])]
            caution_header = CAUTION_RESPONSES[lang] + "\n\n"
            gpt_answer = gpt_route_and_respond(corrected_q, conversation_history, lang)
            full_answer = caution_header + gpt_answer
            if correction_note:
                full_answer = correction_note + "\n\n" + full_answer
            suggestions = []

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
            gr.update(value=history_md), "", suggestions, last_intent_ctx)


def execute_suggestion(suggestion_text: str, intent_id: str, drug: str, n_val: int,
                        day: str, category: str, city: str, month: str,
                        lang: str, chat_history, search_history):
    """Called when user clicks a suggestion button."""
    try:
        answer = execute_hardcoded_intent(intent_id, drug, n_val, day, category, city, month, lang)
        clinical_intents = {"drug_interactions","drug_safety","drug_info","side_effects","dosage","contraindications"}
        if intent_id in clinical_intents:
            header = "*🧪 Clinical data — drug knowledge graph*\n\n" if not answer.lstrip().startswith("*🧪") else ""
        else:
            header = "*📦 Operational data*\n\n"
        full_answer = header + answer
    except Exception as e:
        full_answer = f"⚠️ Something went wrong.\n\n*Details: {str(e)}*"

    chat_history   = list(chat_history or [])
    search_history = list(search_history or [])
    chat_history.append({"role": "user",      "content": suggestion_text})
    chat_history.append({"role": "assistant", "content": full_answer})
    if suggestion_text not in search_history:
        search_history.insert(0, suggestion_text)
    search_history = search_history[:15]
    history_md = "\n".join(f"- {h}" for h in search_history)
    new_ctx = {"intent_id": intent_id, "drug": drug or "", "n": n_val,
              "day": day or "", "category": category or ""}
    followups = get_followup_suggestions(
        intent_id, drug, n_val, day, category, lang, top_n=5
    )
    return (chat_history, search_history,
            gr.update(choices=search_history, value=None),
            gr.update(value=history_md), "", followups, new_ctx)


def filter_drugs(search_text):
    if not search_text or len(search_text) < 2:
        return gr.update(choices=DRUG_NAMES[:20])
    matches = [d for d in DRUG_NAMES if search_text.lower() in d.lower()][:20]
    return gr.update(choices=matches if matches else DRUG_NAMES[:20])


def drug_summary_respond(drug_name, chat_history, search_history, lang):
    if not drug_name:
        return chat_history, search_history, gr.update(), gr.update(), "", []
    try:
        answer      = format_drug_summary(drug_name)
        full_answer = "*📦 Operational data — inventory + batch records*\n\n" + answer
    except Exception as e:
        full_answer = f"⚠️ Error: {str(e)}"
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
            gr.update(value=history_md), "", [])


def export_chat(chat_history):
    if not chat_history:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"pharmacy_chat_{timestamp}.txt"
    lines = ["Netrisyl Pharmacy Assistant — Chat Export", "Sunrise Pharmacy | Harare, Zimbabwe",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*60, ""]
    for m in chat_history:
        role = "Staff" if m["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{m['content']}\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return filename


# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════

with gr.Blocks(title="Netrisyl Pharmacy Assistant", css="""
.suggestion-radio label { cursor: pointer !important; }
.suggestion-radio .wrap { gap: 4px !important; }
.suggestion-radio input[type="radio"] { display: none !important; }
.suggestion-radio .svelte-1gfkn6j, .suggestion-radio [data-testid="radio-item"] {
    border-left: 3px solid #f97316 !important;
    background: #fff8f0 !important;
    padding: 8px 14px !important;
    border-radius: 6px !important;
    margin-bottom: 4px !important;
    cursor: pointer !important;
    width: 100% !important;
    font-size: 13px !important;
}
.suggestion-btn {
    text-align: left !important;
    font-size: 13px !important;
    width: 100% !important;
    display: block !important;
    padding: 8px 14px !important;
    margin-bottom: 4px !important;
    border-radius: 6px !important;
    border-left: 3px solid #f97316 !important;
    background: #fff8f0 !important;
    color: #1a1a1a !important;
    font-weight: normal !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 36px !important;
    line-height: 1.4 !important;
    overflow: visible !important;
    word-wrap: break-word !important;
}
.suggestion-btn:hover {
    background: #fff0e0 !important;
    border-left-color: #ea6c00 !important;
}
.caution-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 8px 12px; border-radius: 4px; }
.lang-btn { min-width: 100px !important; }
""") as demo:

    gr.HTML("""
    <script>
    function scrollChat() {
        const chatbots = document.querySelectorAll('.chatbot,[class*="chatbot"]');
        chatbots.forEach(el => { el.scrollTop = el.scrollHeight; });
        const msgs = document.querySelectorAll('.message-wrap,.messages');
        msgs.forEach(el => { el.scrollTop = el.scrollHeight; });
    }
    const obs = new MutationObserver(scrollChat);
    document.addEventListener('DOMContentLoaded', () => {
        const t = document.querySelector('.gradio-container');
        if (t) obs.observe(t, {childList:true, subtree:true});
        setInterval(scrollChat, 500);
    });
    </script>
    """)

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
                padding:16px 24px;border-radius:10px;margin-bottom:16px;
                display:flex;align-items:center;justify-content:space-between;">
        <img src="https://huggingface.co/spaces/Sylvester1922/Netrisyl_pharmacy_assistant/resolve/main/NI_Logo.png"
             style="height:70px;object-fit:contain;" alt="Netrisyl Insights"
             onerror="this.style.display='none'"/>
        <div style="text-align:center;flex:1;">
            <h1 style="color:white;margin:0;font-size:24px;">💊 Pharmacy Assistant</h1>
            <p style="color:#aed6f1;margin:4px 0 0 0;font-size:13px;">
                Powered by Neo4j Knowledge Graph + GPT-4o-mini | Harare, Zimbabwe
            </p>
        </div>
        <div style="width:180px;"></div>
    </div>
    """)

    # ── Language state ────────────────────────────────────────────
    lang_state = gr.State("en")

    with gr.Row():
        # ── LEFT sidebar ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🌐 Language / Langue")
            with gr.Row():
                btn_en = gr.Button("🇬🇧 English", variant="primary",  size="sm", elem_classes=["lang-btn"])
                btn_fr = gr.Button("🇫🇷 Français", variant="secondary", size="sm", elem_classes=["lang-btn"])
            gr.Markdown("---")
            gr.Markdown("### 🔍 Drug Lookup")
            drug_search   = gr.Textbox(placeholder="Type e.g. amox...", label="Search drug name")
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
**Response types:**
- 📦 *Operational* — direct SQL, no AI
- 🧪 *Clinical* — AI summary + disclaimer
- ⚠️ *Caution* — GPT routing (less reliable)
            """)

        # ── CENTRE — Chat + Suggestions ──────────────────────────
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(label="Pharmacy Assistant", height=380, autoscroll=True)

            # ── Suggestion panel ──────────────────────────────────
            suggestion_state = gr.State([])
            with gr.Column(visible=True) as suggestion_panel:
                suggestion_label = gr.Markdown("", visible=False)
                suggestion_radio = gr.Radio(
                    choices=[],
                    label="",
                    value=None,
                    visible=False,
                    interactive=True,
                    elem_classes=["suggestion-radio"]
                )

            brief_box = gr.Textbox(label="💡 Key Points", placeholder="Ask a question, then click Brief", interactive=False, lines=2)

            with gr.Row():
                msg       = gr.Textbox(placeholder="Type your question... / Posez votre question...", label="", scale=4)
                submit    = gr.Button("Ask / Demander", variant="primary", scale=1)
                brief_btn = gr.Button("💡 Brief", variant="secondary", scale=1)
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Voice / Voix", visible=True)
            with gr.Row():
                export_btn  = gr.Button("📥 Export Chat", variant="secondary", scale=1)
                export_file = gr.File(label="Download", scale=2, visible=False)

        # ── RIGHT sidebar — History ───────────────────────────────
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🕘 Search History")
            history_dropdown = gr.Dropdown(choices=[], label="Re-ask a past question", interactive=True)
            history_display  = gr.Markdown("*No searches yet*")

    gr.HTML("""
    <div style="text-align:center;margin-top:16px;color:#7f8c8d;font-size:12px;">
        Netrisyl Insights · Harare, Zimbabwe · Data. Analytics. Intelligence.
    </div>
    """)

    # ── State ────────────────────────────────────────────────────
    search_history_state = gr.State([])
    last_intent_ctx_state = gr.State({})
    # suggestion_radio replaces individual buttons

    # ── Helper to update suggestion buttons ──────────────────────
    def update_suggestion_buttons(suggestions):
        """Return gr.update for the Radio + label."""
        if suggestions:
            choices = [s[0] for s in suggestions]
            label_update = gr.update(value="### 💡 Did you mean one of these? Click a question for a guaranteed answer:", visible=True)
            radio_update = gr.update(choices=choices, value=None, visible=True)
        else:
            label_update = gr.update(value="", visible=False)
            radio_update = gr.update(choices=[], value=None, visible=False)
        return [label_update, radio_update]

    # ── Main submit handler ───────────────────────────────────────
    def handle_submit(message, chat_history, search_history, lang, last_intent_ctx):
        result = respond(message, chat_history, search_history, lang, last_intent_ctx)
        # result: (cleared_msg, ch, sh, dropdown, history_md, brief, suggestions, new_ctx)
        if len(result) == 8:
            cleared_msg, ch, sh, dd_upd, hist_upd, brief, suggestions, new_ctx = result
        else:
            cleared_msg, ch, sh, dd_upd, hist_upd, brief, suggestions = result
            new_ctx = last_intent_ctx
        btn_updates = update_suggestion_buttons(suggestions)
        if len(btn_updates) == 2:
            label_upd, radio_upd = btn_updates
        else:
            label_upd = gr.update(visible=False)
            radio_upd = gr.update(choices=[], value=None, visible=False)
        return [cleared_msg, ch, sh, dd_upd, hist_upd, brief, suggestions, new_ctx, label_upd, radio_upd]

    submit_outputs = [
        msg, chatbot, search_history_state,
        history_dropdown, history_display, brief_box,
        suggestion_state,
        last_intent_ctx_state,
        suggestion_label,
        suggestion_radio
    ]

    submit.click(handle_submit,
        [msg, chatbot, search_history_state, lang_state, last_intent_ctx_state],
        submit_outputs)
    msg.submit(handle_submit,
        [msg, chatbot, search_history_state, lang_state, last_intent_ctx_state],
        submit_outputs)

    # ── Suggestion button clicks ──────────────────────────────────
    def make_suggestion_handler(btn_idx):
        def handler(suggestions, chat_history, search_history, lang, last_ctx):
            if not suggestions or btn_idx >= len(suggestions):
                return [chat_history, search_history, gr.update(), gr.update(), "", [], last_ctx] + [gr.update(visible=False)]*5 + [gr.update(visible=False)]
            s = suggestions[btn_idx]
            suggestion_text = s[0]
            intent_id = s[1]
            drug      = s[2]
            n_val     = s[3]
            day       = s[4]
            category  = s[5]
            city      = s[6]
            month     = s[7]
            result = execute_suggestion(
                suggestion_text, intent_id, drug, n_val, day, category, city, month, lang, chat_history, search_history
            )
            if len(result) == 7:
                ch, sh, dd_upd, hist_upd, brief, new_suggestions, new_ctx = result
            else:
                ch, sh, dd_upd, hist_upd, brief, new_suggestions = result
                new_ctx = {}
            btn_updates = update_suggestion_buttons(new_suggestions)
            if len(btn_updates) == 2:
                label_upd, radio_upd = btn_updates
            else:
                label_upd = gr.update(visible=False)
                radio_upd = gr.update(choices=[], value=None, visible=False)
            return [ch, sh, dd_upd, hist_upd, brief, new_suggestions, new_ctx, label_upd, radio_upd]
        return handler

    sug_click_outputs = [
        chatbot, search_history_state,
        history_dropdown, history_display, brief_box,
        suggestion_state,
        last_intent_ctx_state,
        suggestion_label,
        suggestion_radio
    ]

    # ── Radio selection handler ───────────────────────────────────
    def handle_radio_select(selected_text, suggestions, chat_history, search_history, lang, last_ctx):
        """Called when user clicks a suggestion in the Radio list."""
        if not selected_text or not suggestions:
            return [chat_history, search_history, gr.update(), gr.update(), "", suggestions, last_ctx, gr.update(visible=False), gr.update(choices=[], value=None, visible=False)]
        # Find the matching suggestion by text
        match = next((s for s in suggestions if s[0] == selected_text), None)
        if not match:
            return [chat_history, search_history, gr.update(), gr.update(), "", suggestions, last_ctx, gr.update(visible=False), gr.update(choices=[], value=None, visible=False)]
        suggestion_text, intent_id, drug, n_val, day, category, city, month = match
        result = execute_suggestion(suggestion_text, intent_id, drug, n_val, day, category, city, month, lang, chat_history, search_history)
        if len(result) == 7:
            ch, sh, dd_upd, hist_upd, brief, new_suggestions, new_ctx = result
        else:
            ch, sh, dd_upd, hist_upd, brief, new_suggestions = result
            new_ctx = {}
        btn_updates = update_suggestion_buttons(new_suggestions)
        if len(btn_updates) == 2:
            label_upd, radio_upd = btn_updates
        else:
            label_upd = gr.update(visible=False)
            radio_upd = gr.update(choices=[], value=None, visible=False)
        return [ch, sh, dd_upd, hist_upd, brief, new_suggestions, new_ctx, label_upd, radio_upd]

    suggestion_radio.change(handle_radio_select,
        [suggestion_radio, suggestion_state, chatbot, search_history_state, lang_state, last_intent_ctx_state],
        sug_click_outputs)

    # ── Language toggle ───────────────────────────────────────────
    def set_lang_en():
        return "en", gr.update(variant="primary"), gr.update(variant="secondary")
    def set_lang_fr():
        return "fr", gr.update(variant="secondary"), gr.update(variant="primary")

    btn_en.click(set_lang_en, [], [lang_state, btn_en, btn_fr])
    btn_fr.click(set_lang_fr, [], [lang_state, btn_en, btn_fr])

    # ── Voice input ───────────────────────────────────────────────
    def transcribe_audio(audio_path, chat_history, search_history, lang):
        if not audio_path:
            return ["", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), "", []] + [gr.update(visible=False)]*5 + [gr.update(visible=False)]
        try:
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
            result = handle_submit(transcript.text, chat_history, search_history, lang)
            return result[:7] + [gr.update(value=None)] + result[7:]
        except Exception:
            return ["", chat_history, search_history, gr.update(), gr.update(), gr.update(value=None), "", []] + [gr.update(visible=False)]*5 + [gr.update(visible=False)]

    audio_input.stop_recording(transcribe_audio,
        [audio_input, chatbot, search_history_state, lang_state],
        [msg, chatbot, search_history_state, history_dropdown, history_display, audio_input, brief_box, suggestion_state, suggestion_label, suggestion_radio])

    # ── History re-ask ────────────────────────────────────────────
    def reask_from_history(selected_question, chat_history, search_history, lang):
        if not selected_question:
            return ["", chat_history, search_history, gr.update(), gr.update(), "", []] + [gr.update(visible=False)]*5 + [gr.update(visible=False)]
        return handle_submit(selected_question, chat_history, search_history, lang)

    history_dropdown.change(reask_from_history,
        [history_dropdown, chatbot, search_history_state, lang_state, last_intent_ctx_state],
        submit_outputs)

    # ── Drug lookup sidebar ───────────────────────────────────────
    drug_search.change(filter_drugs, [drug_search], [drug_dropdown])

    def drug_lookup_handler(drug_name, chat_history, search_history, lang):
        ch, sh, dd, hist, brief, sug = drug_summary_respond(drug_name, chat_history, search_history, lang)
        btn_updates = update_suggestion_buttons(sug)
        if len(btn_updates) == 2:
            label_upd, radio_upd = btn_updates
        else:
            label_upd = gr.update(visible=False)
            radio_upd = gr.update(choices=[], value=None, visible=False)
        return [ch, sh, dd, hist, brief, sug, label_upd, radio_upd]

    drug_lookup_btn.click(drug_lookup_handler,
        [drug_dropdown, chatbot, search_history_state, lang_state],
        [chatbot, search_history_state, history_dropdown, history_display, brief_box, suggestion_state, suggestion_label, suggestion_radio])

    # ── Export ────────────────────────────────────────────────────
    def do_export(chat_history):
        f = export_chat(chat_history)
        return gr.update(value=f, visible=True) if f else gr.update(visible=False)
    export_btn.click(do_export, [chatbot], [export_file])

    # ── Brief ─────────────────────────────────────────────────────
    def do_brief(chat_history, lang):
        if not chat_history:
            return "No response yet." if lang == "en" else "Pas encore de réponse."
        try:
            last_response = ""
            for entry in reversed(chat_history):
                if isinstance(entry, dict) and entry.get("role") == "assistant":
                    content = entry.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(c.get("text","") if isinstance(c,dict) else str(c) for c in content)
                    last_response = str(content)
                    break
                elif isinstance(entry, (list, tuple)) and len(entry) > 1:
                    last_response = str(entry[1] or "")
                    break
            if not last_response:
                return "No response yet." if lang == "en" else "Pas encore de réponse."
            lang_instruction = "Respond in French." if lang == "fr" else "Respond in English."
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    f"Summarise this pharmacy data in 2-3 clear sentences for a pharmacy manager. "
                    f"Focus on the most important numbers and actionable insights. No bullet points or markdown. {lang_instruction}\n\nResponse:\n"
                    + last_response[:1500] + "\n\nBrief summary:"
                )}],
                temperature=0.0, max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Could not generate brief: {str(e)}"

    brief_btn.click(do_brief, [chatbot, lang_state], [brief_box])


demo.launch()