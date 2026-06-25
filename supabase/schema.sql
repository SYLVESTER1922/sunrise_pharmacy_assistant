-- =====================================================================
-- Sunrise Pharmacy Assistant -- Database Schema
-- Client: Sunrise Pharmacy
-- Stack: Supabase (PostgreSQL)
-- Note: lobels_materials and lobels_stores tables excluded (separate project)
-- =====================================================================

CREATE TABLE IF NOT EXISTS inventory (
  product_id          TEXT,
  generic_name        TEXT,
  brand_name          TEXT,
  formulation         TEXT,
  strength            TEXT,
  unit_of_measure     TEXT,
  quantity_in_stock   BIGINT,
  reorder_level       BIGINT,
  cost_price_usd      DOUBLE PRECISION,
  selling_price_usd   DOUBLE PRECISION,
  shelf_location      TEXT,
  category            TEXT,
  supplier_id         TEXT
);

CREATE TABLE IF NOT EXISTS batches (
  batch_id            TEXT,
  product_id          TEXT,
  batch_number        TEXT,
  supplier_id         TEXT,
  date_received       TEXT,
  expiry_date         TEXT,
  quantity_received   BIGINT,
  quantity_remaining  BIGINT
);

CREATE TABLE IF NOT EXISTS drug_knowledge (
  product_id              TEXT,
  generic_name            TEXT,
  drug_class              TEXT,
  indications             TEXT,
  contraindications       TEXT,
  common_side_effects     TEXT,
  adult_dose              TEXT,
  pediatric_dose          TEXT,
  prescription_required   TEXT,
  controlled_substance    TEXT
);

CREATE TABLE IF NOT EXISTS suppliers (
  supplier_id     TEXT,
  supplier_name   TEXT,
  contact_person  TEXT,
  phone           BIGINT,
  email           TEXT,
  city            TEXT,
  lead_time_days  BIGINT,
  payment_terms   TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
  interaction_id          TEXT,
  drug_a                  TEXT,
  drug_b                  TEXT,
  severity                TEXT,
  interaction_description TEXT,
  clinical_recommendation TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
  transaction_id  TEXT,
  date            TEXT,
  product_id      TEXT,
  quantity_sold   BIGINT,
  unit_price      DOUBLE PRECISION,
  total_amount    DOUBLE PRECISION,
  staff_id        TEXT,
  customer_type   TEXT
);