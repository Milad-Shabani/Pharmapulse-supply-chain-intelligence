# Methodology

## 1. Data generation

`src/pharmapulse/data_generation/` builds the synthetic network from
interpretable, seeded components rather than pure noise:

- **Products** (`products.py`): 42 SKUs across 9 therapeutic
  categories, each tagged with a `seasonality_profile`
  (`strong_winter`, `mild_winter`, `winter_bump`, `spring_summer`,
  `strong_summer`, `new_year_bump`, `flat_chronic`, `flat`) that
  encodes *how* that category's real-world demand moves through the
  year - antibiotics and respiratory products spike with the winter
  cold/flu season, allergy medication (cetirizine, loratadine) spikes
  in spring/summer, vitamins spike every January, and chronic-disease
  medication (cardiovascular, diabetes, endocrine) stays essentially
  flat with only a slow secular uptrend.
- **Demand** (`demand_curve.py`): expected daily demand is
  `base(product) x market_weight(center) x seasonal(product, day) x
  trend(product, day) x promo(product, day) x weekday_factor(day)`,
  sampled from a Gamma-Poisson (Negative Binomial) mixture for
  realistic over-dispersion. OTC-style categories (analgesics,
  vitamins, respiratory, GI) get randomly scheduled promotional
  uplift events; prescription categories don't.
- **Production batches** (`batches.py`): each SKU is produced in
  discrete campaigns (a real GMP manufacturing pattern) sized to
  roughly cover 2.5-5 weeks of network demand, with a ~2% QC-failure
  rate, at whichever of the 3 plants is assigned to that SKU's
  category.
- **Inventory** (`inventory_sim.py`): a genuine periodic-review
  (s, S) simulation - reorder point and order-up-to level are
  computed from each series' own average demand, variability, and
  lead time; the day-by-day simulation produces realistic stockouts
  when demand exceeds on-hand stock. This is deliberately a *naive*
  policy (reorder point based on trailing averages, blind to trend
  and seasonality) so that the network's observed fill rate
  **degrades over time as the business grows** (84.5% in 2023 ->
  81.0% in 2024) - this is the business problem the forecasting +
  planning layer described below is built to fix.

All generators are seeded (`numpy.random.default_rng`) for full
reproducibility from `scripts/generate_sample_data.py`.

## 2. Feature engineering

`src/pharmapulse/features/build_features.py` aggregates the ~430K-row
daily fact table to weekly grain per (product, center) - matching how
pharma distribution replenishment cycles actually operate - and builds
lag (1/2/3/4/8/52 weeks), rolling mean/std (4/8/12 weeks), and
calendar (week-of-year, cyclical encodings) features, plus categorical
codes for product/center/category/region/seasonality profile.

## 3. Forecasting model: one global LightGBM, two quantiles

Rather than fitting 588 separate per-series models, a **single global
LightGBM model** is trained across all (product, center) series
together, with series identity encoded as categorical features. This
is the standard architecture for demand forecasting at this scale
(the same approach used in the M5 forecasting competition and in
production systems at large retailers/distributors) - it lets the
model borrow statistical strength across similar SKUs and centers,
which matters most for lower-volume series that don't have enough
history to model individually.

Two models are trained on the identical feature set with different
`objective="quantile"` settings:

- **P50** (median) - the number shown to a planner as "expected
  demand".
- **P95** - used directly as a **quantile-regression-native safety
  buffer** (see below), rather than assuming demand is Normally
  distributed the way the classical `mean + z*sigma` formula does.
  Several categories here (antibiotics, allergy medication) have
  sharply right-skewed, seasonal demand that a Normal assumption
  would under-buffer.

**Backtest**: the final 12 weeks of the 106-week history are held out;
the model is trained on everything before that and evaluated on the
held-out weeks. **Future forecast**: the model is refit on the full
history and rolled forward 12 weeks using a recursive procedure (each
week's own forecast feeds the next week's lag features), producing
genuine out-of-sample projections beyond the observed dataset.

### Results (see `reports/model_performance.csv` for the full table)

| Segment | WAPE | R2 |
|---|---:|---:|
| Network overall | 3.2% | 0.977 |
| Best category (Gastrointestinal) | 2.6% | 0.988 |
| Weakest category (Vitamins - promo-driven) | 3.6% | 0.948 |

P95 interval coverage on the held-out set is 93.8%, close to the
nominal 95% target - a reasonable quantile calibration for a
promotion- and seasonality-heavy demand series.

## 4. Inventory planning layer

- **ABC/XYZ segmentation** (`planning/abc_xyz.py`): ABC by Pareto
  revenue share (A: top 80%, B: next 15%, C: remainder); XYZ by
  coefficient of variation of weekly demand (X: CV<0.5, Y: 0.5-1.0,
  Z: >=1.0). Computed at (product, center) grain so replenishment
  policy can be differentiated down to where decisions are actually
  made.
- **Safety stock / reorder point** (`planning/safety_stock.py`):
  `reorder_point = P50_daily * lead_time + (P95_daily - P50_daily) * lead_time`,
  `order_up_to = reorder_point + P50_daily * review_period`. This
  quantile-spread approach is a direct, distribution-free alternative
  to the textbook Normal-distribution safety-stock formula.
- **Expiry / waste risk** (`planning/expiry_risk.py`): a documented
  FIFO approximation - unexpired, QC-passed batches are consumed
  oldest-expiry-first until the running total matches current network
  stock, and the resulting near-expiry quantity is compared against
  forecasted demand for the same window to flag likely waste.

## 5. Upstream supply chain: raw materials & procurement

`data_generation/raw_materials.py` builds a 51-material, 8-supplier
network with a bill-of-materials (BOM) linking every SKU to what it's
made from (one dedicated API + shared excipients/packaging).
`data_generation/procurement.py` then:

1. Computes daily raw-material consumption directly from the
   production-batch schedule via the BOM (`compute_material_consumption`).
2. Runs a day-by-day (s, S) procurement simulation per material
   (`simulate_procurement`), placing purchase orders against each
   material's assigned supplier and applying that supplier's own
   lead time and on-time delivery rate - a late overseas API shipment
   is a real, simulated event here, not just an assumption.

This is what makes supplier reliability a *measurable* input rather
than a talking point: overseas API suppliers in this network run at
76-87% on-time delivery with 45-60 day lead times, materially worse
than the domestic excipient/packaging suppliers (92-97% on-time,
8-18 day lead times) - exactly the kind of asymmetry a real
procurement team would want surfaced on a dashboard.

`planning/supply_risk.py` turns this into two decision-ready views:
**supplier performance** (on-time rate, average delay, spend, rolled
up per supplier) and **material days-of-cover** (current stock ÷
recent daily consumption, compared against that material's own
supplier lead time - a material whose cover has fallen below its
lead time is one stockout away from delaying a production batch).

## 6. Working capital: the Cash Conversion Cycle

`planning/working_capital.py` computes the classic three-part metric:

    CCC = DIO + DSO - DPO

- **DIO** (Days Inventory Outstanding) is computed directly from the
  *actual* simulated inventory value - finished goods (from the
  inventory ledger) **plus** raw materials (from the procurement
  simulation) - divided by trailing average COGS. This is a genuine
  output of the operational simulation, not an assumed constant.
- **DSO** (Days Sales Outstanding) and **DPO** (Days Payable
  Outstanding) are driven by a mechanistic receivables/payables
  ledger: the AR balance grows with each day's revenue and shrinks
  when the revenue booked `dso_target_days` earlier is collected; AP
  works the same way against COGS with a `dpo_target_days` payment
  lag. This produces genuinely fluctuating (not hard-coded) DSO/DPO
  series that settle near their target collection/payment terms
  after an initial transient.
- Aurora's receivables terms are set long (**75 days**) - realistic
  for healthcare distribution, where hospitals and pharmacy chains
  are notoriously slow payers - while payables terms are comparatively
  tight (**30 days**), which is what pushes the network's steady-state
  CCC to roughly **55-60 days**, matching the target referenced
  throughout the dashboard. All figures are reported as trailing
  60-day averages, the conventional CCC reporting window.

## 7. Reporting layer

Both `reporting/excel_report.py` (a 9-sheet workbook with native,
editable Excel charts) and `reporting/html_dashboard.py` (a
standalone, interactive Plotly dashboard in a red/orange theme) are
built from exactly the same computed tables in
`scripts/run_pipeline.py` - one source of truth, two presentation
formats for two different audiences (a planner working in Excel vs.
an executive skimming a dashboard link).
