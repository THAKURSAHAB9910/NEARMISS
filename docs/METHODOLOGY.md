# METHODOLOGY — NEARMISS
## Predictive Industrial Safety Intelligence System

---

## 1. Domain Foundations & Modeling Philosophy

Traditional occupational safety analytics is strictly retrospective: incident investigators analyze accident reports *after* workers are injured or production is disrupted. 

The **NEARMISS** methodology flips this paradigm by conceptualizing workplace incidents as the terminal outcome of an **accumulating chain of operational warning signals**.

```mermaid
graph LR
    subgraph Precursor_Chain ["Accumulating Precursor Chain (Leading Indicators)"]
        A["Near-Miss Clustering<br/>(Small slips, guard trips, excursions)"] --> E["Terminal Incident<br/>(Loss of containment / Injury)"]
        B["Preventive Maintenance Backlog<br/>(Overdue PMs, neglected tickets)"] --> E
        C["ISO 10816 Mechanical Drift<br/>(Vibration RMS, kurtosis shocks)"] --> E
        D["Human-Shift Fatigue<br/>(Night shifts, cumulative overtime)"] --> E
    end
```

---

## 2. Temporal Data Leakage Prevention

In time-series safety modeling, data leakage is the most frequent cause of inflated, non-deployable model performance. Standard k-fold cross-validation or random train-test splitting leaks future warning trends into past training instances.

### Leakage Prevention Protocols:
1. **Strict Date Cutoffs**:
   - **Training Set**: 2024-01-01 to 2025-06-30 (16,410 rows)
   - **Validation Set**: 2025-07-01 to 2025-09-30 (2,760 rows) — reserved exclusively for probability calibration and threshold optimization.
   - **Test Set**: 2025-10-01 to 2025-12-31 (2,760 rows) — strictly held-out forward-looking evaluation.
2. **Left-Bounded Rolling Aggregations**:
   - All rolling window indicators (7, 14, 30 days) are computed on past observations up to and including day $t$.
   - No rolling aggregation ever peeks into day $t+1$ or beyond.
3. **Target Forward Lookahead Window**:
   - The target $Y_{i, t}$ is binary: does an incident occur on unit $i$ in the interval $[t+1, t+7]$?
   - The current day's incident status ($t$) is excluded from the forward target to prevent tautological self-prediction.

---

## 3. Modeling Architectures

### 3.1 Class Imbalance Handling
Because workplace safety incidents are rare events (~13-17% positive window rate in our benchmark fleet, and often < 3% in general industry):
- Standard objective functions (e.g. cross-entropy or mean squared error) will collapse to the majority class.
- **Random Forest**: Implements `class_weight='balanced'`, dynamically adjusting weights inversely proportional to class frequencies:
  $$w_j = \frac{n}{k \cdot n_j}$$
- **XGBoost**: Sets `scale_pos_weight = N_neg / N_pos`, scaling gradient updates for minority incident instances.

### 3.2 Unsupervised Anomaly Detection (Isolation Forest)
Supervised models learn patterns from known historical incidents. However, industrial systems encounter novel, unprecedented operating states (e.g., simultaneous thermal inversion and pressure oscillation).

NEARMISS runs an **Isolation Forest** on continuous sensor channels (vibration RMS, kurtosis, temperatures, pressure, RPM, duty cycle, acoustic emissions). By isolating observations in random trees, it assigns continuous anomaly scores:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
This provides an independent secondary layer of defense: an asset can be flagged as "Operating in an Unprecedented/Anomalous State" even if the supervised model has never seen that specific failure mode before.

### 3.3 Probability Calibration (Isotonic Regression)
Raw tree ensemble outputs (especially when tuned with `scale_pos_weight` or `class_weight='balanced'`) produce distorted probability distributions. For an industrial safety manager, an uncalibrated score of 0.85 cannot be distinguished from a true 85% risk.

NEARMISS fits an **Isotonic Regression** calibrator on the validation partition:
$$\min \sum_{i=1}^n (y_i - \hat{y}_i)^2 \quad \text{subject to } \hat{y}_i \le \hat{y}_j \text{ for } x_i \le x_j$$
This guarantees that when NEARMISS predicts a 30% incident probability, approximately 30 out of 100 such operational shifts experience an incident.

### 3.4 Survival Analysis & Hazard Dynamics
To investigate time-to-incident dynamics:
1. **Kaplan-Meier Estimator**: Computes non-parametric survival functions $S(t) = P(T > t)$, stratified by warning conditions (e.g., High Near-Miss vs Low Near-Miss cohort):
   $$\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$
2. **Cox Proportional Hazards Model**: Estimates semi-parametric hazard ratios $\text{HR} = \exp(\beta)$ for leading indicators:
   $$h(t \mid X) = h_0(t) \exp(\beta^T X)$$
   Empirical findings verify that vibration velocity RMS multiplies incident hazard by **1.42× per mm/s** ($p < 10^{-40}$) and each near-miss report increases instantaneous hazard by **1.11×** ($p = 0.0028$).

---

## 4. Decision Threshold & Cost-Sensitive Optimization

In industrial safety, symmetric decision thresholds ($p = 0.50$) are economically and ethically unsound:
- Cost of False Negative (Missed Incident): Hospitalization, regulatory fines, plant shutdown = **$50,000**.
- Cost of False Positive (Precautionary Walk-Through): Supervisory check, sensor inspection = **$1,500**.
- Cost Asymmetry Ratio = **33.3 : 1**.

The optimal decision threshold $\tau^*$ is found by minimizing total operational cost:
$$\tau^* = \arg\min_\tau \left[ \text{FN}(\tau) \cdot C_{\text{FN}} + \text{FP}(\tau) \cdot C_{\text{FP}} \right]$$
On the validation set, this optimization selects **$\tau^* = 0.35$** (or $0.05 - 0.35$ depending on tolerance), capturing over 75-80% of impending incidents and saving over **$18M+ in unmitigated risk cost**.
