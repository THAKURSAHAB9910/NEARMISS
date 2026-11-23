# LIMITATIONS — NEARMISS
## Predictive Industrial Safety Intelligence System

---

## 1. Domain & Data Constraints

### 1.1 The Industrial Secrecy Gap
Because genuine plant floor sensor streams and near-miss accident records are proprietary, high-liability corporate assets, public government databases (OSHA, MSHA) publish only post-facto severe injury registries without continuous pre-incident telemetry.
While the NEARMISS synthetic benchmark is rigorously grounded in ISO 10816, Heinrich's Law, and Reason's Swiss Cheese model, real plants exhibit site-specific idiosyncrasies (e.g. unique piping resonance, legacy PLCs) that require fine-tuning.

### 1.2 Near-Miss Reporting Culture & Psychological Safety
The leading-indicator power of NEARMISS heavily relies on prompt, uninhibited near-miss reporting by operating crews.
In industrial environments with punitive cultures or blame-seeking management:
- Workers frequently under-report near-misses (e.g., minor guard trips or small chemical leaks) out of fear of disciplinary action.
- Under-reporting artificially suppresses the `near_miss_sum_7d` and `near_miss_ewm_4d` features, weakening the system's ability to forecast impending incidents.
- NEARMISS must be accompanied by non-punitive safety reporting cultures to achieve its theoretical potential.

---

## 2. Technical & Modeling Limitations

### 2.1 Sensor Drift, Calibration Loss & Missing Telemetry
- The current system assumes that industrial transducers (piezoelectric accelerometers, RTD thermocouples, pressure transmitters) are reasonably well-maintained.
- In harsh chemical or high-vibration environments, sensor mounting adhesives can detach, cables can degrade, and transducers can drift. While the `OperationalAnomalyDetector` identifies erratic readings, severe sensor failure may degrade predictive reliability.

### 2.2 Concept Drift & Plant Re-tooling
- The models assume stationary baseline operational profiles for equipment classes.
- If a centrifugal pump undergoes impeller redesign, changes fluid viscosity, or shifts operating duty points, historical baseline distributions will drift, necessitating scheduled re-training.

### 2.3 Non-Equivalence to Emergency Shutdown (ESD) Systems
- **NEARMISS is an operational advisory and decision-support system, NOT a hard real-time safety instrumented system (SIS / SIL-3)**.
- It is designed for medium-term operational intervention (7-day lookahead window for maintenance dispatch and safety walk-throughs).
- It must **never replace** physical pressure safety relief valves (PSVs), emergency trip switches (ESTOPs), or hardware interlocks.
