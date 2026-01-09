# 🏎️ ApexBrain Enterprise | F1 Intelligence Platform

**ApexBrain** is a production-grade telemetry and strategy analytics SaaS used for high-fidelity Formula 1 data analysis. It utilizes a micro-modular architecture to process FastF1 signals, run Monte Carlo simulations, and train XGBoost degradation models in real-time.

---

## 🏗️ Architecture

The system follows a **Hexagonal Architecture** pattern:

* **`core/`**: Domain Logic (Physics, ML, Strategy). Independent of the UI.
* **`ui/`**: Visual Components (Bento Grids, Cards).
* **`app.py`**: The Controller that wires Data to Views.

### Key Modules
1.  **Physics Engine (`core.analytics`)**: Uses Savitzky-Golay filters to derive G-Forces and detect corner apexes automatically.
2.  **Strategy Oracle (`core.strategy_engine`)**: Runs 1,000+ stochastic race simulations to determine pit window optimality.
3.  **Cortex ML (`core.ml_engine`)**: An XGBoost Regressor that learns tyre degradation curves from live session telemetry.

---

## 🚀 Deployment

### Option A: Docker (Production)
The recommended way to run ApexBrain is via the containerized environment.

```bash
# 1. Build the Enterprise Image
docker build -t apexbrain:pro .

# 2. Deploy
docker run -p 8501:8501 -v $(pwd)/cache:/app/cache apexbrain:pro