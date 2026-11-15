# Week 4 Internship — Logistics Predictive Modeling

## Files
- `Week4_Logistics_Predictive_Modeling_Report.docx` — full report and recommendations.
- `logistics_delivery_dataset.csv` — 1,200-row synthetic shipment dataset.
- `logistics_model.py` — reproducible model training, testing, and validation script.
- `model_comparison.csv` — measured holdout metrics for the three models.
- `actual_vs_predicted.png`, `residual_distribution.png` — model diagnostic figures.
- `requirements.txt` — Python dependencies.

## Run
1. Install Python 3.10+.
2. In this folder run: `pip install -r requirements.txt`
3. Run: `python logistics_model.py`

The dataset is synthetic and intended for learning. Results are illustrative, not operational service-level guarantees. The script uses a chronological 80/20 holdout split and 5-fold expanding-window TimeSeriesSplit validation.
