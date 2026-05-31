# Predictive Hospitality Pipeline: Mitigating Revenue Leakage via Optimized Machine Learning

## Executive Summary
An empty hotel room represents unrecoverable revenue. This project delivers an end-to-end predictive pipeline designed to flag high-risk reservations before arrival, allowing hotel operators to strategically manage inventory, implement targeted pre-payment policies, and safely deploy overbooking strategies. 

By upgrading a baseline XGBoost framework through systemic hyperparameter optimization (`RandomizedSearchCV`), the final production model achieved an **Accuracy of 85%** and an **ROC-AUC of 0.9232**, capturing **74% of all cancellations** while maintaining an elite **85% precision rate**.

---

## Key Business Insights & Visualizations

### 1. Market Segment Cancellation Velocity
* **The Insight:** Group bookings represent a massive operational liability, with a staggering 60%–70% cancellation rate across all travel seasons. Conversely, Direct and Corporate channels maintain a sub-20% cancellation rate year-round, proving to be the most reliable revenue streams.
* **Operational Action:** Implement strict, non-refundable deposit matrices for all group booking contracts and peak-season Online Travel Agency (OTA) reservations.

### 2. Temporal Risk Distribution (Lead Time Thresholds)
* **The Insight:** Log-transformed distribution analysis revealed a major operational pivot point at **20 days out**. Bookings secured closer than 20 days are highly stable. Reservations made further in advance—peaking heavily around the 8-month mark—are heavily dominated by cancellations.
* **Operational Action:** Establish a rolling cancellation fee structure that tightens significantly once a booking crosses the 3-week threshold.

---

## Data Engineering & Feature Architecture
Raw booking data was transformed into an optimized feature matrix to satisfy tree-based ensemble criteria:
* **Cyclical Encoding:** Transformed raw calendar months into `month_sin` and `month_cos` signals using trigonometric mappings ($2 \cdot \pi \cdot \frac{x}{12}$) to preserve chronological continuity.
* **Risk Seasonality:** Engineered a discrete `seasonality_index` tracking regional demand behavior (Low, Medium, High).
* **Behavioral Dynamics:** Built a robust `lead_time_log` using a log-plus-one ($log1p$) transformation to neutralize heavy right-skewed data distributions and stabilize feature variances.
* **Anti-Leakage Protocol:** Systematically identified and removed `assigned_room_type` and `reservation_status_date` columns, preventing structural target leakage before model ingestion.

---

## Model Performance & Hyperparameter Optimization

The model was built using an `XGBoost Classifier`. To maximize the discovery of hidden cancellation patterns without generating excessive false alarms, a hyperparameter grid search was executed over tree architecture parameters (`max_depth`, `learning_rate`, `scale_pos_weight`).

### Performance Evolution:
| Metric | Baseline Model | Optimized Production Model | Net Operational Impact |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 0.9007 | **0.9232** | Enhanced probabilistic separation power. |
| **Cancellations Caught (True Positives)** | 5,766 | **6,544** | **+778 additional rooms** protected from vacancy. |
| **Missed Cancellations (False Negatives)** | 3,074 | **2,296** | Reduced unexpected "no-shows" by **25.3%**. |
| **Precision (Class 1)** | 86% | **85%** | Preserved high accuracy, eliminating customer friction. |
| **Overall Accuracy** | 83% | **85%** | Highly stable predictive baseline across all classes. |

### Final Production Confusion Matrix:
* **True Negatives:** 13,822 (Correctly predicted successful stays)
* **True Positives:** 6,544 (Correctly captured cancellations)
* **False Negatives:** 2,296 (Missed cancellations)
* **False Positives:** 1,180 (False alarms)

---

## Technical Stack
* **Language:** Python 3.13
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Seaborn, Matplotlib
* **Methodologies:** One-Hot Encoding, Ordinal Mapping, RandomizedSearchCV, Cyclical Feature Engineering


👉 [Click here to view the interactive version of the project notebook on NBViewer](YOUR_NBVIEWER_LINK_HERE)