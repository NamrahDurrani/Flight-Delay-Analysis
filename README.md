# Flight-Delay-Analysis
Predicting how many minutes a flight will be delayed using machine learning regression models and a neural network.
# ✈️ Flight Delay Prediction — Regression

> Predicting **how many minutes** a flight will be delayed using machine learning regression models and a neural network.

---

## Dataset

| Property | Detail |
|---|---|
| Rows | 3,593 flights |
| Features | 10 |
| Target | `Arr_Delay` — continuous delay in minutes (0 – 180) |
| Missing values | None |
| Mean delay | 69.8 minutes |

### Top Features by Correlation

| Feature | Correlation |
|---|---|
| Number of flights at origin | **+0.82** |
| Baggage loading time | **+0.78** |
| Late arrival at origin | **+0.67** |
| Airport distance | +0.48 |
| Support crew available | −0.36 |
| Weather, Security, Cleaning, Fueling | < ±0.33 |

The top two features have such clean linear relationships with delay that even Linear Regression competes with complex models — confirmed by scatter plots showing tight diagonal bands.


## Methodology

**Split:** 60% train → 20% validation → 20% test. Validation is used for tuning; test is used exactly once at the end for the final honest score.

**Preprocessing:** Airline codes are Label Encoded for tree models and One-Hot Encoded for the neural network (to avoid false magnitude ordering). Features are standardised to mean=0, std=1 for the neural network only — tree models are scale-invariant and do not need this.

---

## Models

| Model | Role |
|---|---|
| Linear Regression | Baseline — checks whether complexity is justified |
| Random Forest | 200 independent trees averaged together |
| Gradient Boosting | 300 trees built sequentially, each correcting the last |
| Neural Network (MLP) | 3-layer network with Dropout and BatchNormalization |

---

## Results

### Baseline — Test Set

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 10.22 min | 13.06 | 0.8039 |
| Random Forest | 10.25 min | 13.09 | 0.8031 |
| Gradient Boosting | 10.26 min | 13.05 | 0.8041 |
| **Neural Network** | **10.14 min** | **12.80** | **0.8115** |

All four models score almost identically — confirming the data has predominantly linear structure. The neural network edges ahead slightly due to correct One-Hot carrier encoding.

### After Improvements — Test Set

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (best) | 10.14 min | 12.80 | 0.8115 |
| + Feature engineering | 9.96 min | 12.57 | 0.8220 |
| + Hyperparameter tuning | 9.69 min | 12.25 | 0.8307 |
| **Improved Neural Network** | **9.63 min** | **12.21** | **0.8320** |

**Final improvement: MAE reduced by 0.63 minutes (6.1%). R² improved from 0.803 → 0.832.**

In plain terms — the best model predicts delay within about 9.6 minutes on average and explains 83.2% of why some flights are delayed more than others.

---

## Error Analysis

### Bias by Delay Range

| Range | Bias | Meaning |
|---|---|---|
| 0–30 min | **+6.0 min** | Model predicts too high |
| 30–60 min | **+5.3 min** | Model predicts too high |
| 60–90 min | −0.8 min | Accurate — model's comfort zone |
| 90–120 min | **−7.2 min** | Model predicts too low |
| 120–180 min | **−13.6 min** | Model predicts too low |

The model overestimates short delays and underestimates long ones — a classic **regression-to-the-mean** effect. It has seen the most training examples in the 60–90 minute range, so it pulls extreme predictions back toward the centre.


---

*Python · scikit-learn · TensorFlow/Keras · pandas · numpy*
