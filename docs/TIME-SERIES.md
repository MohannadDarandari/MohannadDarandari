# ⏱️ Time Series Forecasting - Advanced Techniques

## Fundamentals

### Components
- **Trend**: Long-term direction
- **Seasonality**: Repeating patterns
- **Cyclicity**: Long-term cycles
- **Noise**: Random fluctuations

### Decomposition
- Additive: Y = Trend + Seasonality + Residual
- Multiplicative: Y = Trend × Seasonality × Residual

---

## Classical Methods

### ARIMA Models
- **AR (Autoregressive)**: Depends on past values
- **I (Integrated)**: Differencing for stationarity
- **MA (Moving Average)**: Depends on past errors
- **ARIMA(p,d,q)**: Parameters selection via ACF/PACF

### Variations
- **SARIMA**: Seasonal ARIMA
- **ARIMAX**: ARIMA with exogenous variables
- **VAR**: Multivariate Autoregressive

### Exponential Smoothing
- Simple exponential smoothing
- Holt's linear trend
- Holt-Winters (with seasonality)
- ETS (Error-Trend-Seasonality)

### Prophet
- Facebook's forecasting tool
- Handles seasonality, trends, holidays
- Robust to missing data
- Interpretable components

---

## Deep Learning Approaches

### Recurrent Neural Networks

#### LSTM (Long Short-Term Memory)
- Memory cells for long-term dependencies
- Forget gate, Input gate, Output gate
- Seq2Seq with attention

#### GRU (Gated Recurrent Unit)
- Simplified LSTM
- Reset gate, Update gate
- Faster training

#### Bidirectional RNNs
- Both directions for encoding
- Better context understanding

### Temporal Convolutional Networks (TCN)
- 1D convolutions for sequences
- Dilated convolutions for large receptive field
- Parallel processing (faster than RNNs)

### Transformers for Time Series
- **Transformer-based**: Self-attention for temporal patterns
- **Informer**: Sparse attention for long sequences
- **Temporal Fusion Transformer**: Variable selection
- **N-BEATS**: Neural basis expansion

### Hybrid Approaches
- Combine classical + deep learning
- Example: LSTM with ARIMA components
- AutoML for model selection

---

## Advanced Techniques

### Ensemble Methods
- Combining multiple models
- Stacking forecasts
- Weighted averages based on performance

### Multi-step Forecasting
- Direct approach: Predict all steps at once
- Iterative approach: Predict one step, use as input
- MIMO (Multiple Input Multiple Output)

### Multivariate Forecasting
- Multiple correlated time series
- VAR, VARIMA models
- Multivariate neural networks

### Anomaly Detection
- Statistical methods: Z-score, IQR
- Isolation Forest
- Autoencoder-based
- LSTM-based anomaly detection

### Uncertainty Quantification
- Quantile regression
- Prediction intervals
- Probabilistic forecasts
- Bayesian approaches

---

## Feature Engineering

### Lagged Features
- Previous values as inputs
- Sliding window approach

### Rolling Statistics
- Mean, std, min, max over windows
- Momentum, trends

### Decomposition Features
- Trend component
- Seasonal component
- Residuals

### Domain-Specific Features
- Hour of day, day of week
- Month, season
- Holiday indicators
- Special events

---

## Evaluation Metrics

### Point Forecasts
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric MAPE

### Probabilistic Forecasts
- **Quantile Loss**: For quantile predictions
- **Continuous Ranked Probability Score (CRPS)**
- **Interval Coverage**

### Directional Accuracy
- Percentage correctly predicting direction
- Useful for financial forecasting

---

## Practical Workflow

```
1. Data Collection & Cleaning
   ↓
2. Exploratory Analysis (ACF, PACF, stationarity tests)
   ↓
3. Feature Engineering
   ↓
4. Train-Test Split (temporal, not random)
   ↓
5. Model Selection & Hyperparameter Tuning
   ↓
6. Validation (backtesting, out-of-sample)
   ↓
7. Ensemble (if multiple models)
   ↓
8. Deployment & Monitoring
```

---

## Tools & Libraries

- **Statsmodels**: Classical methods (ARIMA, exponential smoothing)
- **Prophet**: Facebook's tool for business forecasting
- **Sktime**: Unified interface for time series
- **PyTorch & TensorFlow**: Deep learning approaches
- **XGBoost/LightGBM**: Tree-based forecasting
- **PyTorch Forecasting**: Comprehensive toolkit
- **Darts**: Modern time series library

---

## Best Practices

1. **Respect temporal order**: No shuffling in train-test split
2. **Handle non-stationarity**: Differencing or detrending
3. **Feature scaling**: Normalize inputs properly
4. **Cross-validation**: Use time series CV
5. **Baseline models**: Compare with naive methods
6. **Domain knowledge**: Incorporate business logic
7. **Monitor drift**: Track performance over time

---

*Implementations and case studies available in projects folder.*
