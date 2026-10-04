# Comparative Analysis of RNN Architectures for WTI Crude Oil Price Forecasting

A comparison of three recurrent neural networks (SimpleRNN, GRU and LSTM) for one-month-ahead forecasting of the West Texas Intermediate (WTI) crude oil price, built with TensorFlow/Keras.

**Author:** Megha P (IMS22165), School of Data Science, IISER Thiruvananthapuram
**Supervisor:** Dr. Alwin Poulose

## Overview

Each model reads a window of 90 months of eight input series and predicts the WTI price for the following month. The models are evaluated on the last 28 months of the dataset (September 2022 to December 2024) using RMSE and MAE.

The full write-up, including the model development history, figures and limitations, is in the report PDF in this repository.

## Dataset

The data comes from the [WTI oil price prediction dataset on Kaggle](https://www.kaggle.com/datasets/nurbolatb/wti-oil-price-prediction-dataset), which combines series from the U.S. Energy Information Administration (EIA) and Federal Reserve Economic Data (FRED). It has about 19 years of monthly observations ending in December 2024.

| Role | Columns |
|------|---------|
| Target | `wti` (WTI price in USD per barrel) |
| Inputs | `eur_usd`, `inventory`, `production`, `rigs`, `inflation`, `wti_6m_rolling`, `wti_12m_rolling`, `wti_6m_lag` |

The CSV is not included in this repository. Download it from Kaggle and place it as `training_data_20241224.csv` (see Usage below).

## Method

- All columns are scaled to [0, 1] with a min-max scaler.
- The data is cut into overlapping windows of 90 months, with the 8 input series as features.
- Windows are split chronologically: 80% train, 20% test (28 test windows).
- Final models use three recurrent layers (128, 128, 64 units, tanh), a dense layer with 50 ReLU units and a linear output.
- Training uses Huber loss, batch size 16, up to 100 epochs, `ReduceLROnPlateau` and `EarlyStopping`.

| Setting | SimpleRNN | GRU | LSTM |
|---------|-----------|-----|------|
| Batch normalisation | after each layer | after each layer | none |
| Dropout | 0.3, 0.3, 0.3 | 0.3, 0.3, 0.3 | 0.3, 0.2, 0.2 |
| Optimiser | RMSprop (1e-3) | RMSprop (1e-3) | Adam (5e-4) |

## Results

Errors on the 28 test months, in USD per barrel (percentage of the mean test price in brackets).

| Model | RMSE | MAE |
|-------|------|-----|
| SimpleRNN | 10.00 (12.83%) | 8.18 (10.50%) |
| GRU | 5.91 (7.58%) | 4.85 (6.22%) |
| LSTM | 5.86 (7.52%) | 4.86 (6.23%) |

The GRU and LSTM were clearly more accurate and more stable than the SimpleRNN, and their results are too close to rank against each other. All three models follow the general price level but miss sharp month-to-month movements.

## Limitations

- Small sample (about 110 training windows) and only 28 test months.
- The test windows were also used as validation data, so the reported errors are likely optimistic.
- The scaler was fitted on the full dataset, including the test period.
- No random seed was set and each configuration was run once.
- The LSTM uses a different optimiser and regularisation from the other two models, so differences cannot be attributed to the cell type alone.

## Repository structure

```
.
├── wti_crude_oil_price_forecasting.py   # All experiments (runs 1 to 9), exported from Colab
├── Comparative_Analysis_of_RNN_Architectures_for_WTI_Crude_Oil_Price_Forecasting.pdf
├── requirements.txt
└── README.md
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and save it as `training_data_20241224.csv`.
4. Update the CSV path in the script. It currently points to `/content/training_data_20241224.csv` (the Colab location), so change it to `training_data_20241224.csv` if the file is in the same folder.
5. Run the script:
   ```bash
   python wti_crude_oil_price_forecasting.py
   ```

Note: the script is a Colab export containing all nine runs in order, so it trains every model one after another. The final models are runs 5 (SimpleRNN), 6 (GRU) and 9 (LSTM). Results will vary slightly between runs because no seed is set.

## Future work

- A separate validation period taken from the training data, with the scaler fitted on training data only.
- Fixed seeds and several runs per configuration.
- The same optimiser and regularisation for all cell types.
- The lagged price as an additional input.
- A comparison with classical models such as ARIMA.
- Higher-frequency data for more training windows.

## References

1. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," EMNLP, 2014.
2. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 9(8), 1997.
3. Nurbolat_b, "WTI oil price prediction dataset," Kaggle.
