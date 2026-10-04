# Comparative Analysis of RNN Architectures for WTI Crude Oil Price Forecasting

A comparison of three recurrent neural networks (SimpleRNN, GRU and LSTM) for one-month-ahead forecasting of the West Texas Intermediate (WTI) crude oil price, built with TensorFlow/Keras.

**Author:** Megha P (IMS22165), School of Data Science, IISER Thiruvananthapuram  
**Supervisor:** Dr. Alwin Poulose

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MEGHAPANKAJ/wti-oil-price-rnn-forecasting/blob/main/WTI_Crude_Oil_Price_Forecasting.ipynb)

## Overview

Each model reads a window of 90 months of eight input series and predicts the WTI price for the following month. The models are evaluated on the last 28 months of the dataset (September 2022 to December 2024) using RMSE and MAE.

The full write-up, with the model development history, figures and limitations, is in the [report](Comparative_Analysis_of_RNN_Architectures_for_WTI_Crude_Oil_Price_Forecasting.pdf).

## Repository contents

| File | Description |
| --- | --- |
| [`WTI_Crude_Oil_Price_Forecasting.ipynb`](WTI_Crude_Oil_Price_Forecasting.ipynb) | Notebook with all nine training runs and their saved outputs |
| [`Comparative_Analysis_of_RNN_Architectures_for_WTI_Crude_Oil_Price_Forecasting.pdf`](Comparative_Analysis_of_RNN_Architectures_for_WTI_Crude_Oil_Price_Forecasting.pdf) | Project report |
| `README.md` | This file |

## Dataset

The data comes from the [WTI oil price prediction dataset on Kaggle](https://www.kaggle.com/datasets/nurbolatb/wti-oil-price-prediction-dataset), which combines series from the U.S. Energy Information Administration (EIA) and Federal Reserve Economic Data (FRED). It has about 19 years of monthly observations ending in December 2024.

| Role | Columns |
| --- | --- |
| Target | `wti` (WTI price in USD per barrel) |
| Inputs | `eur_usd`, `inventory`, `production`, `rigs`, `inflation`, `wti_6m_rolling`, `wti_12m_rolling`, `wti_6m_lag` |

The CSV is not included in this repository. Download it from Kaggle as `training_data_20241224.csv` (see [How to run](#how-to-run)).

## Method

- All columns are scaled to [0, 1] with a min-max scaler.
- The data is cut into overlapping windows of 90 months, with the eight input series as features. The target is the WTI price in the month after the window.
- Windows are split chronologically: 80% for training (about 110 windows) and 20% for testing (28 windows).
- The final models use three recurrent layers (128, 128 and 64 units, tanh), a dense layer with 50 ReLU units and a linear output.
- Training uses the Huber loss, a batch size of 16, up to 100 epochs, `ReduceLROnPlateau` and `EarlyStopping`.

| Setting | SimpleRNN | GRU | LSTM |
| --- | --- | --- | --- |
| Batch normalisation | after each layer | after each layer | none |
| Dropout | 0.3, 0.3, 0.3 | 0.3, 0.3, 0.3 | 0.3, 0.2, 0.2 |
| Optimiser (learning rate) | RMSprop (1e-3) | RMSprop (1e-3) | Adam (5e-4) |

## Results

Errors on the 28 test months, in USD per barrel, with the percentage of the mean test price in brackets.

| Model | RMSE | MAE |
| --- | --- | --- |
| SimpleRNN | 10.00 (12.83%) | 8.18 (10.50%) |
| GRU | 5.91 (7.58%) | 4.85 (6.22%) |
| LSTM | 5.86 (7.52%) | 4.86 (6.23%) |

The GRU and the LSTM were clearly more accurate and more stable than the SimpleRNN, and their results are too close to rank against each other. All three models follow the general price level but miss sharp month-to-month movements.

## Notebook guide

The notebook has nine code cells. Each cell is one complete run: it loads the data, builds a model, trains it and reports the errors. The final models are cells 5 (SimpleRNN), 6 (GRU) and 9 (LSTM); the earlier cells show how the configuration was developed.

| Cell | Model | Window (months) | Main settings | RMSE | MAE |
| --- | --- | --- | --- | --- | --- |
| 1 | SimpleRNN | 30 | 1 layer (50 units, ReLU), MSE loss, Adam | 24.24 | 22.30 |
| 2 | SimpleRNN | 30 | 2 layers, dropout 0.2, early stopping | 12.18 | 9.51 |
| 3 | SimpleRNN | 30 | same as cell 2 | 22.79 | 15.64 |
| 4 | SimpleRNN | 60 | 3 layers (128, 128, 64; tanh), dropout 0.3 | 18.64 | 14.49 |
| **5** | **SimpleRNN** | **90** | **batch normalisation, Huber loss, RMSprop** | **10.00** | **8.18** |
| **6** | **GRU** | **90** | **same as cell 5** | **5.91** | **4.85** |
| 7 | LSTM | 90 | same as cell 5 | 6.92 | 5.47 |
| 8 | LSTM | 90 | no batch normalisation, Adam (5e-4) | 6.62 | 5.71 |
| **9** | **LSTM** | **90** | **same as cell 8 with dropout 0.3, 0.2, 0.2** | **5.86** | **4.86** |

Cells 1 to 4 use shorter windows, so their test periods are longer (40 and 34 months) and their errors are not directly comparable with those of cells 5 to 9.

## Limitations

- The sample is small: about 110 training windows and 28 test months.
- The test windows were also used as validation data, so the reported errors are likely to be optimistic.
- The scaler was fitted on the full dataset, including the test period.
- No random seed was set and each configuration was run once, so results vary between runs.
- The LSTM uses a different optimiser and regularisation from the other two models, so differences cannot be attributed to the cell type alone.

## How to run

### Google Colab

1. Open the notebook with the **Open in Colab** button at the top of this page.
2. Download the dataset from Kaggle and upload it to the Colab session as `training_data_20241224.csv`. Files uploaded through the Files panel are placed in `/content/`, which is the path the notebook reads.
3. Run the cells in order from the top (**Runtime > Run all**). Later cells reuse imports from earlier ones.

### Local Jupyter

1. Clone the repository and install the dependencies:

   ```bash
   git clone https://github.com/MEGHAPANKAJ/wti-oil-price-rnn-forecasting.git
   cd wti-oil-price-rnn-forecasting
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn notebook
   ```

2. Download the dataset from Kaggle and save it in this folder as `training_data_20241224.csv`.
3. In each cell, change the path in `pd.read_csv("/content/training_data_20241224.csv")` to `"training_data_20241224.csv"`.
4. Start Jupyter with `jupyter notebook`, open the notebook and run the cells in order.

Running all nine cells trains every model one after another. Because no seed is set, the errors will differ somewhat from the saved outputs.

## Future work

- A separate validation period taken from the training data, with the scaler fitted on the training data only.
- Fixed seeds and several runs per configuration.
- The same optimiser and regularisation for all cell types.
- The lagged price as an additional input.
- A comparison with classical time series models such as ARIMA.
- Higher-frequency data, which would give more training windows.

## References

1. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," EMNLP, 2014.
2. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 9(8), 1997.
3. Nurbolat_b, "WTI oil price prediction dataset," Kaggle. https://www.kaggle.com/datasets/nurbolatb/wti-oil-price-prediction-dataset
