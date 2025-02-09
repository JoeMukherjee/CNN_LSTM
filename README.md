# CNN_LSTM
Time-Series Forecasting Using CNN-RNN Hybrid Model
Submitted by: Pritam Mukherjee
1. Objective
The objective of this task was to develop a hybrid CNN-RNN model to forecast future time-series values. The model aimed to leverage:
•	CNN (Convolutional Neural Network) for feature extraction, capturing short-term dependencies.
•	RNN (Recurrent Neural Network) for sequential modeling, capturing long-term temporal trends.
This approach was implemented on a publicly available dataset of energy consumption to predict future values based on historical data.
2. Dataset Overview
The dataset used contains hourly energy consumption data, representing real-world time-series challenges with:
•	Periodic fluctuations (e.g., daily and weekly patterns).
•	Random spikes and troughs (e.g., holiday consumption, equipment malfunctions).
•	Dependencies on time-of-day, weekends, and external factors.
The data preprocessing steps were tailored to address these characteristics and ensure robust forecasting.
3. Methodology
Data Preprocessing
1.	Aggregation:
a.	Multiple .csv files were merged, and TxnDate and TxnTime columns were combined into a single Datetime index.
2.	Feature Engineering:
a.	Lag Features: Incorporated historical data for up to 7 prior hours.
b.	Rolling Statistics: Added rolling mean, standard deviation, and median to capture local trends.
c.	Time-Based Features: Extracted day-of-week, month, and hour.
d.	Daypart Features: Included a binary indicator for weekends.
e.	Exponential Moving Average (EMA): Captured short-term trends using EMA with a 7-hour span.
f.	Seasonal Decomposition: Split the time-series data into trend, seasonal, and residual components using the seasonal_decompose method.
3.	Normalization:
a.	Data was normalized to bring all features into a uniform scale for model training.
4.	Train-Test Split:
a.	The dataset was divided into training, validation, and test sets for fair model evaluation.

Model Design
The hybrid CNN-RNN model was built to effectively process and predict time-series data:
•	CNN Layers:
o	Extracted short-term patterns using a 1D convolutional layer.
o	Followed by dropout for regularization and batch normalization for stable training.
•	RNN Layers:
o	A single LSTM layer captured long-term dependencies in the data.
•	Output Layer:
o	A Dense layer provided the final prediction.
Model Summary

Layer (type)	Output Shape	Param 
conv1d_1 (Conv1D)	(None, 32, 32)	128
dropout_1 (Dropout)	(None, 32, 32)	0
batch_normalization_1
(Batch Normalization)	(None, 32, 32)	128
lstm_1 (LSTM)	(None, 50)	16600
dense_1 (Dense)	(None, 1)	51

 	Total params: 16,907 (66.04 KB)
 	Trainable params: 16,843 (65.79 KB)
 	Non-trainable params: 64 (256.00 B)

Model Training
•	The model was trained using the Mean Squared Error (MSE) loss function, with Adam optimizer.
•	Hyperparameters such as window size, learning rate, and layer configurations were optimized through experimentation.

4. Evaluation and Results
Evaluation Metrics:
The model's performance was evaluated using RMSE, MAE, and R² metrics:
•	Root Mean Square Error (RMSE):
o	Normalized Scale: 0.0333
o	Original Scale: 0.9094 (significantly less compared to the standard deviation (2.9612).
•	Mean Absolute Error (MAE):
o	Normalized Scale: 0.0144
o	Original Scale: 0.3937
•	R-Squared (R²):
o	Achieved an R² score of 0. 6626, demonstrating moderate correlation between predictions and actual values.



5. Challenges Faced
1.	Preprocessing Complexities:
a.	The data's dependence on external factors like time of day, day of week and weekends made preprocessing non-trivial.
b.	Techniques such as lag features, rolling statistics, and decomposition were necessary to provide meaningful inputs for the model.
2.	Model Selection:
a.	Designing a hybrid CNN-RNN model required extensive experimentation due to limited available resources and prior implementations for this architecture.
3.	Capturing Spikes:
a.	Sudden peaks and falls in the data posed challenges for earlier models but were mitigated by the hybrid model and feature engineering.

