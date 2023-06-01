# Fraud Detection Simulator
This code has been created by Renée Theunissen for her Master Thesis at Jheronimus Academy of Data Science (JADS).

## Use
This code can be used to simulate selection bias and observe the effects over time. 

Download the data from https://www.kaggle.com/datasets/kartik2112/fraud-detection and save the `fraudTrain.csv` and 
`fraudTest.csv` in the fraud_detection folder. Hence, on the same level as this ReadMe. Now the code can be run.

The `fraud detector` can be used for single predictions, while the `transactions over time` can be used to see the 
transactions over time.

In scenarios there are three files that show how the code can be used.
- `metrics_over_time.py` shows how to plot the metrics over time and provided the probability to use active learning
- `RQ2_metrics_per_observed_fraud_rate.py` shows how to plot the metrics for all possible observed fraud rates for males
- `RQ2_random_sampling.py` prints the metrics by using random sampling instead of all possible OFR.

## Code Structure
- Data Creator
  - Data Cleaner
  - Train Test Data Creator
- Fraud Detector
  - Fraud Detector
  - Transactions Over Time
- Information and Metrics
  - Confusion Matrix Metrics
  - Metrics per Observed Fraud Rate
- Scenarios
  - RQ2
    - Logistic Regression (output plots)
    - Random Forest (output plots)
    - RQ2 Metrics per Observed Fraud Rate (example running RQ2)
  - RQ3
    - Logistic Regression
      - M0.3-F0.05 (output plots)
      - Random Sampling (output plots)
    - Random Forest
      - M0.3-F0.05 (output plots)
      - Random Sampling (output plots)
  - RQ4
    - Random Sampling (output plots)
    - Representative (output plots)
    - Uncertainty (output plots)
  - Metrics Over Time (example running RQ3 and RQ4)