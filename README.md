# Fraud Detection Simulator
This code has been created by Renée Theunissen for her Master Thesis at Jheronimus Academy of Data Science (JADS).

## Use
This code can be used to simulate selection bias and observe the effects over time. 

The `fraud detector` can be used for single predictions, while the `transactions over time` can be used to see the 
transactions over time.

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