from information_and_metrics import MetricsPerObservedFraudRate

# Get graph of the metrics
metrics_over_time = MetricsPerObservedFraudRate()
metrics_over_time.plot_metrics_by_observed_fraud(
    classifier_name='RandomForest'
)