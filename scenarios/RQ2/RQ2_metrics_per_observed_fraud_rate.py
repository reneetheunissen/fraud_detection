from information_and_metrics import MetricsPerObservedFraudRate

# Get graph of the metrics
metrics_over_time = MetricsPerObservedFraudRate()
# TPR group
metrics_over_time.plot_metrics_by_observed_fraud(
    metrics_to_use='TPR_GROUP',
    classifier_name='RandomForest'
)
# FDR group
metrics_over_time.plot_metrics_by_observed_fraud(
    metrics_to_use='FDR_GROUP',
    classifier_name='RandomForest'
)
# RPP group
metrics_over_time.plot_metrics_by_observed_fraud(
    metrics_to_use='RPP_GROUP',
    classifier_name='RandomForest'
)
