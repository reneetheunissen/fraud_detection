
# Get graph of the metrics
from information_and_metrics import MetricsPerObservedFraudRate

metrics_over_time = MetricsPerObservedFraudRate()
# TPR group
metrics_over_time.plot_metrics_by_observed_fraud(
    metrics_to_use='TPR_GROUP',
    classifier_name='LogisticRegression'
)
# FDR group
metrics_over_time.plot_metrics_by_observed_fraud(
    metrics_to_use='FDR_GROUP',
    classifier_name='LogisticRegression'
)
# RPP group
metrics_over_time.plot_metrics_by_observed_fraud(
    metrics_to_use='RPP_GROUP',
    classifier_name='LogisticRegression'
)
