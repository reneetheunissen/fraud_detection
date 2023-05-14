from information_and_metrics import MetricsPerObservedFraudRate

metrics_over_time = MetricsPerObservedFraudRate()

metrics_over_time.plot_metrics_by_observed_fraud(
    classifier_name='LogisticRegression',
    plot_title='Logistic Regression',
)

metrics_over_time.plot_metrics_by_observed_fraud(
    classifier_name='RandomForest',
    plot_title='Random Forest'
)
