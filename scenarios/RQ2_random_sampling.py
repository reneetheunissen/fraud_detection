from fraud_detector import FraudDetector
from information_and_metrics import MetricsPerObservedFraudRate, ConfusionMatrixMetrics

metrics_over_time = MetricsPerObservedFraudRate()

num_iterations = 100
male_results = []
female_results = []

for _ in range(num_iterations):
    fraud_detector: FraudDetector = FraudDetector(
        male_fraud_proportion=0.03,
        female_fraud_proportion=0.05,
        sample_size=2500,
        classifier_name='RandomForest',
        al_type_name='',
        random_training_set=True,
    )

    _, informative_test_data = fraud_detector.detect_fraud()
    metric_generator: ConfusionMatrixMetrics = ConfusionMatrixMetrics(informative_test_data)

    male_metrics, female_metrics = metric_generator.get_metrics()
    male_results.append(male_metrics)
    female_results.append(female_metrics)

# Calculate average for each metric for males
male_average_metrics = {}
for key in male_results[0].keys():
    male_average_metrics[key] = sum(result[key] for result in male_results) / num_iterations

# Calculate average for each metric for females
female_average_metrics = {}
for key in female_results[0].keys():
    female_average_metrics[key] = sum(result[key] for result in female_results) / num_iterations

print("Male Average Metrics:")
print(male_average_metrics)

print("Female Average Metrics:")
print(female_average_metrics)
