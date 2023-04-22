from fraud_detector import FraudDetector
from information_and_metrics import ConfusionMatrixMetrics

# # Random sampling
# print("Random sampling:")
# # Initialize fraud detector
# fraud_detector: FraudDetector = FraudDetector(
#     classifier_name='LogisticRegression',
#     male_fraud_proportion=0.2,  # not used
#     female_fraud_proportion=0.1,  # not used
#     sample_size=2500,
#     random_training_set=True
# )
# # Create predictions
# _, informative_test_data_random = fraud_detector.detect_fraud()
# # Initialize confusion matrix
# confusion_matrix_metrics = ConfusionMatrixMetrics(informative_test_data_random)
# # Get confusion matrix metrics
# confusion_matrix_metrics.get_metrics()
#
# print()

# Selection bias
print("Biased data:")
# Initialize fraud detector
fraud_detector: FraudDetector = FraudDetector(
    classifier_name='LogisticRegression',
    male_fraud_proportion=0.66,
    female_fraud_proportion=0.02,
    sample_size=2500,
)
# Create predictions
_, informative_test_data_biased = fraud_detector.detect_fraud()
# Initialize confusion matrix
confusion_matrix_metrics = ConfusionMatrixMetrics(informative_test_data_biased)
# Get confusion matrix
confusion_matrix_metrics.get_metrics()
