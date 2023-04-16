from fraud_detector import FraudDetector
from information_and_metrics import ConfusionMatrix

# Initialize fraud detector
fraud_detector: FraudDetector = FraudDetector(
    classifier_name='LogisticRegression',
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    sample_size=2500,
)
# Initialize confusion matrix
confusion_matrix_creator = ConfusionMatrix()
# Create predictions
_, informative_test_data_biased = fraud_detector.detect_fraud()
# Get confusion matrix
print("Biased data confusion matrix:")
confusion_matrix_creator.get_confusion_matrix(informative_test_data_biased)

print()

# Now get the predictions for the random data set
# Initialize fraud detector
fraud_detector: FraudDetector = FraudDetector(
    classifier_name='LogisticRegression',
    male_fraud_proportion=0.2,  # not used
    female_fraud_proportion=0.1,  # not used
    sample_size=2500,
    random_training_set=True
)
# Create predictions
_, informative_test_data_random = fraud_detector.detect_fraud()
# Get confusion matrix
print("Random data confusion matrix:")
confusion_matrix_creator.get_confusion_matrix(informative_test_data_random)