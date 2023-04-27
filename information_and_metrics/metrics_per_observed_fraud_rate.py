from typing import Optional

from matplotlib import pyplot as plt

from fraud_detector import FraudDetector
from information_and_metrics import ConfusionMatrixMetrics


class MetricsPerObservedFraudRate:

    @staticmethod
    def _get_metrics(metrics_to_use: str,
                     male_metrics: dict[str, float],
                     female_metrics: dict[str, float]
                     ) -> Optional[tuple[list[float], list[float], list[str]]]:
        if metrics_to_use == 'TPR_GROUP':
            male_metrics = [male_metrics['TPR'], male_metrics['TNR'], male_metrics['FPR'], male_metrics['FNR']]
            female_metrics = [female_metrics['TPR'], female_metrics['TNR'],
                              female_metrics['FPR'], female_metrics['FNR']]
            return male_metrics, female_metrics, ['TPR', 'TNR', 'FPR', 'FNR']
        if metrics_to_use == 'FDR_GROUP':
            male_metrics = [male_metrics['FDR'], male_metrics['FOR'], male_metrics['PPV'], male_metrics['NPV']]
            female_metrics = [female_metrics['FDR'], female_metrics['FOR'],
                              female_metrics['PPV'], female_metrics['NPV']]
            return male_metrics, female_metrics, ['FDR', 'FOR', 'PPV', 'NPV']
        if metrics_to_use == 'RPP_GROUP':
            male_metrics = [male_metrics['RPP'], male_metrics['RNP'], male_metrics['ACC']]
            female_metrics = [female_metrics['RPP'], female_metrics['RNP'], female_metrics['ACC']]
            return male_metrics, female_metrics, ['RPP', 'RNP', 'ACC']
        return None

    def plot_metrics_by_observed_fraud(self,
                                       metrics_to_use: str,
                                       classifier_name: str,
                                       sample_size: int = 2500,
                                       ):
        # Create empty dictionaries to store metrics per observed fraud rate
        metric1_males, metric2_males, metric3_males, metric4_males = {}, {}, {}, {}
        metric1_females, metric2_females, metric3_females, metric4_females = {}, {}, {}, {}

        # Iterate over observed fraud rates
        for fraud_rate in range(1, 101):
            observed_fraud_rate = fraud_rate / 100

            # Initialize fraud detector
            fraud_detector: FraudDetector = FraudDetector(
                classifier_name=classifier_name,
                male_fraud_proportion=observed_fraud_rate,
                female_fraud_proportion=0.1,
                sample_size=sample_size,
            )
            # Create predictions
            _, informative_test_data_random = fraud_detector.detect_fraud()
            # Initialize confusion matrix
            confusion_matrix_metrics = ConfusionMatrixMetrics(informative_test_data_random)
            # Get confusion matrix metrics
            male_metrics, female_metrics = confusion_matrix_metrics.get_metrics()

            # Extract metrics
            male_metrics, female_metrics, labels = self._get_metrics(metrics_to_use, male_metrics, female_metrics)

            # Store metrics in dictionary
            metric1_males[observed_fraud_rate] = male_metrics[0]
            metric1_females[observed_fraud_rate] = female_metrics[0]

            metric2_males[observed_fraud_rate] = male_metrics[1]
            metric2_females[observed_fraud_rate] = female_metrics[1]

            metric3_males[observed_fraud_rate] = male_metrics[2]
            metric3_females[observed_fraud_rate] = female_metrics[2]

            if len(labels) > 3:
                metric4_males[observed_fraud_rate] = male_metrics[3]
                metric4_females[observed_fraud_rate] = female_metrics[3]

        # Create a dictionary to map labels to colors
        colors = {labels[0]: 'red', labels[1]: 'blue', labels[2]: 'green'}
        if len(labels) > 3:
            colors[labels[3]] = 'purple'

        # Plot the metrics for males
        plt.plot(list(metric1_males.keys()), list(metric1_males.values()), label=labels[0], color=colors[labels[0]])
        plt.plot(list(metric2_males.keys()), list(metric2_males.values()), label=labels[1], color=colors[labels[1]])
        plt.plot(list(metric3_males.keys()), list(metric3_males.values()), label=labels[2], color=colors[labels[2]])
        if len(labels) > 3:
            plt.plot(list(metric4_males.keys()), list(metric4_males.values()), label=labels[3], color=colors[labels[3]])

        # Plot the metrics for females as dotted lines with the same color as males
        plt.plot(list(metric1_males.keys()), list(metric1_females.values()), label=labels[0],
                 linestyle='dotted', color=colors[labels[0]])
        plt.plot(list(metric2_males.keys()), list(metric2_females.values()), label=labels[1],
                 linestyle='dotted', color=colors[labels[1]])
        plt.plot(list(metric3_males.keys()), list(metric3_females.values()), label=labels[2],
                 linestyle='dotted', color=colors[labels[2]])
        if len(labels) > 3:
            plt.plot(list(metric4_males.keys()), list(metric4_females.values()), label=labels[3],
                     linestyle='dotted', color=colors[labels[3]])

        # Add axis labels and legend outside of the plot
        plt.xlabel('Observed Fraud Rate of Males')
        plt.ylabel('Metric Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Show the plot
        plt.show()


