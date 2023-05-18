from typing import Optional

from matplotlib import pyplot as plt

from fraud_detector import FraudDetector
from information_and_metrics import ConfusionMatrixMetrics


class MetricsPerObservedFraudRate:

    def __init__(self) -> None:
        self._labels: list[str] = ['FPR', 'FNR', 'FDR', 'FOR', 'RPP', 'ACC']
        # Create empty dictionaries to store metrics per observed fraud rate
        self._metric1_males, self._metric2_males, self._metric3_males = {}, {}, {}
        self._metric4_males, self._metric5_males, self._metric6_males = {}, {}, {}
        self._metric1_females, self._metric2_females, self._metric3_females = {}, {}, {}
        self._metric4_females, self._metric5_females, self._metric6_females = {}, {}, {}

    @staticmethod
    def _get_metrics(male_metrics: dict[str, float],
                     female_metrics: dict[str, float]
                     ) -> Optional[tuple[list[float], list[float]]]:
        male_metrics = [male_metrics['FPR'], male_metrics['FNR'], male_metrics['FDR'],
                        male_metrics['FOR'], male_metrics['RPP'], male_metrics['ACC']]
        female_metrics = [female_metrics['FPR'], female_metrics['FNR'], female_metrics['FDR'],
                          female_metrics['FOR'], female_metrics['RPP'], female_metrics['ACC']]
        return male_metrics, female_metrics

    def plot_metrics_by_observed_fraud(self,
                                       classifier_name: str,
                                       plot_title: str,
                                       sample_size: int = 2500,
                                       ):
        # Iterate over observed fraud rates
        for fraud_rate in range(1, 101):
            observed_fraud_rate = fraud_rate / 100

            # Initialize fraud detector
            fraud_detector: FraudDetector = FraudDetector(
                classifier_name=classifier_name,
                male_fraud_proportion=observed_fraud_rate,
                female_fraud_proportion=0.1,
                sample_size=sample_size,
                al_type_name='None',
            )
            # Create predictions
            _, informative_test_data_random, _ = fraud_detector.detect_fraud()
            # Initialize confusion matrix
            confusion_matrix_metrics = ConfusionMatrixMetrics(informative_test_data_random)
            # Get confusion matrix metrics
            male_metrics, female_metrics = confusion_matrix_metrics.get_metrics()

            # Extract metrics
            male_metrics, female_metrics = self._get_metrics(male_metrics, female_metrics)

            # Store metrics in dictionary
            self._metric1_males[observed_fraud_rate] = male_metrics[0]
            self._metric1_females[observed_fraud_rate] = female_metrics[0]

            self._metric2_males[observed_fraud_rate] = male_metrics[1]
            self._metric2_females[observed_fraud_rate] = female_metrics[1]

            self._metric3_males[observed_fraud_rate] = male_metrics[2]
            self._metric3_females[observed_fraud_rate] = female_metrics[2]

            self._metric4_males[observed_fraud_rate] = male_metrics[3]
            self._metric4_females[observed_fraud_rate] = female_metrics[3]

            self._metric5_males[observed_fraud_rate] = male_metrics[4]
            self._metric5_females[observed_fraud_rate] = female_metrics[4]

            self._metric6_males[observed_fraud_rate] = male_metrics[5]
            self._metric6_females[observed_fraud_rate] = female_metrics[5]

        self._plot('FPR', plot_title)
        self._plot('FDR', plot_title)
        self._plot('RPP', plot_title)

    def _plot(self, group_to_use: str, plot_title: str) -> None:
        """
        Plots the graph with the relevant metrics.
        :param group_to_use: which group of metrics to plot
        """
        # Create a dictionary for lines and colors
        colors: dict[str, str] = {'male': '#1A98A6', 'female': '#E1AD01'}
        lines: list[str] = ['solid', 'dashed']

        if group_to_use == 'FPR':
            metric1_males, metric2_males = self._metric1_males, self._metric2_males
            metric1_females, metric2_females = self._metric1_females, self._metric2_females
            labels = self._labels[:2]
        elif group_to_use == 'FDR':
            metric1_males, metric2_males = self._metric3_males, self._metric4_males
            metric1_females, metric2_females = self._metric3_females, self._metric4_females
            labels = self._labels[2:4]
        else:
            metric1_males, metric2_males = self._metric5_males, self._metric6_males
            metric1_females, metric2_females = self._metric5_females, self._metric6_females
            labels = self._labels[4:]

        # Plot the metrics for males
        plt.plot(list(metric1_males.keys()), list(metric1_males.values()), label=f'{labels[0]} males',
                 color=colors['male'], linestyle=lines[0])
        plt.plot(list(metric2_males.keys()), list(metric2_males.values()), label=f'{labels[1]} males',
                 color=colors['male'], linestyle=lines[1])

        # Plot the metrics for females as dotted lines with the same color as males
        plt.plot(list(metric1_males.keys()), list(metric1_females.values()), label=f'{labels[0]} females',
                 color=colors['female'], linestyle=lines[0])
        plt.plot(list(metric2_males.keys()), list(metric2_females.values()), label=f'{labels[1]} females',
                 color=colors['female'], linestyle=lines[1])

        # Add axis labels and legend outside of the plot
        plt.xlabel('Observed Fraud Rate of Males')
        plt.ylabel('Metric Value')
        plt.title(plot_title)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(f'{group_to_use}.png')

        plt.show()
