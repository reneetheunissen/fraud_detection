from math import floor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from fraud_detector import FraudDetector
from information_and_metrics import ConfusionMatrixMetrics


class TransactionsOverTime:

    def __init__(self,
                 male_fraud_proportion: float,
                 female_fraud_proportion: float,
                 classifier_name: str,
                 sample_size: int = 2500,
                 random_training_set: bool = False,
                 amount_of_days: int = 3,
                 ) -> None:
        self._fraud_detector: FraudDetector = FraudDetector(
            male_fraud_proportion, female_fraud_proportion, sample_size, classifier_name, random_training_set
        )
        self._amount_of_days: int = amount_of_days
        self._historical_data_all_labeled: DataFrame = self._fraud_detector.historical_data
        self._fpr_males, self._fnr_males, self._fdr_males = [], [], []
        self._for_males, self._rpp_males, self._acc_males = [], [], []
        self._fpr_females, self._fnr_females, self._fdr_females = [], [], []
        self._for_females, self._rpp_females, self._acc_females = [], [], []
        self._alerts_males, self._alerts_females, self._alerts_fraud = [], [], []
        self._ofr_males, self._ofr_females, self._tfr_males, self._tfr_females = [], [], [], []
        self._random: bool = random_training_set

    def start_transactions(self) -> None:
        """
        Goes of the specified amount of days and evaluates the transactions for fraud.
        :return:
        """
        test_sets = self._split_test_set()
        number_of_alerts: int = int(len(test_sets[0]) * 0.1)
        print(f'The number of alerts per day is {number_of_alerts} out of {len(test_sets[0])} transactions.')

        for day in range(1, self._amount_of_days + 1):
            print(f'Day {day}')
            print(f"Fraud in test set: {len(test_sets[day - 1][test_sets[day-1]['is_fraud'] == 1]) / len(test_sets[day-1])}")
            self._fraud_detector.test_transactions = test_sets[day - 1]
            predictions, informative_data = self._fraud_detector.detect_fraud()

            # Get metrics
            confusion_matrix_metrics = ConfusionMatrixMetrics(informative_data)
            male_metrics, female_metrics = confusion_matrix_metrics.get_metrics()
            self._append_metrics(male_metrics, female_metrics, informative_data)

            # Predictions is a dataframe with columns 'non-fraud' and 'fraud' and the index is the test index
            if self._random:
                alerts_index = predictions.sample(n=number_of_alerts, random_state=0).index
                alerts = informative_data[informative_data.index.isin(alerts_index) == True]
            else:
                # Sort based on certainty of fraud
                predictions = predictions.sort_values(by='fraud', ascending=False)
                alerts_index = predictions.iloc[:number_of_alerts].index
                alerts = informative_data[informative_data.index.isin(alerts_index) == True]

            self._alerts_males.append(len(alerts[alerts['gender_M'] == 1]) / len(alerts))
            self._alerts_females.append(1 - self._alerts_males[-1])
            self._alerts_fraud.append(len(alerts[alerts['is_fraud'] == 1]) / len(alerts))

            # Add alerts to historical data
            informative_data.drop(columns=['predicted'], axis=1, inplace=True)
            self._fraud_detector.historical_data = pd.concat(
                [
                    self._fraud_detector.historical_data,
                    informative_data[informative_data.index.isin(alerts_index) == True]
                ]
            )
        print("Days ended.")

    def plot(self) -> None:
        """
        Plots all metric plots.
        """
        self._plot_metric(self._fpr_males, self._fnr_males, self._fpr_females, self._fnr_females, 'FPR', 'FNR')
        self._plot_metric(self._fnr_males, self._for_males, self._fdr_females, self._for_females, 'FDR', 'FOR')
        self._plot_metric(self._rpp_males, self._acc_males, self._rpp_females, self._acc_females, 'RPP', 'ACC')
        self._plot_metric(self._ofr_males, self._tfr_males, self._ofr_females, self._tfr_females, 'OFR', 'TFR')

        plt.plot(self._alerts_males, label='Males')
        plt.plot(self._alerts_females, label='Females', linestyle='--')
        plt.plot(self._alerts_fraud, label='Fraud')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.show()

    @staticmethod
    def _plot_metric(
            metric1_males: list[float], metric2_males: list[float],
            metric1_females: list[float], metric2_females: list[float],
            metric1_name: str, metric2_name: str,
    ) -> None:
        """
        Plots the metrics.
        """
        plt.plot(metric1_males, label=f'{metric1_name} males', color='blue')
        plt.plot(metric1_females, label=f'{metric1_name} females', linestyle='--', color='blue')

        plt.plot(metric2_males, label=f'{metric2_name} males', color='orange')
        plt.plot(metric2_females, label=f'{metric2_name} females', linestyle='--', color='orange')
        if metric1_name == 'OFR':
            plt.ylim(0, 0.5)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.show()

    def _split_test_set(self) -> list[DataFrame]:
        """
        Splits the entirety of the test set into smaller test set. One test set is provided for each day.
        :return: A list of test sets, one for each day
        """
        test_set = self._create_test_set()
        test_set.reset_index(drop=True, inplace=True)
        transactions_per_day: int = floor(len(test_set) / self._amount_of_days)
        divided_test_sets = []

        for day in range(1, self._amount_of_days + 1):
            if day == 1:
                divided_test_sets.append(test_set.sample(n=transactions_per_day, random_state=0))
            else:
                test_set.drop(index=divided_test_sets[-1].index, inplace=True)
                divided_test_sets.append(test_set.sample(n=transactions_per_day, random_state=0))

        return divided_test_sets

    def _create_test_set(self) -> DataFrame:
        """
        Creates the test set with 10% fraud.

        :return: The test dataframe
        """
        test_data, combined_data = self._fraud_detector.train_test_creator.get_test_and_combined_data()
        test_data = combined_data[combined_data.index.isin(test_data.index) == True]
        fraud_data = test_data[test_data['is_fraud'] == 1]
        non_fraud_data = test_data[test_data['is_fraud'] == 0]

        current_fraud_rate = round(len(fraud_data) / len(test_data), 3)
        desired_fraud_rate: float = 0.1

        if current_fraud_rate < desired_fraud_rate:
            print("FRAUD RATE ADJUSTMENT!")
            transactions_to_remove = int(len(test_data) - len(fraud_data) / desired_fraud_rate)
            print(f"Number of transactions that will be removed: {transactions_to_remove}")
            non_fraud_indices = non_fraud_data.index
            index_to_drop = np.random.choice(non_fraud_indices, size=transactions_to_remove, replace=False)
            test_data.drop(index=index_to_drop, inplace=True)

            print(len(fraud_data) / len(test_data))

        return test_data

    def _append_metrics(self, male_metrics, female_metrics, informative_data: DataFrame) -> None:
        """
        Appends the given metrics to the list of metrices.
        :param male_metrics: The metrics for males
        :param female_metrics: The metrics for females.
        :param informative_data: The data of the transactions of the current day
        """
        self._fpr_males.append(male_metrics['FPR'])
        self._fnr_males.append(male_metrics['FNR'])
        self._fdr_males.append(male_metrics['FDR'])
        self._for_males.append(male_metrics['FOR'])
        self._rpp_males.append(male_metrics['RPP'])
        self._acc_males.append(male_metrics['ACC'])

        self._fpr_females.append(female_metrics['FPR'])
        self._fnr_females.append(female_metrics['FNR'])
        self._fdr_females.append(female_metrics['FDR'])
        self._for_females.append(female_metrics['FOR'])
        self._rpp_females.append(female_metrics['RPP'])
        self._acc_females.append(female_metrics['ACC'])

        historical_data_males = self._fraud_detector.historical_data[
            self._fraud_detector.historical_data['gender_M'] == 1
            ]
        historical_data_females = self._fraud_detector.historical_data[
            self._fraud_detector.historical_data['gender_F'] == 1
            ]
        self._ofr_males.append(len(historical_data_males[historical_data_males['is_fraud'] == 1])\
                          / len(historical_data_males))
        self._ofr_females.append(len(historical_data_females[historical_data_females['is_fraud'] == 1])\
                          / len(historical_data_females))

        self._historical_data_all_labeled = pd.concat([self._historical_data_all_labeled, informative_data])
        all_males = self._historical_data_all_labeled[self._historical_data_all_labeled['gender_M'] == 1]
        all_females = self._historical_data_all_labeled[self._historical_data_all_labeled['gender_F'] == 1]
        self._tfr_males.append(len(all_males[all_males['is_fraud'] == 1]) / len(all_males))
        self._tfr_females.append(len(all_females[all_females['is_fraud'] == 1]) / len(all_females))
