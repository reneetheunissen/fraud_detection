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
                 title_scenario: str,
                 al_type_name: str,
                 sample_size: int = 6500,
                 random_training_set: bool = False,
                 amount_of_days: int = 25,
                 percentage_alerts: float = 0.01,
                 active_learning: bool = False,
                 percentage_active_learning: float = 0.1,
                 ) -> None:
        self._fraud_detector: FraudDetector = FraudDetector(
            male_fraud_proportion, female_fraud_proportion, sample_size, classifier_name, al_type_name,
            random_training_set, active_learning
        )
        self._amount_of_days: int = amount_of_days
        self._historical_data_all_labeled: DataFrame = self._fraud_detector.historical_data
        self._historical_data_original: DataFrame = self._historical_data_all_labeled.copy()
        self._fpr_males, self._fnr_males, self._fdr_males = [], [], []
        self._for_males, self._rpp_males, self._acc_males = [], [], []
        self._fpr_females, self._fnr_females, self._fdr_females = [], [], []
        self._for_females, self._rpp_females, self._acc_females = [], [], []
        self._alerts_males, self._alerts_females, self._alerts_fraud_males, self._alerts_fraud_females = [], [], [], []
        self._ofr_males, self._ofr_females, self._tfr_males, self._tfr_females = [], [], [], []
        self._random: bool = random_training_set
        self._title_scenario: str = title_scenario
        self._percentage_alerts: float = percentage_alerts
        self._active_learning: bool = active_learning
        self._percentage_active_learning: float = percentage_active_learning
        self._al_type_name: str = al_type_name
        self._classifier_name: str = classifier_name

    def start_transactions(self) -> None:
        """
        Goes of the specified amount of days and evaluates the transactions for fraud.
        """
        test_sets = self._split_test_set()
        number_of_alerts: int = int(len(test_sets[0]) * self._percentage_alerts)
        number_of_active_learning: int = int(self._percentage_active_learning * number_of_alerts)

        for day in range(1, self._amount_of_days + 1):
            self._fraud_detector.test_transactions = test_sets[day - 1]
            predictions, informative_data = self._fraud_detector.detect_fraud()

            # Get metrics
            confusion_matrix_metrics = ConfusionMatrixMetrics(informative_data)
            male_metrics, female_metrics = confusion_matrix_metrics.get_metrics()
            self._append_metrics(male_metrics, female_metrics, informative_data)

            # Predictions is a dataframe with columns 'non-fraud' and 'fraud' and the index is the test index
            if self._random:
                alerts_index = predictions.sample(n=number_of_alerts).index
                alerts = informative_data[informative_data.index.isin(alerts_index) == True]
            else:
                # Sort based on certainty of fraud
                predictions = predictions.sort_values(by='fraud', ascending=False)
                if self._active_learning is True:
                    alerts_index = predictions.iloc[:(number_of_alerts - number_of_active_learning)].index
                    alerts = informative_data[informative_data.index.isin(alerts_index) == True]

                    if self._al_type_name == 'uncertainty':
                        predictions['max_probability'] = np.max(predictions, axis=1)
                        most_uncertain_indices = predictions.sort_values(
                            'max_probability'
                        ).iloc[:number_of_active_learning].index
                        exploratory_alerts = informative_data[
                            informative_data.index.isin(most_uncertain_indices) == True
                            ]
                    elif self._al_type_name == 'representative':
                        centroid_labeled_instances = np.mean(self._fraud_detector.historical_data, axis=0)
                        # Using the Euclidean distance for distance calculations
                        informative_data['representativeness'] = -1 * np.linalg.norm(
                            self._fraud_detector.test_transactions - centroid_labeled_instances,
                            axis=1,
                        )
                        exploratory_alerts = informative_data.sort_values(
                            by='representativeness', ascending=False
                        ).iloc[:number_of_active_learning]
                    else:
                        exploratory_alerts = informative_data[
                            ~informative_data.index.isin(alerts_index)
                        ].sample(n=number_of_active_learning)

                    alerts = pd.concat([alerts, exploratory_alerts])
                    alerts_index = alerts.index

                else:
                    alerts_index = predictions.iloc[:number_of_alerts].index
                    alerts = informative_data[informative_data.index.isin(alerts_index) == True]

            self._alerts_males.append(len(alerts[alerts['gender_M'] == 1]) / len(alerts))
            self._alerts_females.append(1 - self._alerts_males[-1])
            fraud_alerts: DataFrame = alerts[alerts['is_fraud'] == 1]
            self._alerts_fraud_males.append(len(fraud_alerts[fraud_alerts['gender_M'] == 1]) / len(alerts))
            self._alerts_fraud_females.append(len(fraud_alerts[fraud_alerts['gender_F'] == 1]) / len(alerts))

            # Add alerts to historical data
            if informative_data.get('representativeness') is not None:
                informative_data.drop(columns=['predicted', 'representativeness'], axis=1, inplace=True)
            else:
                informative_data.drop(columns=['predicted'], axis=1, inplace=True)
            self._fraud_detector.historical_data = pd.concat(
                [
                    self._fraud_detector.historical_data,
                    informative_data[informative_data.index.isin(alerts_index) == True]
                ]
            )

    def plot(self, n_iterations: int = 5) -> None:
        """
        Plots all metric plots.
        :param n_iterations: The number of iterations to use
        """
        for iteration in range(1, n_iterations + 1):
            self.start_transactions()
            print(f"Iteration number {iteration} ended.")
            # Reset the historical data for the next iteration
            self._fraud_detector.historical_data = self._historical_data_original.copy()

        fpr_males_avg, fnr_males_avg = self._get_averages(self._fpr_males, n_iterations), \
                                       self._get_averages(self._fnr_males, n_iterations)
        fpr_females_avg, fnr_females_avg = self._get_averages(self._fpr_females, n_iterations), \
                                           self._get_averages(self._fnr_females, n_iterations)

        fdr_males_avg, for_males_avg = self._get_averages(self._fdr_males, n_iterations), \
                                       self._get_averages(self._for_males, n_iterations)
        fdr_females_avg, for_females_avg = self._get_averages(self._fdr_females, n_iterations), \
                                           self._get_averages(self._for_females, n_iterations)

        rpp_males_avg, acc_males_avg = self._get_averages(self._rpp_males, n_iterations), \
                                       self._get_averages(self._acc_males, n_iterations)
        rpp_females_avg, acc_females_avg = self._get_averages(self._rpp_females, n_iterations), \
                                           self._get_averages(self._acc_females, n_iterations)

        ofr_males_avg, tfr_males_avg = self._get_averages(self._ofr_males, n_iterations), \
                                       self._get_averages(self._tfr_males, n_iterations)
        ofr_females_avg, tfr_females_avg = self._get_averages(self._ofr_females, n_iterations), \
                                           self._get_averages(self._tfr_females, n_iterations)

        alerts_males_avg = self._get_averages(self._alerts_males, n_iterations)
        alerts_females_avg = self._get_averages(self._alerts_females, n_iterations)
        alerts_fraud_males_avg = self._get_averages(self._alerts_fraud_males, n_iterations)
        alerts_fraud_females_avg = self._get_averages(self._alerts_fraud_females, n_iterations)

        self._plot_metrics(fpr_males_avg, fnr_males_avg, fpr_females_avg, fnr_females_avg, 'FPR', 'FNR')
        self._plot_metrics(fdr_males_avg, for_males_avg, fdr_females_avg, for_females_avg, 'FDR', 'FOR')
        self._plot_metrics(ofr_males_avg, tfr_males_avg, ofr_females_avg, tfr_females_avg, 'OFR', 'TFR')
        self._plot_metrics(alerts_males_avg, alerts_fraud_males_avg, alerts_females_avg, alerts_fraud_females_avg,
                           'Alert proportion', 'Fraud proportion')

        self._plot_single_metric(rpp_males_avg, rpp_females_avg, 'RPP')
        self._plot_single_metric(acc_males_avg, acc_females_avg, 'ACC')

    def _get_averages(self, nested_list: list[list[float]], n_iterations: int) -> list[float]:
        """
        Gets the average over all simulations per day.
        :param nested_list: The nested list of all values per iterations
        :param n_iterations: The number of iterations
        :returns: List of averages
        """
        average_list = []
        for day in range(1, self._amount_of_days + 1):
            average: float = 0

            for iteration in range(n_iterations):
                index_to_obtain: int = day + iteration * self._amount_of_days - 1
                average += nested_list[index_to_obtain]

            average_list.append(average / n_iterations)
        return average_list

    def _plot_single_metric(
            self, metric1_males: list[float], metric1_females: list[float], metric1_name: str,
    ) -> None:
        """
        Plots the metrics.
        """
        colors: dict[str, str] = {'male': '#1A98A6', 'female': '#E1AD01'}
        linestyles: list[str] = ['solid', 'dashed']

        plt.plot(metric1_males, label=f'{metric1_name} males', color=colors['male'], linestyle=linestyles[0])
        plt.plot(metric1_females, label=f'{metric1_name} females', color=colors['female'], linestyle=linestyles[0])

        if metric1_name in ['RPP']:
            plt.ylim(0, 0.25)
        elif metric1_name in ['ACC']:
            plt.ylim(0.5, 1)
        else:
            plt.ylim(0, 1)

        # Add axis labels and legend outside of the plot
        plt.xlabel('Amount of days')
        plt.ylabel('Metric Value')
        plt.title(f"{self._title_scenario} with {self._percentage_alerts * 100}% alerts")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plot_name: str = f'{self._classifier_name}-{metric1_name}{int(self._percentage_alerts * 100)}'
        if self._active_learning:
            plot_name = f'{plot_name}-{int(self._percentage_active_learning * 100)}-{self._al_type_name}'
        plt.savefig(f'{plot_name}.png')

    def _plot_metrics(
            self,
            metric1_males: list[float], metric2_males: list[float],
            metric1_females: list[float], metric2_females: list[float],
            metric1_name: str, metric2_name: str,
    ) -> None:
        """
        Plots the metrics.
        """
        colors: dict[str, str] = {'male': '#1A98A6', 'female': '#E1AD01'}
        linestyles: list[str] = ['solid', 'dashed']

        plt.plot(metric1_males, label=f'{metric1_name} males', color=colors['male'], linestyle=linestyles[0])
        plt.plot(metric1_females, label=f'{metric1_name} females', color=colors['female'], linestyle=linestyles[0])

        plt.plot(metric2_males, label=f'{metric2_name} males', color=colors['male'], linestyle=linestyles[1])
        plt.plot(metric2_females, label=f'{metric2_name} females', color=colors['female'], linestyle=linestyles[1])
        if metric1_name in ['FDR']:
            plt.ylim(0, 0.5)
        else:
            plt.ylim(0, 1)

        # Add axis labels and legend outside of the plot
        plt.xlabel('Amount of days')
        plt.ylabel('Metric Value')
        plt.title(f"{self._title_scenario} with {self._percentage_alerts * 100}% alerts")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plot_name: str = f'{self._classifier_name}-{metric1_name}{int(self._percentage_alerts * 100)}'
        if self._active_learning:
            plot_name = f'{plot_name}-{int(self._percentage_active_learning * 100)}-{self._al_type_name}'
        plt.savefig(f'{plot_name}.png')

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
                divided_test_sets.append(test_set.sample(n=transactions_per_day))
            else:
                test_set.drop(index=divided_test_sets[-1].index, inplace=True)
                divided_test_sets.append(test_set.sample(n=transactions_per_day))

        return divided_test_sets

    def _create_test_set(self) -> DataFrame:
        """
        Creates the test set with 10% fraud.

        :return: The test dataframe
        """
        test_data, combined_data = self._fraud_detector.train_test_creator.get_test_and_combined_data()
        test_data = combined_data[combined_data.index.isin(test_data.index) == True]
        test_data.reset_index(drop=True, inplace=True)
        fraud_data = test_data[test_data['is_fraud'] == 1]
        non_fraud_data = test_data[test_data['is_fraud'] == 0]

        fraud_data_males = fraud_data[fraud_data['gender_M'] == 1]
        fraud_data_females = fraud_data[fraud_data['gender_F'] == 1]

        current_fraud_rate_males = round(len(fraud_data_males) / len(test_data), 3)
        current_fraud_rate_females = round(len(fraud_data_females) / len(test_data), 3)
        desired_fraud_rate: float = 0.1

        if current_fraud_rate_males < desired_fraud_rate:
            transactions_to_remove = int(
                len(test_data[test_data['gender_M'] == 1]) - len(fraud_data_males) / desired_fraud_rate
            )
            non_fraud_indices = non_fraud_data[non_fraud_data['gender_M'] == 1].index
            index_to_drop = np.random.choice(non_fraud_indices, size=transactions_to_remove, replace=False)
            test_data.drop(index=index_to_drop, inplace=True)

        if current_fraud_rate_females < desired_fraud_rate:
            transactions_to_remove = int(
                len(test_data[test_data['gender_F'] == 1]) - len(fraud_data_females) / desired_fraud_rate
            )
            non_fraud_indices = non_fraud_data[non_fraud_data['gender_F'] == 1].index
            index_to_drop = np.random.choice(non_fraud_indices, size=transactions_to_remove, replace=False)
            test_data.drop(index=index_to_drop, inplace=True)

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
