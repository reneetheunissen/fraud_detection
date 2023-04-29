import os
from math import floor

import pandas as pd
from pandas import DataFrame

from fraud_detector import FraudDetector


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

    def start_transactions(self) -> None:
        """
        Goes of the specified amount of days and evaluates the transactions for fraud.
        :return:
        """
        test_sets = self._split_test_set()
        number_of_alerts = floor(len(test_sets[0]) * 0.05)
        print(f'The number of alerts per day is {number_of_alerts}')

        for day in range(1, self._amount_of_days + 1):
            self._fraud_detector.test_transactions = test_sets[day - 1]
            predictions, informative_data = self._fraud_detector.detect_fraud()

            # Predictions is a dataframe with columns 'non-fraud' and 'fraud' and the index is the test index
            # Sort based on certainty of fraud
            predictions = predictions.sort_values(by='fraud', ascending=False)
            alerts_index = predictions.iloc[:number_of_alerts].index
            print(len(alerts_index))

            # Add alerts to historical data
            informative_data.drop(columns=['predicted'], axis=1, inplace=True)
            self._fraud_detector.historical_data = pd.concat(
                [
                    self._fraud_detector.historical_data,
                    informative_data[informative_data.index.isin(alerts_index) == True]
                ]
            )
            print("NEXT DAY!")

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
        fraud_data = test_data[test_data['is_fraud'] == 1]
        non_fraud_data = test_data[test_data['is_fraud'] == 0]

        current_fraud_rate = round(len(fraud_data) / len(test_data), 1)
        desired_fraud_rate = 0.1

        while current_fraud_rate > desired_fraud_rate:
            non_fraud_indices = non_fraud_data.index
            test_data.drop(index=non_fraud_indices.sample(1), inplace=True)

            current_fraud_rate = round(len(fraud_data) / len(test_data), 1)

        return test_data
