from math import ceil

import pandas as pd
from pandas import DataFrame

from data_creator import DataCleaner


class TrainTestCreator:

    def __init__(self) -> None:
        # Train and test are split to represent realistic fraud scenarios in terms of size
        # They are turned around since we test on more data than we want to train on for this scenario
        self._train_data: DataFrame = self._import_data('../fraudTest.csv')
        self._test_data: DataFrame = self._import_data('../fraudTrain.csv')
        self._combined_data: DataFrame = DataCleaner().clean_data(pd.concat([self._train_data, self._test_data]))

    def _import_data(self, file_name: str) -> DataFrame:
        """
        Imports the data
        :param file_name: name of the file including csv
        :return: The imported data as DataFrame
        """
        return pd.read_csv(file_name, index_col='Unnamed: 0')

    def create_test_set(self) -> DataFrame:
        """
        Creates the test set with 10% fraud.

        :return: The test dataframe
        """
        test_data = self._combined_data[self._combined_data.index.isin(self._test_data.index) == True]
        fraud_data = test_data[test_data['is_fraud'] == 1]
        non_fraud_data = test_data[test_data['is_fraud'] == 0]

        return pd.concat([
            fraud_data[fraud_data['gender_M'] == 1].sample(n=2000, random_state=0),
            non_fraud_data[non_fraud_data['gender_M'] == 1].sample(n=18000, random_state=0),
            fraud_data[fraud_data['gender_F'] == 1].sample(n=2000, random_state=0),
            non_fraud_data[non_fraud_data['gender_F'] == 1].sample(n=18000, random_state=0),
        ])

    def create_random_train_data(self, sample_size: int) -> DataFrame:
        """
        Creates the train data set with random sampling.
        :param sample_size: The sample size
        :return: The train dataframe
        """
        train_data = self._combined_data[self._combined_data.index.isin(self._train_data.index) == True]
        # First we need to adjust the training data to ensure we get around 10% fraud in it
        # We get all fraud data and a random sample of non-fraud data
        train_data_fraud = train_data[train_data['is_fraud'] == 1]
        train_data_non_fraud = train_data[train_data['is_fraud'] == 0].sample(
            n=(len(train_data_fraud) * 9),
            random_state=0
        )
        train_data = pd.concat([train_data_fraud, train_data_non_fraud])
        return train_data.sample(n=sample_size, random_state=0)

    def create_train_data(
            self,
            male_fraud_proportion: float,
            female_fraud_proportion: float,
            sample_size: int
    ) -> DataFrame:
        """
        Creates the train set with the specified percentage of fraud.

        Male fraud cannot go above 981 and female fraud not above 1164.

        :param male_fraud_proportion: the proportion of fraud for males
        :param female_fraud_proportion: the proportion of fraud for females
        :param sample_size: the sample size
        :return: The train dataframe
        """
        train_data = self._combined_data[self._combined_data.index.isin(self._train_data.index) == True]
        fraud_data = train_data[train_data['is_fraud'] == 1]
        non_fraud_data = train_data[train_data['is_fraud'] == 0]

        fraud_number_males: int = ceil(male_fraud_proportion * sample_size)
        non_fraud_number_males: int = ceil((1 - male_fraud_proportion) * sample_size)
        fraud_number_females: int = ceil(female_fraud_proportion * sample_size)
        non_fraud_number_females: int = ceil((1 - female_fraud_proportion) * sample_size)

        return pd.concat([
            fraud_data[fraud_data['gender_M'] == 1].sample(n=fraud_number_males, random_state=0),
            non_fraud_data[non_fraud_data['gender_M'] == 1].sample(n=non_fraud_number_males, random_state=0),
            fraud_data[fraud_data['gender_F'] == 1].sample(n=fraud_number_females, random_state=0),
            non_fraud_data[non_fraud_data['gender_F'] == 1].sample(n=non_fraud_number_females, random_state=0),
        ])
