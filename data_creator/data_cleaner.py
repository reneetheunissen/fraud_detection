from typing import Union

import pandas as pd
from pandas import DataFrame


class DataCleaner:
    """Data cleaner class specifically for the simulated fraud detection data set."""

    @staticmethod
    def drop_columns(
            data: DataFrame,
            columns: Union[str, list[str]]
    ) -> DataFrame:
        """
        Drops the columns that are specified in the parameter columns.

        :param columns: List of column names
        :param data: The dataframe to delete the column names from
        :return: A dataframe without the column names
        """
        return data.drop(columns=columns, axis=1)

    @staticmethod
    def _dob_to_age(data: DataFrame) -> DataFrame:
        """
        Transforms the date of birth to age and deletes the date of birth column

        :return: A dataframe with age and without date of birth
        """
        data['dob'] = data['dob'].apply(lambda x: int(x[:4]))
        data['age'] = 2023 - data['dob']
        return data.drop(columns='dob', axis=1)

    @staticmethod
    def _merge_categories(data: DataFrame) -> DataFrame:
        """
        Cleans the categories in the data by merging similar categories together.

        This includes groceries, misc, and shopping.

        :param data: The data to be cleaned

        :return: A dataframe with cleaned up categories
        """
        groceries: list[str] = ['grocery_pos', 'grocery_net']
        data.loc[data.category.isin(groceries), 'category'] = 'grocery'

        misc: list[str] = ['misc_pos', 'misc_net']
        data.loc[data.category.isin(misc), 'category'] = 'misc'

        shopping: list[str] = ['shopping_pos', 'shopping_net']
        data.loc[data.category.isin(shopping), 'category'] = 'shopping'

        return data

    @staticmethod
    def _prepare_for_predictor(data: DataFrame) -> DataFrame:
        """
        Prepares the data for training and testing by dropping unnecessary columns and one-hot encoding categorical
        variables.

        :param data: The data to prepare
        :return: The dataframe ready for predictor
        """
        data.drop(columns=['trans_num', 'city', 'job', 'merchant'], axis=1, inplace=True)
        categorical_data: list[str] = ['category', 'gender', 'state']
        data_dummies = pd.get_dummies(data[categorical_data])
        data = pd.concat([data, data_dummies], axis=1)
        return data.drop(columns=categorical_data, axis=1)

    def clean_data(self, data: DataFrame) -> DataFrame:
        """
        Cleans the data by dropping columns, transforming the date of birth to age, merging categories, and preparing
        the data for use in a machine learning model.

        :param data: The data set to be cleaned
        :return: A cleaned data set
        """
        # Drop the unnecessary columns
        cleaned_data = self.drop_columns(
            data,
            columns=['trans_date_trans_time', 'first', 'last', 'street', 'lat', 'long']
        )

        # Transform the date of birth to ages
        cleaned_data = self._dob_to_age(cleaned_data)

        # Clean up the categories
        cleaned_data = self._merge_categories(cleaned_data)

        # Prepare for training and testing
        cleaned_data = self._prepare_for_predictor(cleaned_data)

        return cleaned_data
