from typing import Optional, Union

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from data_creator import TrainTestCreator


class FraudDetector:
    """
    Detects fraudulent transactions. Suitable for logistic regression and random forest classifiers.
    """

    def __init__(
            self,
            male_fraud_proportion: float,
            female_fraud_proportion: float,
            sample_size: int,
            classifier_name: str,
            random_training_set: bool = False,
            active_learning: bool = False,
            al_type_name: str = '',
    ) -> None:
        self.train_test_creator: TrainTestCreator = TrainTestCreator()
        if not random_training_set:
            self.historical_data: DataFrame = self.train_test_creator.create_train_data(
                male_fraud_proportion,
                female_fraud_proportion,
                sample_size
            )
        else:
            self.historical_data: DataFrame = self.train_test_creator.create_random_train_data(sample_size)
        self.test_transactions: DataFrame = self.train_test_creator.create_small_test_set()
        self._fraudulent_transactions: DataFrame = self.historical_data[self.historical_data['is_fraud'] == 1]
        self._non_fraudulent_transactions: DataFrame = self.historical_data[self.historical_data['is_fraud'] == 0]
        self.predictor: Predictor
        self._classifier: Union[LogisticRegression, RandomForestClassifier] = \
            self._initialize_classifier(classifier_name)
        self._active_learning: bool = active_learning
        self._al_type_name: str = al_type_name

    def detect_fraud(self) -> tuple[DataFrame, DataFrame]:
        """
        Detects fraud and returns a dataframe of the predictions with information on their actual label
        and a dataframe on the test data with all information including actual and predicted label.

        :returns the predictions and the informative data set with all information, including predictions
        """
        # Initialize the classifier and predict
        self.predictor = Predictor(
            historical_data=self.historical_data,
            test_data=self.test_transactions,
            classifier=self._classifier,
        )
        predictions = self.predictor.run_model()

        # Transform predictions to dataframe
        predictions = pd.DataFrame(
            predictions,
            columns=['not fraud', 'fraud'],
            index=self.predictor.X_test.index
        )

        # Get an informative test set
        informative_test_data = self.predictor.X_test.copy()
        informative_test_data['is_fraud'] = self.predictor.y_test
        informative_test_data['predicted'] = predictions['fraud']
        informative_test_data['predicted'] = informative_test_data['predicted'].apply(lambda x: 1 if x > 0.5 else 0)

        return predictions, informative_test_data

    @staticmethod
    def _initialize_classifier(classifier_name: str) -> Union[LogisticRegression, RandomForestClassifier]:
        """
        Initializes a classifier
        
        :param classifier_name: name of the classifier
        :return: Classifier
        """
        if classifier_name == 'LogisticRegression':
            return LogisticRegression(random_state=0, max_iter=1000)
        else:
            return RandomForestClassifier(random_state=0)


class Predictor:
    """
    Splits the historical data into X and y and runs the classifier on the data.
    """

    def __init__(self,
                 historical_data: DataFrame,
                 test_data: DataFrame,
                 classifier: Union[LogisticRegression, RandomForestClassifier],
                 target_column_name: str = 'is_fraud',
                 ) -> None:
        self._historical_data: DataFrame = historical_data
        self._test_data: DataFrame = test_data
        self._classifier: Union[LogisticRegression, RandomForestClassifier] = classifier
        self._target_column_name: str = target_column_name
        self._X_train: Optional[DataFrame] = None
        self._y_train: Optional[DataFrame] = None
        self.X_test: Optional[DataFrame] = None
        self.y_test: Optional[DataFrame] = None
        self.pipeline: Optional[Pipeline] = None

    def _split_x_y(self, data_set: DataFrame) -> tuple[DataFrame, list[int]]:
        """
        Splits the data set into X and y for training and testing.

        :param data_set: The data set to be split
        :return: X and y
        """
        return data_set.drop(columns=self._target_column_name), data_set[self._target_column_name].to_numpy()

    def run_model(self) -> list[list[float]]:
        """
        Prepares the data for running the model, runs the model, and returns the predictions.

        :return: the predictions
        """
        # Split the data into X and y
        self._X_train, self._y_train = self._split_x_y(self._historical_data)
        self.X_test, self.y_test = self._split_x_y(self._test_data)

        # Create the classifier
        self.pipeline = make_pipeline(StandardScaler(), self._classifier)
        self.pipeline.fit(self._X_train, self._y_train)

        return self.pipeline.predict_proba(self.X_test)
