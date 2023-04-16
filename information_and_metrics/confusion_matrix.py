from pandas import DataFrame
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:

    def get_confusion_matrix(self, informative_data_set: DataFrame) -> None:
        """
        Prints the confusion matrix for males and for females.
        :param informative_data_set: The data set with gender, predictions, and actual fraud information.
        """
        print('Males:')
        print(confusion_matrix(
            informative_data_set[informative_data_set['gender_M'] == 1].is_fraud.to_numpy(),
            informative_data_set[informative_data_set['gender_M'] == 1].predicted.to_numpy())
        )
        print()
        print('Females:')
        print(confusion_matrix(
            informative_data_set[informative_data_set['gender_F'] == 1].is_fraud.to_numpy(),
            informative_data_set[informative_data_set['gender_F'] == 1].predicted.to_numpy())
        )
