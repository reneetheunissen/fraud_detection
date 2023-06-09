from pandas import DataFrame
from sklearn.metrics import confusion_matrix


class ConfusionMatrixMetrics:
    """
    Gets the standard fairness metrics based on the confusion matrix.
    """

    def __init__(self, informative_data_set: DataFrame) -> None:
        """
        :param informative_data_set: The data set with gender, predictions, and actual fraud information.
        """
        self._informative_data_set = informative_data_set

    def get_metrics(self) -> tuple[dict, dict]:
        """
        Gets the confusion matrix metrics for males and for females.

        :returns the metrics of males and females are dictionary
        """
        male_metrics = self._get_metrics('gender_M')
        female_metrics = self._get_metrics('gender_F')

        return male_metrics, female_metrics

    def _get_metrics(self, group_name: str) -> dict[str, float]:
        """
        Prints the metrics for the specified group.

        :param group_name: The name of the column that specifies to group to look at
        :return: the metrics per group
        """
        cnf_mat = confusion_matrix(
            self._informative_data_set[self._informative_data_set[group_name] == 1].is_fraud.to_numpy(),
            self._informative_data_set[self._informative_data_set[group_name] == 1].predicted.to_numpy()
        )

        TP = cnf_mat[1][1]
        FP = cnf_mat[0][1]
        FN = cnf_mat[1][0]
        TN = cnf_mat[0][0]

        # Fall out or false positive rate
        FPR = round(FP / (FP + TN), 3)

        # False negative rate
        FNR = round(FN / (TP + FN), 3)

        # False discovery rate
        FDR = round(FP / (TP + FP), 3)

        # False omission rate
        FOR = round(FN / (FN + TN), 3)

        # Rate of positive predictions
        RPP = round((FP + TP) / (TN + TP + FN + FP), 3)

        # Overall accuracy
        ACC = round((TP + TN) / (TP + FP + FN + TN), 3)

        return {
            'FPR': FPR,
            'FNR': FNR,
            'FDR': FDR,
            'FOR': FOR,
            'RPP': RPP,
            'ACC': ACC,
        }
