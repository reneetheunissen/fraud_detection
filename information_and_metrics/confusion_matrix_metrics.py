from fairlearn.metrics import demographic_parity_ratio
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, recall_score, precision_score, balanced_accuracy_score, roc_auc_score


class ConfusionMatrixMetrics:

    def __init__(self, informative_data_set: DataFrame) -> None:
        """
        :param informative_data_set: The data set with gender, predictions, and actual fraud information.
        """
        self._informative_data_set = informative_data_set

    def get_metrics(self) -> None:
        """
        Prints the confusion matrix metrics for males and for females.
        """
        print('Males:')
        self._get_metrics('gender_M')

        print()

        print('Females:')
        self._get_metrics('gender_F')

        print()

        # demographic parity
        demographic_parity = round(demographic_parity_ratio(
            self._informative_data_set.is_fraud.to_numpy(),
            self._informative_data_set.predicted.to_numpy(),
            sensitive_features=self._informative_data_set.gender_M.to_numpy(),
        ), 3)
        print(f'Demographic parity: {demographic_parity}')

    def _get_metrics(self, group_name: str) -> None:
        """
        Prints the metrics for the specified group.
        :param group_name: The name of the column that specifies to group to look at
        :return:
        """
        cnf_mat = confusion_matrix(
            self._informative_data_set[self._informative_data_set[group_name] == 1].is_fraud.to_numpy(),
            self._informative_data_set[self._informative_data_set[group_name] == 1].predicted.to_numpy()
        )

        TP = cnf_mat[1][1]
        FP = cnf_mat[0][1]
        FN = cnf_mat[1][0]
        TN = cnf_mat[0][0]

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = round(TP / (TP + FN), 3)
        print(f'TPR: {TPR}')

        # Specificity or true negative rate
        TNR = round(TN / (TN + FP), 3)
        print(f'TNR: {TNR}')

        # Precision or positive predictive value
        PPV = round(TP / (TP + FP), 3)
        print(f'PPV: {PPV}')

        # Negative predictive value
        NPV = round(TN / (TN + FN), 3)
        print(f'NPV: {NPV}')

        # Fall out or false positive rate
        FPR = round(FP / (FP + TN), 3)
        print(f'FPR: {FPR}')

        # False negative rate
        FNR = round(FN / (TP + FN), 3)
        print(f'FNR: {FNR}')

        # False discovery rate
        FDR = round(FP / (TP + FP), 3)
        print(f'FDR: {FDR}')

        # False ommission rate
        FOR = round(FN / (FN + TN), 3)
        print(f'FOR: {FOR}')

        # Rate of positive predictions
        RPP = round((FP + TP) / (TN + TP + FN + FP), 3)
        print(f'RPP: {RPP}')

        # Rate of negative predictions
        RNP = round((FN + TN) / (TN + TP + FN + FP), 3)
        print(f'RNP: {RNP}')

        # Overall accuracy
        ACC = round((TP + TN) / (TP + FP + FN + TN), 3)
        print(f'ACC: {ACC}')

        # Recall
        recall = round(recall_score(
            self._informative_data_set[self._informative_data_set[group_name] == 1].is_fraud.to_numpy(),
            self._informative_data_set[self._informative_data_set[group_name] == 1].predicted.to_numpy()
        ), 3)
        print(f'Recall: {recall}')

        # Precision
        precision = round(precision_score(
            self._informative_data_set[self._informative_data_set[group_name] == 1].is_fraud.to_numpy(),
            self._informative_data_set[self._informative_data_set[group_name] == 1].predicted.to_numpy()
        ), 3)
        print(f'Precision: {precision}')

        # ROC-AUC
        roc_auc = round(roc_auc_score(
            self._informative_data_set[self._informative_data_set[group_name] == 1].is_fraud.to_numpy(),
            self._informative_data_set[self._informative_data_set[group_name] == 1].predicted.to_numpy()
        ), 3)
        print(f'ROC-AUC: {roc_auc}')
