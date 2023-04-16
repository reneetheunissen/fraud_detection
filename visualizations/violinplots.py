from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


class Violinplot:

    def __init__(self, data: DataFrame) -> None:
        self._data: DataFrame = data

    def visualize(
            self,
            column_name: str,
            title: str,
            y_label: str,
            fraud_column: str = 'is_fraud',
            dataset: Optional[DataFrame] = None,
            show_outliers: bool = False
    ) -> None:
        """
        Visualizes violin plots for total, male, and female
        :param column_name: name of the column to visualize
        :param title: Title of the plot
        :param y_label: y label of the plot
        :param fraud_column: The name of the column containing fraud
        :param dataset: The data set to visualize
        :param show_outliers: Show an additional plot with outliers
        """
        if dataset is None:
            dataset = self._data
        ax = sns.violinplot(data=dataset, x='gender', y=column_name, hue=fraud_column, cut=0, palette="pastel")
        ax.set(
            title=title,
            ylabel=y_label,
            xlabel="Gender"
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        if not show_outliers:
            max_value = 2 * (dataset[column_name].describe()['75%'] + 1.5 * (
                        dataset[column_name].describe()['75%'] - dataset[column_name].describe()['25%']))
            ax.set_ylim(ymax=max_value, ymin=0)

        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        plt.legend(title="fraud", loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()