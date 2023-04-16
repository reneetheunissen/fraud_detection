from typing import Optional
from matplotlib import pyplot as plt
from pandas import DataFrame


class Boxplot:

    def __init__(self, data: DataFrame) -> None:
        self._data: DataFrame = data

    def visualize(
            self,
            column_name: str,
            title: str,
            y_label: str,
            fraud_column: str = 'is_fraud',
            data: Optional[DataFrame] = None,
            show_outliers: bool = False
    ) -> None:
        """
        Visualizes boxplots for total, male, and female
        :param column_name: name of the column to visualize
        :param title: Title of the plot
        :param y_label: y label of the plot
        :param fraud_column: The name of the column containing fraud
        :param data: The data set to visualize
        :param show_outliers: Show an additional plot with outliers
        """
        if data is None:
            data = self._data
        data_fraud = data[data[fraud_column] >= 0.5][column_name].to_numpy()
        data_non_fraud = data[data[fraud_column] < 0.5][column_name].to_numpy()
        male_data = data[data['gender'] == 'M']
        male_data_fraud = male_data[male_data[fraud_column] >= 0.5][column_name].to_numpy()
        male_data_non_fraud = male_data[male_data[fraud_column] < 0.5][column_name].to_numpy()
        female_data = data[data['gender'] == 'F']
        female_data_fraud = female_data[female_data[fraud_column] >= 0.5][column_name].to_numpy()
        female_data_non_fraud = female_data[female_data[fraud_column] < 0.5][column_name].to_numpy()

        boxplot_data = [data_non_fraud, data_fraud, male_data_non_fraud, male_data_fraud, female_data_non_fraud,
                        female_data_fraud]

        self._plot_boxplot(boxplot_data, title, y_label, show_outliers)

    def _plot_boxplot(self,
                      boxplot_data,
                      title: str,
                      y_label: str,
                      show_outliers: bool = False
                      ) -> None:
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

        # Creating plot
        plt.boxplot(boxplot_data, labels=['Total', 'Total fraud', 'Male', 'Male fraud', 'Female', 'Female fraud'],
                    showfliers=show_outliers)

        plt.ylabel(y_label)
        plt.title(title)

        # show plot
        plt.show()
