from fraud_detector import TransactionsOverTime

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    # random_training_set=True,
    amount_of_days=25,
    title_scenario="Biased Data, Logistic Regression",
    percentage_alerts=0.05,
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    # random_training_set=True,
    amount_of_days=25,
    title_scenario="Biased Data, Logistic Regression",
    percentage_alerts=0.01,
)
bank_process.plot(n_iterations=25)
