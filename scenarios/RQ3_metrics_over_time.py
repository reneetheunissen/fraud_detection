from fraud_detector import TransactionsOverTime

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=100,
    title_scenario="Biased, LR, uncertainty",
    percentage_alerts=0.05,
)
bank_process.plot(n_iterations=100)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=100,
    title_scenario="Biased, LR, uncertainty",
    percentage_alerts=0.01,
)
bank_process.plot(n_iterations=100)
