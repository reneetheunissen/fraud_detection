from fraud_detector import TransactionsOverTime

print("Random Sampling")
bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.2,
    female_fraud_proportion=0.1,
    classifier_name='RandomForest',
    sample_size=6500,
    random_training_set=True,
    amount_of_days=25,
    title_scenario="Random Sampling",
    percentage_alerts=0.05,
)
bank_process.plot(n_iterations=25)

print()

print("Selection bias scenario 2")
bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.1,
    classifier_name='RandomForest',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Over-representation",
    percentage_alerts=0.05,
)
bank_process.plot(n_iterations=25)
