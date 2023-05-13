from fraud_detector import TransactionsOverTime

percentage_alerts = 0.05

print("100 DAYS!")

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=100,
    title_scenario="Biased, LR",
    percentage_alerts=percentage_alerts,
    active_learning=False,
    percentage_active_learning=0.5,
    al_type_name='representative',
)
bank_process.plot(n_iterations=100)