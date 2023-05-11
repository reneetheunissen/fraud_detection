from fraud_detector import TransactionsOverTime

percentage_alerts = 0.1

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Biased, LR, uncertainty, 10% exploration",
    percentage_alerts=percentage_alerts,
    active_learning=True,
    percentage_active_learning=0.1,
    al_type_name='uncertainty',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Biased, LR, uncertainty, 25% exploration",
    percentage_alerts=percentage_alerts,
    active_learning=True,
    percentage_active_learning=0.25,
    al_type_name='uncertainty',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Biased, LR, uncertainty, 50% exploration",
    percentage_alerts=percentage_alerts,
    active_learning=True,
    percentage_active_learning=0.5,
    al_type_name='uncertainty',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='RandomForest',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Biased, RF, uncertainty, 10% exploration",
    percentage_alerts=percentage_alerts,
    active_learning=True,
    percentage_active_learning=0.1,
    al_type_name='uncertainty',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='RandomForest',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Biased, RF, uncertainty, 25% exploration",
    percentage_alerts=percentage_alerts,
    active_learning=True,
    percentage_active_learning=0.25,
    al_type_name='uncertainty',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='RandomForest',
    sample_size=6500,
    amount_of_days=25,
    title_scenario="Biased, RF, uncertainty, 50% exploration",
    percentage_alerts=percentage_alerts,
    active_learning=True,
    percentage_active_learning=0.5,
    al_type_name='uncertainty',
)
bank_process.plot(n_iterations=25)