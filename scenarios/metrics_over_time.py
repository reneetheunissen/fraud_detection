from fraud_detector import TransactionsOverTime

classifier = 'LogisticRegression'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, LR",
    percentage_alerts=0.01,
    active_learning=False,
    percentage_active_learning=0.1,
    al_type_name='representative',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, LR",
    percentage_alerts=0.05,
    active_learning=False,
    percentage_active_learning=0.1,
    al_type_name='representative',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, LR",
    percentage_alerts=0.1,
    active_learning=False,
    percentage_active_learning=0.1,
    al_type_name='representative',
)
bank_process.plot(n_iterations=25)

classifier = 'RandomForest'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF",
    percentage_alerts=0.01,
    active_learning=False,
    percentage_active_learning=0.1,
    al_type_name='representative',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF",
    percentage_alerts=0.05,
    active_learning=False,
    percentage_active_learning=0.1,
    al_type_name='representative',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF",
    percentage_alerts=0.1,
    active_learning=False,
    percentage_active_learning=0.1,
    al_type_name='representative',
)
bank_process.plot(n_iterations=25)

