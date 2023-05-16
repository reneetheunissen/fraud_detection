from fraud_detector import TransactionsOverTime

al_type = 'representative'
classifier = 'RandomForest'
percentage_active_learning = 0.25

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF, 25% exploration",
    percentage_alerts=0.01,
    active_learning=True,
    percentage_active_learning=percentage_active_learning,
    al_type_name=al_type,
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF, 25% exploration",
    percentage_alerts=0.05,
    active_learning=True,
    percentage_active_learning=percentage_active_learning,
    al_type_name=al_type,
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF, 25% exploration",
    percentage_alerts=0.1,
    active_learning=True,
    percentage_active_learning=percentage_active_learning,
    al_type_name=al_type,
)
bank_process.plot(n_iterations=25)

percentage_active_learning = 0.5

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF, 50% exploration",
    percentage_alerts=0.01,
    active_learning=True,
    percentage_active_learning=percentage_active_learning,
    al_type_name=al_type,
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF, 50% exploration",
    percentage_alerts=0.05,
    active_learning=True,
    percentage_active_learning=percentage_active_learning,
    al_type_name=al_type,
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    sample_size=6500,
    amount_of_days=25,
    random_training_set=True,
    title_scenario="Biased, RF, 50% exploration",
    percentage_alerts=0.1,
    active_learning=True,
    percentage_active_learning=percentage_active_learning,
    al_type_name=al_type,
)
bank_process.plot(n_iterations=25)
