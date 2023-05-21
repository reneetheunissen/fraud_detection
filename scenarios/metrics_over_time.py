from fraud_detector import TransactionsOverTime

classifier = 'LogisticRegression'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR, 10% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.10
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR, 25% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.25
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR, 50% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.50
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR, 10% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.10
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR, 25% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.25
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR, 50% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.50
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.10,
    title_scenario="Biased, LR, 10% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.10
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.10,
    title_scenario="Biased, LR, 25% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.25
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.10,
    title_scenario="Biased, LR, 50% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.50
)
bank_process.plot(n_iterations=25)

classifier = 'RandomForest'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR, 10% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.10
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR, 25% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.25
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR, 50% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.50
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR, 10% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.10
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR, 25% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.25
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR, 50% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.50
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.10,
    title_scenario="Biased, LR, 10% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.10
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.10,
    title_scenario="Biased, LR, 25% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.25
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.10,
    title_scenario="Biased, LR, 50% exploration",
    active_learning=True,
    al_type_name='unrepresentative',
    percentage_active_learning=0.50
)
bank_process.plot(n_iterations=25)