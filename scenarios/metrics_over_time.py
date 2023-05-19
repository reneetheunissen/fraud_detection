from fraud_detector import TransactionsOverTime

classifier = 'LogisticRegression'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, LR",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, LR",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.1,
    title_scenario="Biased, LR",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

classifier = 'RandomForest'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.01,
    title_scenario="Biased, RF",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.05,
    title_scenario="Biased, RF",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=False,
    percentage_alerts=0.1,
    title_scenario="Biased, RF",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

classifier = 'LogisticRegression'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=True,
    percentage_alerts=0.01,
    title_scenario="Random, LR",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=True,
    percentage_alerts=0.05,
    title_scenario="Random, LR",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=True,
    percentage_alerts=0.1,
    title_scenario="Random, LR",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

classifier = 'RandomForest'

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=True,
    percentage_alerts=0.01,
    title_scenario="Random, RF",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=True,
    percentage_alerts=0.05,
    title_scenario="Random, RF",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)

bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name=classifier,
    random_training_set=True,
    percentage_alerts=0.1,
    title_scenario="Random, RF",
    active_learning=False,
    al_type_name='',
)
bank_process.plot(n_iterations=25)
