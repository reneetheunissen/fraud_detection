from fraud_detector import TransactionsOverTime

# print("Random Sampling")
# bank_process: TransactionsOverTime = TransactionsOverTime(
#     male_fraud_proportion=0.2,
#     female_fraud_proportion=0.1,
#     classifier_name='RandomForest',
#     sample_size=6500,
#     random_training_set=True,
#     amount_of_days=50,
# )
# bank_process.start_transactions()
# bank_process.plot()
#
# print()
#
# print("Selection bias scenario 1")
# bank_process: TransactionsOverTime = TransactionsOverTime(
#     male_fraud_proportion=0.2,
#     female_fraud_proportion=0.1,
#     classifier_name='LogisticRegression',
#     sample_size=6500,
#     amount_of_days=50,
# )
# bank_process.start_transactions()
# bank_process.plot()
#
# print()
#
# print("Selection bias scenario 2")
# bank_process: TransactionsOverTime = TransactionsOverTime(
#     male_fraud_proportion=0.3,
#     female_fraud_proportion=0.1,
#     classifier_name='RandomForest',
#     sample_size=6500,
#     amount_of_days=50,
# )
# bank_process.start_transactions()
# bank_process.plot()
#
# print()
#
print("Selection bias scenario 3")
bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.3,
    female_fraud_proportion=0.05,
    classifier_name='LogisticRegression',
    sample_size=6500,
    amount_of_days=50,
)
bank_process.start_transactions()
bank_process.plot()