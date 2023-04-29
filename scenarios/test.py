from fraud_detector import TransactionsOverTime

print("Random Sampling")
bank_process: TransactionsOverTime = TransactionsOverTime(
    male_fraud_proportion=0.2,
    female_fraud_proportion=0.1,
    classifier_name='LogisticRegression',
    random_training_set=True,
)
bank_process.start_transactions()

# print()
#
# print("Selection bias scenario 1")
# bank_process: TransactionsOverTime = TransactionsOverTime(
#     male_fraud_proportion=0.2,
#     female_fraud_proportion=0.1,
#     classifier_name='LogisticRegression',
# )
# bank_process.start_transactions()
#
# print()
#
# print("Selection bias scenario 2")
# bank_process: TransactionsOverTime = TransactionsOverTime(
#     male_fraud_proportion=0.3,
#     female_fraud_proportion=0.1,
#     classifier_name='LogisticRegression',
# )
# bank_process.start_transactions()
#
# print()
#
# print("Selection bias scenario 3")
# bank_process: TransactionsOverTime = TransactionsOverTime(
#     male_fraud_proportion=0.3,
#     female_fraud_proportion=0.05,
#     classifier_name='LogisticRegression',
# )
# bank_process.start_transactions()
