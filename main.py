import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

print("loading dataset...")

# run divider.py first to get the simplified labels
labeled_dataset = pd.read_csv('data/labeled_processed.csv')

print("dataset loaded!")

train, test = train_test_split(
    labeled_dataset,
    test_size=0.2,
    random_state=1,
    stratify=labeled_dataset['event_type']
)

fields_to_drop = [
    'sample', 'file_name', 'ML_1', 'ML_1_1', 'ML_1_1_1', 'ML_1_2','ML_2','ML_2_1_1','ML_2_1_2','ML_2_1_3','ML_2_2','ML_2_3',
    'ML_2_3_1','ML_2_4_1','ML_2_4_2','ML_2_5_1','ML_2_6','ML_2_7_1','ML_2_7_2','ML_3','ML_3_1','ML_3_2','ML_3_3','ML_3_4',
    'ML_3_5', 'event_type']

train_features = train.drop(fields_to_drop, axis=1)
train_labels = train['event_type']

test_features = test.drop(fields_to_drop, axis=1)
test_labels = test['event_type']

classifier = CatBoostClassifier(
    iterations=75,
    random_seed=2,
    learning_rate=0.2,
    custom_loss=['AUC', 'Accuracy']
)

print("training...")

classifier.fit(
    train_features, train_labels,
    eval_set=(test_features, test_labels),
    verbose=1,
    #plot=True
)
