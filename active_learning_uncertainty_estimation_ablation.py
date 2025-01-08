import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from tqdm import tqdm

from ATTIKON_Dataset import read_dataset_as_df as read_attikon
from CUBS_Dataset import read_dataset as read_cubs
from utils import create_feature_extractor, create_model, select_random_frame, print_evaluation, rep

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()
kseed = 0
np.random.seed(kseed)
random.seed(kseed)
tf.random.set_seed(kseed)
tf.keras.utils.set_random_seed(kseed)
tqdm.pandas()
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

cubs = read_cubs('ClinicalDatabase-CUBS.csv')

# Prepare the image data and labels
X = np.stack(cubs.image.values)  # Each row has two images
y = cubs.risk.values

X_train_ind, X_test_ind, y_train, y_test = train_test_split(np.arange(len(X)), y, test_size=0.4, stratify=y)
X_test_ind, X_valid_ind, y_test, y_valid = train_test_split(X_test_ind, y_test, test_size=0.5,
                                                            stratify=y_test)  # Split the data into training, validation and testing sets keeping data from the same patient together

# Unfold the data from each patient
X = X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])
y = np.repeat(y, 2)

temp = np.zeros(2 * len(X_train_ind))
temp[::2, ...] = X_train_ind * 2
temp[1::2, ...] = X_train_ind * 2 + 1
X_train_ind = temp.astype(int)

temp = np.zeros(2 * len(X_valid_ind))
temp[::2, ...] = X_valid_ind * 2
temp[1::2, ...] = X_valid_ind * 2 + 1
X_valid_ind = temp.astype(int)

temp = np.zeros(2 * len(X_test_ind))
temp[::2, ...] = X_test_ind * 2
temp[1::2, ...] = X_test_ind * 2 + 1
X_test_ind = temp.astype(int)

X_train = X[X_train_ind]
y_train = np.repeat(y_train, 2)

X_valid = X[X_valid_ind]
y_valid = np.repeat(y_valid, 2)

X_test = X[X_test_ind]
y_test = np.repeat(y_test, 2)

print(np.mean(y_train), np.mean(y_valid), np.mean(y_test))

carotid_df = read_attikon()

carotid_df['image'] = carotid_df.video.apply(select_random_frame)

feature_extractor = create_feature_extractor()
X_train = feature_extractor.predict(X_train)
X_valid = feature_extractor.predict(X_valid)
X_test = feature_extractor.predict(X_test)
X_attikon = feature_extractor.predict(np.stack(carotid_df.image.values))

model = create_model(neurons=[1024, 256], dropout_rate=0.1, lr=0.001, kreg=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3), breg=None, load_weights=True)

print_evaluation(model, X_test, y_test)
print_evaluation(model, X_attikon, carotid_df.risk.values)

df = pd.read_pickle('BHI/scores.pickle')

data = np.stack(df.X.values)
rep_rank = np.zeros((len(data)), dtype=int)

for i in tqdm(range(len(rep_rank))):
    rep_rank[df.apply(lambda x: rep(x, data, rep_rank), axis=1).argmax()] = i + 1

df['rep_rank'] = rep_rank-1

skf = StratifiedKFold(n_splits=5)

# Initialize a list to store the test scores
test_preds = []
valid_preds = []
y_tests = []
test_scores = []

# Loop over the training and validation indices
for train_index, test_index in skf.split(np.stack(df.X.values), df.risk.values):

    test_preds.append([])
    valid_preds.append([])
    test_scores.append({})

    train = df.iloc[train_index]
    test = df.iloc[test_index]

    X_test = np.stack(test.X.values)
    y_tests.append(test.risk.values)
    y_test = y_tests[-1]
    print('Test:\n', test.risk.value_counts())

    for iteration in np.append(train.rep_rank.min() - 1, train.rep_rank.sort_values().values):

        try:
            selected = train[train.rep_rank <= iteration]
            if selected.risk.nunique() < 2:
                continue

            X_train = np.stack(selected.X.values)
            y_train = selected.risk.values

            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=kseed,
                                                                  stratify=y_train)
        except ValueError:
            print('skippig iteration due to error\n\n\n\n')
            continue
        else:
            if len(X_train) < 2 or len(X_valid) < 2:
                print('skippig iteration due training/validation size\n\n\n\n')
                continue

        y_train_one = tf.keras.utils.to_categorical(y_train)
        y_val_one = tf.keras.utils.to_categorical(y_valid)

        model = create_model(neurons=[1024, 256], dropout_rate=0.1, lr=0.001,
                             kreg=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3), breg=None)
        class_weights = {i: j for i, j in enumerate(
            class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train))}
        history = model.fit(X_train, y_train_one, epochs=200, validation_data=(X_valid, y_val_one),
                            class_weight=class_weights, verbose=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                                        mode='max',
                                                                        patience=50,
                                                                        verbose=1,
                                                                        restore_best_weights=True)]
                            )

        # Evaluate the model on the validation data and append the test score
        test_preds[-1].append(model.predict(X_test, verbose=0)[:, 1])
        valid_preds[-1].append(model.predict(X_valid, verbose=0)[:, 1])
        test_scores[-1][len(selected)] = roc_auc_score(y_test, test_preds[-1][-1])
        print(30 * '- - ' + f'\nSelected {len(selected)} samples with {selected.risk.values.mean()} positive. ' +
              f'{len(y_train)} for training with {y_train.mean()} positive. ' +
              f'{len(y_valid)} for validation with {y_valid.mean()} positive. Test auc: {roc_auc_score(y_test, test_preds[-1][-1])}.\n' + 30 * '- - ' + '\n\n')

# Convert the test scores to a DataFrame
test_scores = pd.DataFrame(test_scores)
test_scores = test_scores.reindex(sorted(test_scores.columns), axis=1)
test_scores.fillna(test_scores.mean(), inplace=True)

print(test_scores)

test_scores.to_pickle('results_uncertainty_estimation_ablation.pkl')
