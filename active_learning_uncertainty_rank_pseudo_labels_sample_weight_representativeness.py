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
    df['representativeness'] = df.apply(lambda x: rep(x, data, rep_rank), axis=1)
    rep_rank[(df['uncertainty'] + df['representativeness']).argmax()] = i + 1

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

    for iteration in np.append(train.rep_rank.min()-1, train.rep_rank.sort_values().values):

        selected = train[train.rep_rank<=iteration]

        X_train_selected = np.stack(selected.X.values) if len(selected) else selected.X.values
        y_train_selected = selected.risk.values

        certain = train[train.uncertainty<=0.571]
        certain = certain[~certain.index.isin(selected.index)]
        if len(certain):
            X_train_certain = np.stack(certain.X.values)
            y_train_certain = certain.pred_label.values

        if len(X_train_selected)-2>=(len(X_train_selected)+len(certain))*0.2:
            print(f'{len(X_train_selected)} selected and {len(certain)} so selecting {int((len(X_train_selected)+len(certain))*0.2)} for validation from selected.')
            X_train_selected, X_valid, y_train_selected, y_valid = train_test_split(X_train_selected, y_train_selected,
                                                                                    test_size=int((len(X_train_selected)+len(certain))*0.2),
                                                                                    random_state=kseed, stratify=y_train_selected)

        elif len(X_train_selected)<(len(X_train_selected)+len(certain))*0.2:
            remaining_size = int((len(X_train_selected)+len(certain))*0.2)-len(X_train_selected)
            print(f'{len(X_train_selected)} selected and {len(certain)} certain, so selecting {len(X_train_selected)} for validation from selected and {remaining_size} for validation from certain.')
            if len(X_train_selected):
                X_valid, y_valid = X_train_selected.copy(), y_train_selected.copy()
                X_train_selected, y_train_selected = np.zeros((0, X_train_selected.shape[1])), np.zeros((0))
            else:
                X_train_selected = np.empty((0, X_train_certain.shape[1]))
                y_train_selected = np.empty((0, ))
                X_valid = np.empty(X_train_selected.shape)
                y_valid = np.empty(y_train_selected.shape)
            X_valid = np.concatenate([X_valid, X_train_certain[:remaining_size]])
            y_valid = np.concatenate([y_valid, y_train_certain[:remaining_size]])
            X_train_certain = X_train_certain[remaining_size:]
            y_train_certain = y_train_certain[remaining_size:]
        else:
            remaining_size = len(X_train_selected) - int((len(X_train_selected)+len(certain))*0.2)
            print(f'{len(X_train_selected)} selected and {len(certain)} so selecting {len(X_train_selected)-remaining_size} for validation from selected.')
            X_valid, y_valid = X_train_selected[:(len(X_train_selected)-remaining_size)], y_train_selected[:(len(X_train_selected)-remaining_size)]
            X_train_selected, y_train_selected = X_train_selected[(len(X_train_selected)-remaining_size):], y_train_selected[(len(X_train_selected)-remaining_size):]


        X_train = np.concatenate([X_train_certain, X_train_selected])
        y_train = np.concatenate([y_train_certain, y_train_selected])
        sample_weights = np.concatenate([0.5*np.ones_like(y_train_certain), np.ones_like(y_train_selected)])


        y_train_one = tf.keras.utils.to_categorical(y_train)
        y_val_one = tf.keras.utils.to_categorical(y_valid)

        model = create_model(neurons=[1024, 256], dropout_rate=0.1, lr=0.001, kreg=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3), breg=None)
        class_weights = {i: j for i, j in enumerate(class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train))}
        history = model.fit(X_train, y_train_one, epochs=200, validation_data=(X_valid, y_val_one),
                            sample_weight=sample_weights, verbose=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                                        mode='max',
                                                                        patience=50,
                                                                        verbose=1,
                                                                        restore_best_weights=True)]
                           )


        # Evaluate the model on the validation data and append the test score
        test_preds[-1].append(model.predict(X_test, verbose=0)[:,1])
        valid_preds[-1].append(model.predict(X_valid, verbose=0)[:,1])
        test_scores[-1][len(selected)] = roc_auc_score(y_test, test_preds[-1][-1])
        print(30*'- - ' + f'\nCertain about {len(certain)} samples with {certain.pred_label.values.mean()} positive. '+
              f'Selected {len(selected)} samples with {selected.risk.values.mean()} positive. '+
              f'{len(y_train)} for training with {y_train.mean()} positive. '+
              f'{len(y_valid)} for validation with {y_valid.mean()} positive. Test auc: {roc_auc_score(y_test, test_preds[-1][-1])}.\n' + 30*'- - ' + '\n\n')

# Convert the test scores to a DataFrame
test_scores = pd.DataFrame(test_scores)
test_scores = test_scores.reindex(sorted(test_scores.columns), axis=1)
test_scores.fillna(test_scores.mean(), inplace=True)

print(test_scores)

test_scores.to_pickle('results_uncertainty_rank_pseudo_labels_sample_weight_representativeness.pkl')
