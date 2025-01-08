import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm

from CUBS_Dataset import read_dataset as read_cubs
from utils import create_feature_extractor, create_model, variation_ratio, predictive_entropy, total_variance, mi, evaluate, print_evaluation, split_correct_incorrect

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

feature_extractor = create_feature_extractor()
X_train = feature_extractor.predict(X_train)
X_valid = feature_extractor.predict(X_valid)
X_test = feature_extractor.predict(X_test)

y_train_one = tf.keras.utils.to_categorical(y_train)
y_val_one = tf.keras.utils.to_categorical(y_valid)

model = create_model(neurons=[1024, 256], dropout_rate=0.1, lr=0.001, kreg=tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3), breg=None)
class_weights = {i: j for i, j in enumerate(class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train))}
history = model.fit(X_train, y_train_one, epochs=200, validation_data=(X_valid, y_val_one),
                    class_weight=class_weights,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                                mode='max',
                                                                patience=50,
                                                                verbose=1,
                                                                restore_best_weights=True)]
                    )

print_evaluation(model, X_test, y_test)

pred_cor, pred_mis = split_correct_incorrect(model, X_test, y_test)

scores = pd.DataFrame(columns=['accuracy', 'auc', 'youdens', 'j', 'fun', 'accuracy_thrs', 'auc_thrs', 'youdens_thrs'])

for j in range(10, 150, 10):
    for fun in [variation_ratio, predictive_entropy, mi, total_variance]:
        cor = np.apply_along_axis(fun, 1, pred_cor[j])
        mis = np.apply_along_axis(fun, 1, pred_mis[j])
        cor, mis = cor / max(cor.max(), mis.max()), mis / max(cor.max(), mis.max())

        eval_results = evaluate(np.concatenate([np.zeros_like(cor), np.ones_like(mis)]), np.concatenate([cor, mis]))
        eval_results['j'] = j
        eval_results['fun'] = fun.__name__
        new_row = pd.concat([pd.DataFrame.from_dict(eval_results).iloc[0], pd.DataFrame.from_dict(eval_results).iloc[1]], axis=0)
        new_row.index = [f'{col}_thrs' if i >= len(eval_results.keys()) else col for i, col in enumerate(new_row.index)]
        new_row = new_row.to_frame().T.drop(['j_thrs', 'fun_thrs'], axis=1)

        # Reset the index
        new_row.reset_index(drop=True, inplace=True)
        scores.loc[len(scores)] = new_row.iloc[0]

        plt.show()

fun_names = {'variation_ratio': 'Variation Ratio', 'predictive_entropy': 'Predictive Entropy', 'mi': 'Mutual Information', 'total_variance': 'Total Variance'}

for ev in ['AUC', 'Accuracy', 'Youdens']:
    fig = plt.figure(figsize=(10, 6))

    # Loop through each unique category and plot
    for category in ['variation_ratio', 'predictive_entropy', 'mi', 'total_variance']:
        subset = scores[scores['fun'] == category]
        plt.plot(subset['j'], subset[ev.lower()], marker='o', label=fun_names[category])

    plt.xlabel('Iterations')
    plt.ylabel(ev)
    fig.get_axes()[0].yaxis.set_major_formatter(FuncFormatter(lambda x, y: f'{int(x * 100)}%'))
    plt.legend(title='Function')
    plt.grid(True)
    plt.show()

for fun, j in [(variation_ratio, 150), (predictive_entropy, 10), (mi, 20), (total_variance, 130)]:
    cor = np.apply_along_axis(fun, 1, pred_cor[j])
    mis = np.apply_along_axis(fun, 1, pred_mis[j])
    cor, mis = cor / max(cor.max(), mis.max()), mis / max(cor.max(), mis.max())

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(cor, density=False, alpha=0.5, stacked=True, bins=30, range=(0, 1), color='g', label="Correct count")
    ax.hist(mis, density=False, alpha=0.5, stacked=True, bins=30, range=(0, 1), color='r', label="Incorrect count")
    ax.grid(which='both')
    a = ax.twinx()
    a.plot(np.unique(np.concatenate([cor, mis])), [((cor <= t) * 1).sum() / (((cor <= t) * 1).sum() + ((mis <= t) * 1).sum()) for t in np.unique(np.concatenate([cor, mis]))], 'g-',
           linewidth=0.5, label="Correct %")
    a.plot(np.unique(np.concatenate([cor, mis])), [((mis >= t) * 1).sum() / (((cor >= t) * 1).sum() + ((mis >= t) * 1).sum()) for t in np.unique(np.concatenate([cor, mis]))], 'r-',
           linewidth=0.5, label="Incorrect %")
    a.yaxis.set_major_formatter(FuncFormatter(lambda x, y: f'{int(x * 100)}%'))

    # Customize the plot
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Samples count")
    a.set_ylabel("Samples %")

    fig.suptitle(f'{fun_names[fun.__name__]}, {j} Iterations')
    eval_results = evaluate(np.concatenate([np.zeros_like(cor), np.ones_like(mis)]), np.concatenate([cor, mis]))
    ax.axvline(eval_results['youdens'][1], ls='-.', c='black', label='Uncertainty Threshold')

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = a.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Create the combined legend
    ax.legend(handles, labels, loc='lower left')
    plt.show()
