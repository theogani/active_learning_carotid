from collections import Counter

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import stats
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, balanced_accuracy_score, precision_recall_curve, f1_score, confusion_matrix, recall_score, auc
from tqdm import tqdm


def select_random_frame(x):
    return np.repeat(x[np.random.randint(x.shape[0])][..., np.newaxis], 3, axis=-1)


def create_feature_extractor(pooling='max', lr=0.001):
    """
    This function creates a ResNet50 model.

    Parameters:
    pooling (str): the type of global pooling to be applied, default is 'max'
    lr (float): learning rate, default is 0.001

    Returns:
    model (tf.keras.models.Model): the compiled model
    """
    tf.keras.backend.clear_session()

    i = tf.keras.layers.Input([None, None, 3], dtype="uint8")
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, "float32"))(i)
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input(x)

    # load a specified pre-trained model
    pretrained_model = tf.keras.applications.InceptionV3(include_top=False, pooling=pooling, weights='imagenet', input_tensor=preprocess_input)
    # freeze the weights of the pre-trained layers
    for layer in pretrained_model.layers:
        layer.trainable = False
    # add the pre-trained model and custom classifier to the new model
    x = pretrained_model.output
    model = tf.keras.models.Model(i, outputs=pretrained_model.output)
    # model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(multi_label=True, num_labels=2, label_weights=[0, 1], name='auc'),
                           tf.keras.metrics.AUC(multi_label=True, num_labels=2, label_weights=[0, 1], curve='PR', name='auprc')]
                  )
    return model


def create_model(neurons, dropout_rate, lr, kreg, breg, load_weights=False):
    """
    This function creates a ResNet50 model.

    Parameters:
    pooling (str): the type of global pooling to be applied, default is 'max'
    lr (float): learning rate, default is 0.001

    Returns:
    model (tf.keras.models.Model): the compiled model
    """
    tf.keras.backend.clear_session()

    i = tf.keras.layers.Input([2048], dtype="float32")
    x = i
    for n in neurons:
        x = tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=kreg,
                                  bias_regularizer=breg)(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(2, kernel_regularizer=kreg, bias_regularizer=breg)(x)
    out = tf.keras.activations.softmax(x)
    model = tf.keras.models.Model(i, outputs=out)
    if load_weights:
        model.load_weights('BHI/model.weights.h5')
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(multi_label=True, num_labels=2, label_weights=[0, 1], name='auc'),
                           tf.keras.metrics.AUC(multi_label=True, num_labels=2, label_weights=[0, 1], curve='PR', name='auprc')]
                  )
    return model


def variation_ratio(arr):
    """
    Compute the variation ratio of a 1D array.

    Parameters:
    data (array-like): Input array.

    Returns:
    float: Variation ratio of the input array.
    """
    if len(arr) == 0:
        raise ValueError("Input array must not be empty.")

    temp = Counter((np.asarray(arr) * 100).astype(int))
    total = sum(temp.values())
    for i, j in temp.items():
        temp[i] = j / total

    arr = np.array(list(temp.values()))
    mode, count = stats.mode(arr)
    total_count = len(arr)

    vr = 1 - (count / total_count)
    return vr


def predictive_entropy(arr):
    """
    Calculate the predictive entropy of a probability distribution.

    Parameters:
    arr (1D array-like): An array representing a probability distribution.

    Returns:
    float: The predictive entropy of the distribution.
    """
    # Ensure the probabilities are a numpy array

    temp = Counter((np.asarray(arr) * 100).astype(int))
    total = sum(temp.values())
    for i, j in temp.items():
        temp[i] = j / total

    probs = np.array(list(temp.values()))

    # Check if the probabilities sum to 1, if not normalize them
    if not np.isclose(np.sum(probs), 1.0):
        probs = probs / np.sum(probs)

    # Calculate the entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))  # Adding epsilon to avoid log(0)

    return entropy


def mi(arr):
    # Ensure the probabilities are a numpy array
    arr = np.asarray(arr)

    # Calculate the entropy
    entropy = -np.sum(np.array([1 - arr.mean(), arr.mean()]) * np.log(np.array([1 - arr.mean(), arr.mean()]) + 1e-10))  # Adding epsilon to avoid log(0)
    return entropy - ((arr * np.log(arr + 1e-10)) + ((1 - arr) * np.log((1 - arr) + 1e-10))).mean()


def total_variance(arr):
    """
    Calculate the total variance of the values in a 1D array.

    Parameters:
    arr (numpy.ndarray): A 1D array of numerical values.

    Returns:
    float: The total variance of the array values.
    """
    # Convert input to numpy array if it's not already
    arr = np.asarray(arr)

    # Compute the mean of the array
    mean = np.mean(arr)

    # Compute the squared differences from the mean
    squared_diffs = (arr - mean) ** 2

    # Compute the total variance
    total_variance = np.sum(squared_diffs) / len(arr)

    return total_variance


def find_optimal_threshold(y_true, y_pred_prob, metric_fn, **kwargs):
    """
    Find the optimal threshold for a given metric function.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred_prob (array-like): Probability predictions for the positive class.
    metric_fn (function): A function that calculates the metric given true and predicted labels.

    Returns:
    float: The optimal threshold.
    float: The maximum metric value.
    """
    _, _, thresholds = roc_curve(y_true, y_pred_prob)
    acc = [metric_fn(y_true, (y_pred_prob > thrs) * 1, **kwargs) for thrs in thresholds]
    opt_met, opt_thr = max(zip(acc, thresholds), key=lambda x: x[0])

    return round(opt_met, 3), round(opt_thr, 3)


def evaluate(y_true, y_pred_prob):
    """
    Evaluate the performance of a binary classification.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred_prob (array-like): Probability predictions for the positive class.
    threshold (float, optional): Threshold to convert probabilities to binary predictions. Default is 0.5.

    Returns:
    dict: A dictionary containing various evaluation metrics.
    """

    # Calculate evaluation metrics
    accuracy = find_optimal_threshold(y_true, y_pred_prob, accuracy_score)
    auc = round(roc_auc_score(y_true, y_pred_prob), 3)

    # Youden's Index (Sensitivity + Specificity - 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    youdens_index = (round(np.max(tpr - fpr), 3), round(thresholds[np.argmax(tpr - fpr)], 3))

    # Create a dictionary to store the results
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'youdens': youdens_index,
    }

    return results


def print_evaluation(model=None, x_in=None, y_in=None):
    y_pred = model.predict(x_in)

    _, _, thresholds = roc_curve(y_in, y_pred[:, 1])
    acc = [balanced_accuracy_score(y_in, (y_pred[:, 1] > thrs) * 1) for thrs in thresholds]

    thr = max(zip(acc, thresholds), key=lambda x: x[0])[1]

    y_pred_labels = (y_pred[:, 1] > thr) * 1

    # Calculate metrics
    accuracy = accuracy_score(y_in, y_pred_labels)
    roc_auc = roc_auc_score(y_in, y_pred[:, 1])
    f1 = f1_score(y_in, y_pred_labels)
    confusion = confusion_matrix(y_in, y_pred_labels)
    recall = recall_score(y_in, y_pred_labels)
    specificity = recall_score(y_in, y_pred_labels, pos_label=0)
    balanced_accuracy = balanced_accuracy_score(y_in, y_pred_labels)

    print("Confusion Matrix:")
    print(confusion)

    print("Accuracy: ", accuracy)
    print("Sensitivity: ", recall)
    print("Specificity: ", specificity)
    print("Balanced accuracy: ", balanced_accuracy)
    print("ROC AUC: ", roc_auc)
    print("F1 Score: ", f1)

    precision, recall, thresholds = precision_recall_curve(y_in, y_pred[:, 1])

    # Calculate trapezoidal approximation for AUC under PR curve
    auc_prc = auc(recall, precision)

    print(f"AUC under Precision-Recall Curve: {auc_prc:.4f}")
    return


def split_correct_incorrect(model=None, x_in=None, y_in=None):
    pred_cor = dict()
    pred_mis = dict()

    y_pred = model.predict(x_in)

    _, _, thresholds = roc_curve(y_in, y_pred[:, 1])
    acc = [balanced_accuracy_score(y_in, (y_pred[:, 1] > thrs) * 1) for thrs in thresholds]

    thr = max(zip(acc, thresholds), key=lambda x: x[0])[1]

    y_pred_labels = (y_pred[:, 1] > thr) * 1

    for j in tqdm(range(10, 150, 10)):
        pred = np.zeros(len(x_in), j)
        for i in range(j):
            pred[:, i] = model(x_in, training=True).numpy()[:, 1]

        pred_cor[j] = pred[(y_in == y_pred_labels), :]
        pred_mis[j] = pred[(y_in != y_pred_labels), :]
    return pred_cor, pred_mis


def plot_results(main, reference, label, reference_label='Random sampling'):
    fig = plt.figure(figsize=(7, 4))
    (_, _, bars) = plt.errorbar(list(range(71 - main.shape[1], 71)), main.mean(axis=0), yerr=main.std(axis=0), fmt='-o', label=label, capsize=2,
                                linewidth=0.5,
                                markersize=4)
    for bar in bars:
        bar.set_linestyle((0, (2, 4)))

    (_, _, bars) = plt.errorbar(list(range(71 - reference.shape[1], 71)), reference.mean(axis=0), yerr=reference.std(axis=0), fmt='-o', label=reference_label, capsize=2,
                                linewidth=0.5,
                                markersize=4)
    for bar in bars:
        bar.set_linestyle((0, (2, 4)))

    plt.legend()
    fig.get_axes()[0].yaxis.set_major_formatter(FuncFormatter(lambda x, y: f'{int(x * 100)}%'))
    fig.get_axes()[0].set_xlim(-3, 73)
    plt.xlabel('Annotated samples')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.show()
    return

def mmd_rbf(X, Y, gamma=1.0):
    from sklearn import metrics
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def rep(x, data, rep_rank):
    if rep_rank[x.name]!=0:
        return -np.inf
    if rep_rank.sum():
        pre = mmd_rbf(data[np.where(rep_rank!=0)], data)
        post = mmd_rbf(np.vstack([np.array(x.X), data[np.where(rep_rank!=0)]]), data)
        return (pre - post) / pre
    else:
        return -mmd_rbf(np.vstack([np.array(x.X), data[np.where(rep_rank!=0)]]), data)
