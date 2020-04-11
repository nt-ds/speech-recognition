# --- Import libraries ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


# --- Function ---
def plot_confusion_matrix(real_labels,
                          predicted_labels,
                          target_names,
                          title="Confusion Matrix"):
    """
    Print classification report and plot confusion matrix

    Arguments
    ---------
    real_labels:        test images' real labels

    predicted_labels:   test images' predicted labels

    target_names:       given classification classes such as [0, 1, 2]
                        the class names, for example: ["bed", "dog", "wow"]

    Optional Arguments
    ------------------
    title:              the text to display at the top of the confusion matrix plot

    Citation
    --------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Source
    ------
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """

    # get classification report and save it
    report = pd.DataFrame(classification_report(real_labels,
                                                predicted_labels,
                                                target_names=target_names,
                                                output_dict=True)).transpose()
    report.to_csv(
        "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\result\\best_report.csv",
        index=True)

    # print classification report
    print(classification_report(real_labels, predicted_labels, target_names=target_names))

    # get confusion matrix info
    cm = confusion_matrix(real_labels, predicted_labels)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy = np.trace(cm) / float(np.sum(cm))

    # create confusion matrix plot and save it
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title(title)
    plt.colorbar(fraction=0.045, pad=0.025)

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label\naccuracy={:0.4f}; inaccuracy={:0.4f}".format(accuracy, 1 - accuracy))
    plt.savefig(
        "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\result\\best_result.png",
        bbox_inches="tight")
