from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


def calculate_acc(predictions, true_labels):
    # If an array has the shape (N, 1), convert it to (N)
    if predictions.ndim == 2:
        predictions = predictions[:, 0]
    if true_labels.ndim == 2:
        true_labels = true_labels[:, 0]

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.2f}")


def plot_conf_mat(predictions, true_labels):
    if predictions.ndim == 2:
        predictions = predictions[:, 0]
    if true_labels.ndim == 2:
        true_labels = true_labels[:, 0]

    # Plot the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["unacc", "acc", "good", "vgood"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
