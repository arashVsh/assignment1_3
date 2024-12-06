import numpy as np

classes = ["unacc", "acc", "good", "vgood"]


class MyNaiveBayes:
    def __init__(self):
        self.class_probs = {cls: 0 for cls in classes}
        self.feature_probs = {cls: {} for cls in classes}

    def fit(self, X, y):
        total_samples = len(y.values)

        # Calculate class probabilities
        for cls in classes:
            cls_samples = []
            for row in range(total_samples):
                if y.values[row][0] == cls:
                    cls_samples.append(X.values[row])

            self.class_probs[cls] = (
                len(cls_samples) / total_samples
            )  # Probability of class 'cls'

            # Initialize feature probabilities for class 'cls'
            self.feature_probs[cls] = {}

            # Calculate feature probabilities
            feature_number = 0
            for feature in X.columns:
                unique_values = X[feature].unique()  # Unique values for this feature
                value_counts = {value: 0 for value in unique_values}

                for cls_sample in cls_samples:
                    value_counts[cls_sample[feature_number]] += 1

                # Store probabilities for each unique value in this feature
                self.feature_probs[cls][feature] = {}
                for value in unique_values:
                    count = value_counts.get(value, 0)
                    smoothed_prob = (count + 1) / (
                        len(cls_samples) + len(unique_values)
                    )  # As mentioned in class notes
                    self.feature_probs[cls][feature][value] = smoothed_prob

                feature_number += 1

    def predict(self, X):
        predictions = []
        X_array = X.values
        n_rows, n_features = X_array.shape

        for row in range(n_rows):
            row = X_array[row]
            class_scores = {}

            # Compute scores for each class
            for cls in classes:
                class_score = np.log(self.class_probs[cls])
                for feature_index in range(n_features):
                    feature = X.columns[feature_index]
                    feature_value = row[feature_index]
                    class_score += np.log(
                        self.feature_probs[cls][feature].get(feature_value, 1e-9)
                    )
                class_scores[cls] = class_score

            # Select class with the highest score
            best_class = None
            best_score = float("-inf")
            for cls, score in class_scores.items():
                if score > best_score:
                    best_class = cls
                    best_score = score

            predictions.append(best_class)

        return np.array(predictions)
