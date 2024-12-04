import numpy as np

classes = ["unacc", "acc", "good", "vgood"]
n_classes = 4


class MyNaiveBayes:
    def __init__(self):
        self.class_probs = {cls: 0 for cls in classes}
        self.feature_probs = {cls: {} for cls in classes}

    def fit(self, X, y):
        total_samples = len(y.values)

        # Calculate class probabilities
        for cls in classes:
            cls_samples = []
            for i in range(total_samples):
                if y.values[i][0] == cls:
                    cls_samples.append(X.values[i])
                    
            self.class_probs[cls] = (
                len(cls_samples) / total_samples
            )  # Probability of class 'cls'

            # Initialize feature probabilities for class 'cls'
            self.feature_probs[cls] = {}

            # Calculate feature probabilities
            col_number = 0
            for col in X.columns:
                unique_values = X[col].unique()  # Unique values for feature 'col'
                value_counts = {value: 0 for value in unique_values}
                m = len(unique_values)  # Number of unique values in column 'col'

                for cls_sample in cls_samples:
                    value_counts[cls_sample[col_number]] += 1
                    
                # Store probabilities for each unique value in this feature
                self.feature_probs[cls][col] = {}
                for value in unique_values:
                    count = value_counts.get(value, 0)  # Observed count
                    smoothed_prob = (count + 1) / (
                        len(cls_samples) + m
                    )  # As mentioned in class notes
                    self.feature_probs[cls][col][value] = smoothed_prob
                
                col_number += 1

    def predict(self, X):
        predictions = []
        X_array = X.values
        n_rows, n_cols = X_array.shape

        for i in range(n_rows):
            row = X_array[i]
            class_scores = {}

            # Compute scores for each class
            for cls in classes:
                class_score = np.log(self.class_probs[cls])
                for j in range(n_cols):
                    col = X.columns[j]
                    feature_value = row[j]
                    class_score += np.log(
                        self.feature_probs[cls][col].get(feature_value, 1e-9)
                    )
                class_scores[cls] = class_score

            # Select class with the highest score
            best_class = None
            max_score = float("-inf")
            for cls, score in class_scores.items():
                if score > max_score:
                    best_class = cls
                    max_score = score

            predictions.append(best_class)

        return np.array(predictions)
