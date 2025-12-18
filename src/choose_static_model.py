import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class StaticModelSelector:
    """
    This class is responsible for:
        - Loading processed static ASL letter data
        - Training it on multiple ML models
        - Comparing their performance
        - Saving the best-performing model
    """

    def __init__(self, data_dir="../data/processed", output_dir="../models"):
        # Paths for data and saved models
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Models to compare
        self.models = {
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
            ),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
            ),
        }

        # Storage for results
        self.results = {}
        self.best_model = None
        self.best_score = 0.0
        self.best_name = None

    def load_data(self):
        """
        Loading processed static ASL letter data.
        Uses training and validation sets only.
        """
        self.X_train = np.load(self.data_dir / "static_X_train.npy")
        self.y_train = np.load(self.data_dir / "static_y_train.npy")
        self.X_val = np.load(self.data_dir / "static_X_val.npy")
        self.y_val = np.load(self.data_dir / "static_y_val.npy")

    def train_and_evaluate(self):
        """
        Train each model and evaluate it on the validation set.
        The best model is selected based on validation accuracy.
        """
        for name, model in self.models.items():
            print(f"\nTraining {name}")

            # Train model
            model.fit(self.X_train, self.y_train)

            # Validate model
            preds = model.predict(self.X_val)
            acc = accuracy_score(self.y_val, preds)

            self.results[name] = acc
            print(f"{name} validation accuracy: {acc:.4f}")

            # Track best model
            if acc > self.best_score:
                self.best_score = acc
                self.best_model = model
                self.best_name = name

    def plot_results(self):
        """
        Visualize validation accuracy for all tested models.
        Accuracy values are shown directly on the bars.
        """
        names = list(self.results.keys())
        scores = list(self.results.values())

        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, scores)

        plt.ylabel("Validation Accuracy")
        plt.title("Static ASL Model Comparison")
        plt.xticks(rotation=30, ha="right")

        # Show exact accuracy on each bar
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score * 100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Zoom into high-accuracy range for better comparison
        plt.ylim(0.95, 1.0)
        plt.tight_layout()
        plt.show()

    def save_best_model(self):
        """
        Save the best-performing model.
        """
        path = self.output_dir / f"static_best_model_{self.best_name}.joblib"
        joblib.dump(self.best_model, path)

        print("\nBest static model:")
        print(f"Model: {self.best_name}")
        print(f"Accuracy: {self.best_score:.4f}")

    def run(self):
        """
        Full pipeline:
        load data -> train the models -> compare the models -> save the best model
        """
        self.load_data()
        self.train_and_evaluate()
        self.plot_results()
        self.save_best_model()


if __name__ == "__main__":
    selector = StaticModelSelector()
    selector.run()
