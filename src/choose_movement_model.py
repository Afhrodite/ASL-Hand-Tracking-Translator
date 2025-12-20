import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Bidirectional,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class MovementModelSelector:
    """
    A class to train and compare different models for movement-based ASL letters (J and Z).
    Handles LSTM, GRU, 1D CNN, Bidirectional LSTM, and Stacked LSTM.
    """

    def __init__(self, data_dir="../data/processed", output_dir="../models"):
        # Paths for data and where to save models
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Dictionary to store model objects and results
        self.models = {}
        self.results = {}

        # To track the best model
        self.best_model = None
        self.best_score = 0.0
        self.best_name = None

    def load_data(self, letter):
        """
        Load preprocessed movement datasets for a given letter (J or Z)
        and one-hot encode the labels.
        """
        self.letter = letter

        # Load training and validation data
        self.X_train = np.load(self.data_dir / f"movement_{letter}_X_train.npy")
        self.y_train = np.load(self.data_dir / f"movement_{letter}_y_train.npy")
        self.X_val = np.load(self.data_dir / f"movement_{letter}_X_val.npy")
        self.y_val = np.load(self.data_dir / f"movement_{letter}_y_val.npy")

        # One-hot encode labels (0 or 1)
        self.y_train = to_categorical(self.y_train, num_classes=2)
        self.y_val = to_categorical(self.y_val, num_classes=2)

        # Input shape for Keras layers
        self.input_shape = self.X_train.shape[1:]

    def build_models(self):
        """
        Build and compile different deep learning models to try:
        LSTM, GRU, 1D CNN, Bidirectional LSTM, and Stacked LSTM.
        """
        # Simple LSTM
        lstm = Sequential(
            [LSTM(64, input_shape=self.input_shape), Dense(2, activation="softmax")]
        )
        lstm.compile(
            optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # GRU
        gru = Sequential(
            [GRU(64, input_shape=self.input_shape), Dense(2, activation="softmax")]
        )
        gru.compile(
            optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # 1D CNN for sequence features
        cnn = Sequential(
            [
                Conv1D(
                    64, kernel_size=3, activation="relu", input_shape=self.input_shape
                ),
                MaxPooling1D(2),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(2, activation="softmax"),
            ]
        )
        cnn.compile(
            optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Bidirectional LSTM (reads sequence both directions)
        bilstm = Sequential(
            [
                Bidirectional(LSTM(64), input_shape=self.input_shape),
                Dense(2, activation="softmax"),
            ]
        )
        bilstm.compile(
            optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Stacked LSTM (two LSTM layers stacked)
        stacked_lstm = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=self.input_shape),
                LSTM(32),
                Dense(2, activation="softmax"),
            ]
        )
        stacked_lstm.compile(
            optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Save all models in a dictionary
        self.models = {
            "LSTM": lstm,
            "GRU": gru,
            "CNN_1D": cnn,
            "BiLSTM": bilstm,
            "Stacked_LSTM": stacked_lstm,
        }

    def train_and_evaluate(self, epochs=20, batch_size=16):
        """
        Train each model and evaluate validation accuracy.
        Keep track of the best-performing model.
        """
        for name, model in self.models.items():
            print(f"\nTraining {name} for letter {self.letter}")

            history = model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

            # Max validation accuracy for comparison
            val_acc = max(history.history["val_accuracy"])
            self.results[name] = val_acc

            print(f"{name} validation accuracy: {val_acc:.4f}")

            # Update best model if this one is better
            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_model = model
                self.best_name = name

    def plot_results(self):
        """
        Create a bar chart showing the validation accuracy of each model.
        """
        names = list(self.results.keys())
        scores = list(self.results.values())

        plt.figure(figsize=(8, 4))
        bars = plt.bar(names, scores, color="skyblue")

        # Title at the top
        plt.title(
            f"Movement Model Comparison ({self.letter})", fontsize=14, weight="bold"
        )
        plt.ylabel("Validation Accuracy")

        # Show accuracy on top of each bar
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{score * 100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

        plt.ylim(0.5, 1.05)  # Keep some space above the bars
        plt.tight_layout()
        plt.show()

    def save_best_model(self):
        """
        Save the best-performing model for this letter.
        """
        path = self.output_dir / f"movement_{self.letter}_best_{self.best_name}.keras"
        self.best_model.save(path)

        print("\nBest movement model:")
        print(f"Letter: {self.letter}")
        print(f"Model: {self.best_name}")
        print(f"Accuracy: {self.best_score:.4f}")

    def run(self, letter):
        """
        Run the full workflow: load data, build models, train, evaluate, plot, and save.
        """
        self.load_data(letter)
        self.build_models()
        self.train_and_evaluate()
        self.plot_results()
        self.save_best_model()


if __name__ == "__main__":
    # Create a selector object and train models for J and Z
    selector = MovementModelSelector()
    selector.run("J")
    selector.run("Z")
