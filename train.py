import argparse
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(C=1.0):
    """
    Εκπαιδεύει ένα Iris classification pipeline με LogisticRegression
    και επιστρέφει την ακρίβεια στο test set.
    """
    # Φόρτωση του Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    X = df[iris.feature_names]
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Ορισμός του pipeline
    model_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=C))
        ]
    )

    # Εκπαίδευση
    model_pipeline.fit(X_train, y_train)

    # Αξιολόγηση
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}")
    joblib.dump(model_pipeline, "iris_model.pkl")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization strength for LogisticRegression (C parameter)",
    )
    args = parser.parse_args()

    train(C=args.C)
