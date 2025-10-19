from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Step 1: Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Step 2: Predictions
    y_pred = model.predict(X_test)

    # Step 3: Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n✅ Model Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Step 4: Save model for later use (deployment)
    joblib.dump(model, "fraud_detection_model.pkl")
    print("\n💾 Model saved as fraud_detection_model.pkl")

    return model

