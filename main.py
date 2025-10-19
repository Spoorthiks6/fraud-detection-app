# main.py
from dataset import load_dataset       # or download_and_load_dataset
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_model

def main():
    # Step 1: Load dataset
    df = load_dataset("creditcard.csv")  # change path if needed

    # Step 2: Preprocess data
    X_train_res, X_test_scaled, y_train_res, y_test = preprocess_data(df)

    # Step 3: Train model and evaluate
    model = train_and_evaluate_model(X_train_res, X_test_scaled, y_train_res, y_test)

if __name__ == "__main__":
    main()
