import catboost as cb
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(
        model_path: str,
        test_path: str
    ):
    # Load the model
    model = cb.CatBoostClassifier()
    model.load_model(model_path)
    
    # Load the test data
    df_test = pd.read_csv(test_path)
    df_test.fillna('NA', inplace=True)
    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test['target']
    
    # Predict probabilities and binary labels
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print the evaluation metrics
    print(f"ROC-AUC: {roc_auc:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained CatBoost model file")
    parser.add_argument("--test-path", type=str, required=True, help="Path to the test dataset CSV file")
    args = parser.parse_args()
    
    metrics = evaluate_model(
        model_path=args.model_path,
        test_path=args.test_path
    )

    # Optionally, save the evaluation metrics to a file
    output_file = args.model_path.replace('.cbm', '_evaluation_metrics.json')
    pd.Series(metrics).to_json(output_file)
    print(f"Evaluation metrics saved to {output_file}")
