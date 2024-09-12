import catboost as cb
import pandas as pd
import sklearn as sk
import numpy as np
import argparse

from sklearn.metrics import roc_auc_score
        
        
def train_and_evaluate(
        train_path: str,
        validation_path: str,
        test_path: str
    ):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(validation_path)
    df_test = pd.read_csv(test_path)
    
    df_train.fillna('NA', inplace=True)
    df_val.fillna('NA', inplace=True)
    df_test.fillna('NA', inplace=True)

    X_train, y_train = df_train.iloc[:, :-1], df_train['target']
    X_val, y_val = df_val.iloc[:, :-1], df_val['target']
    X_test, y_test = df_test.iloc[:, :-1], df_test['target']

    features = {
        'numerical': ['retail_price', 'age'],
        'static': ['gender', 'country', 'city'],
        'dynamic': ['brand', 'department', 'category']
    }

    train_pool = cb.Pool(
        X_train,
        y_train,
        cat_features=features.get("static"),
        text_features=features.get("dynamic"),
    )

    validation_pool = cb.Pool(
        X_val,
        y_val,
        cat_features=features.get("static"),
        text_features=features.get("dynamic"),
    )
    
    test_pool = cb.Pool(
        X_test,
        y_test,
        cat_features=features.get("static"),
        text_features=features.get("dynamic"),
    )

    # Text processing options
    text_processing_options = {
        "tokenizers": [
            {"tokenizer_id": "SemiColon", "delimiter": ";", "lowercasing": "false"}
        ],
        "dictionaries": [{"dictionary_id": "Word", "gram_order": "1"}],
        "feature_processing": {
            "default": [
                {
                    "dictionaries_names": ["Word"],
                    "feature_calcers": ["BoW"],
                    "tokenizers_names": ["SemiColon"],
                }
            ],
        },
    }

    # Train the model
    model = cb.CatBoostClassifier(
        iterations=200,
        loss_function="Logloss",
        random_state=42,
        verbose=1,
        auto_class_weights="SqrtBalanced",
        use_best_model=True,
        text_processing=text_processing_options,
        eval_metric='AUC'
    )


    model.fit(
        train_pool, 
        eval_set=validation_pool, 
        verbose=10
    )
    
    roc_train = roc_auc_score(y_true=y_train, y_score=model.predict(X_train))
    roc_eval  = roc_auc_score(y_true=y_val, y_score=model.predict(X_val))
    roc_test  = roc_auc_score(y_true=y_test, y_score=model.predict(X_test))
    print(f"ROC-AUC for train set      : {roc_train:.2f}")
    print(f"ROC-AUC for validation set : {roc_eval:.2f}")
    print(f"ROC-AUC for test.      set : {roc_test:.2f}")
    
    return {"model": model, "scores": {"train": roc_train, "eval": roc_eval, "test": roc_test}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--validation-path", type=str)
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--output-dir", type=str)
    args, _ = parser.parse_known_args()
    _ = train_and_evaluate(
        args.train_path,
        args.validation_path,
        args.test_path)
