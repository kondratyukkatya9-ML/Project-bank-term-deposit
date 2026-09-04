from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(model, X_train, y_train, X_val, y_val, name, params=""):
    
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    pr_train = average_precision_score(y_train, train_proba)
    pr_val = average_precision_score(y_val, val_proba)

    return {
        'model': name,
        'params': params,
        'pr_auc_train': pr_train,
        'pr_auc_val': pr_val,
        'pr_auc_gap': pr_train - pr_val,
        'roc_auc_train': roc_auc_score(y_train, train_proba),
        'roc_auc_val': roc_auc_score(y_val, val_proba),
    }