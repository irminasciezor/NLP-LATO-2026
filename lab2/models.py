from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp

MODELS = {
    "nb":     MultinomialNB,
    "rf":     RandomForestClassifier,
    "mlp":    MLPClassifier,
    "logreg": LogisticRegression,
}

GRIDS = {
    "nb":     {"alpha": [0.1, 0.5, 1.0]},
    "rf":     {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
    "mlp":    {"hidden_layer_sizes": [(128,), (256, 128)]},
    "logreg": {"C": [0.1, 1, 10]},
}

DEFAULT_PARAMS = {
    "nb":     {},
    "rf":     {"n_jobs": -1},
    "mlp":    {"max_iter": 300},
    "logreg": {"max_iter": 1000, "n_jobs": -1},
}


def _prepare_X(X, method: str):
    if method == "nb":
        if sp.issparse(X):
            return X
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    return X


def run_experiment(
    X, y, method_name: str, model_key: str,
    use_gridsearch: bool, seed: int
) -> dict:
    X_proc = _prepare_X(X, model_key)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf_class = MODELS[model_key]
    params = DEFAULT_PARAMS[model_key].copy()

    if "random_state" in clf_class().get_params():
        params["random_state"] = seed

    clf = clf_class(**params)

    if use_gridsearch:
        gs = GridSearchCV(clf, GRIDS[model_key], cv=3, n_jobs=-1, scoring="f1_macro")
        gs.fit(X_train, y_train)
        clf = gs.best_estimator_
    else:
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    feature_importance = _get_feature_importance(clf, model_key)

    return {
        "model_key": model_key,
        "embedding": method_name,
        "accuracy": acc,
        "macro_f1": f1,
        "seed": seed,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "clf": clf,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def _get_feature_importance(clf, model_key: str):
    try:
        if model_key == "rf":
            return clf.feature_importances_
        if model_key == "logreg":
            return clf.coef_
        if model_key == "nb":
            return clf.feature_log_prob_
    except AttributeError:
        pass
    return None


def resolve_models(method_arg: str) -> list[str]:
    if method_arg == "all":
        return list(MODELS.keys())
    return [m.strip() for m in method_arg.split(",") if m.strip() in MODELS]


SEEDS = [42, 1337, 2024]
