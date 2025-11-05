from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import numpy as np

def train_model(X_train, y_train, X_val, y_val, win_size=(24, 24), save_path="haar_adaboost.pkl"):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    weak = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=weak,
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, digits=4))

    joblib.dump({"model": model, "scaler": scaler, "win_size": win_size}, save_path)
    print(f"[OK] Modelo guardado en {save_path}")
    return model, scaler
