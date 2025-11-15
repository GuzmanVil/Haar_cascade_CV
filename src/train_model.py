from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
import joblib
import numpy as np

def train_cascade(X_train, y_train, X_val, y_val, win_size=(24, 24), num_stages=5, save_path="haar_cascade.pkl"):
    """
    Entrena una cascada de clasificadores AdaBoost secuenciales.
    Cada etapa filtra los falsos positivos de la anterior.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    cascada = []
    X_stage, y_stage = X_train.copy(), y_train.copy()

    for stage in range(num_stages):
        print(f"\n===== Etapa {stage+1}/{num_stages} =====")

        # === Entrenamiento inicial con todas las features ===
        weak = DecisionTreeClassifier(max_depth=1, random_state=42)
        model = AdaBoostClassifier(
            estimator=weak,
            n_estimators=80 + stage * 20,   # cada etapa un poco más compleja
            learning_rate=0.5,
            random_state=42
        )

        model.fit(X_stage, y_stage)

        # Reporte de validación previo a la selección
        y_pred = model.predict(X_val)
        print(classification_report(y_val, y_pred, digits=4))

        # === Selección de features ===
        selector = SelectFromModel(model, threshold="mean", prefit=True)
        X_stage_sel = selector.transform(X_stage)
        X_val_sel = selector.transform(X_val)
        print(f"[INFO] Features seleccionadas: {X_stage_sel.shape[1]} / {X_stage.shape[1]}")

        # === Reentrenar modelo con las features seleccionadas ===
        weak = DecisionTreeClassifier(max_depth=1, random_state=42)
        model = AdaBoostClassifier(
            estimator=weak,
            n_estimators=80 + stage * 40,
            learning_rate=0.5,
            random_state=42
        )
        model.fit(X_stage_sel, y_stage)

        # Validar el nuevo modelo ya reducido
        y_pred = model.predict(X_val_sel)
        print("[INFO] Reporte tras reentrenar con features seleccionadas:")
        print(classification_report(y_val, y_pred, digits=4))

        # === Filtrar falsos positivos (hard negative mining) ===
        probs = model.predict_proba(X_stage_sel)[:, 1]
        hard_neg_mask = (y_stage == 0) & (probs > (0.55 + 0.05*stage))
        keep_idx = ~hard_neg_mask

        X_stage = X_stage[keep_idx]
        y_stage = y_stage[keep_idx]

        print(f"[INFO] Próxima etapa: {X_stage.shape[0]} muestras restantes")

        # Guardar la etapa
        cascada.append({
            "model": model,
            "selector": selector,
            "threshold": 0.5 + stage * 0.05  # umbral progresivo
        })

    # === Guardar cascada completa ===
    joblib.dump({"cascade": cascada, "scaler": scaler, "win_size": win_size}, save_path)
    print(f"\n[OK] Cascada entrenada y guardada en {save_path}")
    return cascada, scaler
