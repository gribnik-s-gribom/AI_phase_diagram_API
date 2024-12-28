from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier  # Используем XGBClassifier для использования predict_proba
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from io import StringIO
import warnings
import xgboost as xgb

# Игнорируем конкретное предупреждение
warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")

# Инициализация приложения
app = FastAPI()

# Пути к файлам
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_data.csv"

# Загрузка данных
def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y

# Загрузка модели
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Инициализация данных и модели
X, y = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Метрики модели
@app.get("/metrics")
def get_metrics():
    try:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (Class 0)": precision_score(y_test, y_pred, pos_label=0),
            "Precision (Class 1)": precision_score(y_test, y_pred, pos_label=1),
            "Recall (Class 0)": recall_score(y_test, y_pred, pos_label=0),
            "Recall (Class 1)": recall_score(y_test, y_pred, pos_label=1),
            "F1-score (Class 0)": f1_score(y_test, y_pred, pos_label=0),
            "F1-score (Class 1)": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_test, y_pred_prob),
        }
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ROC-кривая
@app.get("/roc-curve")
def get_roc_curve():
    try:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Кросс-валидация
@app.get("/cross-validation")
def perform_cross_validation():
    try:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            roc_auc = roc_auc_score(y_test_fold, model.predict_proba(X_test_fold)[:, 1])
            cm = confusion_matrix(y_test_fold, y_pred_fold)

            results.append({
                "fold": fold,
                "confusion_matrix": cm.tolist(),
                "roc_auc": roc_auc
            })

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Кросс-валидация с визуализацией ROC-AUC
@app.get("/cross-validation/plot")
def get_roc_auc_plot():
    try:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_auc_scores = []

        for train_idx, test_idx in kfold.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            roc_auc_scores.append(roc_auc_score(y_test_fold, y_pred_prob_fold))

        # Построение графика
        plt.figure(figsize=(10, 6))
        folds = np.arange(1, len(roc_auc_scores) + 1)
        plt.plot(folds, roc_auc_scores, marker='o', markersize=10, linestyle='-', color='royalblue', linewidth=2, label='ROC-AUC Score')
        for i, score in enumerate(roc_auc_scores):
            plt.text(folds[i], score + 0.01, f'{score:.4f}', ha='center', fontsize=12, color='black')
        plt.ylim(0.8, 1.1)
        plt.xticks(folds)
        plt.xlabel('Fold Number', fontsize=14)
        plt.ylabel('ROC-AUC Score', fontsize=14)
        plt.title('5-fold Cross-Validation ROC-AUC', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()

        # Сохранение графика в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Важность признаков
@app.get("/feature-importance")
def feature_importance():
    try:
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# SHAP-анализ
@app.get("/shap-analysis")
def get_shap_analysis():
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Построение SHAP-графика
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)

        # Сохранение графика в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Предсказания
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    EXPECTED_COLUMNS = ["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
    try:
        content = await file.read()
        data = pd.read_csv(StringIO(content.decode("utf-8")))
        if not all(col in data.columns for col in EXPECTED_COLUMNS):
            raise ValueError(f"Missing columns: {', '.join(EXPECTED_COLUMNS)}")

        predictions = model.predict(data[EXPECTED_COLUMNS])
        prediction_probs = model.predict_proba(data[EXPECTED_COLUMNS])[:, 1]

        results = pd.DataFrame({
            "Sample": data.index,
            "Prediction": predictions,
            "Probability": prediction_probs
        })
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
