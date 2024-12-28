import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from xgboost import XGBClassifier
import io

# Пути к файлам
MODEL_PATH = "xgboost_model.pkl"
DATA_PATH = "Main_data.csv"


# Загрузка данных
@st.cache_data
def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data[["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    return X, y


# Загрузка модели
@st.cache_data
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


# Инициализация
X, y = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Главный интерфейс
st.title("Модель прогнозирования (XGBoost)")

# Метрики модели
st.header("Метрики модели")
if st.button("Рассчитать метрики"):
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
    st.json(metrics)

# ROC-кривая
st.header("ROC-кривая")
if st.button("Построить ROC-кривую"):
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
    st.pyplot(plt)

# Кросс-валидация
st.header("Кросс-валидация")
if st.button("Выполнить кросс-валидацию"):
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
            "Fold": fold,
            "Confusion Matrix": cm.tolist(),
            "ROC-AUC": roc_auc,
        })

    st.json(results)

# Важность признаков
st.header("Важность признаков")
if st.button("Показать важность признаков"):
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(plt)

# SHAP-анализ
st.header("SHAP-анализ")
if st.button("Построить SHAP-анализ"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, show=False)
    st.pyplot(plt)

# Предсказания
st.header("Предсказания")
uploaded_file = st.file_uploader("Загрузите файл CSV для предсказаний", type="csv")
if uploaded_file is not None:
    EXPECTED_COLUMNS = ["T", "E", "C", "FM", "Xfm", "AFM", "Xafm"]
    try:
        data = pd.read_csv(uploaded_file)
        if not all(col in data.columns for col in EXPECTED_COLUMNS):
            st.error(f"Отсутствуют необходимые колонки: {', '.join(EXPECTED_COLUMNS)}")
        else:
            predictions = model.predict(data[EXPECTED_COLUMNS])
            prediction_probs = model.predict_proba(data[EXPECTED_COLUMNS])[:, 1]

            results = pd.DataFrame({
                "Sample": data.index,
                "Prediction": predictions,
                "Probability": prediction_probs,
            })
            st.write(results)
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
