import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Eficiencia Presupuestal - Ica", layout="wide")
st.title("📊 Predicción de Eficiencia Presupuestal - Municipalidad Distrital de Pueblo Nuevo (2019-2024)")

@st.cache_resource
def load_model():
    return joblib.load("modelo_eficiencia_presupuestal.pkl")

model = load_model()

st.sidebar.header("📁 Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';', decimal=',')

    if 'Categoría Presupuestal' in df.columns:
        df = df.drop(columns=['Categoría Presupuestal'])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if 'Eficiencia Presupuestal' in df.columns:
        X = df.drop('Eficiencia Presupuestal', axis=1)
        y = df['Eficiencia Presupuestal']
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.success("✅ Datos cargados y predicciones realizadas")
        st.metric("R² Score", f"{r2:.2f}", help="Porcentaje de varianza explicada por el modelo")
        st.metric("Mean Squared Error", f"{mse:.4f}")

        df_resultados = df.copy()
        df_resultados['Predicción'] = y_pred

        st.subheader("📈 Importancia de las características")
        importances = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)

        st.subheader("🔥 Mapa de Calor de Correlaciones")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("📋 Tabla de resultados")
        st.dataframe(df_resultados, use_container_width=True)

        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar predicciones en CSV",
            data=csv,
            file_name='predicciones_eficiencia.csv',
            mime='text/csv'
        )
    else:
        st.warning("El archivo debe contener la columna 'Eficiencia Presupuestal' para evaluar el modelo.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")