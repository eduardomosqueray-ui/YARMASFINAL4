# Debe direccionar VS Code a la carpeta con los archivos:
# 1.- Archivo
# 2.- Abrir carpeta. Debe dar click en la carpeta que contiene los archivos de interés
#3.- A la izquierda, en el explorador deberá poder visualizar todos los archivos
#------------------------------------------------------------------------------------------------

# CÓDIGO STREAMLIT
# Ir a:   Ver/Terminal
# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv'): coloca este código
#   python -m venv venv

#---------------------------------------------------------------------------------------
# Luego de crear el ambiente virtual, lo activas
#   .\venv\Scripts\activate   # En Windows
#---------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Cuando vuelva a iniciar sesión, debe volver a activar el ambiente virtual, ya no lo debe crear.
# En este caso debes abrir la carpeta con los archivos del caso.
#---------------------------------------------------------------------------------------------


# Instala la versión específica de scikit-learn
#   pip install scikit-learn==1.2.2
# Instala otras dependencias, incluyendo Streamlit
#  pip install streamlit pandas joblib
#-------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
#      Get-ExecutionPolicy                           Si es Restricted; ejecuta
#      Set-ExecutionPolicy RemoteSigned              Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate



import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
#import pyautogui

# Cargar el modelo de regresión
regressor = load('Modelopipeline.joblib')

# Cargar el encoder
#with open('encoderpipeline.pickle', 'rb') as f:
#    encoder = pickle.load(f)

# Inicializar variables
#rd_spend = administration = marketing_spend = 0.0
#selected_state = "New York"
if regressor is not None:

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("ℹ️ Información del Modelo")

        st.markdown("""
        **Tipo de Modelo:** Regresión Lineal con Pipeline

        **Variables de entrada:**
        - 📊 Edad (numérica)
        - 👥 Sexo (categórica)
        - 🗺️ Departamento (categórica)
        - 🏥 Tipo de Atención (categórica)
        - 📅 Días de Atención (numérica)
        - 📋 Grupo CIE10 (categórica)

        **Variable de salida:**
        - 💰 Monto Bruto (S/)
        """)

        st.divider()
        st.caption("Desarrollado para el Examen Final de Machine Learning")

    # ==================== FORMULARIO ====================

    st.subheader("📋 Ingrese los datos del paciente")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 👤 Datos Personales")

        edad = st.number_input(
            "Edad (en años)",
            min_value=0,
            max_value=120,
            value=50,
            step=1
        )

        sexo = st.selectbox(
            "Sexo",
            options=sexo,
            format_func=lambda x: "Femenino" if x == "F" else "Masculino"
        )

        departamento = st.selectbox(
            "Departamento de Procedencia",
            options=DEPARTAMENTO_OPTIONS
        )

    with col2:

        st.markdown("### 🏥 Datos de Atención")

        tipo_atencion = st.selectbox(
            "Tipo de Atención",
            options=TIPO_ATENC_OPTIONS
        )

        dias_atencion = st.number_input(
            "Días de Atención",
            min_value=0,
            max_value=365,
            value=0,
            step=1
        )

        grupo_cie10 = st.selectbox(
            "Grupo CIE10",
            options=GRUPO_CIE10_OPTIONS
        )

    st.markdown("---")

    predecir = st.button(
        "🔮 PREDECIR MONTO BRUTO",
        use_container_width=True,
        type="primary"
    )

    # ==================== PREDICCIÓN ====================

    if predecir:

        try:

            obs = pd.DataFrame({
                'edad': [edad],
                'SEXO': [sexo],
                'DEPARTAMENTO_PAC': [departamento],
                'TIPO_ATENC': [tipo_atencion],
                'DIAS_ATENCIÓN': [dias_atencion],
                'ATE_GRUPOCIE10': [grupo_cie10]
            })

            monto_predicho = regressor.predict(obs)[0]

            st.markdown("---")

            st.subheader("💰 Resultado de la Predicción")

            st.markdown(
                f"""
                <div style="
                    background-color:#f8f9fa;
                    padding:25px;
                    border-radius:15px;
                    border-left:8px solid #28a745;
                    text-align:center;
                    margin-bottom:25px;
                ">
                    <h3>Monto Bruto Estimado</h3>
                    <h1 style="color:#28a745;">
                        S/ {monto_predicho:,.2f}
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Edad",
                    f"{edad} años"
                )

            with col2:
                st.metric(
                    "Sexo",
                    "Femenino" if sexo == "F" else "Masculino"
                )

            with col3:
                st.metric(
                    "Departamento",
                    departamento
                )

            with col4:
                st.metric(
                    "Días Atención",
                    dias_atencion
                )

            with st.expander(
                "📋 Ver detalle de los datos ingresados",
                expanded=True
            ):

                resumen = pd.DataFrame({
                    "Variable": [
                        "Edad",
                        "Sexo",
                        "Departamento",
                        "Tipo de Atención",
                        "Días de Atención",
                        "Grupo CIE10"
                    ],
                    "Valor": [
                        f"{edad} años",
                        "Femenino" if sexo == "F" else "Masculino",
                        departamento,
                        tipo_atencion,
                        f"{dias_atencion} días",
                        grupo_cie10
                    ]
                })

                st.dataframe(
                    resumen,
                    use_container_width=True,
                    hide_index=True
                )

            st.success("✅ Predicción realizada correctamente")

            st.balloons()

        except Exception as e:

            st.error("❌ Error al realizar la predicción")

            st.code(str(e))
# Colocar el botón "Resetear" debajo del botón "Predecir"
#if st.sidebar.button("Resetear"):
    # Resetear inputs
  #  reset_inputs()

#   R&D Spend	Administration	Marketing Spend  Ciudad  
#	  142107.34  	91391.77	366168.42         Florida    ---->

# Cambiar los valores.
# Para asignar valores: ver los rangos de las cuantitativas ( MÍNIMO --MÁXIMO)
# eso determinan  cómo predice el modelo. 

#  streamlit run streamlitpipelines.py       en la consola
#  pip freeze > requirements.txt