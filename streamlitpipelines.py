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
rd_spend = administration = marketing_spend = 0.0
selected_state = "New York"

# Streamlit app
st.title("Modelo de Regresión")
st.markdown("##### Si colocas un valor negativo, aparecerá un error y no podrás completar otros campos. La predicción será incorrecta.")

# Sidebar para la entrada del usuario
st.sidebar.header("Campos a Evaluar")

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Monto Bruto en Salud",
    page_icon="🏥",
    layout="wide"
)

# ==================== DEFINICIÓN DE VARIABLES CATEGÓRICAS ====================
# Opciones para SEXO
SEXO_OPTIONS = ['F', 'M']

# Opciones para DEPARTAMENTO_PAC (todos los departamentos listados)
DEPARTAMENTO_OPTIONS = [
    'LORETO', 'LAMBAYEQUE', 'PIURA', 'AREQUIPA', 'LIMA', 'LA LIBERTAD', 
    'ANCASH', 'HUANCAVELICA', 'JUNIN', 'CUSCO', 'CALLAO', 'TACNA', 
    'UCAYALI', 'CAJAMARCA', 'AYACUCHO', 'HUANUCO', 'PUNO', 'SAN MARTIN', 
    'TUMBES', 'APURIMAC', 'ICA', 'MOQUEGUA', 'AMERICA', 'PASCO', 
    'AMAZONAS', 'MADRE DE DIOS', 'ASIA', 'EUROPA'
]

# Opciones para TIPO_ATENC
TIPO_ATENC_OPTIONS = [
    'AMBULATORIO', 'HOSPITALIZACIÓN', 'OTRO', 'EMERGENCIA'
]

# Opciones para ATE_GRUPOCIE10
GRUPO_CIE10_OPTIONS = [
    'ENFERMEDAD RARA O HUERFANA', 'LINFOMA', 'INSUFICIENCIA RENAL CRONICA TERMINAL',
    'CANCER DE MAMA', 'CANCER DE CUELLO UTERINO', 'CANCER DE ESTOMAGO',
    'CANCER DE COLON', 'LEUCEMIA', 'CANCER DE PROSTATA',
    'TRASPLANTE DE PROGENITORES HEMATOPOYÉTICOS', 'TRASPLANTE RENAL', 'TRASPLANTE HEPÁTICO'
]


# ==================== INTERFAZ DE USUARIO ====================
st.title("🏥 Sistema de Predicción de Monto Bruto en Servicios de Salud - YARMAS MOSQUERA")
st.markdown("---")

# Cargar modelo
with st.spinner("Cargando modelo predictivo..."):
    modelo = cargar_modelo()

if modelo is not None:
    # Barra lateral con información
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
        st.caption("Desarrollado para el Examen Final de ML")
    
    # Formulario principal
    st.subheader("📋 Ingrese los datos del paciente")
    
    # Crear 2 columnas para el formulario
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 👤 Datos Personales")
        
        edad = st.number_input(
            "**Edad** (en años)", 
            min_value=0, 
            max_value=120, 
            value=50, 
            step=1,
            help="Ingrese la edad del paciente"
        )
        
        sexo = st.selectbox(
            "**Sexo**", 
            options=SEXO_OPTIONS,
            help="Seleccione el sexo del paciente",
            format_func=lambda x: "Femenino" if x == 'F' else "Masculino"
        )
        
        departamento = st.selectbox(
            "**Departamento de Procedencia**", 
            options=DEPARTAMENTO_OPTIONS,
            help="Seleccione el departamento donde reside el paciente"
        )
        
    with col2:
        st.markdown("### 🏥 Datos de Atención")
        
        tipo_atencion = st.selectbox(
            "**Tipo de Atención**", 
            options=TIPO_ATENC_OPTIONS,
            help="Seleccione el tipo de atención médica"
        )
        
        dias_atencion = st.number_input(
            "**Días de Atención**", 
            min_value=0, 
            max_value=365, 
            value=0, 
            step=1,
            help="Número de días que duró la atención"
        )
        
        grupo_cie10 = st.selectbox(
            "**Grupo CIE10**", 
            options=GRUPO_CIE10_OPTIONS,
            help="Seleccione el diagnóstico según clasificación CIE10"
        )
    
    # Botón de predicción
    st.markdown("---")
    
    col_button, col_empty = st.columns([2, 3])
    with col_button:
        predecir = st.button(
            "🔮 **PREDECIR MONTO BRUTO**", 
            type="primary", 
            use_container_width=True
        )
    
    if predecir:
        with st.spinner("Calculando predicción..."):
            try:
                # Realizar predicción
                monto_predicho = predecir_monto(
                    modelo, edad, sexo, departamento, 
                    tipo_atencion, dias_atencion, grupo_cie10
                )
                
                # Mostrar resultado
                st.markdown("---")
                st.subheader("💰 Resultado de la Predicción")
                
                # Crear métricas en columnas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="**Monto Bruto Estimado**",
                        value=f"S/ {monto_predicho:,.2f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="**Edad**",
                        value=f"{edad} años",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="**Sexo**",
                        value="Femenino" if sexo == 'F' else "Masculino",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        label="**Días de Atención**",
                        value=f"{dias_atencion} días",
                        delta=None
                    )
                
                # Mostrar tabla con resumen de datos ingresados
                with st.expander("📊 Ver detalle de los datos ingresados", expanded=True):
                    datos_resumen = pd.DataFrame({
                        'Variable': [
                            'Edad', 'Sexo', 'Departamento', 
                            'Tipo de Atención', 'Días de Atención', 'Grupo CIE10'
                        ],
                        'Valor': [
                            f"{edad} años", 
                            'Femenino' if sexo == 'F' else 'Masculino',
                            departamento, 
                            tipo_atencion, 
                            f"{dias_atencion} días",
                            grupo_cie10
                        ]
                    })
                    st.dataframe(datos_resumen, use_container_width=True, hide_index=True)
                
                # Mostrar nota informativa
                # st.info("💡 **Nota:** Esta predicción es una estimación basada en el modelo de regresión lineal. El monto real puede variar según otros factores no considerados en el modelo.")
                
                # Efecto visual
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")
                st.info("Por favor, verifica que todos los datos sean correctos y que el modelo esté funcionando adecuadamente.")

# Colocar el botón "Resetear" debajo del botón "Predecir"
if st.sidebar.button("Resetear"):
    # Resetear inputs
    reset_inputs()

#   R&D Spend	Administration	Marketing Spend  Ciudad  
#	  142107.34  	91391.77	366168.42         Florida    ---->

# Cambiar los valores.
# Para asignar valores: ver los rangos de las cuantitativas ( MÍNIMO --MÁXIMO)
# eso determinan  cómo predice el modelo. 

#  streamlit run streamlitpipelines.py       en la consola
#  pip freeze > requirements.txt