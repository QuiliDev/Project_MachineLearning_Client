# ⚖️ Predicción de Disputas en Quejas de Clientes

Este proyecto es una aplicación interactiva desarrollada con **Streamlit** que utiliza **Machine Learning (XGBoost)** para predecir el riesgo de disputa en quejas de clientes.  
La herramienta está pensada para que tanto analistas como usuarios no técnicos puedan cargar casos (individuales o masivos) y obtener una predicción clara e interpretaciones visuales.

---

## 🚀 Funcionalidades principales

- **Predicción individual**:  
  Completa un formulario con los datos de una queja y obtén:
  - La probabilidad de que se convierta en disputa.  
  - Explicaciones SHAP de las variables más influyentes.  
  - Recomendaciones de acción.  
  - Visualizaciones dinámicas (velocímetro, histogramas, rankings).  

- **Scoring masivo**:  
  Sube un archivo CSV con múltiples quejas en formato original o de trabajo y recibe:  
  - Predicciones en lote.  
  - Descarga de resultados en CSV o Excel.  
  - Tablas resaltadas según nivel de riesgo.  
  - Distribución de riesgo global.  

- **Página de ayuda / About**:  
  Explicación de umbrales, definiciones de campos y ejemplos de interpretación para el cliente.  

---

## 🛠️ Tecnologías utilizadas

- **Python 3.10+**
- **Streamlit** (interfaz interactiva)
- **scikit-learn / XGBoost** (modelado ML)
- **pandas / numpy** (procesamiento de datos)
- **matplotlib / plotly** (gráficas)
- **PyYAML / joblib** (configuración y persistencia)

---

## 📂 Estructura del proyecto

```bash
.
├── app/
│   ├── app.py                 # Página principal
│   ├── pages/                 # Módulos multipágina
│   │   ├── 1_Home.py
│   │   ├── 2_Prediccion_individual.py
│   │   ├── 3_Scoring_masivo.py
│   │   └── 4_Ayuda_About.py
│   ├── loaders.py             # Carga de modelos y catálogos
│   ├── utils.py               # Funciones auxiliares
│   └── explain.py             # Explicaciones SHAP
├── data/                      # Datos de entrada (originales y transformados)
├── models/                    # Modelos entrenados y configuraciones
├── pyproject.toml             # Dependencias (manejado con uv)
└── README.md
```

---

## ▶️ Cómo ejecutar el proyecto localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/QuiliDev/Project_MachineLearning_Client.git
cd Project_MachineLearning_Client
```

### 2. Crear y activar el entorno virtual

```bash
uv venv
```

En Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS/Linux:


### 3. Instalar dependencias

```bash
uv sync
```

### 4. Ejecutar la aplicación

```bash
streamlit run app/home.py
```

La app quedará disponible en [http://localhost:8501](http://localhost:8501).

---

## 📊 Ejemplo de uso

1. Ingresa a **Predicción individual**, completa el formulario (o carga un ejemplo aleatorio) y obtén la probabilidad de disputa junto con la explicación visual.
2. O bien, entra en **Scoring masivo**, sube un CSV y recibe resultados en lote con descarga directa en CSV o Excel.
3. Explora la sección **Ayuda / About** para entender cómo interpretar los resultados y umbrales.

---

