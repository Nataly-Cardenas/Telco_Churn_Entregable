# Modelo de Clasificación Binaria para la Predicción de Abandono de Clientes a partir de Datos de Suscripción: Dataset Telco Customer Churn

***Análisis de Retención de Clientes y Riesgo de Abandono (Churn) con Machine Learning***

**Laura Rivera · Natalý Cárdenas** Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia
`sriveral@uninorte.edu.co` · `nizaquita@uninorte.edu.co`

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Asignatura
:class-card: meta-card
Visualización de Datos y Machine Learning
:::

:::{grid-item-card} Autoras
:class-card: meta-card
Laura Rivera & Natalý Cárdenas

*Departamento de Matemáticas, Física y Ciencia de Datos*
*Universidad del Norte*
:::

:::{grid-item-card} Metodología
:class-card: meta-card
EDA · Feature Engineering · Modelos de Clasificación
:::

::::

## Introducción del proyecto

## Contexto del problema
El customer churn es la cancelación o abandono de un servicio. Comprender las características relacionadas con el abandono permite orientar futuras estrategias de retención. En Machine Learning se plantea como una clasificación binaria: identificar clientes que abandonan y clientes que permanecen.

Según los [datos de Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), cada fila representa un cliente y Churn indica abandono durante el último mes. El conjunto de datos incluye información sobre:

- Clientes que se dieron de baja en el último mes: la columna se llama "Abandono".
Servicios que cada cliente ha contratado: teléfono, líneas múltiples, internet, seguridad en línea, copia de seguridad en línea, protección de dispositivos, soporte técnico y transmisión de TV y películas.

- Información de la cuenta del cliente: cuánto tiempo lleva siendo cliente, contrato, método de pago, facturación electrónica, cargos mensuales y cargos totales.

- Información demográfica sobre los clientes: género, rango de edad y si tienen pareja o personas a su cargo.
Inspiración

## Objetivo general
Desarrollar y evaluar modelos de Machine Learning para predecir la cancelación de clientes de telecomunicaciones con el dataset Telco Customer Churn, partiendo de una regresión logística y comparando alternativas mediante una metodología reproducible que priorice la generalización y evite la fuga de información.

### Objetivos específicos
- Comprender la estructura, calidad y distribución de los datos, así como las asociaciones entre las características de los clientes y el churn, mediante análisis exploratorio del conjunto de entrenamiento.

- Diseñar un preprocesamiento adecuado para valores faltantes, variables categóricas y escalas numéricas, ajustando sus parámetros exclusivamente con los datos de entrenamiento de cada partición de validación.

- Construir una regresión logística como modelo inicial de referencia y evaluar sus aciertos, errores y capacidad para identificar clientes que abandonan.

- Comparar modelos y estrategias metodológicas adicionales con el baseline bajo el mismo protocolo de validación, considerando el desbalance y controlando el sobreajuste.

- Evaluar la mejora respecto al baseline con métricas apropiadas para clasificación y probabilidades, reservando el conjunto test para la evaluación final del modelo seleccionado.

- Interpretar los resultados y las limitaciones del modelo para formular conclusiones sobre su utilidad potencial en retención de clientes, sin atribuir causalidad a las asociaciones observadas.

## Descripción de las variables del proyecto
Diccionario resumido de los metadatos de Kaggle:

| Variable | Tipo conceptual | Significado | Papel |
|---|---|---|---|
| customerID | Identificador | Código del cliente | Integridad; no predictor |
| gender | Categórica | Género registrado (Masculino/Femenino) | Candidato |
| SeniorCitizen | Binaria | Adulto mayor (1/0) | Candidato |
| Partner | Binaria | Tiene pareja (1/0) | Candidato |
| Dependents | Binaria | Tiene dependientes | Candidato |
| tenure | Numérica discreta | Antigüedad en meses | Candidato |
| PhoneService | Categórica | Servicio telefónico | Candidato |
| MultipleLines | Categórica | Líneas múltiples | Candidato |
| InternetService | Categórica | Servicio de internet | Candidato |
| OnlineSecurity | Categórica | Seguridad en línea | Candidato |
| OnlineBackup | Categórica | Respaldo en línea | Candidato |
| DeviceProtection | Categórica | Protección de dispositivos | Candidato |
| TechSupport | Categórica | Soporte técnico | Candidato |
| StreamingTV | Categórica | Televisión por streaming | Candidato |
| StreamingMovies | Categórica | Películas por streaming | Candidato |
| Contract | Categórica | Modalidad contractual | Candidato |
| PaperlessBilling | Categórica | Facturación sin papel | Candidato |
| PaymentMethod | Categórica | Método de pago | Candidato |
| MonthlyCharges | Numérica | Cargo mensual | Candidato |
| TotalCharges | Numérica; puede venir como texto | Cargos totales | Candidato |
| Churn | Binaria | Abandono durante el último mes | Objetivo Yes/No |


El identificador no representa una característica explicativa. Los demás campos son candidatos.


## Estructura del Análisis

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 1. Análisis Exploratorio (EDA)
:class-card: step-card
Evaluación de distribuciones, tratamiento de valores faltantes, detección de sesgo y análisis de correlación entre variables candidatas.
:::

:::{grid-item-card} 2. Modelado y Predicción
:class-card: step-card
Entrenamiento de algoritmos supervisados, optimización de hiperparámetros y evaluación de métricas orientadas a la detección de riesgo (Recall / ROC-AUC).
:::

::::



:::{seealso} Enlaces de interés
**Conjunto de datos**
- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Repositorio del proyecto**
- [GitHub](https://github.com/Nataly-Cardenas/Telco_Churn_Entregable)
