# Modelo de Clasificación Binaria para la Predicción de Abandono de Clientes a partir de Datos de Suscripción: Dataset Telco Customer Churn

***Análisis de Retención de Clientes y Riesgo de Abandono (Churn)***

**Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita**  
Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia  
`sriveral@uninorte.edu.co` · `nizaquita@uninorte.edu.co`

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Asignatura
:class-card: meta-card
Visualización de Datos y Machine Learning
:::

:::{grid-item-card} Conjunto de datos
:class-card: meta-card
Telco Customer Churn (IBM) · 7,043 clientes · 26.5 % de abandono
:::

:::{grid-item-card} Metodología
:class-card: meta-card
EDA · Preprocesamiento · Modelado Avanzado · Interpretación
:::

::::

## Introducción
La pérdida de clientes (*churn*) es uno de los principales riesgos financieros de las empresas de telecomunicaciones: cada cancelación supone la pérdida de ingresos recurrentes, y retener a un cliente existente suele ser considerablemente más económico que adquirir uno nuevo. Anticipar qué clientes tienen mayor probabilidad de abandonar el servicio permite dirigir ofertas, mejoras de servicio y acciones de retención de forma selectiva, en lugar de aplicarlas a toda la base de clientes.

El problema presenta dos dificultades. Los clientes que abandonan son una minoría (alrededor de uno de cada cuatro), por lo que un modelo puede acertar en la mayoría de los casos sin detectar a casi ninguno de ellos; y las variables disponibles describen el estado del cliente en un momento dado, lo que obliga a interpretar los hallazgos como asociaciones y no como causas del abandono.

Este proyecto aborda la predicción del abandono con el conjunto público **Telco Customer Churn** de IBM, que combina los servicios contratados, las características de la cuenta y los datos demográficos de cada cliente. A partir de un análisis exploratorio que identifica los perfiles con mayor tasa de abandono, se construye un modelo base de regresión logística y se compara con alternativas más complejas bajo un mismo protocolo de validación, priorizando la detección de los clientes en riesgo sin generar un volumen excesivo de falsas alertas. Los resultados se ponen a disposición en este Jupyter Book, en un paper y en un tablero interactivo.

## Contexto del problema
El *customer churn* es la cancelación o abandono de un servicio. Según los [datos de Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), cada fila del conjunto representa un cliente y la columna `Churn` indica si abandonó el servicio durante el último mes. El conjunto de datos incluye información sobre:

- **Abandono:** clientes que se dieron de baja en el último mes (columna `Churn`).
- **Servicios contratados:** teléfono, líneas múltiples, internet, seguridad en línea, copia de seguridad en línea, protección de dispositivos, soporte técnico y transmisión de TV y películas.
- **Información de la cuenta:** antigüedad como cliente, tipo de contrato, método de pago, facturación electrónica, cargos mensuales y cargos totales.
- **Información demográfica:** género, rango de edad y si tienen pareja o personas a su cargo.

## Objetivo general
Identificar a los clientes de una empresa de telecomunicaciones con mayor riesgo de abandonar el servicio y los factores asociados a ese abandono, construyendo y comparando de forma reproducible modelos de clasificación que detecten la mayor cantidad posible de clientes en riesgo sin generar un exceso de falsas alertas.

### Objetivos específicos
1. **Describir el problema y los datos:** caracterizar el conjunto Telco Customer Churn, su calidad (valores faltantes, atípicos y tipos de variables) y el nivel de desbalance de la variable objetivo.
2. **Reservar un conjunto de prueba:** separar el 20 % de los clientes mediante una partición estratificada antes del análisis, de modo que ninguna decisión se tome con los datos destinados a la evaluación final.
3. **Identificar los perfiles de abandono:** analizar la distribución de cada variable y su asociación con el abandono, para determinar qué segmentos de clientes (tipo de contrato, servicios, antigüedad, método de pago) concentran las mayores tasas de cancelación.
4. **Preparar los datos sin fuga de información:** definir el tratamiento de faltantes, la codificación de variables categóricas, el escalado y las variables derivadas, ajustándolos únicamente con los datos de entrenamiento de cada partición.
5. **Establecer un modelo de referencia:** entrenar una regresión logística como modelo base y cuantificar sus aciertos y errores, en especial los clientes que abandonan y no son detectados.
6. **Comparar alternativas de forma controlada:** evaluar 112 configuraciones (7 modelos × 4 técnicas de balanceo × 4 métodos de optimización de hiperparámetros) con validación cruzada anidada y el F1 de la clase de abandono como métrica principal, contrastando estadísticamente sus diferencias, la calibración de probabilidades y el costo computacional.
7. **Explicar las predicciones:** identificar con SHAP y LIME las variables que más influyen en el riesgo de abandono, sin atribuirles un efecto causal.
8. **Poner los resultados a disposición:** presentar el análisis en un Jupyter Book, un paper y un tablero interactivo que permitan consultar los hallazgos sin necesidad de ejecutar el código.

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


## Estructura del libro

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 1. Marco Teórico
:class-card: step-card
Conceptos que sustentan el análisis: abandono de clientes, clasificación, desbalance, validación, métricas e interpretabilidad.
:::

:::{grid-item-card} 2. Análisis Exploratorio (EDA)
:class-card: step-card
Calidad de los datos, distribuciones, tasas de abandono por segmento y asociación entre variables.
:::

:::{grid-item-card} 3. Preprocesamiento
:class-card: step-card
Imputación, codificación de variables categóricas, escalado y modelo base, ajustados exclusivamente con los datos de entrenamiento.
:::

:::{grid-item-card} 4. Modelado Avanzado
:class-card: step-card
Comparación de 112 configuraciones con validación cruzada anidada, evaluación en test e interpretación de los modelos.
:::

:::{grid-item-card} 5. Paper
:class-card: step-card
Versión web y PDF del artículo con la metodología y los resultados del proyecto.
:::

:::{grid-item-card} 6. Guía del Repositorio
:class-card: step-card
Estructura del repositorio, requisitos, orden de ejecución de los notebooks y construcción del libro.
:::

::::

:::{seealso} Enlaces de interés
**Conjunto de datos**
- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Repositorios del proyecto**
- [GitHub — Jupyter Book y notebooks](https://github.com/Nataly-Cardenas/Telco_Churn_Entregable)
- [GitHub — Dashboard interactivo](https://github.com/Nataly-Cardenas/Dash_Telco_Churn)
:::
