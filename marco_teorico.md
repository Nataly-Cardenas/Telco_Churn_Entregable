# Marco Teórico

***Fundamentos de visualización de datos y aprendizaje automático para la predicción del abandono de clientes***

Este capítulo reúne los conceptos que sustentan el proyecto. Se organiza en dos bloques que se corresponden con la asignatura: primero, la **visualización de datos** como herramienta de exploración, análisis y comunicación; después, los fundamentos de **Machine Learning** empleados para construir, validar, comparar e interpretar los modelos de clasificación.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Visualización de datos
:class-card: step-card
Análisis exploratorio, codificación visual, medidas de asociación y comunicación de resultados.
:::

:::{grid-item-card} Machine Learning
:class-card: step-card
Clasificación binaria, desbalance, modelos de referencia, validación, métricas, comparación estadística e interpretabilidad.
:::

::::

## 1. Abandono de clientes (*customer churn*)

El abandono de clientes es la cancelación o no renovación de un servicio por parte de un cliente durante un periodo determinado. En telecomunicaciones genera pérdidas recurrentes de ingresos, y detectarlo a tiempo permite focalizar ofertas y acciones de retención en los clientes con mayor riesgo, lo que suele ser más económico que adquirir clientes nuevos.

El problema presenta dos dificultades. La primera es que la clase de interés (clientes que abandonan) es **minoritaria**, de modo que un modelo puede obtener una exactitud alta sin detectar a casi ningún cliente en riesgo. La segunda es que las variables disponibles describen el **estado del cliente en un corte**, lo que limita la lectura prospectiva de cualquier métrica y obliga a interpretar los resultados como asociaciones y no como relaciones causales. En este proyecto se utiliza el conjunto público Telco Customer Churn de IBM (IBM, 2018).

## 2. Visualización de datos

### 2.1 Análisis exploratorio visual

El análisis exploratorio de datos (EDA) propuesto por Tukey (Tukey, 1977) consiste en examinar los datos antes de formular modelos, con el fin de descubrir su estructura, detectar anomalías, valores faltantes o atípicos, y generar hipótesis. La visualización es su herramienta principal: un gráfico bien construido revela patrones que los resúmenes numéricos ocultan.

En este proyecto se emplean, entre otros, los siguientes tipos de gráficos:

| Gráfico | Propósito en el análisis |
|---|---|
| Histogramas y curvas de densidad | Forma de la distribución, asimetría y concentración de variables numéricas como `tenure`, `MonthlyCharges` y `TotalCharges`. |
| Diagramas de caja (*boxplots*) | Comparación de distribuciones entre clientes que abandonan y que permanecen, y detección de atípicos con el criterio IQR. |
| Barras de conteo y de tasas | Composición de clases y tasa de abandono por categoría (contrato, método de pago, servicio de internet, etc.). |
| Mapas de calor | Matrices de correlación y de asociación entre predictores. |
| Barras de métricas y de importancia | Comparación del desempeño de los modelos y de la contribución de cada variable (SHAP). |

### 2.2 Principios de diseño y codificación visual

Una visualización efectiva traduce datos en **marcas** (puntos, barras, líneas) y **canales** visuales (posición, longitud, color, tamaño). Munzner (Munzner, 2014) muestra que no todos los canales se perciben con igual precisión: la posición sobre una escala común y la longitud son los más precisos para comparar magnitudes, mientras que el color es más adecuado para distinguir categorías o resaltar una clase de interés. Por esa razón, las comparaciones de tasas de abandono y de métricas se representan con barras ordenadas, y el color se reserva para diferenciar a los clientes que abandonan.

Tufte (Tufte, 2001) propone maximizar la proporción de tinta dedicada a los datos (*data-ink ratio*), eliminar elementos decorativos que no aportan información y evitar distorsiones de escala. Estos principios orientan el diseño de los gráficos del libro: ejes rotulados, escalas honestas, etiquetas con los valores relevantes y títulos que expresan el hallazgo principal.

### 2.3 Medidas de asociación que acompañan a los gráficos

Las visualizaciones del EDA se complementan con medidas numéricas que cuantifican la fuerza de las relaciones observadas:

- **Correlación punto-biserial** ($r_{pb}$): asociación entre una variable numérica y una binaria, como `tenure` y `Churn`.
- **V de Cramér** (Cramér, 1946): asociación entre dos variables categóricas a partir del estadístico $\chi^2$ de una tabla de contingencia de $r \times k$ con $n$ observaciones:

$$
V = \sqrt{\frac{\chi^2 / n}{\min(r-1,\ k-1)}}, \qquad 0 \le V \le 1.
$$

- **Coeficiente de Spearman** ($\rho$): correlación basada en rangos, adecuada cuando las variables no siguen una distribución normal.
- **Corrección de Holm** (Holm, 1979): ajuste de los valores $p$ cuando se realizan múltiples contrastes simultáneos, para controlar la tasa de error de tipo I.

### 2.4 Comunicación de resultados y tableros interactivos

Además de explorar, la visualización sirve para **comunicar**. Los resultados de un modelo (comparaciones de F1, curvas y áreas ROC, importancia de variables) deben presentarse de forma que un público no técnico, como un área comercial o de retención, pueda entender qué segmentos están en riesgo y por qué. Los tableros interactivos permiten filtrar segmentos, comparar distribuciones y explorar los resultados sin escribir código; en este proyecto se acompañan con un tablero construido con Plotly Dash.

## 3. Clasificación binaria supervisada

En aprendizaje supervisado se dispone de un conjunto de entrenamiento $\{(x_i, y_i)\}_{i=1}^{n}$, donde $x_i \in \mathbb{R}^p$ es el vector de características de un cliente y $y_i \in \{0, 1\}$ indica si abandonó ($1$) o permaneció ($0$). El objetivo es aprender una función $f$ que estime la probabilidad $P(y = 1 \mid x)$ o asigne directamente una clase a clientes nuevos, de forma que el modelo **generalice** a datos no vistos y no solo memorice el conjunto de entrenamiento.

### 3.1 Preprocesamiento y fuga de información

Antes de entrenar es necesario imputar valores faltantes, codificar variables categóricas (por ejemplo, con *One-Hot Encoding*) y estandarizar las variables numéricas. Todas estas transformaciones aprenden parámetros de los datos (medias, desviaciones, categorías), por lo que deben ajustarse **únicamente con el conjunto de entrenamiento** de cada partición. Si se ajustan con todos los datos, información del conjunto de validación o prueba se filtra al modelo (**fuga de información** o *data leakage*) y las métricas resultan optimistas. Encadenar las transformaciones y el clasificador en un *pipeline* garantiza este orden.

## 4. Desbalance de clases

Cuando una clase es mucho menos frecuente que la otra, los clasificadores tienden a favorecer a la mayoritaria. Las estrategias más comunes son:

- **SMOTE** (*Synthetic Minority Over-sampling Technique*) (Chawla et al., 2002): genera ejemplos sintéticos de la clase minoritaria interpolando entre un ejemplo $x_i$ y uno de sus $k$ vecinos más cercanos $x_{nn}$ de la misma clase:

$$
x_{\text{nuevo}} = x_i + \lambda\,(x_{nn} - x_i), \qquad \lambda \sim U(0, 1).
$$

- **ADASYN** (He et al., 2008): variante adaptativa de SMOTE que genera más ejemplos sintéticos en las regiones donde la clase minoritaria es más difícil de aprender.
- **Ponderación de clases**: asigna un peso mayor a los errores sobre la clase minoritaria dentro de la función de pérdida, sin modificar los datos.

El remuestreo se aplica solo sobre el entrenamiento; los conjuntos de validación y prueba conservan su distribución original. En Python estas técnicas están disponibles en *imbalanced-learn* (Lemaître et al., 2017).

## 5. Modelos de clasificación

Se consideran siete familias de modelos clásicos, implementadas en scikit-learn (Pedregosa et al., 2011) y XGBoost.

**Regresión logística** (Hosmer et al., 2013). Modela $P(y = 1 \mid x) = \sigma(w^\top x + b)$ con $\sigma(z) = 1/(1 + e^{-z})$ y minimiza una pérdida logarítmica regularizada, donde $C$ es el inverso de la regularización y $\Omega$ es la norma $L_1$ o $L_2$:

$$
\min_{w,b}\ \Omega(w) + C \sum_{i=1}^{n} \log\!\left(1 + e^{-y_i (w^\top x_i + b)}\right), \qquad y_i \in \{-1, +1\}.
$$

Por su interpretabilidad se utiliza como modelo base (*baseline*).

**K vecinos más cercanos (KNN)** (Cover y Hart, 1967). Clasifica $x$ por el voto de sus $k$ vecinos más cercanos $\mathcal{N}_k(x)$ según la distancia de Minkowski de orden $p$, con pesos uniformes o inversamente proporcionales a la distancia:

$$
\hat{y}(x) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} w_i\, \mathbb{1}[y_i = c], \qquad d_p(x, x') = \Big(\sum_j |x_j - x'_j|^p\Big)^{1/p}.
$$

**Naive Bayes gaussiano** (Domingos y Pazzani, 1997). Supone independencia condicional de las variables dada la clase, con densidades gaussianas:

$$
\hat{y}(x) = \arg\max_{c}\ \pi_c \prod_{j=1}^{p} \mathcal{N}\!\left(x_j;\ \mu_{cj},\ \sigma^2_{cj} + \epsilon\right).
$$

**Árbol de decisión** (Breiman et al., 1984). Divide recursivamente el espacio de variables eligiendo en cada nodo la partición que más reduce una medida de impureza, como el índice de Gini o la entropía, donde $p_c$ es la proporción de la clase $c$ en el nodo:

$$
I_{\text{Gini}}(t) = 1 - \sum_c p_c^2, \qquad I_{\text{ent}}(t) = -\sum_c p_c \log p_c.
$$

**Random Forest** (Breiman, 2001). Combina $T$ árboles entrenados con muestras *bootstrap* $\mathcal{D}_t$ y un subconjunto aleatorio de variables en cada nodo, lo que reduce la varianza de un árbol individual:

$$
\hat{y}(x) = \text{mayoría}\{h_t(x; \mathcal{D}_t)\}_{t=1}^{T}.
$$

**XGBoost** (Chen y Guestrin, 2016). Construye un modelo aditivo $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$ en el que cada árbol nuevo corrige los errores de los anteriores, minimizando una pérdida regularizada donde $T_f$ es el número de hojas y $w$ los pesos de las hojas:

$$
\mathcal{L} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i) + \sum_t \Omega(f_t), \qquad \Omega(f) = \gamma T_f + \tfrac{1}{2}\lambda \lVert w \rVert^2.
$$

**Máquina de vectores de soporte (SVM) con kernel RBF** (Cortes y Vapnik, 1995). Busca el hiperplano de margen máximo en un espacio transformado mediante el truco del kernel, con $k(x_i, x) = \exp(-\gamma \lVert x_i - x \rVert^2)$:

$$
\min_{w,b,\xi}\ \tfrac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.a.}\quad y_i(w^\top \phi(x_i) + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0.
$$

## 6. Optimización de hiperparámetros

Los hiperparámetros (por ejemplo, $C$ en la regresión logística o la profundidad de un árbol) no se aprenden durante el entrenamiento y deben elegirse por validación. Se comparan cuatro estrategias de búsqueda:

- **Grid Search**: evalúa exhaustivamente todas las combinaciones de una rejilla predefinida.
- **Random Search** (Bergstra y Bengio, 2012): muestrea combinaciones al azar; con el mismo presupuesto suele explorar mejor los hiperparámetros relevantes que la rejilla.
- **TPE** (*Tree-structured Parzen Estimator*) (Bergstra et al., 2011): optimización bayesiana que modela qué regiones del espacio producen buenos resultados y propone candidatos en ellas; está implementado en Optuna (Akiba et al., 2019).
- **Algoritmo genético**: evoluciona una población de configuraciones mediante selección, cruce y mutación; se implementa con DEAP (Fortin et al., 2012).

## 7. Validación cruzada anidada

En la validación cruzada de $K$ particiones (*K-fold*) los datos se dividen en $K$ bloques; cada bloque se usa una vez como validación y los restantes como entrenamiento, y el desempeño se resume con la media y la desviación estándar de las $K$ estimaciones. La estratificación conserva en cada bloque la proporción original de clases.

Si los mismos datos se usan para elegir hiperparámetros y para reportar el desempeño, la estimación resulta optimista (Varma y Simon, 2006). La **validación cruzada anidada** separa ambas tareas: un bucle **interno** selecciona los hiperparámetros y un bucle **externo** estima el desempeño del procedimiento completo sobre datos que no intervinieron en la selección.

```{admonition} Esquema de la validación cruzada anidada
:class: tip
1. Dividir el conjunto de desarrollo en $K_{ext}$ particiones externas.
2. Para cada partición externa, reservarla como evaluación.
3. Con el resto de los datos, buscar hiperparámetros mediante $K_{int}$ particiones internas.
4. Reajustar el *pipeline* completo con los mejores hiperparámetros y evaluar en la partición externa reservada.
5. Reportar la media $\pm$ desviación estándar de las métricas externas.
```

## 8. Métricas de evaluación

A partir de la matriz de confusión (verdaderos positivos $TP$, falsos positivos $FP$, falsos negativos $FN$ y verdaderos negativos $TN$), y tomando como clase positiva el abandono:

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2\,TP}{2\,TP + FP + FN}.
$$

- **Precision**: de los clientes señalados en riesgo, qué proporción realmente abandona (controla las falsas alertas).
- **Recall** (sensibilidad): de los clientes que abandonan, qué proporción detecta el modelo.
- **F1-score**: media armónica de Precision y Recall; resume el equilibrio entre detectar el abandono y evitar falsas alertas, y es más informativa que la exactitud (*accuracy*) cuando las clases están desbalanceadas.
- **AUC-ROC**: área bajo la curva que relaciona la tasa de verdaderos positivos con la de falsos positivos para todos los umbrales; equivale a la probabilidad de que el modelo asigne mayor puntaje a un cliente que abandona que a uno que permanece.

Para evaluar la **calidad de las probabilidades** se utilizan:

$$
\text{Brier} = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i)^2, \qquad \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\big|\,\text{acc}(B_m) - \text{conf}(B_m)\,\big|,
$$

donde el *Expected Calibration Error* (ECE) agrupa las predicciones en $M$ intervalos $B_m$ y compara la frecuencia observada de abandono con la probabilidad media predicha. La **calibración sigmoide de Platt** (Platt, 1999) ajusta una función logística sobre los puntajes del modelo para que las probabilidades sean más fieles.

## 9. Comparación estadística de modelos

Las diferencias pequeñas entre modelos pueden deberse a la variabilidad de las particiones. Para compararlos de forma rigurosa (Demšar, 2006):

- **Prueba de Friedman** (Friedman, 1937): contraste no paramétrico basado en rangos que evalúa si existen diferencias entre varias configuraciones evaluadas sobre los mismos bloques.
- **Prueba de Nemenyi y diferencia crítica**: comparación *post hoc* por pares; dos configuraciones difieren si la distancia entre sus rangos medios supera la diferencia crítica.
- **Prueba de DeLong** (DeLong et al., 1988): compara las AUC de dos modelos evaluados sobre los mismos clientes, teniendo en cuenta la correlación entre sus puntajes.
- **Intervalos *bootstrap***: remuestreos con reemplazo del conjunto de prueba para estimar intervalos de confianza de las métricas.

## 10. Interpretabilidad de los modelos

Los modelos de conjunto, como Random Forest o XGBoost, son difíciles de interpretar directamente. Las técnicas de explicación permiten entender qué variables impulsan las predicciones y se comunican principalmente mediante **visualizaciones**:

- **SHAP** (*SHapley Additive exPlanations*) (Lundberg y Lee, 2017): basado en los valores de Shapley de la teoría de juegos cooperativos, descompone cada predicción en contribuciones aditivas de las variables, $f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j$. El promedio de $|\phi_j|$ sobre los clientes da una medida de importancia global, que se representa con gráficos de barras y de resumen (*beeswarm*).
- **LIME** (*Local Interpretable Model-agnostic Explanations*) (Ribeiro et al., 2016): aproxima el modelo alrededor de una predicción concreta con un modelo simple e interpretable, lo que permite explicar casos individuales.

Estas explicaciones describen el comportamiento del modelo y no deben interpretarse como relaciones causales sobre el abandono.

## Referencias

- Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. En *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 2623–2631).
- Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for hyper-parameter optimization. En *Advances in Neural Information Processing Systems*, 24.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A. y Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
- Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Cover, T. y Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, 44(3), 837–845.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Domingos, P. y Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29, 103–130.
- Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M. y Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research*, 13, 2171–2175.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675–701.
- He, H., Bai, Y., Garcia, E. A. y Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. En *2008 IEEE International Joint Conference on Neural Networks* (pp. 1322–1328).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hosmer, D. W., Lemeshow, S. y Sturdivant, R. X. (2013). *Applied Logistic Regression* (3.ª ed.). Wiley.
- IBM. (2018). *Telco Customer Churn* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Lemaître, G., Nogueira, F. y Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, 18(17), 1–5.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural Information Processing Systems*, 30 (pp. 4765–4774).
- Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.
- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144).
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2.ª ed.). Graphics Press.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Varma, S. y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
