# Marco Teórico

Este capítulo resume los conceptos que sustentan el análisis del abandono de clientes y las decisiones metodológicas del proyecto.

## 1. Abandono de clientes (*churn*)

El abandono de clientes es la cancelación o no renovación de un servicio durante un periodo determinado. En telecomunicaciones genera pérdidas recurrentes de ingresos, por lo que detectarlo a tiempo permite focalizar las acciones de retención en los clientes de mayor riesgo. Su predicción se plantea como un problema de **clasificación binaria supervisada**: a partir de las características $x_i$ de cada cliente se estima la probabilidad $P(y_i = 1 \mid x_i)$ de que abandone ($y_i = 1$) o permanezca ($y_i = 0$).

## 2. Análisis exploratorio de datos

El análisis exploratorio (EDA) examina los datos antes de modelar para conocer su estructura, detectar valores faltantes o atípicos y generar hipótesis (Tukey, 1977). Se apoya en gráficos —histogramas, diagramas de caja y barras de tasas de abandono por categoría— y en medidas de asociación con la variable objetivo: la **correlación punto-biserial** para variables numéricas, la **V de Cramér** para variables categóricas (Cramér, 1946) y el coeficiente de **Spearman** entre predictores no normales. Cuando se realizan muchos contrastes a la vez, la **corrección de Holm** ajusta los valores $p$ para controlar los falsos hallazgos (Holm, 1979).

## 3. Desbalance de clases y fuga de información

Cuando la clase de interés es minoritaria, los modelos tienden a favorecer a la mayoritaria. **SMOTE** genera ejemplos sintéticos de la clase minoritaria interpolando entre un cliente y uno de sus vecinos más cercanos de la misma clase (Chawla et al., 2002); **ADASYN** concentra esos ejemplos en las regiones más difíciles de aprender (He et al., 2008), y la **ponderación de clases** penaliza más los errores sobre la clase minoritaria.

Toda transformación que aprende de los datos (imputación, codificación, escalado y balanceo) debe ajustarse **solo con el conjunto de entrenamiento** de cada partición. Si se ajusta con todos los datos, información de validación o prueba se filtra al modelo (*data leakage*) y las métricas resultan optimistas.

## 4. Modelos de clasificación

| Modelo | Idea principal |
|---|---|
| Regresión logística (Hosmer et al., 2013) | Modelo lineal que estima $P(y = 1 \mid x) = 1/(1 + e^{-(w^\top x + b)})$; interpretable, se usa como modelo base. |
| K vecinos más cercanos (Cover y Hart, 1967) | Clasifica a un cliente según la clase mayoritaria de sus $k$ vecinos más cercanos. |
| Naive Bayes gaussiano (Domingos y Pazzani, 1997) | Aplica el teorema de Bayes suponiendo independencia entre variables dada la clase. |
| Árbol de decisión (Breiman et al., 1984) | Divide los datos con reglas sucesivas que reducen la impureza (Gini o entropía). |
| Random Forest (Breiman, 2001) | Combina muchos árboles entrenados con muestras y variables aleatorias para reducir la varianza. |
| XGBoost (Chen y Guestrin, 2016) | Agrega árboles de forma secuencial, cada uno corrigiendo los errores de los anteriores. |
| SVM con kernel RBF (Cortes y Vapnik, 1995) | Busca la frontera de máximo margen en un espacio transformado por el kernel. |

## 5. Validación y optimización de hiperparámetros

Los hiperparámetros se eligen con métodos de búsqueda como **Grid Search**, **Random Search** (Bergstra y Bengio, 2012), **TPE**, una optimización bayesiana (Bergstra et al., 2011), o **algoritmos genéticos**. Si los mismos datos se usan para elegir hiperparámetros y para reportar el desempeño, la estimación es optimista (Varma y Simon, 2006). La **validación cruzada anidada** lo evita: un bucle interno selecciona los hiperparámetros y un bucle externo mide el desempeño sobre datos que no intervinieron en esa selección.

## 6. Métricas de evaluación

Tomando el abandono como clase positiva:

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2\,TP}{2\,TP + FP + FN}.
$$

El **Recall** indica qué proporción de los clientes que abandonan detecta el modelo y la **Precision** qué proporción de las alertas son correctas; el **F1** resume ambas y es más informativo que la exactitud con clases desbalanceadas. El **AUC-ROC** mide la capacidad de ordenar a los clientes por riesgo, y el **puntaje de Brier** y el **ECE** evalúan qué tan fieles son las probabilidades predichas. Para comparar modelos se emplean las pruebas de **Friedman** y **Nemenyi** (Demšar, 2006) y la prueba de **DeLong** para diferencias de AUC (DeLong et al., 1988).

## 7. Interpretabilidad

**SHAP** descompone cada predicción en contribuciones aditivas de las variables a partir de los valores de Shapley (Lundberg y Lee, 2017), lo que permite obtener la importancia global de cada variable. **LIME** explica predicciones individuales aproximando el modelo, alrededor de cada caso, con un modelo simple (Ribeiro et al., 2016). Ambas describen el comportamiento del modelo y no relaciones causales.

## Referencias

- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for hyper-parameter optimization. En *Advances in Neural Information Processing Systems*, 24.
- Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
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
- He, H., Bai, Y., Garcia, E. A. y Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. En *2008 IEEE International Joint Conference on Neural Networks* (pp. 1322–1328).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hosmer, D. W., Lemeshow, S. y Sturdivant, R. X. (2013). *Applied Logistic Regression* (3.ª ed.). Wiley.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural Information Processing Systems*, 30 (pp. 4765–4774).
- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144).
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Varma, S. y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
