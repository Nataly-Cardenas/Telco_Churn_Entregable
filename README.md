# Telco Customer Churn

Proyecto de análisis y clasificación del abandono de clientes de telecomunicaciones.

**[Leer el Jupyter Book](https://nataly-cardenas.github.io/Telco_Churn_Entregable/)**

## Orden del libro

1. [EDA](EDA.ipynb)
2. [PREPROCESAMIENTO](PREPROCESAMIENTO.ipynb)
3. [MODELADO_AVANZADO](MODELADO_AVANZADO.ipynb)

El índice se define explícitamente en `myst.yml`. EDA es la página inicial.
Los notebooks se publican con sus salidas guardadas: la construcción del libro no los ejecuta ni modifica.

## Construcción local

Con Python 3.12 y Node.js 24 instalados:

```sh
python -m pip install -r requirements-book.txt
jupyter book build --html
```

El sitio se genera en `_build/html`. Las dependencias del análisis original están en `requirements.txt`; no son necesarias para construir el libro con las salidas existentes.

## Publicación

GitHub Actions construye y publica el libro en GitHub Pages al actualizar `main`.
El flujo también verifica que los tres notebooks conserven sus huellas SHA-256 originales.

Se incluyen los notebooks, los datos de la carpeta principal y la configuración del libro. La carpeta local `resultados_modelado/` (aproximadamente 6 GB de artefactos de experimentos) y `revision_tmp/` no se suben. Las salidas guardadas dentro de los notebooks sí se muestran en el libro. Para volver a ejecutar el modelado que consulta esos artefactos se requiere conservar la carpeta local de resultados.
