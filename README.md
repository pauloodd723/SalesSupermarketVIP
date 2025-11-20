# Minería de Datos – Proyecto Final
# 📌 Descripción del Proyecto

Este proyecto analiza datos de ventas y productos para identificar clientes VIP y segmentar productos, utilizando técnicas de Data Mining y Machine Learning.
Se aplicó la metodología CRISP-DM para guiar todo el flujo de trabajo, desde la comprensión del negocio hasta el despliegue de modelos.

## 📊 Objetivos

Analizar el comportamiento de ventas y descuentos.

Identificar clientes con potencial de convertirse en VIP.

Segmentar productos mediante clustering.

Desarrollar un API que permita predecir clientes VIP y asignar clusters automáticamente.

## 🔍 Metodología CRISP-DM

1. Comprensión del negocio
Entender las necesidades de la empresa: identificar clientes VIP y optimizar estrategias de ventas y descuentos.

2. Comprensión de los datos
Se trabajaron cuatro datasets (Annex1 a Annex4) con información de productos, ventas y precios mayoristas.

3. Preparación de los datos

    Limpieza de nulos y duplicados.

    Corrección de cantidades negativas y normalización de columnas categóricas.

    Cálculo de nuevas métricas: Revenue, Margin y columna VIP según cuartil 75 de Revenue.

4. Análisis exploratorio de datos (EDA)

    Identificación de top productos y categorías.

    Visualización de precios, pérdidas, descuentos y ventas por fecha.

5. Modelado

    Modelos principales para predecir clientes VIP: RandomForest y XGBoost (se selecciona el mejor según AUC-ROC).

    Modelos secundarios obligatorios: SVM (secundario para mostrar cumplimiento del requerimiento).

    Clustering de productos: K-Means (para segmentación de productos).

6. Evaluación

    Métricas: Accuracy, F1-score, AUC-ROC para clasificación.

    Silhouette Score para K-Means.

7. Despliegue

    Guardado de modelos y preprocesadores (scaler y LabelEncoders) con joblib.

    API construida en Flask para predicción en tiempo real.

MineriaProyectoFinal/
│
├─ api.py                     # API Flask para predicciones
├─ MineriaProyectoFinal_clean.csv  # Dataset limpio y unificado
├─ clf_vip_model.pkl           # Modelo final VIP (RandomForest/XGBoost)
├─ svm_vip_model.pkl           # Modelo secundario SVM
├─ kmeans_model.pkl            # Modelo K-Means
├─ scaler.pkl                  # Escalador de features
├─ le_category.pkl             # LabelEncoder categorías
├─ le_discount.pkl             # LabelEncoder descuentos
├─ le_sale.pkl                 # LabelEncoder tipo de venta
├─ README.md                   # Este archivo
└─ notebook_colab.ipynb        # Notebook Colab con todo el flujo

## ⚙️ Requisitos e Instalación

Python 3.12 recomendado, con librerías:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask joblib

## 🚀 Uso de la API

Ejecutar api.py: python api.py

Enviar peticiones POST con los datos del cliente/producto para recibir predicciones:

    Probabilidad de ser VIP (modelo principal).

    Probabilidad secundaria con SVM.

    Cluster asignado por K-Means.


## 📈 Explicación de Modelos

    RandomForest/XGBoost: Modelado principal para predicción VIP, se selecciona el mejor según AUC

    SVM: Modelo secundario, obligatorio según requerimientos, sirve como referencia de clasificación

    K-Means: Segmentación de productos según características y ventas, usado para clustering 

## 📌 Notas importantes

    El dataset final está limpio y listo para análisis o despliegue.

    Los modelos se entrenan con un subsample de 13k registros para acelerar el entrenamiento sin perder representatividad.