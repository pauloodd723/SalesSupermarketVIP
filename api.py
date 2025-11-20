from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

# --- Cargar modelos y preprocessors ---
vip_model = joblib.load("clf_vip_model.pkl")  # XGBoost o RandomForest final
svm_model = joblib.load("svm_vip_model.pkl")  # SVM secundario
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
le_category = joblib.load("le_category.pkl")
le_discount = joblib.load("le_discount.pkl")
le_sale = joblib.load("le_sale.pkl")

# --- Inicializar Flask ---
app = Flask(__name__, template_folder="templates")

# --- Mapas español → valores del encoder ---
map_discount = {"Sí": "Yes", "No": "No"}
map_sale = {"Venta": "sale", "Devolución": "return"}

# --- Función de preprocesamiento ---
def preprocess_input(data):
    try:
        category_val = le_category.transform([data["category"]])[0]
        discount_val = le_discount.transform([map_discount[data["discount"]]])[0]
        sale_val = le_sale.transform([map_sale[data["sale_flag"]]])[0]

        # DataFrame con las features
        df = pd.DataFrame([{
            "Quantity Sold (kilo)": float(data["quantity"]),
            "Loss Rate (%)": float(data["loss_rate"]),
            "Category_Code_Num": category_val,
            "Discount_Flag": discount_val,
            "Sale_Flag": sale_val
        }])

        # Escalar
        df_scaled = scaler.transform(df)
        return df_scaled
    except Exception as e:
        raise ValueError(f"Error en preprocesamiento: {e}")

# --- Ruta principal / formulario ---
@app.route("/premium_classifier", methods=["GET", "POST"])
def premium_classifier():
    if request.method == "GET":
        categories = le_category.classes_.tolist()
        return render_template("form.html", categories=categories)
    
    # --- POST ---
    try:
        data = request.form
        X_input = preprocess_input(data)

        # --- Predicción VIP principal (XGBoost / RandomForest) ---
        vip_pred_prob = vip_model.predict_proba(X_input)[0]
        prob_vip = vip_pred_prob[1]  # probabilidad de ser VIP

        # Solo dos categorías
        if prob_vip > 0.5:
            vip_msg = "Podría convertirse en cliente VIP"
        else:
            vip_msg = "No es probable convertirse en cliente VIP"

        vip_prob_str = [round(p*100,2) for p in vip_pred_prob]

        # --- Predicción SVM secundaria ---
        svm_pred_prob = svm_model.predict_proba(X_input)[0]
        svm_prob_str = [round(p*100,2) for p in svm_pred_prob]

        # --- K-Means segmentación ---
        cluster = kmeans_model.predict(X_input)[0]

        return render_template(
            "result.html",
            vip_msg=vip_msg,
            vip_prob=vip_prob_str,
            svm_prob=svm_prob_str,
            cluster=cluster
        )
    except Exception as e:
        return f"Error en predicción: {e}"

# --- Ejecutar servidor ---
if __name__ == "__main__":
    app.run(debug=True)
