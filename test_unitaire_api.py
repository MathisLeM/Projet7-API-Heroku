import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Récupérez le répertoire actuel du fichier api.py
current_directory = os.path.dirname(os.path.abspath(__file__))

# Charger le modèle en dehors de la clause if __name__ == "__main__":
model_path = os.path.join(current_directory, "saved_model_v3.pkl")
model = joblib.load(model_path)

# Charger les données de prédiction
csv_path = os.path.join(current_directory, "df300.csv")
df = pd.read_csv(csv_path)

# Charger les données personnelles
info_path = os.path.join(current_directory, "personal_info.csv")
personal_info_df = pd.read_csv(info_path)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data['SK_ID_CURR']

    # Rechercher l'échantillon
    sample = df[df['SK_ID_CURR'] == sk_id_curr]

    # Vérifiez si l'échantillon n'est pas vide
    if sample.empty:
        return jsonify({"error": f"SK_ID_CURR {sk_id_curr} not found in the dataset"}), 400

    # Supprimer la colonne ID et TARGET pour la prédiction
    sample = sample.drop(columns=['SK_ID_CURR', 'TARGET'])

    # Prédire
    prediction = model.predict_proba(sample)
    proba = prediction[0][1]  # Probabilité de la seconde classe

    # Retourner la probabilité
    return jsonify({
        'probability': proba * 100,
        'feature_names': sample.columns.tolist(),
        'feature_values': sample.values[0].tolist()
    })

@app.route("/info", methods=['POST'])
def info():
    data = request.json
    sk_id_curr = data['SK_ID_CURR']

    # Rechercher les informations personnelles
    info_sample = personal_info_df[personal_info_df['SK_ID_CURR'] == sk_id_curr]

    # Vérifiez si l'échantillon n'est pas vide
    if info_sample.empty:
        return jsonify({"error": f"SK_ID_CURR {sk_id_curr} not found in the dataset"}), 400

    # Retourner les informations personnelles
    return jsonify(info_sample.to_dict(orient='records')[0])

@app.route("/distribution", methods=['POST'])
def distribution():
    data = request.json
    feature = data['feature']
    sk_id_curr = data['SK_ID_CURR']

    if feature not in df.columns:
        return jsonify({"error": f"Feature {feature} not found in the dataset"}), 400

    # Rechercher la valeur de la caractéristique pour le client donné
    client_value = df[df['SK_ID_CURR'] == sk_id_curr][feature].values[0]

    # Retourner les données de distribution
    distribution_data = df[feature].tolist()
    
    return jsonify({
        'feature': feature,
        'client_value': client_value,
        'distribution': distribution_data
    })

if __name__ == "__main__":
    port = os.environ.get("PORT", 8501)
    app.run(debug=False, host="0.0.0.0", port=int(port))
