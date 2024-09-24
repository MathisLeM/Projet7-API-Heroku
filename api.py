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
        """
    Prédit la probabilité d'un événement pour un échantillon basé sur un identifiant fourni.

    Cette fonction reçoit une requête POST avec un identifiant unique (`SK_ID_CURR`) au format JSON.
    Elle recherche l'échantillon correspondant dans un DataFrame, puis utilise un modèle de machine learning
    pour prédire la probabilité d'une seconde classe (par exemple, un événement spécifique). 

    Si l'échantillon correspondant n'est pas trouvé dans le DataFrame, la fonction renvoie un message d'erreur.

    Returns:
        flask.Response: Un objet JSON contenant :
            - `probability` (float) : La probabilité prédite pour la seconde classe.
            - `feature_names` (list) : La liste des noms de caractéristiques utilisées pour la prédiction.
            - `feature_values` (list) : La liste des valeurs des caractéristiques de l'échantillon.
        En cas d'erreur (si `SK_ID_CURR` n'est pas trouvé), un message d'erreur est renvoyé avec un code de statut 400.

    Raises:
        KeyError: Si `SK_ID_CURR` n'est pas présent dans les données de la requête JSON.
    """
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
        """
    Récupère les informations personnelles associées à un identifiant unique.

    Cette fonction reçoit une requête POST contenant un identifiant unique (`SK_ID_CURR`) au format JSON.
    Elle recherche les informations personnelles correspondantes dans un DataFrame, puis renvoie ces informations
    sous forme d'un dictionnaire JSON.

    Si l'échantillon correspondant n'est pas trouvé dans le DataFrame, la fonction renvoie un message d'erreur.

    Returns:
        flask.Response: Un objet JSON contenant les informations personnelles de l'échantillon, ou un message 
        d'erreur si `SK_ID_CURR` n'est pas trouvé dans le jeu de données.
        
        Exemple de réponse JSON :
        {
            'SK_ID_CURR': 100001,
            'NAME': 'John Doe',
            'AGE': 45,
            ...
        }

    Raises:
        KeyError: Si `SK_ID_CURR` n'est pas présent dans les données de la requête JSON.
    """
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
        """
    Récupère la distribution d'une caractéristique spécifique et la valeur associée à un client donné.

    Cette fonction reçoit une requête POST contenant un identifiant unique (`SK_ID_CURR`) et une caractéristique (`feature`) au format JSON.
    Elle vérifie si la caractéristique demandée existe dans le DataFrame, puis renvoie la distribution de cette caractéristique
    pour tous les clients, ainsi que la valeur spécifique associée au client identifié par `SK_ID_CURR`.

    Si la caractéristique n'est pas trouvée dans le DataFrame, la fonction renvoie un message d'erreur.

    Args:
        None. Les données sont extraites de la requête JSON avec les clés :
            - `feature` (str): Le nom de la caractéristique pour laquelle la distribution est demandée.
            - `SK_ID_CURR` (int): L'identifiant du client dont la valeur de la caractéristique est recherchée.

    Returns:
        flask.Response: Un objet JSON contenant :
            - `feature` (str): Le nom de la caractéristique demandée.
            - `client_value` (float, int): La valeur de la caractéristique pour le client identifié.
            - `distribution` (list): La liste de toutes les valeurs de cette caractéristique dans le DataFrame.
        En cas d'erreur (par exemple, si la caractéristique n'existe pas), un message d'erreur est renvoyé avec un code de statut 400.

    Raises:
        KeyError: Si `feature` ou `SK_ID_CURR` n'est pas présent dans les données de la requête JSON.
    """
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
