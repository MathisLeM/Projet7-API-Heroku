import requests
import pytest

BASE_URL = "https://p7-ocr-api-mathis-d22bcf66c298.herokuapp.com"

# Test du point de terminaison /predict
def test_predict():
    url = f"{BASE_URL}/predict"
    # Exemple de données pour l'API
    data = {"SK_ID_CURR": 100004}
    response = requests.post(url, json=data)
    
    # Vérifie que la requête est réussie
    assert response.status_code == 200
    
    # Vérifie que la réponse contient une probabilité
    json_data = response.json()
    assert "probability" in json_data
    assert 0 <= json_data["probability"] <= 100, "Probabilité hors de l'intervalle 0-100"

# Test du point de terminaison /info
def test_info():
    url = f"{BASE_URL}/info"
    data = {"SK_ID_CURR": 100004}
    response = requests.post(url, json=data)

    # Vérifie que la requête est réussie
    assert response.status_code == 200
    
    # Vérifie que la réponse contient des informations personnelles
    json_data = response.json()
    assert "SK_ID_CURR" in json_data
    assert "INCOME_PER_PERSON" in json_data

# Test du point de terminaison /distribution
def test_distribution():
    url = f"{BASE_URL}/distribution"
    data = {"SK_ID_CURR": 100004, "feature": "INCOME_PER_PERSON"}
    response = requests.post(url, json=data)

    # Vérifie que la requête est réussie
    assert response.status_code == 200

    # Vérifie que la réponse contient la valeur pour le client et la distribution de la caractéristique
    json_data = response.json()
    assert "client_value" in json_data
    assert "distribution" in json_data
    assert isinstance(json_data["distribution"], list), "La distribution doit être une liste"
