import json
from fastapi import FastAPI
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from starlette.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import urllib3
import requests

app = FastAPI()

SPOTIPY_CLIENT_ID = "f7233b70e8b945c587390fc6cee8882a"
SPOTIPY_CLIENT_SECRET = "14aea31fdb0143d484cbc95527bae3df"
SPOTIPY_REDIRECT_URI = "http://127.0.0.1:8000"


@app.get("/")
async def root():
    auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                            client_secret=SPOTIPY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    playlistOne = sp.playlist('1NRtRsGpMkpZO3r68kezIM')
    playlistTwo = sp.playlist('6uY36xN4g5RNP5RgKo53ue')
    playlistThree = sp.playlist('4mdphiZLuZ35QfV9eS9TiY')

    flat_list = []
    ids = []
    for sublist in [playlistOne["tracks"]['items'], playlistTwo["tracks"]['items'], playlistThree["tracks"]['items']]:
        for item in sublist:
            ids.append(item['track']['id'])
            flat_list.append(item['track'])

    chunks = list(separer_en_chunks(ids, 100))

    flat_list_features = []
    for chunk in chunks:
        http = urllib3.PoolManager()
        chunk = ','.join(chunk)
        r = http.request('GET', 'https://api.spotify.com/v1/audio-features?ids=' + chunk, headers={
            'Authorization': 'Bearer ' + get_access_token()})
        for item in json.loads(r.data.decode())["audio_features"]:
            flat_list_features.append(item)

    y = json.dumps(flat_list)
    x = json.dumps(flat_list_features)

    jsonResp = JSONResponse(
        content=[flat_list, flat_list_features])

    df = pd.read_json(y)
    df2 = pd.read_json(x)
    df.to_csv(r'D:\MLSpotify\tracks.csv', index=None)
    df2.to_csv(r'D:\MLSpotify\tracks_features.csv', index=None)

    return jsonResp


def separer_en_chunks(liste, taille_chunk):
    for i in range(0, len(liste), taille_chunk):
        yield liste[i:i + taille_chunk]


@app.get("/train")
async def train():
    input_file = "tracks_features.csv"

    # Charger les données à partir du fichier CSV
    X = pd.read_csv(input_file)

    X = X.drop(
        ['id', 'type', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature', 'valence', 'tempo', 'mode',
         'key', 'loudness'], axis=1)

    # # Mise à l'échelle des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Trouver le nombre optimal de clusters (k) en utilisant la méthode du coude (elbow method)
    # Somme des carrés des distances par rapport aux centres de cluster
    sse = []
    k_values = range(2, 10)  # Testez différentes valeurs de k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    # Tracer la courbe du coude (elbow curve)
    plt.plot(k_values, sse, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Curve')
    plt.show()

    # Choisissez le nombre optimal de clusters en fonction de la courbe du coude
    # À partir du graphique, déterminez le point où l'amélioration des clusters devient marginale

    # Utilisez le nombre optimal de clusters pour le modèle K-means
    k = 4  # Choisissez le nombre optimal de clusters obtenu à partir de la courbe du coude
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Calcul de l'indice de silhouette
    silhouette_avg = silhouette_score(X_scaled, labels)
    print("Silhouette Score:", silhouette_avg)

    # Affichage des données d'origine
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', label='Original Data')

    # Affichage du clustering avec les labels
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', label='Cluster Centers', marker='x')
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, label='Clusters')
    plt.xlabel('Feature 1 (danceability)')
    plt.ylabel('Feature 2 (energy)')
    plt.title('Clustering with K-means')
    plt.legend()
    plt.show()

    with open("model.pkl", "wb") as file:
        pickle.dump(kmeans, file)

    return JSONResponse([plt.show(), silhouette_avg])


# Définition de la route pour les prédictions
@app.get("/predict")
def predict(track_url: str = '', ):
    auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                            client_secret=SPOTIPY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    track = json.dumps(sp.track(track_url))

    http = urllib3.PoolManager()
    r = http.request('GET', 'https://api.spotify.com/v1/audio-features/' + json.loads(track)["id"], headers={
        'Authorization': 'Bearer ' + get_access_token()})

    data = json.loads(r.data.decode())

    df = pd.DataFrame([data])

    df.to_csv(r'D:\MLSpotify\predict.csv', index=None)

    with open("model.pkl", "rb") as file:
        kmeans = pickle.load(file)

    input_file = "predict.csv"

    CSVData = pd.read_csv(input_file)

    CSVData = CSVData.drop(
        ['id', 'type', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature', 'valence', 'tempo', 'mode',
         'key', 'loudness'], axis=1)

    predictions = kmeans.predict(CSVData)

    return JSONResponse(content=predictions.tolist())


@app.get("/auth")
def auth():
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": SPOTIPY_CLIENT_ID,
        "client_secret": SPOTIPY_CLIENT_SECRET
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        json_response = response.json()
        access_token = json_response["access_token"]

        with open("spotifyKey.txt", "w") as file:
            file.write(access_token + "\n")

        return JSONResponse(content="OK")
    else:
        return JSONResponse(content="ERROR")

def get_access_token():
    with open("spotifyKey.txt", "r") as file:
        access_token = file.read().strip()
    return access_token
