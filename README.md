# API MLSpotify
## Prérequis
modifier les 3 variables dans le main.py
```shell
SPOTIPY_CLIENT_ID =
SPOTIPY_CLIENT_SECRET =
SPOTIPY_REDIRECT_URI =
```
## Routes
```shell
/auth
```
Générer le token de connexion spotify il sera stocké dans un fichier

```shell
/train
```
entraine le modèle KMean est l'exporte dans un fichier .pkl

```shell
/predict?url=https://open.spotify.com/track/1tYj0VcVtlF3mY8GW9UMID?si=11a4c531368144c1
```
fait une prediction selon une track fournis