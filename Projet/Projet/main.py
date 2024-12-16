from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json

app = FastAPI()

# === Configuration des templates et des fichiers statiques ===
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Chargement et préparation des données ===
df_embeddings = pd.read_csv('main_with_compressed_embeddings_unet.csv')
df_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Préparation des IDs et des dates
df_metadata['release_date'] = pd.to_datetime(df_metadata['release_date'], errors='coerce')
df_embeddings['imdb_id'] = df_embeddings['imdb_id'].astype(str)
df_metadata['imdb_id'] = df_metadata['imdb_id'].astype(str)

# Fusion des deux DataFrames
df = pd.merge(df_embeddings, df_metadata, on='imdb_id', how='left')

# Conversion des embeddings
df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
embeddings_matrix = np.vstack(df['embeddings'].values)

# === Dictionnaires pour les filtres ===
collection_dict = df[['imdb_id', 'belongs_to_collection']].dropna().set_index('imdb_id')['belongs_to_collection'].to_dict()

# === Listes pour les API ===
titles = df['title'].tolist()
all_genres = sorted(set(
    g['name'] for genres in df['genres'].dropna().apply(ast.literal_eval) 
    for g in genres if 'name' in g
))
all_collections = sorted(set(df['belongs_to_collection'].dropna().unique()))
all_decades = sorted(set((row['release_date'].year // 10 * 10) for _, row in df.iterrows() if pd.notna(row['release_date'])))

# === Routes pour les pages HTML ===
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Affiche la page d'accueil."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/suggestions", response_class=HTMLResponse)
async def show_suggestions_page(request: Request):
    """Affiche la page des suggestions avec les filtres."""
    return templates.TemplateResponse("suggestions.html", {
        "request": request,
        "selected_film": None,
        "recommendations": [],
        "all_genres": all_genres,
        "all_collections": all_collections,
        "all_decades": all_decades
    })

@app.post("/suggestions", response_class=HTMLResponse)
async def get_suggestions(
    request: Request,
    film: str = Form(...),
    top_n: int = Form(5),
    genres_filter: list[str] = Form(default=[]),
    collections_filter: list[str] = Form(default=[]),
    year_range_filter: list[int] = Form(default=[])
):
    """Calcule et affiche les recommandations de films."""
    film_index = df.index[df['title'] == film][0]
    selected_film = df.iloc[film_index]

    # Calcul des similarités
    film_embedding = embeddings_matrix[film_index].reshape(1, -1)
    similarities = cosine_similarity(film_embedding, embeddings_matrix)[0]

    selected_id = selected_film['imdb_id']
    selected_collection = collection_dict.get(selected_id, None)
    selected_genres = set(
        genre['name'] for genre in ast.literal_eval(selected_film['genres']) if 'name' in genre
    ) if pd.notna(selected_film['genres']) else set()
    selected_year = selected_film['release_date'].year if pd.notna(selected_film['release_date']) else None

    scores = []
    for idx, row in df.iterrows():
        if idx == film_index:
            continue

        # Vérifier les filtres
        row_genres = set(genre['name'] for genre in ast.literal_eval(row['genres']) if 'name' in genre) if pd.notna(row['genres']) else set()
        row_collection = collection_dict.get(row['imdb_id'], None)
        row_year = row['release_date'].year if pd.notna(row['release_date']) else None

        if genres_filter and not row_genres.intersection(genres_filter):
            continue
        if collections_filter and row_collection not in collections_filter:
            continue
        if year_range_filter and (row_year is None or row_year // 10 * 10 not in year_range_filter):
            continue

        # Calcul des scores
        embedding_score = similarities[idx]
        collection_score = 1.0 if row_collection == selected_collection else 0.0
        genre_score = len(selected_genres & row_genres) / max(len(selected_genres | row_genres), 1)
        runtime_score = 1 - (abs(selected_film['runtime'] - row['runtime']) / max(selected_film['runtime'], row['runtime'], 1)) if pd.notna(selected_film['runtime']) and pd.notna(row['runtime']) else 0
        time_score = 1 - abs((selected_year - row_year) / max(selected_year, row_year, 1)) if selected_year and row_year else 0

        total_score = (0.4 * collection_score) + (0.3 * embedding_score) + (0.15 * genre_score) + (0.1 * runtime_score) + (0.05 * time_score)
        scores.append((idx, total_score))
        

   
    # Trier et récupérer les recommandations
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    # Print : Nombre de recommandations
    print(f"\nNombre de recommandations : {len(scores)}\n")

    # Print : Détails des scores pour chaque film recommandé
    for idx, score in scores:
        row = df.iloc[idx]
        print(f"Film recommandé : {row['title']}")
        print(f"    Embedding Score : {similarities[idx]:.3f}")
        print(f"    Collection Score : {1.0 if collection_dict.get(row['imdb_id']) == selected_collection else 0.0:.3f}")
        print(f"    Genre Score      : {len(selected_genres & set(g['name'] for g in ast.literal_eval(row['genres']) if 'name' in g)) / max(len(selected_genres | set(g['name'] for g in ast.literal_eval(row['genres']) if 'name' in g)), 1):.3f}")
        print(f"    Runtime Score    : {1 - (abs(selected_film['runtime'] - row['runtime']) / max(selected_film['runtime'], row['runtime'], 1)) if pd.notna(selected_film['runtime']) and pd.notna(row['runtime']) else 0:.3f}")
        print(f"    Time Score       : {1 - abs((selected_year - row['release_date'].year) / max(selected_year, row['release_date'].year, 1)) if selected_year and pd.notna(row['release_date']) else 0:.3f}")
        print(f"    Total Score      : {score:.3f}\n")

    recommendations = [
        {
            "title": df.iloc[idx]['title'],
            "genres": ", ".join([g['name'] for g in ast.literal_eval(df.iloc[idx]['genres'])]) if pd.notna(df.iloc[idx]['genres']) else "Non spécifié",
            "release_year": df.iloc[idx]['release_date'].year if pd.notna(df.iloc[idx]['release_date']) else "Inconnue",
            "runtime": int(df.iloc[idx]['runtime']) if pd.notna(df.iloc[idx]['runtime']) else "Non spécifié",
            "score": round(score, 2)
        }
        for idx, score in scores
    ]
    # Print pour afficher les suggestions envoyées
    print(json.dumps(recommendations, indent=4))

    print(f"Film: {film}, Top N: {top_n}, Genres Filter: {genres_filter}, Collections Filter: {collections_filter}, Year Range Filter: {year_range_filter}")



    return templates.TemplateResponse("suggestions.html", {
        "request": request,
        "selected_film": film,
        "recommendations": recommendations
    })

# === Routes pour les API ===
@app.get("/api/film-titles")
async def get_film_titles():
    """Retourne les titres des films."""
    return {"titles": titles}

@app.get("/api/genres")
async def get_genres():
    """Retourne la liste des genres."""
    return {"genres": all_genres}

@app.get("/api/collections")
async def get_collections():
    """Retourne la liste des collections."""
    return {"collections": all_collections}

@app.get("/api/decades")
async def get_decades():
    """Retourne la liste des décennies."""
    return {"decades": all_decades}
