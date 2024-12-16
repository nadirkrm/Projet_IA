import pandas as pd
from sentence_transformers import SentenceTransformer

# Chemin vers le fichier CSV
csv_file = r'C:\Users\papis\Desktop\EI4\Projet_IA\cleaned_overview.csv'

# Lire le fichier CSV avec l'encodage approprié (ici ISO-8859-1)
df = pd.read_csv(csv_file)

# Filtrer les lignes où 'overview' est manquant ou contient 'No overview found'
df_filtered = df[df['overview'].notna() & (df['overview'] != 'No overview found')]

# Charger le modèle pré-entraîné SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Récupérer la colonne 'overview' et la convertir en liste
overviews = df_filtered['overview'].tolist()

# Générer les embeddings pour chaque résumé de film
embeddings = model.encode(overviews)

# Ajouter les embeddings dans le DataFrame sous forme de liste
df_filtered['embeddings'] = embeddings.tolist()

# Sauvegarder le DataFrame avec les embeddings dans un nouveau fichier CSV
output_file = r'C:\Users\papis\Desktop\EI4\Projet_IA\cleaned_ov_with_embeddings_filtered.csv'
df_filtered.to_csv(output_file, index=False)

print(f"Embeddings générés et sauvegardés dans {output_file}")
