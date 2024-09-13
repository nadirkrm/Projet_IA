# # from sentence_transformers import SentenceTransformer

# # # Charger un modèle pré-entraîné
# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # # Liste de phrases à encoder
# # phrases = [
# #     "Le chat dort sur le canapé.",
# #     "Un félin se repose sur un sofa.",
# #     "Il fait beau aujourd'hui."
# # ]

# # # Générer les embeddings pour les phrases
# # embeddings = model.encode(phrases)

# # # Afficher les embeddings
# # for i, embedding in enumerate(embeddings):
# #     print(f"Embedding de la phrase {i+1}:")
# #     print(embedding)
# #     print("\n")
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np

# # # Charger un modèle pré-entraîné
# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # # Liste de phrases à encoder
# # phrases = [
# #     "Le chat dort sur le canapé.",
# #     "Un félin se repose sur un sofa.",
# #     "Il fait beau aujourd'hui."
# # ]

# # # Générer les embeddings pour les phrases
# # embeddings = model.encode(phrases)

# # # Afficher les embeddings
# # for i, embedding in enumerate(embeddings):
# #     print(f"Embedding de la phrase {i+1}:")
# #     print(embedding)
# #     print("\n")

# # # Calculer la similarité cosinus entre les phrases
# # similarity_matrix = cosine_similarity(embeddings)

# # # Afficher la matrice de similarité
# # print("Matrice de similarité cosinus entre les phrases :")
# # print(np.round(similarity_matrix, 2))

# # # Interprétation de la similarité :
# # for i in range(len(phrases)):
# #     for j in range(i + 1, len(phrases)):
# #         print(f"Similarité entre la phrase {i+1} et la phrase {j+1}: {similarity_matrix[i][j]:.2f}")
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np

# # # Charger un modèle pré-entraîné
# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # # Liste de phrases à encoder
# # phrases = [
# #     "Le chat dort sur le canapé.",
# #     "Un félin se repose sur un sofa.",
# #     "Il fait beau aujourd'hui."
# # ]

# # # Générer les embeddings pour les phrases
# # embeddings = model.encode(phrases)

# # # Calculer la similarité cosinus entre les phrases
# # similarity_matrix = cosine_similarity(embeddings)

# # # Afficher la matrice de similarité
# # print("Matrice de similarité cosinus entre les phrases :")
# # print(np.round(similarity_matrix, 2))

# # # Interprétation de la similarité :
# # for i in range(len(phrases)):
# #     for j in range(i + 1, len(phrases)):
# #         print(f"Similarité entre la phrase {i+1} et la phrase {j+1}: {similarity_matrix[i][j]:.2f}")
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Essayer un autre modèle
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Liste de phrases à encoder
# phrases = [
#     "Le chat dort sur le canapé.",
#     "Un félin se repose sur un sofa.",
#     "Il fait beau aujourd'hui."
# ]

# # Générer les embeddings pour les phrases
# embeddings = model.encode(phrases)

# # Calculer la similarité cosinus entre les phrases
# similarity_matrix = cosine_similarity(embeddings)

# # Afficher la matrice de similarité
# print("Matrice de similarité cosinus entre les phrases :")
# print(np.round(similarity_matrix, 2))

# # Interprétation de la similarité :
# for i in range(len(phrases)):
#     for j in range(i + 1, len(phrases)):
#         print(f"Similarité entre la phrase {i+1} et la phrase {j+1}: {similarity_matrix[i][j]:.2f}")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Charger un modèle pré-entraîné
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Liste de phrases longues à encoder
phrases = [
    "Le chat, qui a des yeux verts éclatants, est assis confortablement sur le mur en pierre, profitant du doux soleil du matin.",
    "Le chat, qui a des yeux verts éclatants, est assis confortablement sur le mur en pierre, profitant du doux soleil du matin.",
    "Le chien, un golden retriever joyeux, joue dans le jardin verdoyant en courant après une balle que son propriétaire lui lance régulièrement.",
    "Il pleut beaucoup aujourd'hui, et les gouttes d'eau tombent doucement sur les fenêtres, créant une mélodie apaisante dans la maison.",
    "Les enfants, en fin de journée scolaire, jouent avec enthousiasme dans le parc voisin, où les balançoires et les toboggans sont les attractions principales.",
    "Un oiseau, dont les plumes sont d'un bleu vif, chante gaiement dans l'arbre majestueux situé au milieu du jardin, apportant une touche de joie à l'environnement.",
    "Le soleil brille intensément dans le ciel bleu clair, et les rayons de lumière réchauffent agréablement l'air tandis que les gens se promènent dans le parc.",
    "Je prépare un repas spécial pour ce soir, en suivant une recette délicieuse qui combine des épices exotiques avec des légumes frais et savoureux.",
    "Les fleurs colorées dans le jardin sont magnifiques, avec des pétales éclatants qui se balancent doucement au gré du vent léger et parfumé.",
    "Je lis un livre passionnant sur les voyages à travers le monde, qui décrit des aventures fascinantes dans des destinations lointaines et exotiques.",
    "La température est agréable aujourd'hui, parfaite pour une promenade en plein air, avec un léger vent qui rafraîchit l'air tout en restant doux et agréable.",
    "The temperature is pleasant today, perfect for an outdoor walk, with a light breeze that cools the air while remaining mild and pleasant."
]

# Générer les embeddings pour les phrases
embeddings = model.encode(phrases)

# Calculer la similarité cosinus entre les phrases
similarity_matrix = cosine_similarity(embeddings)

# Afficher la matrice de similarité
print("Matrice de similarité cosinus entre les phrases :")
print(np.round(similarity_matrix, 2))

# Interprétation de la similarité
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        print(f"Similarité entre la phrase {i+1} et la phrase {j+1}: {similarity_matrix[i][j]:.2f}")
