import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from schemas.recommendation_schemas import (
    RecommendPlacesRequest,
    RecommendedPlace,
    RecommendPlacesResponse
)


PLACES_FILE = "places.json"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_places():
    with open(PLACES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def place_to_text(place: dict) -> str:
    return f"{place['name']}, {place['country']}. {place['description']}"


places_catalog = load_places()

places_texts = [place_to_text(place) for place in places_catalog]

places_embeddings = embedding_model.encode(
    places_texts,
    normalize_embeddings=True
)


def visited_place_to_text(visited_place) -> str:
    matched_place = next(
        (
            place for place in places_catalog
            if place["name"].lower() == visited_place.name.lower()
        ),
        None
    )

    if matched_place:
        return place_to_text(matched_place)

    parts = [visited_place.name]

    if visited_place.country:
        parts.append(visited_place.country)

    text = ", ".join(parts)

    if visited_place.description:
        text += f". {visited_place.description}"

    return text


def recommend_places(data: RecommendPlacesRequest) -> RecommendPlacesResponse:
    visited_places_lower = {place.name.lower() for place in data.visitedPlaces}

    visited_texts = [
        visited_place_to_text(visited_place)
        for visited_place in data.visitedPlaces
    ]

    visited_embeddings = embedding_model.encode(
        visited_texts,
        normalize_embeddings=True
    )

    user_embedding = np.mean(visited_embeddings, axis=0).reshape(1, -1)

    scores = cosine_similarity(user_embedding, places_embeddings)[0]

    ranked_indexes = np.argsort(scores)[::-1]

    recommendations = []

    for index in ranked_indexes:
        place = places_catalog[index]

        if place["name"].lower() in visited_places_lower:
            continue

        recommendations.append(
            RecommendedPlace(
                name=place["name"],
                country=place["country"],
                description=place["description"],
                score=round(float(scores[index]), 4),
                imageUrl=place.get("imageUrl"),
                imageAuthor=place.get("imageAuthor"),
                imageAuthorUrl=place.get("imageAuthorUrl"),
                imageSource=place.get("imageSource")
            )
        )

        if len(recommendations) >= data.count:
            break

    return RecommendPlacesResponse(
        recommendations=recommendations
    )