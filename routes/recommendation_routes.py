from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from schemas.recommendation_schemas import RecommendPlacesRequest
from services.recommendation_service import recommend_places


recommendation_bp = Blueprint("recommendation", __name__)


@recommendation_bp.post("/recommend-places")
def recommend_places_endpoint():
    try:
        data = RecommendPlacesRequest.model_validate(request.get_json())

        response = recommend_places(data)

        return jsonify(response.model_dump()), 200

    except ValidationError as ex:
        return jsonify({
            "error": "Invalid request.",
            "details": ex.errors()
        }), 400

    except Exception as ex:
        return jsonify({
            "error": "Unexpected error.",
            "details": str(ex)
        }), 500