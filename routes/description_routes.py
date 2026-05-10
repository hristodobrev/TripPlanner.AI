from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from schemas.description_schemas import DescriptionRequest
from services.description_service import generate_description


description_bp = Blueprint("description", __name__)


@description_bp.post("/generate-description")
def generate_description_endpoint():
    try:
        data = DescriptionRequest.model_validate(request.get_json())

        last_error = None

        for _ in range(2):
            try:
                response = generate_description(data)
                return jsonify(response.model_dump()), 200
            except Exception as ex:
                last_error = str(ex)

        return jsonify({
            "error": "Model did not return valid JSON.",
            "details": last_error
        }), 500

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