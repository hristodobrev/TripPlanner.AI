import json
from pyexpat.errors import messages
import re
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = Flask(__name__)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir="./models"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir="./models",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


class DescriptionRequest(BaseModel):
    placeName: str = Field(min_length=2, max_length=100)
    placeLocation: str | None = Field(default=None, max_length=100)
    maxLength: int = Field(default=300, ge=50, le=1000)


class DescriptionResponse(BaseModel):
    placeName: str
    description: str = Field(min_length=20, max_length=1000)


def build_prompt(data: DescriptionRequest) -> list[dict]:
    place_location_text = f", Location: {data.placeLocation}" if data.placeLocation else ""

    return [
        {
            "role": "system",
            "content": (
                "You are a travel assistant. "
                "Return ONLY valid JSON. "
                "Do not include markdown. "
                "Do not include explanations. "
                "The JSON must match this schema exactly: "
                '{"placeName": "string", "description": "string"}'
            )
        },
        {
            "role": "user",
            "content": (
                f"Generate a short travel description. "
                f"Place: {data.placeName}{place_location_text}. "
                f"The description must be around and no more than {data.maxLength} characters. "
                "Return only JSON."
            )
        }
    ]


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group())


def generate_description(data: DescriptionRequest) -> DescriptionResponse:
    messages = build_prompt(data)

    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    result = pipe(
        prompt,
        max_new_tokens=250,
        temperature=0.3,
        do_sample=True,
        return_full_text=False
    )

    generated_text = result[0]["generated_text"]

    if isinstance(generated_text, list):
        generated_text = generated_text[-1]["content"]

    parsed_json = extract_json(generated_text)

    return DescriptionResponse.model_validate(parsed_json, strict=True)


@app.post("/generate-description")
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)