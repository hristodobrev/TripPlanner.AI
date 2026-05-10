import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from schemas.description_schemas import DescriptionRequest, DescriptionResponse
from utils.json_utils import extract_json


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