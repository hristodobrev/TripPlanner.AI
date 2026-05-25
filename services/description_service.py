import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

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
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    parsed_json = extract_json(generated_text)

    return DescriptionResponse.model_validate(parsed_json, strict=True)
