import json
import re


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group())