import json
import os
import time
from pathlib import Path

import httpx


API_URL = "https://places.googleapis.com/v1/places:autocomplete"
API_KEY_ENV_NAME = "GooglePlaces__ApiKey"
PLACES_FILE = Path("places.json")
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_DELAY_SECONDS = 0.2


def load_places() -> list[dict]:
    with PLACES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_places(places: list[dict]) -> None:
    with PLACES_FILE.open("w", encoding="utf-8") as file:
        json.dump(places, file, ensure_ascii=False, indent=2)
        file.write("\n")


def get_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV_NAME)

    if not api_key:
        raise RuntimeError(f"Missing environment variable: {API_KEY_ENV_NAME}")

    return api_key


def get_place_id(client: httpx.Client, api_key: str, name: str, country: str) -> str | None:
    response = client.post(
        API_URL,
        headers={
            "X-Goog-FieldMask": (
                "suggestions.placePrediction.placeId,"
                "suggestions.placePrediction.structuredFormat.*"
            ),
            "X-Goog-Api-Key": api_key
        },
        json={
            "input": f"{name}, {country}",
            "includedPrimaryTypes": ["country", "locality"]
        },
        timeout=REQUEST_TIMEOUT_SECONDS
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Google Places API error: {response.status_code} - {response.text}"
        )

    data = response.json()
    suggestions = data.get("suggestions", [])

    if not suggestions:
        return None

    prediction = suggestions[0].get("placePrediction")

    if not prediction:
        return None

    return prediction.get("placeId")


def main() -> None:
    api_key = get_api_key()
    places = load_places()

    with httpx.Client() as client:
        for index, place in enumerate(places, start=1):
            if place.get("placeId"):
                print(f"{index}/{len(places)} Skipping {place['name']} - already has placeId")
                continue

            name = place["name"]
            country = place["country"]

            print(f"{index}/{len(places)} Fetching placeId for {name}, {country}...")

            try:
                place_id = get_place_id(client, api_key, name, country)

                if place_id:
                    place["placeId"] = place_id
                    save_places(places)
                    print(f"Saved placeId for {name}: {place_id}")
                else:
                    print(f"No placeId found for {name}, {country}")

                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as ex:
                print(f"Stopping because of error for {name}, {country}: {ex}")
                save_places(places)
                break

    print("Done.")


if __name__ == "__main__":
    main()
