import json
import os
import time
import requests

from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PLACES_FILE = "places.json"

if not ACCESS_KEY:
    raise Exception("Missing UNSPLASH_ACCESS_KEY in .env")


def save_places(places):
    with open(PLACES_FILE, "w", encoding="utf-8") as file:
        json.dump(places, file, ensure_ascii=False, indent=2)


def search_city_photo(city: str, country: str):
    response = requests.get(
        "https://api.unsplash.com/search/photos",
        headers={
            "Authorization": f"Client-ID {ACCESS_KEY}"
        },
        params={
            "query": f"{city} {country}",
            "orientation": "landscape",
            "per_page": 1
        },
        timeout=20
    )

    if response.status_code != 200:
        raise Exception(
            f"Unsplash API error: {response.status_code} - {response.text}"
        )

    data = response.json()

    if not data["results"]:
        return None

    photo = data["results"][0]

    return {
        "imageUrl": photo["urls"]["regular"],
        "imageAuthor": photo["user"]["name"],
        "imageAuthorUrl": photo["user"]["links"]["html"],
        "imageSource": "Unsplash"
    }


with open(PLACES_FILE, "r", encoding="utf-8") as file:
    places = json.load(file)


for index, place in enumerate(places, start=1):
    if place.get("imageUrl"):
        print(f"{index}/{len(places)} Skipping {place['name']} - already has imageUrl")
        continue

    name = place["name"]
    country = place["country"]

    print(f"{index}/{len(places)} Fetching image for {name}, {country}...")

    try:
        image_data = search_city_photo(name, country)

        if image_data:
            place.update(image_data)
            save_places(places)
            print(f"Saved image for {name}")
        else:
            print(f"No image found for {name}")

        time.sleep(1)

    except Exception as ex:
        print(f"Stopping because of error for {name}: {ex}")

        # Save everything collected so far before stopping
        save_places(places)
        break


print("Done.")