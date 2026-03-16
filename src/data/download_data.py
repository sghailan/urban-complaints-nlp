from pathlib import Path
import requests

DATA_URL = "https://datos.madrid.es/dataset/300114-0-madrid-decide/resource/300114-2-madrid-decide/download/300114-2-madrid-decide.csv"

def download_dataset(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(DATA_URL, headers=headers, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)