from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_URL = "https://datos.madrid.es/dataset/300114-0-madrid-decide/resource/300114-2-madrid-decide/download/300114-2-madrid-decide.csv"
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "madrid_decide.csv"

def download_dataset(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)

if __name__ == "__main__":
    download_dataset(DATA_URL, OUTPUT_PATH)