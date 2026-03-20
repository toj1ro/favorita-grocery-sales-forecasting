import os
import urllib.request
import zipfile
import sys

# https://drive.google.com/file/d/1m6OXyjWhBm8Z8RORSjCHNHUpgq_ANXVb/view
FILE_ID = "1m6OXyjWhBm8Z8RORSjCHNHUpgq_ANXVb"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
ZIP_PATH = "data/data.zip"


def download_file(url: str, output_path: str):
    print(f"Скачивание данных")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 8192

            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        print(f"\r  Прогресс: {percent:.1f}%", end="", flush=True)

        print(f"\n  Сохранено: {output_path}")
        return True
    except Exception as e:
        print(f"\n  Ошибка скачивания: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str):
    print(f"\nРаспаковка {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  Распаковано в: {extract_to}")
        return True
    except Exception as e:
        print(f"  Ошибка распаковки: {e}")
        return False


def main():
    os.makedirs("data", exist_ok=True)

    required_files = ["train.csv", "test.csv", "stores.csv", "items.csv",
                      "oil.csv", "holidays_events.csv", "transactions.csv"]
    existing_files = [f for f in required_files if os.path.exists(f"data/{f}")]

    if len(existing_files) == len(required_files):
        print("Все файлы данных уже существуют!")
        return 0

    # Скачиваем архив
    if not download_file(DOWNLOAD_URL, ZIP_PATH):
        print("\nНе удалось скачать данные.")
        print("Попробуйте скачать вручную:")
        print(f"  {DOWNLOAD_URL}")
        return 1

    if not extract_zip(ZIP_PATH, "data"):
        return 1

    os.remove(ZIP_PATH)
    print(f"\nАрхив {ZIP_PATH} удален")

    missing = [f for f in required_files if not os.path.exists(f"data/{f}")]
    if missing:
        print(f"\nВнимание: отсутствуют файлы: {', '.join(missing)}")
        return 1

    print("\nВсе файлы успешно загружены и распакованы!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
