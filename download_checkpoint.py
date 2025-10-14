from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import List, Dict
from TTS.utils.manage import ModelManager
import os
import shutil

# ----------------------------
# CLI аргументы
# ----------------------------
@dataclass
class DownloadArgs:
    output_path: str = field(
        default="checkpoints",
        metadata={"help": "Куды класці файлы мадэлі"}
    )
    version: str = field(
        default="v2.0.2",
        metadata={"help": "Рэвізія/тэг у coqui/XTTS-v2 (напрыклад: v2.0.2, main, v2.0.3)"}
    )
    include_dvae: bool = field(
        default=True,
        metadata={"help": "Цягнуць DVAE файлы (dvae.pth, mel_stats.pth)"}
    )
    include_speakers: bool = field(
        default=True,
        metadata={"help": "Цягнуць speakers_xtts.pth (рэкамендуецца)"}
    )
    clean: bool = field(
        default=False,
        metadata={"help": "Перачысціць каталог, калі ў ім іншая версія"}
    )

# ----------------------------
# Канстанты
# ----------------------------
REPO_ID = "coqui/XTTS-v2"
REQUIRED_CORE = ["model.pth", "config.json", "vocab.json"]
DVAE_FILES = ["dvae.pth", "mel_stats.pth"]
SPEAKERS_FILE = ["speakers_xtts.pth"]  # карысна для XTTS-v2

def build_hf_links(version: str, files: List[str]) -> Dict[str, str]:
    """
    Сфарміраваць прамыя спасылкі на файлы ў Hugging Face:
    https://huggingface.co/<repo>/resolve/<revision>/<filename>
    """
    base = f"https://huggingface.co/{REPO_ID}/resolve/{version}"
    return {fname: f"{base}/{fname}" for fname in files}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def missing_files(path: str, files: List[str]) -> List[str]:
    return [f for f in files if not os.path.isfile(os.path.join(path, f))]

def write_version_marker(path: str, version: str) -> None:
    with open(os.path.join(path, ".xtts_version"), "w", encoding="utf-8") as f:
        f.write(version.strip())

def read_version_marker(path: str) -> str:
    marker = os.path.join(path, ".xtts_version")
    if os.path.isfile(marker):
        try:
            with open(marker, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""

def download_xtts(output_path: str = "checkpoints", version: str = "v2.0.2",
                  include_dvae: bool = True, include_speakers: bool = True,
                  clean: bool = False) -> None:
    """
    Спампоўвае патрэбныя файлы XTTS-v2 для зададзенай версіі (рэвізіі/tag) у HF.
    Каталог — як у зыходным кодзе: <output_path>/XTTS_v2.0_original_model_files
    """
    # !!! ВАЖНА: назва каталога як у арыгінала
    target_dir = os.path.join(output_path, "XTTS_v2.0_original_model_files")
    ensure_dir(target_dir)

    # Праверка маркера версіі
    existing_version = read_version_marker(target_dir)
    if existing_version and existing_version != version:
        msg = (f"⚠ Знойдзена іншая версія ў каталогу: {existing_version}. "
               f"Запытана: {version}.")
        if clean:
            print(msg + " Чысцім каталог па тваёй камандзе (--clean).")
            # акуратна чысцім змесціва, але пакідаем сам каталог
            for name in os.listdir(target_dir):
                p = os.path.join(target_dir, name)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.remove(p)
                    else:
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"   Не атрымаўся выдаліць {p}: {e}")
        else:
            print(msg + " Працягнем: дацягнем толькі адсутнія файлы.")

    files = REQUIRED_CORE.copy()
    if include_dvae:
        files += DVAE_FILES
    if include_speakers:
        files += SPEAKERS_FILE

    to_get = missing_files(target_dir, files)
    if not to_get:
        print(f"✔ Усё ўжо спампавана ў: {target_dir}")
        # абнаўляем маркер версіі, каб адпавядаў фактычнаму стану
        if existing_version != version:
            write_version_marker(target_dir, version)
        return

    links = build_hf_links(version, to_get)
    print(f"> Спампоўваю XTTS-v2 ({version}) у: {target_dir}")
    ModelManager._download_model_files(list(links.values()), target_dir, progress_bar=True)

    still_missing = missing_files(target_dir, files)
    if still_missing:
        print("⚠ Не атрымалася спампаваць:", still_missing)
    else:
        write_version_marker(target_dir, version)
        print(f"✔ Гатова! Файлы ({len(files)} шт.) у {target_dir}")

def main():
    parser = HfArgumentParser(DownloadArgs)
    args = parser.parse_args()
    download_xtts(
        output_path=args.output_path,
        version=args.version,
        include_dvae=args.include_dvae,
        include_speakers=args.include_speakers,
        clean=args.clean
    )

if __name__ == "__main__":
    main()
