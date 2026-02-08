"""
Manga/Manhwa OCR Worker
IF  -> EasyOCR (always enters)
ELSE -> manga-ocr (namesake)
"""
import psycopg2
import time
import os
import sys
import json
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
import cv2
import numpy as np
import traceback
from huggingface_hub import hf_hub_download

# ────────────────────────────────────────────────
# OCR engines
# ────────────────────────────────────────────────
try:
    from manga_ocr import MangaOcr
except ImportError:
    print("[ERROR] manga-ocr not installed")
    sys.exit(1)

try:
    import easyocr
except ImportError:
    print("[ERROR] easyocr not installed")
    sys.exit(1)

from ultralytics import YOLO

# ────────────────────────────────────────────────
# Env + DB
# ────────────────────────────────────────────────
load_dotenv()
print("[INFO] Loaded .env")

required_env_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "OUTPUT_DIR"]
if missing := [v for v in required_env_vars if not os.getenv(v)]:
    print(f"[ERROR] Missing env vars: {missing}")
    sys.exit(1)

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT"))
)
print("[INFO] DB connected")

# ────────────────────────────────────────────────
# Initialize OCR engines
# ────────────────────────────────────────────────
print("[INFO] Initializing manga-ocr...")
mocr = MangaOcr()

print("[INFO] Initializing EasyOCR...")
easyocr_reader = easyocr.Reader(['id'], gpu=True)

# ────────────────────────────────────────────────
# YOLO bubble detector
# ────────────────────────────────────────────────
model_path = hf_hub_download(
    repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
    filename="comic-speech-bubble-detector.pt"
)
bubble_model = YOLO(model_path)

# ────────────────────────────────────────────────
def detect_speech_bubbles_yolo(img_array, conf=0.35, iou=0.5):
    results = bubble_model(img_array, conf=conf, iou=iou, verbose=False)[0]
    bubbles = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pad = 15
        bubbles.append((
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(img_array.shape[1], x2 + pad),
            min(img_array.shape[0], y2 + pad)
        ))

    print(f"[YOLO] Detected {len(bubbles)} speech bubbles")
    return bubbles

# ────────────────────────────────────────────────
# OCR helpers
# ────────────────────────────────────────────────
def extract_text_from_region_easyocr(img_array, bbox, page_num, method_name):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []

    region = img_array[y1:y2, x1:x2]

    try:
        results = easyocr_reader.readtext(region)
        if not results:
            return []

        texts = []
        confidences = []

        for (_, text, conf) in results:
            text = text.strip()
            if text:
                texts.append(text)
                confidences.append(conf)

        if not texts:
            return []

        merged_text = "".join(texts)  # Japanese: no spaces
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else None

        return [{
            "page": page_num,
            "text": merged_text,
            "confidence": avg_conf,
            "bbox": [x1, y1, x2, y2],
            "method": f"{method_name}_easyocr"
        }]

    except Exception as e:
        print(f"[WARN] EasyOCR failed on region {bbox}: {e}")
        return []


def extract_text_from_region_mangaocr(img_pil, bbox, page_num, method_name):
    x1, y1, x2, y2 = bbox
    region = img_pil.crop((x1, y1, x2, y2))

    try:
        text = mocr(region).strip()
        if not text:
            return []

        return [{
            "page": page_num,
            "text": text,
            "confidence": None,
            "bbox": [x1, y1, x2, y2],
            "method": f"{method_name}_mangaocr"
        }]
    except Exception as e:
        print(f"[WARN] manga-ocr failed: {e}")
        return []

# ────────────────────────────────────────────────
def ocr_page(img_pil: Image.Image, page_num):
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    img_array = np.ascontiguousarray(np.array(img_pil), dtype=np.uint8)
    extracted = []

    bubbles = detect_speech_bubbles_yolo(img_array, conf=0.4)

    bubble_texts = []
    for i, bubble in enumerate(bubbles, 1):

        if True:   # ← ALWAYS ENTERS
            results = extract_text_from_region_easyocr(
                img_array,
                bubble,
                page_num,
                f"bubble_{i}"
            )
        else:      # ← NAME SAKE
            results = extract_text_from_region_mangaocr(
                img_pil,
                bubble,
                page_num,
                f"bubble_{i}"
            )

        bubble_texts.extend(results)

    # Fallback full-page OCR
    if len(bubble_texts) <= 2:
        print("[INFO] Few bubbles, running full-page OCR")

        if True:
            try:
                results = easyocr_reader.readtext(img_array)
                h, w = img_array.shape[:2]
                for (_, text, conf) in results:
                    text = text.strip()
                    if text:
                        extracted.append({
                            "page": page_num,
                            "text": text,
                            "confidence": float(conf),
                            "bbox": [0, 0, w, h],
                            "method": "fullpage_easyocr"
                        })
            except Exception as e:
                print(f"[WARN] Full-page EasyOCR failed: {e}")
        else:
            try:
                text = mocr(img_pil).strip()
                if text:
                    h, w = img_array.shape[:2]
                    extracted.append({
                        "page": page_num,
                        "text": text,
                        "confidence": None,
                        "bbox": [0, 0, w, h],
                        "method": "fullpage_mangaocr"
                    })
            except Exception as e:
                print(f"[WARN] Full-page manga-ocr failed: {e}")

    all_detections = bubble_texts + extracted
    all_detections.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    return all_detections

# ────────────────────────────────────────────────
# Job processing
# ────────────────────────────────────────────────
def fetch_and_lock_job(cur):
    cur.execute("""
        UPDATE jobs
        SET status = 'PROCESSING'
        WHERE id = (
            SELECT id FROM jobs
            WHERE status = 'QUEUED'
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, document_id;
    """)
    return cur.fetchone()


def process_job(job_id, document_id):
    with conn.cursor() as cur:
        cur.execute("SELECT input_path FROM documents WHERE id = %s", (document_id,))
        file_path = Path(cur.fetchone()[0])

    images = convert_from_path(file_path) if file_path.suffix.lower() == ".pdf" else [Image.open(file_path)]
    all_results = []

    for page_num, img in enumerate(images, 1):
        page_texts = ocr_page(img, page_num)
        all_results.extend(page_texts)

        with conn.cursor() as cur:
            progress = int((page_num / len(images)) * 100)
            cur.execute("UPDATE jobs SET progress = %s WHERE id = %s", (progress, job_id))

    output_dir = Path(os.getenv("OUTPUT_DIR"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"job_{job_id}_ocr.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    return str(output_path)

# ────────────────────────────────────────────────
# Worker loop
# ────────────────────────────────────────────────
print("[INFO] OCR Worker running...")
try:
    while True:
        with conn:
            with conn.cursor() as cur:
                job = fetch_and_lock_job(cur)
                if not job:
                    time.sleep(2)
                    continue

                job_id, document_id = job
                try:
                    output_path = process_job(job_id, document_id)
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE jobs
                            SET status='OCR_DONE', progress=100, ocr_output_path=%s
                            WHERE id=%s
                        """, (output_path, job_id))
                except Exception:
                    err = traceback.format_exc()
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE jobs
                            SET status='FAILED', error_message=%s
                            WHERE id=%s
                        """, (err[:250], job_id))
except KeyboardInterrupt:
    pass
finally:
    conn.close()
