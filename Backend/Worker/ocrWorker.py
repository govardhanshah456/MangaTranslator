"""
Manga/Manhwa OCR Worker - Using manga-ocr (Japanese specialized)
Processes PDFs/images from DB queue, uses preprocessing + manga-ocr
Focus: Japanese manga (vertical text, bubbles, SFX, fancy fonts)
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
# OCR engine: manga-ocr
# ────────────────────────────────────────────────
try:
    from manga_ocr import MangaOcr
except ImportError:
    print("[ERROR] manga-ocr not installed. Run: pip install manga-ocr")
    sys.exit(1)

load_dotenv()
print("[INFO] Loaded .env variables")

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

# manga-ocr is Japanese only → we ignore OCR_LANG for now
print("[INFO] Initializing manga-ocr (Japanese manga specialized model)...")
try:
    mocr = MangaOcr()           # loads ~440 MB model on first run
    print("[INFO] manga-ocr initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize manga-ocr: {e}")
    sys.exit(1)

from ultralytics import YOLO

# Load once at startup (e.g. after manga_ocr init)
model_path = hf_hub_download(
    repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
    filename="comic-speech-bubble-detector.pt"
)
bubble_model = YOLO(model_path)
# ────────────────────────────────────────────────
def detect_speech_bubbles_yolo(img_array, conf=0.35, iou=0.5):
    """
    Returns list of (x1,y1,x2,y2) tuples
    """
    print("No crash")
    print(img_array)
    try:
        results = bubble_model(img_array, conf=conf, iou=iou, verbose=False)[0]
    except Exception as e:
        import traceback
        print("[YOLO ERROR]")
        traceback.print_exc()
        raise

    print("Crash")
    bubbles = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Optional: add padding around box
        pad = 15
        bubbles.append((
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(img_array.shape[1], x2 + pad),
            min(img_array.shape[0], y2 + pad)
        ))
    
    print(f"[YOLO] Detected {len(bubbles)} speech bubbles")
    return bubbles


def pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_image_for_ocr(img_array):
    """
    Light preprocessing — manga-ocr is already quite robust,
    so we keep it minimal (mainly denoising + slight contrast)
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    # Mild contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=8)

    return denoised


def detect_speech_bubbles(img_array, min_area=300, max_area_ratio=0.35):
    """
    (kept almost as-is — works reasonably well for many manhwa/manhua too)
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    # Try to catch white/light bubbles
    _, thresh_white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(gray, 40, 140)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    combined = cv2.bitwise_or(thresh_white, dilated)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours_result = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

    bubbles = []
    h, w = gray.shape
    max_area = h * w * max_area_ratio

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        aspect_ratio = w_rect / h_rect if h_rect > 0 else 1.0
        if aspect_ratio < 0.2 or aspect_ratio > 5.5:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.35:          # reasonably lenient
                continue

        pad = 18
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + w_rect + pad)
        y2 = min(h, y + h_rect + pad)

        bubbles.append((x1, y1, x2, y2))

    return bubbles


# ────────────────────────────────────────────────
# Core OCR function using manga-ocr
# ────────────────────────────────────────────────
def extract_text_from_region_mangaocr(img_pil: Image.Image, bbox, page_num, method_name=""):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []

    region = img_pil.crop((x1, y1, x2, y2))

    try:
        text = mocr(region)                     # main call — returns clean string
        text = text.strip()
        if not text or len(text) < 1:
            return []

        # manga-ocr does not give confidence or char-level boxes
        # → we return the whole region as one text block
        return [{
            "page": page_num,
            "text": text,
            "confidence": None,                 # not available
            "bbox": [x1, y1, x2, y2],
            "method": f"{method_name}_mangaocr"
        }]

    except Exception as e:
        print(f"[WARN] manga-ocr failed on region {bbox}: {e}")
        return []


def ocr_page_mangaocr(img_pil: Image.Image, page_num=1):
    """
    Main OCR routine using manga-ocr + bubble detection
    """
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    img_array = np.ascontiguousarray(np.array(img_pil), dtype=np.uint8)
    extracted = []

    # Step 1: Detect likely text bubbles
    bubbles = detect_speech_bubbles_yolo(img_array, conf=0.4)  # tune conf 0.3–0.5
    print(f"[INFO] Page {page_num} - Detected {len(bubbles)} potential speech bubbles")

    # Step 2: OCR each bubble
    bubble_texts = []
    for i, bubble in enumerate(bubbles, 1):
        results = extract_text_from_region_mangaocr(img_pil, bubble, page_num, f"bubble_{i}")
        bubble_texts.extend(results)

    # Step 3: Fallback — full page if almost nothing found
    if len(bubble_texts) <= 2:
        print(f"[INFO] Page {page_num} - Few bubbles → trying full-page OCR")
        try:
            full_text = mocr(img_pil)
            full_text = full_text.strip()
            if full_text:
                h, w = img_array.shape[:2]
                extracted.append({
                    "page": page_num,
                    "text": full_text,
                    "confidence": None,
                    "bbox": [0, 0, w, h],
                    "method": "fullpage_mangaocr"
                })
        except Exception as e:
            print(f"[WARN] Full-page manga-ocr failed: {e}")

    # Combine
    all_detections = bubble_texts + extracted
    print(f"[INFO] Page {page_num} - Collected {len(all_detections)} text regions")

    # Minimal post-processing (you can keep your merge_nearby_text_detections if desired)
    # For manga-ocr we often don't need aggressive merging because it already handles multi-line

    # Sort roughly top-to-bottom, left-to-right
    all_detections.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

    # Debug image
    debug_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for e in all_detections:
        x1, y1, x2, y2 = e["bbox"]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = e["text"][:12] + "…" if len(e["text"]) > 12 else e["text"]
        cv2.putText(debug_img, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for bubble in bubbles:
        x1, y1, x2, y2 = bubble
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 128, 0), 1)

    cv2.imwrite(f"debug_mangaocr_page_{page_num}.png", debug_img)

    return all_detections


# ────────────────────────────────────────────────
# The rest stays almost the same
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
    print(f"[INFO] Job {job_id} | Document {document_id}")

    with conn.cursor() as cur:
        cur.execute("SELECT input_path FROM documents WHERE id = %s", (document_id,))
        file_path = Path(cur.fetchone()[0])

    if file_path.suffix.lower() == ".pdf":
        images = convert_from_path(file_path)
    else:
        images = [Image.open(file_path)]

    all_results = []
    total_pages = len(images)

    for page_num, img in enumerate(images, 1):
        print(f"[INFO] Processing page {page_num}/{total_pages}")
        page_texts = ocr_page_mangaocr(img, page_num)

        all_results.extend(page_texts)

        if page_texts:
            sample = page_texts[0]["text"][:60] + "..." if len(page_texts[0]["text"]) > 60 else page_texts[0]["text"]
            print(f"[INFO] Page {page_num} - {len(page_texts)} regions | Sample: {sample}")
        else:
            print(f"[WARN] Page {page_num} - No text detected")

        with conn.cursor() as cur:
            progress = int((page_num / total_pages) * 100)
            cur.execute("UPDATE jobs SET progress = %s WHERE id = %s", (progress, job_id))

    output_dir = Path(os.getenv("OUTPUT_DIR"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"job_{job_id}_ocr.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved {len(all_results)} entries → {output_path}")
    return str(output_path)


# Worker loop
try:
    print("[INFO] manga-ocr Worker running (Japanese manga specialized)...")
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
                            SET status = 'DONE', progress = 100, output_path = %s
                            WHERE id = %s
                        """, (output_path, job_id))
                    print(f"[INFO] Job {job_id} completed")
                except Exception as e:
                    error = traceback.format_exc()
                    error_short = error[:250] + "..." if len(error) > 250 else error
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE jobs
                            SET status = 'FAILED', error_message = %s
                            WHERE id = %s
                        """, (error_short, job_id))
                    print(f"[ERROR] Job {job_id} failed:\n{error_short}")
except KeyboardInterrupt:
    print("[INFO] Stopped by user")
finally:
    conn.close()
    print("[INFO] DB closed")