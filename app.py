import os
import re
import json
import uuid
import logging
import traceback
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import pymupdf

# --- DUAL OCR INTEGRATION (Tesseract + EasyOCR) ---
try:
    import pytesseract
    from pytesseract import Output as TESS_OUTPUT
    TESSERACT_OK = True
except Exception:
    pytesseract = None
    TESS_OUTPUT = None
    TESSERACT_OK = False
    logging.warning("Tesseract not installed — running only on EasyOCR")

try:
    import easyocr
    EASYOCR_OK = True
except Exception:
    easyocr = None
    EASYOCR_OK = False
    logging.warning("EasyOCR not installed — running only on Tesseract")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Optional AI model support - Enabled by default for max accuracy
MODEL_LOCK = threading.RLock()
_MODEL_STATE = {
    "yolo_world": {"enabled": os.getenv("CABINSPECT_ENABLE_YOLO", "1") == "1",
                        "loaded": False, "error": None, "instance": None},
    "florence2": {"enabled": os.getenv("CABINSPECT_ENABLE_FLORENCE", "1") == "1",
                   "loaded": False, "error": None, "processor": None, "instance": None},
}
# Optimized Models (Grounding DINO removed for better speed)
YOLO_MODEL_ID = os.getenv("CABINSPECT_YOLO_MODEL", "yolov8m-world.pt")
FLORENCE_MODEL_ID = os.getenv("CABINSPECT_FLORENCE_MODEL", "microsoft/Florence-2-large")

_easyocr_reader = None
OCR_LOCK = threading.Lock()

def get_ocr_reader():
    global _easyocr_reader
    with OCR_LOCK:
        if _easyocr_reader is None and EASYOCR_OK:
            _easyocr_reader = easyocr.Reader(['en'], gpu=True)
    return _easyocr_reader

 
# CONFIGURATION
 

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
TEMP_DIR   = BASE_DIR / "temp"
CONFIG_FILE = BASE_DIR / "config.json"

for folder in (UPLOAD_DIR, REPORT_DIR, TEMP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CabinSpect")

ALLOWED_FILES = {
    "cabinet_image":       {"jpg", "jpeg", "png"},
    "engineering_drawing": {"pdf"},
    "qc_checklist":        {"json", "csv", "xlsx"},
}

DEFAULT_CHECKS = [
    ("component_presence", "Component Presence"),
    ("component_position", "Component Position"),
    ("component_order",    "Component Order"),
    ("label_verification", "Label Verification"),
    ("component_condition","Component Visual Condition"),
    ("mounting_verification", "Mounting Verification"),
    ("drawing_revision",   "Drawing Revision"),
    ("image_quality",      "Image Quality"),
]

DEFAULT_CONFIG = {
    "components": ["MCB", "POWER SUPPLY", "PLC", "K1", "K2"],
    "expected_positions": {},
    "thresholds": {
        "ocr_confidence": 0.35,  
        "position_tolerance": 0.25,
        "detection_confidence": 0.50,
        "yolo_confidence": 0.15,
        "missing_severity": "FAIL",
    },
}

_config_cache = None

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return _merge_config(raw)
        except Exception:
            logger.exception("Failed to read config.json — using defaults")
    return _merge_config({})

def _merge_config(cfg):
    base = json.loads(json.dumps(DEFAULT_CONFIG))
    if isinstance(cfg, dict):
        if isinstance(cfg.get("components"), list):
            base["components"] = [str(x) for x in cfg["components"] if str(x).strip()]
        if isinstance(cfg.get("expected_positions"), dict):
            base["expected_positions"] = {
                k: {"x": float(v.get("x", 0.5)), "y": float(v.get("y", 0.5))}
                for k, v in cfg["expected_positions"].items()
                if isinstance(v, dict) and "x" in v and "y" in v
            }
        if isinstance(cfg.get("thresholds"), dict):
            base["thresholds"].update({
                k: v for k, v in cfg["thresholds"].items()
                if k in base["thresholds"]
            })
    return base

def get_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache

def save_config(cfg):
    global _config_cache
    merged = _merge_config(cfg)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    _config_cache = merged
    return merged

 
# UTILITIES
 

def generate_inspection_id():
    return "CSP-" + datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()

def extension(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""

def validate_file(file, category):
    if not file or not file.filename:
        return False, "File missing"
    ext = extension(file.filename)
    if ext not in ALLOWED_FILES[category]:
        return False, f"Unsupported format: .{ext}"
    return True, "Valid"

def inspection_dir(inspection_id):
    safe_id = secure_filename(inspection_id)
    if not safe_id or safe_id != inspection_id:
        raise ValueError("Invalid inspection ID")
    folder = (UPLOAD_DIR / safe_id).resolve()
    if UPLOAD_DIR.resolve() not in folder.parents:
        raise ValueError("Invalid inspection directory")
    return folder

def evidence_dir(inspection_id):
    d = inspection_dir(inspection_id) / "evidence"
    d.mkdir(exist_ok=True)
    return d

def find_input(folder, category):
    matches = list(folder.glob(category + ".*"))
    if not matches:
        raise FileNotFoundError(f"{category} file not found")
    return matches[0]

def normalize(text):
    return re.sub(r"\s+", " ", str(text).upper().strip())

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s)).strip("_") or "component"

def classify_region(x_norm, y_norm):
    h = "left" if x_norm < 0.5 else "right"
    v = "upper" if y_norm < 0.5 else "lower"
    return f"{v}-{h}"

 
# COMPONENT LABEL PATTERNS
 

DESIGNATOR_PATTERN = r"\b(?:QF|KM|KA|KT|SB|HL|SA|QS|PS|K|X|F|M|Q)\d{1,4}\b"
COMPONENT_TERMS = [
    "MCB", "MCCB", "ELCB", "RCCB", "PLC", "POWER SUPPLY", "CONTACTOR",
    "RELAY", "TERMINAL BLOCK", "EARTHING", "PE", "FUSE", "INDICATOR",
    "PUSHBUTTON", "PUSH BUTTON",
]

DEFAULT_COMPONENT_ALIASES = {
    "MCB":          ["MCB", "MC8", "MCS"],
    "POWER SUPPLY": ["POWER SUPPLY", "POWERSUPPLY", "POWER-SUPPLY", "PS"],
    "PLC":          ["PLC", "PLG"],
    "K1":           ["K1"],
    "K2":           ["K2"],
}

def build_aliases(cfg):
    aliases = {k: list(v) for k, v in DEFAULT_COMPONENT_ALIASES.items()}
    for comp in cfg.get("components", []):
        if comp not in aliases:
            aliases[comp] = [comp]
    return aliases

def _merge_word_boxes(words):
    x = min(w["x"] for w in words)
    y = min(w["y"] for w in words)
    x2 = max(w["x"] + w["w"] for w in words)
    y2 = max(w["y"] + w["h"] for w in words)
    return {
        "text": " ".join(w["text"] for w in words),
        "x": x, "y": y, "w": x2 - x, "h": y2 - y,
        "cx": (x + x2) / 2, "cy": (y + y2) / 2,
        "conf": min(w["conf"] for w in words),
    }

def find_components_in_words(words, img_w, img_h, aliases, min_conf):
    normalized_words = []
    for w in words:
        t = re.sub(r"[^A-Z0-9 ]+", "", w["text"].upper()).strip()
        if not t:
            continue
        normalized_words.append({**w, "norm": t})

    detected = []
    for comp_name, variants in aliases.items():
        matches = []
        for variant in variants:
            vtokens = [t for t in re.split(r"[^A-Z0-9]+", variant.upper()) if t]
            if not vtokens:
                continue

            if len(vtokens) == 1:
                target = vtokens[0]
                for w in normalized_words:
                    if w["norm"] == target or target in w["norm"].split():
                        matches.append((w, w["conf"]))
            else:
                for i in range(len(normalized_words) - len(vtokens) + 1):
                    window = normalized_words[i:i + len(vtokens)]
                    if all(window[j]["norm"] == vtokens[j] for j in range(len(vtokens))):
                        merged = _merge_word_boxes(window)
                        matches.append((merged, min(w["conf"] for w in window)))

        if not matches:
            continue

        best, conf = max(matches, key=lambda x: x[1])
        cx_norm = best["cx"] / img_w
        cy_norm = best["cy"] / img_h
        detected.append({
            "name": comp_name,
            "bbox": {
                "x": float(best["x"]), "y": float(best["y"]),
                "w": float(best["w"]), "h": float(best["h"]),
            },
            "center": {"x": float(best["cx"]), "y": float(best["cy"])},
            "position": {"x": float(cx_norm), "y": float(cy_norm)},
            "region": classify_region(cx_norm, cy_norm),
            "confidence": float(conf),
            "matched_text": best["text"],
        })
    return detected

def _extract_dynamic_components(words, img_w, img_h, existing_names, min_conf):
    dynamic = []
    existing_norm = {normalize(n) for n in existing_names}
    for w in words:
        if w["conf"] < min_conf:
            continue
        t = normalize(w["text"])
        if not t or len(t) < 2 or len(t) > 15:
            continue
        is_designator = re.fullmatch(DESIGNATOR_PATTERN, t) is not None
        is_term = t in [normalize(x) for x in COMPONENT_TERMS]
        if not (is_designator or is_term):
            continue
        if t in existing_norm:
            continue
        cx_norm = w["cx"] / img_w
        cy_norm = w["cy"] / img_h
        dynamic.append({
            "name": t,
            "bbox": {"x": float(w["x"]), "y": float(w["y"]),
                     "w": float(w["w"]), "h": float(w["h"])},
            "center": {"x": float(w["cx"]), "y": float(w["cy"])},
            "position": {"x": float(cx_norm), "y": float(cy_norm)},
            "region": classify_region(cx_norm, cy_norm),
            "confidence": float(w["conf"]),
            "matched_text": w["text"],
            "dynamically_extracted": True,
        })
        existing_norm.add(t)
    return dynamic

 
# IMAGE PREPROCESSING HELPERS
 

def _deskew(gray_img):
    try:
        coords = np.column_stack(np.where(gray_img < 128))
        if len(coords) < 500:
            return gray_img, 0.0
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if abs(angle) < 0.3 or abs(angle) > 20:
            return gray_img, 0.0
        h, w = gray_img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            gray_img, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, float(angle)
    except Exception:
        return gray_img, 0.0

def _preprocess_drawing_image(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray, _ = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10,
    )
    return enhanced, binary

def _detect_panel_boundary(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)
        edged = cv2.dilate(edged, None, iterations=2)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        H, W = img.shape[:2]
        img_area = float(H * W)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.25:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")
        return None
    except Exception:
        return None

def _order_corners(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")

def _four_point_transform(img, pts):
    rect = _order_corners(pts)
    (tl, tr, br, bl) = rect
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    max_w = int(max(w1, w2))
    max_h = int(max(h1, h2))
    if max_w < 50 or max_h < 50:
        return img, False
    dst = np.array([[0, 0], [max_w - 1, 0],
                    [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))
    return warped, True

def _find_enclosing_rect(binary_img, label_bbox, search_mult=3.0):
    try:
        x, y, w, h = [int(v) for v in label_bbox]
        H, W = binary_img.shape[:2]
        pad_x = max(20, int(w * (search_mult - 1) / 2))
        pad_y = max(20, int(h * (search_mult - 1) / 2))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        roi = binary_img[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0

        contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        label_cx = x + w / 2
        label_cy = y + h / 2
        label_area = max(1, w * h)

        best = None
        best_score = 0.0
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw < w * 1.05 or ch < h * 1.05:
                continue
            area = cw * ch
            if area < label_area * 1.4:
                continue
            ar = max(cw, ch) / max(1, min(cw, ch))
            if ar > 8:
                continue
            cx_abs = x1 + cx + cw / 2
            cy_abs = y1 + cy + ch / 2
            d = ((cx_abs - label_cx) ** 2 + (cy_abs - label_cy) ** 2) ** 0.5
            max_d = max(cw, ch)
            if d > max_d * 1.1:
                continue
            lx, ly = x - x1, y - y1
            contains = (cx <= lx and cy <= ly and
                        cx + cw >= lx + w and cy + ch >= ly + h)
            proximity = 1.0 / (1.0 + d / max_d)
            score = (1.0 if contains else 0.6) * proximity
            if score > best_score:
                best_score = score
                best = (x1 + cx, y1 + cy, cw, ch)

        if best is None:
            return None, 0.0
        return best, min(1.0, best_score)
    except Exception:
        return None, 0.0


 
# OPTIONAL AI MODEL MANAGER
 

def model_status():
    return {
        "yolo_world": {k: v for k, v in _MODEL_STATE["yolo_world"].items()
                           if k != "instance"},
        "florence2": {k: v for k, v in _MODEL_STATE["florence2"].items()
                      if k not in ("instance", "processor")},
    }

def _load_yolo():
    state = _MODEL_STATE["yolo_world"]
    if not state["enabled"]:
        return None
    with MODEL_LOCK:
        if state["loaded"]:
            return state["instance"]
        if state["error"]:
            return None
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(YOLO_MODEL_ID)
            state["instance"] = model
            state["loaded"] = True
            logger.info("YOLO-World loaded: %s", YOLO_MODEL_ID)
            return model
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("YOLO-World could not be loaded")
            return None


def _load_florence():
    state = _MODEL_STATE["florence2"]
    if not state["enabled"]:
        return None, None
    with MODEL_LOCK:
        if state["loaded"]:
            return state["processor"], state["instance"]
        if state["error"]:
            return None, None
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
            processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_ID, torch_dtype=dtype, trust_remote_code=True
            ).to(device)
            model.eval()
            state.update({"processor": processor, "instance": model, "device": device, "loaded": True})
            logger.info("Florence-2 loaded: %s on %s", FLORENCE_MODEL_ID, device)
            return processor, model
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Florence-2 could not be loaded")
            return None, None

def run_yolo_world(image_bgr, component_names, threshold=0.10):
    detector = _load_yolo()
    if detector is None:
        return []
    try:
        labels = list(dict.fromkeys([str(n).strip().lower() for n in component_names if str(n).strip()]))
        if not labels:
            return []
        
        detector.set_classes(labels)
        results = detector.predict(image_bgr, conf=float(threshold), verbose=False)
        
        output = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = labels[cls_id]
                output.append({
                    "label": label,
                    "score": conf,
                    "bbox": {"x": float(x1), "y": float(y1),
                             "w": float(x2 - x1), "h": float(y2 - y1)},
                })
        return output
    except Exception:
        logger.exception("YOLO-World inference failed")
        return []

def run_florence_caption(image_bgr):
    processor, model = _load_florence()
    if processor is None or model is None:
        return None
    try:
        import torch
        from PIL import Image
        state = _MODEL_STATE["florence2"]
        device = state.get("device", "cpu")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=pil, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=180, num_beams=3)
        decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            decoded, task=prompt, image_size=(pil.width, pil.height)
        )
        return str(parsed.get(prompt, parsed))[:1500]
    except Exception:
        logger.exception("Florence-2 inference failed")
        return None

def _canonical_model_label(label, component_names):
    norm = normalize(label)
    for name in component_names:
        if normalize(name) == norm:
            return name
    for name in component_names:
        n = normalize(name)
        if n and (n in norm or norm in n):
            return name
    return None

 
# IMAGE ANALYSIS (Phase 2) - DUAL OCR ENGINE
 

def analyse_image(path, cfg):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Unable to read cabinet image")

    H0, W0 = image.shape[:2]

    alignment_mode = "full_image"
    panel_confidence = 0.0
    panel_pts = _detect_panel_boundary(image)
    if panel_pts is not None:
        warped, ok = _four_point_transform(image, panel_pts)
        if ok and warped.shape[0] >= 200 and warped.shape[1] >= 300:
            image = warped
            alignment_mode = "perspective_corrected"
            panel_confidence = 0.85
        else:
            alignment_mode = "panel_boundary_low_quality"
            panel_confidence = 0.35

    H, W = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    scale = 2
    processed = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    words = []
    
    # --- 1. Tesseract (Fast CPU OCR Backup) ---
    if TESSERACT_OK:
        try:
            tess_processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            data = pytesseract.image_to_data(tess_processed, config="--psm 11", output_type=TESS_OUTPUT.DICT)
            for i, t in enumerate(data["text"]):
                if not t.strip(): continue
                conf = float(data["conf"][i]) / 100.0
                if conf < 0.1: continue
                x = data["left"][i] / scale
                y = data["top"][i] / scale
                w = data["width"][i] / scale
                h = data["height"][i] / scale
                words.append({
                    "text": t.strip(), "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                    "cx": x + w / 2, "cy": y + h / 2, "conf": conf, "source": "tesseract"
                })
        except Exception as e:
            logger.warning("Tesseract issue: %s", e)

    # --- 2. EasyOCR (Primary GPU OCR) ---
    if EASYOCR_OK:
        try:
            reader = get_ocr_reader()
            data = reader.readtext(processed)
            for (bbox, text, conf) in data:
                if conf < 0.1 or not text.strip(): continue
                x = min([p[0] for p in bbox]) / scale
                y = min([p[1] for p in bbox]) / scale
                w = (max([p[0] for p in bbox]) - x*scale) / scale
                h = (max([p[1] for p in bbox]) - y*scale) / scale
                words.append({
                    "text": text.strip(), "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                    "cx": float(x + w / 2), "cy": float(y + h / 2), "conf": float(conf), "source": "easyocr"
                })
        except Exception:
            logger.exception("EasyOCR failed for %s", path)

    # Combine all words - Our matching logic will naturally pick the highest confidence text from either OCR engine
    ocr_text = " ".join([w["text"] for w in words])

    aliases = build_aliases(cfg)
    min_conf = float(cfg.get("thresholds", {}).get("ocr_confidence", 0.35))
    raw_components = find_components_in_words(words, W, H, aliases, min_conf)

    components = []
    for comp in raw_components:
        lb = (int(comp["bbox"]["x"]), int(comp["bbox"]["y"]),
              int(comp["bbox"]["w"]), int(comp["bbox"]["h"]))
        region_bbox, region_conf = _find_enclosing_rect(gray, lb)

        if region_bbox is not None and region_conf >= 0.35:
            rx, ry, rw, rh = region_bbox
            cx = rx + rw / 2
            cy = ry + rh / 2
            region_dict = {"x": float(rx), "y": float(ry),
                           "w": float(rw), "h": float(rh)}
        else:
            cx = comp["center"]["x"]
            cy = comp["center"]["y"]
            region_dict = None
            region_conf = 0.0

        cx_norm = cx / W
        cy_norm = cy / H
        components.append({
            **comp,
            "region_bbox": region_dict,
            "region_confidence": float(region_conf),
            "center": {"x": float(cx), "y": float(cy)},
            "position": {"x": float(cx_norm), "y": float(cy_norm)},
            "region": classify_region(cx_norm, cy_norm),
        })

    known_names = [c["name"] for c in components]
    dynamic = _extract_dynamic_components(words, W, H, known_names, min_conf)
    for d in dynamic:
        d["region_bbox"] = None
        d["region_confidence"] = 0.0
        components.append(d)

    component_names = list(aliases.keys())
    yolo_threshold = float(cfg.get("thresholds", {}).get("yolo_confidence", 0.15))
    yolo_detections = run_yolo_world(image, component_names, yolo_threshold)
    yolo_used = []
    yolo_best = {}
    for det in yolo_detections:
        canonical = _canonical_model_label(det.get("label", ""), component_names)
        if not canonical:
            continue
        if canonical not in yolo_best or float(det.get("score", 0.0)) > float(yolo_best[canonical].get("score", 0.0)):
            yolo_best[canonical] = det
            
    for canonical, det in yolo_best.items():
        existing = next((c for c in components if c.get("name") == canonical), None)
        box = det["bbox"]
        cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
        region = {"x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"]}
        if existing is None:
            existing = {
                "name": canonical, "bbox": region,
                "center": {"x": cx, "y": cy},
                "position": {"x": cx / W, "y": cy / H},
                "region": classify_region(cx / W, cy / H),
                "confidence": float(det["score"]), "ocr_confidence": 0.0,
                "matched_text": "", "region_bbox": region,
                "region_confidence": float(det["score"]),
                "detection_source": "yolo_world_only",
            }
            components.append(existing)
        else:
            existing["label_bbox"] = existing.get("bbox")
            existing["bbox"] = region
            existing["region_bbox"] = region
            existing["center"] = {"x": cx, "y": cy}
            existing["position"] = {"x": cx / W, "y": cy / H}
            existing["region"] = classify_region(cx / W, cy / H)
            existing["region_confidence"] = float(det["score"])
            existing["detection_source"] = "yolo_world+ocr"
            existing["yolo_confidence"] = float(det["score"])
        yolo_used.append({"name": canonical, "score": float(det["score"]), "bbox": region})

    aligned_path = path.parent / "analysis_image.jpg"
    cv2.imwrite(str(aligned_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    florence_caption = run_florence_caption(image)

    image_quality = "Readable" if (W >= 500 and H >= 300) else "Low resolution"

    return {
        "width": int(W), "height": int(H),
        "original_width": int(W0), "original_height": int(H0),
        "ocr_text": ocr_text,
        "detected_components": components,
        "detected_labels": [c["name"] for c in components],
        "image_quality": image_quality,
        "ocr_available": (EASYOCR_OK or TESSERACT_OK),
        "alignment_mode": alignment_mode,
        "panel_confidence": panel_confidence,
        "yolo_available": _MODEL_STATE["yolo_world"]["loaded"],
        "yolo_detections": yolo_used,
        "florence_caption": florence_caption,
        "analysis_image": str(aligned_path),
    }

 
# PDF / DRAWING ANALYSIS (Phase 1)
 

def _search_component_rect(page, comp_name):
    flags = getattr(pymupdf, "TEXT_IGNORECASE", 0)
    try:
        rects = page.search_for(comp_name, flags=flags)
    except Exception:
        try:
            rects = page.search_for(comp_name)
        except Exception:
            return None
    if not rects:
        return None
    plausible = [r for r in rects if r.width * r.height > 150]
    pool = plausible or rects
    return max(pool, key=lambda r: r.width * r.height)

def _ocr_words_from_image(img_gray, scale=1.0):
    out = []
    
    if TESSERACT_OK:
        try:
            data = pytesseract.image_to_data(img_gray, config="--psm 11", output_type=TESS_OUTPUT.DICT)
            for i, t in enumerate(data["text"]):
                if not t.strip(): continue
                conf = float(data["conf"][i]) / 100.0
                if conf < 0.1: continue
                x, y = data["left"][i] / scale, data["top"][i] / scale
                w, h = data["width"][i] / scale, data["height"][i] / scale
                out.append({"text": t.strip(), "x": x, "y": y, "w": w, "h": h,
                            "cx": x + w / 2, "cy": y + h / 2, "conf": conf, "source": "tesseract"})
        except Exception:
            pass

    if EASYOCR_OK:
        try:
            reader = get_ocr_reader()
            data = reader.readtext(img_gray)
            for (bbox, text, conf) in data:
                if conf < 0.1 or not text.strip(): continue
                x = min([p[0] for p in bbox]) / scale
                y = min([p[1] for p in bbox]) / scale
                w = (max([p[0] for p in bbox]) - x*scale) / scale
                h = (max([p[1] for p in bbox]) - y*scale) / scale
                out.append({"text": text.strip(), "x": x, "y": y, "w": w, "h": h,
                            "cx": x + w / 2, "cy": y + h / 2, "conf": float(conf), "source": "easyocr"})
        except Exception:
            pass
            
    return out

def _analyse_pdf_page(page, page_index, aliases, min_conf):
    pw, ph = page.rect.width, page.rect.height
    embedded_text = page.get_text("text") or ""

    scale = 300.0 / 72.0
    pix = page.get_pixmap(dpi=300)
    img_array = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img_w = img_h = 0
    words = []
    components = []
    binary = None

    if img is not None and img.size > 0:
        img_h, img_w = img.shape[:2]
        enhanced, binary = _preprocess_drawing_image(img)
        words = _ocr_words_from_image(enhanced, scale=1.0)
        raw_components = find_components_in_words(words, img_w, img_h, aliases, min_conf)

        for comp in raw_components:
            lb = (int(comp["bbox"]["x"]), int(comp["bbox"]["y"]),
                  int(comp["bbox"]["w"]), int(comp["bbox"]["h"]))
            region_bbox, region_conf = _find_enclosing_rect(binary, lb)
            if region_bbox is not None and region_conf >= 0.35:
                rx, ry, rw, rh = region_bbox
                cx_px = rx + rw / 2
                cy_px = ry + rh / 2
                region_dict = {"x": float(rx), "y": float(ry),
                               "w": float(rw), "h": float(rh)}
            else:
                cx_px = comp["center"]["x"]
                cy_px = comp["center"]["y"]
                region_dict = None
                region_conf = 0.0

            cx_norm = cx_px / img_w
            cy_norm = cy_px / img_h
            components.append({
                **comp,
                "region_bbox": region_dict,
                "region_confidence": float(region_conf),
                "center": {"x": float(cx_px), "y": float(cy_px)},
                "position": {"x": float(cx_norm), "y": float(cy_norm)},
                "region": classify_region(cx_norm, cy_norm),
                "position_source": "drawing_ocr",
                "page": page_index + 1,
            })

        known_names = [c["name"] for c in components]
        dynamic = _extract_dynamic_components(words, img_w, img_h, known_names, min_conf)
        for d in dynamic:
            d["region_bbox"] = None
            d["region_confidence"] = 0.0
            d["position_source"] = "drawing_ocr_dynamic"
            d["page"] = page_index + 1
            components.append(d)

    found_names = {c["name"] for c in components}
    for comp_name in aliases.keys():
        if comp_name in found_names:
            continue
        rect = _search_component_rect(page, comp_name)
        if not rect:
            continue
        cx = ((rect.x0 + rect.x1) / 2) / pw
        cy = ((rect.y0 + rect.y1) / 2) / ph
        components.append({
            "name": comp_name,
            "bbox": {"x": float(rect.x0), "y": float(rect.y0),
                     "w": float(rect.width), "h": float(rect.height)},
            "center": {"x": float((rect.x0 + rect.x1) / 2),
                       "y": float((rect.y0 + rect.y1) / 2)},
            "position": {"x": float(cx), "y": float(cy)},
            "region": classify_region(cx, cy),
            "region_bbox": None,
            "region_confidence": 0.0,
            "position_source": "pdf_search",
            "confidence": 0.90,
            "page": page_index + 1,
        })
        found_names.add(comp_name)

    normalized_text = normalize(embedded_text)
    for comp_name in aliases.keys():
        if comp_name in found_names:
            continue
        if comp_name in normalized_text:
            components.append({
                "name": comp_name,
                "bbox": None, "center": None, "position": None,
                "region": None,
                "region_bbox": None, "region_confidence": 0.0,
                "position_source": "text_only",
                "confidence": 0.60,
                "page": page_index + 1,
            })
            found_names.add(comp_name)

    return {
        "page": page_index + 1,
        "text": embedded_text,
        "width": int(img_w or pw),
        "height": int(img_h or ph),
        "components": components,
    }

def analyse_pdf(path, cfg):
    doc = pymupdf.open(str(path))
    pages_data = []
    full_text = ""
    aliases = build_aliases(cfg)
    min_conf = float(cfg.get("thresholds", {}).get("ocr_confidence", 0.35))
    drawing_components = {}

    try:
        for idx, page in enumerate(doc):
            try:
                page_dict = _analyse_pdf_page(page, idx, aliases, min_conf)
            except Exception:
                logger.exception("Failed to analyse PDF page %d", idx + 1)
                page_dict = {"page": idx + 1, "text": "", "width": 0,
                             "height": 0, "components": []}
            pages_data.append(page_dict)
            full_text += "\n" + page_dict["text"]
            for comp in page_dict["components"]:
                name = comp["name"]
                prev = drawing_components.get(name)
                if prev is None or comp.get("confidence", 0) > prev.get("confidence", 0):
                    drawing_components[name] = comp
    finally:
        doc.close()

    drawing_match = re.search(r"\bGA[-\s]?\d+\b", full_text, re.IGNORECASE)
    revision_match = re.search(
        r"\b(?:REV(?:ISION)?)[\s.:_-]*([A-Z0-9]+)\b", full_text, re.IGNORECASE
    )

    return {
        "page_count": len(pages_data),
        "text": full_text,
        "pages": pages_data,
        "detected_components": list(drawing_components.values()),
        "drawing_number": drawing_match.group(0) if drawing_match else None,
        "revision": revision_match.group(1).upper() if revision_match else None,
        "text_extracted": bool(full_text.strip()),
    }

 
# ALIGNMENT (Phase 3)
 

def align_panel_images(drawing_data, image_data, cfg):
    alignment = {
        "drawing_page_count": drawing_data.get("page_count", 0),
        "image_alignment_mode": image_data.get("alignment_mode", "unknown"),
        "image_panel_confidence": float(image_data.get("panel_confidence", 0.0)),
        "image_size": [image_data.get("width", 0), image_data.get("height", 0)],
        "drawing_size": [
            drawing_data.get("pages", [{}])[0].get("width", 0) if drawing_data.get("pages") else 0,
            drawing_data.get("pages", [{}])[0].get("height", 0) if drawing_data.get("pages") else 0,
        ],
    }
    dc = drawing_data.get("detected_components", [])
    ic = image_data.get("detected_components", [])
    positioned_d = sum(1 for c in dc if c.get("position"))
    positioned_i = sum(1 for c in ic if c.get("position"))
    alignment["drawing_components_with_position"] = positioned_d
    alignment["image_components_with_position"] = positioned_i

    panel_conf = alignment["image_panel_confidence"]
    if panel_conf >= 0.7 and positioned_d >= 1 and positioned_i >= 1:
        alignment["confidence"] = "high"
        alignment["comparable"] = True
        alignment["reason"] = "Panel detected and components localized in both sources."
    elif panel_conf >= 0.35 and positioned_d >= 1 and positioned_i >= 1:
        alignment["confidence"] = "medium"
        alignment["comparable"] = True
        alignment["reason"] = "Panel detection partial; positions compared with reduced confidence."
    else:
        alignment["confidence"] = "low"
        alignment["comparable"] = False
        alignment["reason"] = (
            "Panel alignment unreliable (image or drawing could not be reliably localized). "
            "Position mismatches and swaps will not be reported as definitive."
        )
    return alignment

 
# CHECKLIST ANALYSIS
 

def analyse_checklist(path):
    ext = extension(path.name)
    if ext == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sections"), list):
            checks = []
            for section in data["sections"]:
                sname = str(section.get("section_name", "General"))
                for item in section.get("checks", []):
                    checks.append({**item, "section_name": sname})
        else:
            checks = data.get("checks", []) if isinstance(data, dict) else []
        if not isinstance(checks, list):
            raise ValueError("QC checklist must contain checks[] or sections[].checks[]")
    elif ext == "csv":
        checks = pd.read_csv(path).fillna("").to_dict(orient="records")
    elif ext == "xlsx":
        sheets = pd.read_excel(path, sheet_name=None)
        if not sheets:
            raise ValueError("QC Excel has no sheets")
        checks = pd.concat(sheets.values(), ignore_index=True).fillna("").to_dict(orient="records")
    else:
        raise ValueError("Unsupported QC checklist format")

    normalized = []
    for i, chk in enumerate(checks, 1):
        cid = str(chk.get("id") or chk.get("tag") or f"CHECK-{i:03d}").strip()
        name = str(chk.get("name") or chk.get("criteria") or cid).strip()
        status_val = str(chk.get("status", "")).strip().lower()
        enabled_val = chk.get("enabled", True)
        if isinstance(enabled_val, str):
            enabled_val = enabled_val.strip().lower() not in ("false", "0", "no", "off", "disabled")
        normalized.append({
            "id": cid,
            "name": name,
            "section": str(chk.get("section_name", "General")),
            "enabled": bool(enabled_val) and status_val not in ("skip", "disabled"),
            "source_status": chk.get("status", "Not checked"),
        })
    return normalized

 
# COMPONENT MATCHING + SPATIAL COMPARISON
 

def compute_relative_positions(components):
    positioned = [c for c in components if c.get("position")]
    if len(positioned) < 2:
        for c in components:
            if c.get("position"):
                c["relative_position"] = {"x": 0.5, "y": 0.5}
        return components
    xs = [c["position"]["x"] for c in positioned]
    ys = [c["position"]["y"] for c in positioned]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max_x - min_x or 1.0
    ry = max_y - min_y or 1.0
    for c in components:
        if c.get("position"):
            c["relative_position"] = {
                "x": (c["position"]["x"] - min_x) / rx,
                "y": (c["position"]["y"] - min_y) / ry,
            }
    return components

def detect_swaps(component_results, tolerance, min_conf):
    named = {}
    for c in component_results:
        if (c.get("detected_present") and c.get("position")
                and c.get("expected_position")
                and c.get("confidence", 0) >= min_conf
                and c.get("position_status") not in ("UNKNOWN",)):
            named[c["name"]] = c

    names = list(named.keys())
    swaps = []
    swap_tol = min(tolerance * 0.8, 0.18)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, cb = named[a], named[b]
            ea, eb = ca["expected_position"], cb["expected_position"]
            ra, rb = ca["position"], cb["position"]

            d_a_eb = ((ra["x"] - eb["x"]) ** 2 + (ra["y"] - eb["y"]) ** 2) ** 0.5
            d_b_ea = ((rb["x"] - ea["x"]) ** 2 + (rb["y"] - ea["y"]) ** 2) ** 0.5
            d_a_ea = ((ra["x"] - ea["x"]) ** 2 + (ra["y"] - ea["y"]) ** 2) ** 0.5
            d_b_eb = ((rb["x"] - eb["x"]) ** 2 + (rb["y"] - eb["y"]) ** 2) ** 0.5

            swapped = (d_a_eb <= swap_tol and d_b_ea <= swap_tol
                       and (d_a_ea > tolerance * 0.5 or d_b_eb > tolerance * 0.5))
            if swapped:
                swaps.append((a, b))
    return swaps

def compare_components(image_data, drawing_data, cfg, alignment=None):
    image_comps = {c["name"]: c for c in image_data.get("detected_components", [])}
    drawing_comps = {c["name"]: c for c in drawing_data.get("detected_components", [])}

    image_list = list(image_comps.values())
    compute_relative_positions(image_list)
    image_comps = {c["name"]: c for c in image_list}

    drawing_list = list(drawing_comps.values())
    compute_relative_positions(drawing_list)
    drawing_comps = {c["name"]: c for c in drawing_list}

    thresh = cfg.get("thresholds", {})
    min_ocr_conf = float(thresh.get("ocr_confidence", 0.35))
    tolerance = float(thresh.get("position_tolerance", 0.25))
    missing_severity = str(thresh.get("missing_severity", "FAIL")).upper()
    align_ok = bool(alignment and alignment.get("comparable", True))

    results = []
    missing_components = []
    position_checks = []

    drawing_names = set(drawing_comps.keys())
    image_names = set(image_comps.keys())

    detected_ok_count = len([n for n in image_names
                             if image_comps[n].get("confidence", 0) >= min_ocr_conf])

    for name in sorted(drawing_names):
        dcomp = drawing_comps[name]
        icomp = image_comps.get(name)
        expected_pos = dcomp.get("position")
        expected_rel = dcomp.get("relative_position")

        if icomp is None:
            if detected_ok_count >= 2 and missing_severity == "FAIL":
                status = "FAIL"
                reason = (f"{name} not detected in the cabinet photograph. "
                          f"Other components were detected with acceptable OCR confidence, "
                          f"so this label may be genuinely missing — manual verification required.")
            else:
                status = "REVIEW"
                reason = (f"{name} is required by the drawing but was not detected in the image. "
                          f"OCR may have missed it — manual verification required.")

            results.append({
                "id": f"COMP_{safe_name(name)}",
                "name": name,
                "status": status,
                "reason": reason,
                "drawing_present": True,
                "detected_present": False,
                "confidence": 0.0,
                "position_status": "MISSING",
                "expected_position": expected_pos,
                "actual_position": None,
                "swapped_with": None,
                "position_source": dcomp.get("position_source", "unknown"),
                "evidence": [],
            })
            missing_components.append(name)
            continue

        confidence = float(icomp.get("confidence", 0.0))
        actual_pos = icomp.get("position")
        actual_rel = icomp.get("relative_position")

        position_status = "UNKNOWN"
        position_reason = ""
        if not align_ok:
            position_status = "UNKNOWN"
            position_reason = "Panel alignment unreliable — position comparison skipped."
        elif expected_rel and actual_rel:
            dx = abs(expected_rel["x"] - actual_rel["x"])
            dy = abs(expected_rel["y"] - actual_rel["y"])
            dist = (dx * dx + dy * dy) ** 0.5
            src = dcomp.get("position_source", "unknown")
            if src in ("drawing_ocr", "drawing_ocr_dynamic", "pdf_search"):
                if dist <= tolerance:
                    position_status = "MATCH"
                else:
                    position_status = "MISMATCH"
                    position_reason = (
                        f"Relative layout differs from drawing by {dist:.2f} (normalized)."
                    )
            else:
                position_status = "UNKNOWN"
                position_reason = "Drawing position could not be reliably extracted."
        elif expected_pos is None:
            position_status = "UNKNOWN"
            position_reason = "Expected position not available from the drawing."
        else:
            position_status = "UNKNOWN"
            position_reason = "Actual position not available."

        if icomp.get("detection_source") == "yolo_world_only":
            status = "REVIEW"
            reason = f"{name} object candidate detected by YOLO-World, but its printed identity was not confirmed by OCR."
        elif confidence < min_ocr_conf:
            status = "REVIEW"
            reason = f"{name} label detected but OCR confidence is low ({confidence:.2f})."
        elif icomp.get("region_bbox") is None:
            status = "REVIEW"
            reason = f"{name} detected; physical component boundary uncertain — manual verification."
        elif position_status == "MATCH":
            status = "PASS"
            reason = f"{name} detected and position matches the drawing."
        elif position_status == "MISMATCH":
            status = "FAIL"
            reason = f"{name} detected but appears misplaced. {position_reason}"
        else:
            status = "REVIEW"
            reason = f"{name} detected; {position_reason or 'position could not be verified'}."

        results.append({
            "id": f"COMP_{safe_name(name)}",
            "name": name,
            "status": status,
            "reason": reason,
            "drawing_present": True,
            "detected_present": True,
            "confidence": confidence,
            "position_status": position_status,
            "expected_position": expected_pos,
            "actual_position": actual_pos,
            "expected_relative_position": expected_rel,
            "relative_position": actual_rel,
            "swapped_with": None,
            "position_source": dcomp.get("position_source", "unknown"),
            "region_confidence": float(icomp.get("region_confidence", 0.0)),
            "evidence": [],
            "image_bbox": icomp.get("bbox"),
            "image_region_bbox": icomp.get("region_bbox"),
        })

    for name in sorted(image_names - drawing_names):
        icomp = image_comps[name]
        conf = float(icomp.get("confidence", 0.0))
        if conf < min_ocr_conf:
            continue
        results.append({
            "id": f"EXTRA_{safe_name(name)}",
            "name": name,
            "status": "REVIEW",
            "reason": f"{name} detected in the photograph but not present in the drawing — verify.",
            "drawing_present": False,
            "detected_present": True,
            "confidence": conf,
            "position_status": "UNEXPECTED",
            "expected_position": None,
            "actual_position": icomp.get("position"),
            "swapped_with": None,
            "position_source": "none",
            "region_confidence": float(icomp.get("region_confidence", 0.0)),
            "evidence": [],
            "image_bbox": icomp.get("bbox"),
            "image_region_bbox": icomp.get("region_bbox"),
        })

    if align_ok:
        swaps = detect_swaps(results, tolerance, min_ocr_conf)
    else:
        swaps = []
    for a, b in swaps:
        for r in results:
            if r["name"] == a:
                r["swapped_with"] = b
                r["position_status"] = "MISMATCH"
                r["status"] = "FAIL"
                r["reason"] = f"{a} appears swapped with {b} compared with the drawing."
            elif r["name"] == b:
                r["swapped_with"] = a
                r["position_status"] = "MISMATCH"
                r["status"] = "FAIL"
                r["reason"] = f"{b} appears swapped with {a} compared with the drawing."

    swapped_strings = [f"{a} ⇄ {b}" for a, b in swaps]

    for r in results:
        if r.get("expected_position") or r.get("actual_position"):
            position_checks.append({
                "component": r["name"],
                "expected": r.get("expected_position"),
                "actual": r.get("actual_position"),
                "status": r.get("position_status"),
                "reason": r.get("reason"),
            })

    return results, missing_components, swapped_strings, position_checks

 
# QC CHECKLIST DERIVATION
 

def derive_qc_status(check, component_results, image_data, drawing_data, alignment=None):
    text = normalize(f"{check['id']} {check['name']}")

    if "PRESENCE" in text:
        failed = [c for c in component_results if c.get("detected_present") is False]
        passed = [c for c in component_results if c.get("detected_present") is True]
        unverified = [c for c in passed if c.get("detection_source") == "yolo_world_only"]
        if failed:
            names = ", ".join(c["name"] for c in failed)
            return "FAIL", f"{len(failed)} drawing component(s) were not identified in the cabinet image: {names}. Confirm manually because OCR can miss labels."
        if unverified:
            names = ", ".join(c["name"] for c in unverified)
            return "REVIEW", f"Object candidates found but printed identity is unconfirmed for: {names}."
        if passed:
            return "PASS", f"OCR/component association found all {len(passed)} expected component label(s)."
        return "REVIEW", "No component labels could be reliably detected."

    if "POSITION" in text or "ARRANGEMENT" in text:
        if alignment and not alignment.get("comparable", True):
            return "REVIEW", "Panel alignment unreliable — position comparison deferred."
        bad = [c for c in component_results if c.get("position_status") == "MISMATCH"]
        good = [c for c in component_results if c.get("position_status") == "MATCH"]
        if bad:
            names = ", ".join(c["name"] for c in bad)
            return "FAIL", f"{len(bad)} component(s) appear misplaced: {names}."
        if good and len(good) == len([c for c in component_results if c.get("detected_present")]):
            return "PASS", "All detected components are in the expected spatial regions."
        return "REVIEW", "Some components could not be spatially verified."

    if "ORDER" in text or "SWAP" in text:
        swapped = [c for c in component_results if c.get("swapped_with")]
        if swapped:
            pairs = set()
            for c in swapped:
                pairs.add(tuple(sorted([c["name"], c["swapped_with"]])))
            return "FAIL", f"Component swap(s): {', '.join(f'{a} ⇄ {b}' for a, b in pairs)}."
        if len([c for c in component_results if c.get("position_status") == "MATCH"]) >= 2:
            return "PASS", "Component ordering matches the drawing."
        return "REVIEW", "Component ordering could not be fully verified."

    if "LABEL" in text:
        detected = [c for c in component_results if c.get("detected_present")]
        low = [c["name"] for c in detected
               if float(c.get("confidence", 0.0)) < 0.45
               or c.get("detection_source") == "yolo_world_only"]
        if not detected:
            return "REVIEW", "No component labels could be reliably read from the cabinet image."
        if low:
            return "REVIEW", "Label identity needs manual verification for: " + ", ".join(low) + "."
        return "PASS", f"OCR read {len(detected)} component label(s); identity is not a substitute for wiring/terminal verification."

    if any(w in text for w in ("CONDITION", "MOUNTING", "WIRING", "TERMINAL",
                               "FERRULE", "TORQUE", "CLEARANCE", "DAMAGE", "WEAR")):
        return "REVIEW", ("This criterion requires physical inspection or close-up imagery; "
                          "cannot be verified from an overview photograph.")

    if "IMAGE QUALITY" in text:
        q = image_data.get("image_quality", "")
        w = image_data.get("width", 0)
        h = image_data.get("height", 0)
        if q == "Readable":
            return "PASS", f"Image resolution: {w} × {h}."
        return "REVIEW", f"Low image resolution ({w} × {h})."

    if "REVISION" in text:
        rev = drawing_data.get("revision")
        if rev:
            return "PASS", f"Drawing revision detected: {rev}."
        return "REVIEW", "Drawing revision could not be extracted from the PDF."

    if "DRAWING" in text:
        dc = drawing_data.get("detected_components", [])
        if dc:
            return "PASS", f"{len(dc)} component label(s) extracted from the drawing."
        return "REVIEW", "No component labels could be extracted from the drawing."

    return "REVIEW", "No automated verification rule is configured for this checklist item."

 
# EVIDENCE GENERATION
 

def generate_evidence_images(inspection_id, image_path, component_results,
                             missing_components, swap_strings):
    try:
        ev_dir = evidence_dir(inspection_id)
    except Exception as e:
        logger.error("Evidence dir failed: %s", e)
        return [], {}

    aligned_path = image_path.parent / "analysis_image.jpg"
    evidence_source = aligned_path if aligned_path.exists() else image_path
    img = cv2.imread(str(evidence_source))
    if img is None:
        return [], {}

    H, W = img.shape[:2]
    annotated = img.copy()
    evidence_list = []
    per_result_evidence = {}

    def _region_box(r):
        return r.get("image_region_bbox") or r.get("image_bbox")

    for r in component_results:
        box = _region_box(r)
        if not r.get("detected_present") or not box:
            continue
        x1, y1 = max(0, int(box["x"])), max(0, int(box["y"]))
        x2, y2 = min(W, int(box["x"] + box["w"])), min(H, int(box["y"] + box["h"]))
        color = (60, 220, 90) if r["status"] == "PASS" else (
            (60, 60, 220) if r["status"] == "FAIL" else (60, 200, 230)
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{r['name']} ({r.get('confidence', 0):.2f})"
        cv2.putText(annotated, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for name in missing_components:
        row = next((c for c in component_results if c["name"] == name), None)
        if not row or not row.get("expected_position"):
            continue
        ex = int(row["expected_position"]["x"] * W)
        ey = int(row["expected_position"]["y"] * H)
        cv2.rectangle(annotated, (ex - 55, ey - 40), (ex + 55, ey + 40), (60, 60, 220), 3)
        cv2.putText(annotated, f"MISSING: {name}", (max(10, ex - 55), max(20, ey - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 2)

    drawn_pairs = set()
    for r in component_results:
        if not r.get("swapped_with"):
            continue
        pair = tuple(sorted([r["name"], r["swapped_with"]]))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)
        partner = next((c for c in component_results
                        if c["name"] == r["swapped_with"]), None)
        if not partner:
            continue
        b1, b2 = _region_box(r), _region_box(partner)
        if not b1 or not b2:
            continue
        c1 = (int(b1["x"] + b1["w"] / 2), int(b1["y"] + b1["h"] / 2))
        c2 = (int(b2["x"] + b2["w"] / 2), int(b2["y"] + b2["h"] / 2))
        cv2.arrowedLine(annotated, c1, c2, (60, 180, 240), 2, tipLength=0.05)
        cv2.arrowedLine(annotated, c2, c1, (240, 180, 60), 2, tipLength=0.05)

    ann_path = ev_dir / "annotated.jpg"
    cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    ann_url = f"/api/inspections/{inspection_id}/evidence/annotated.jpg"
    evidence_list.append({"url": ann_url, "thumbnail": ann_url,
                          "label": "Annotated cabinet image"})

    for r in component_results:
        if not r.get("detected_present"):
            continue
        box = _region_box(r)
        if not box:
            continue
        pad = 20
        x1 = max(0, int(box["x"] - pad)); y1 = max(0, int(box["y"] - pad))
        x2 = min(W, int(box["x"] + box["w"] + pad)); y2 = min(H, int(box["y"] + box["h"] + pad))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        fname = f"crop_{safe_name(r['name'])}.jpg"
        cv2.imwrite(str(ev_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        url = f"/api/inspections/{inspection_id}/evidence/{fname}"
        ev_obj = {"url": url, "thumbnail": url, "label": f"{r['name']} detected"}
        evidence_list.append(ev_obj)
        per_result_evidence.setdefault(r["name"], []).append(ev_obj)

    for name in missing_components:
        row = next((c for c in component_results if c["name"] == name), None)
        if not row or not row.get("expected_position"):
            continue
        ex = int(row["expected_position"]["x"] * W)
        ey = int(row["expected_position"]["y"] * H)
        pad = 80
        x1 = max(0, ex - pad); y1 = max(0, ey - pad)
        x2 = min(W, ex + pad); y2 = min(H, ey + pad)
        region = img[y1:y2, x1:x2].copy()
        if region.size == 0:
            continue
        cv2.rectangle(region, (ex - x1 - 50, ey - y1 - 38),
                      (ex - x1 + 50, ey - y1 + 38), (60, 60, 220), 3)
        fname = f"missing_{safe_name(name)}.jpg"
        cv2.imwrite(str(ev_dir / fname), region, [cv2.IMWRITE_JPEG_QUALITY, 88])
        url = f"/api/inspections/{inspection_id}/evidence/{fname}"
        ev_obj = {"url": url, "thumbnail": url, "label": f"Missing region: {name}"}
        evidence_list.append(ev_obj)
        per_result_evidence.setdefault(name, []).append(ev_obj)

    for a, b in drawn_pairs:
        ra = next((c for c in component_results if c["name"] == a), None)
        rb = next((c for c in component_results if c["name"] == b), None)
        if not ra or not rb:
            continue
        ba, bb = _region_box(ra), _region_box(rb)
        if not ba or not bb:
            continue

        def _crop_pad(box, pad=30):
            x1 = max(0, int(box["x"] - pad)); y1 = max(0, int(box["y"] - pad))
            x2 = min(W, int(box["x"] + box["w"] + pad)); y2 = min(H, int(box["y"] + box["h"] + pad))
            return img[y1:y2, x1:x2]

        ca, cb = _crop_pad(ba), _crop_pad(bb)
        if ca.size == 0 or cb.size == 0:
            continue
        
        target_h = max(ca.shape[0], cb.shape[0], 220)
        def _resize_h(im, hh):
            if im.shape[0] == hh:
                return im
            r = hh / im.shape[0]
            return cv2.resize(im, (int(im.shape[1] * r), hh))
        ca, cb = _resize_h(ca, target_h), _resize_h(cb, target_h)
        gap_px = 20
        canvas_w = ca.shape[1] + cb.shape[1] + gap_px * 3
        canvas_h = target_h + 70
        canvas_img = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
        canvas_img[60:60 + target_h, gap_px:gap_px + ca.shape[1]] = ca
        canvas_img[60:60 + target_h, gap_px * 2 + ca.shape[1]:gap_px * 2 + ca.shape[1] + cb.shape[1]] = cb
        cv2.putText(canvas_img, f"{a} detected", (gap_px, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(canvas_img, f"{b} detected", (gap_px * 2 + ca.shape[1], 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.putText(canvas_img, "SWAPPED", (canvas_w // 2 - 60, canvas_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 220), 2)
        fname = f"swap_{safe_name(a)}__{safe_name(b)}.jpg"
        cv2.imwrite(str(ev_dir / fname), canvas_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        url = f"/api/inspections/{inspection_id}/evidence/{fname}"
        ev_obj = {"url": url, "thumbnail": url,
                  "label": f"Swap comparison: {a} ⇄ {b}"}
        evidence_list.append(ev_obj)
        per_result_evidence.setdefault(a, []).append(ev_obj)
        per_result_evidence.setdefault(b, []).append(ev_obj)

    return evidence_list, per_result_evidence

 
# INSPECTION PIPELINE
 

def process_inspection(inspection_id):
    cfg = get_config()
    folder = inspection_dir(inspection_id)

    image_path = find_input(folder, "cabinet_image")
    pdf_path = find_input(folder, "engineering_drawing")
    checklist_path = find_input(folder, "qc_checklist")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_image = executor.submit(analyse_image, image_path, cfg)
        future_drawing = executor.submit(analyse_pdf, pdf_path, cfg)
        
        image_data = future_image.result()
        drawing_data = future_drawing.result()

    checklist_data = analyse_checklist(checklist_path)
    if not checklist_data:
        checklist_data = [
            {"id": cid, "name": name, "section": "General",
             "enabled": True, "source_status": "Not checked"}
            for cid, name in DEFAULT_CHECKS
        ]

    alignment = align_panel_images(drawing_data, image_data, cfg)

    component_results, missing_components, swapped_strings, position_checks = \
        compare_components(image_data, drawing_data, cfg, alignment)

    qc_results = []
    for check in checklist_data:
        if not check["enabled"]:
            continue
        status, reason = derive_qc_status(check, component_results,
                                          image_data, drawing_data, alignment)
        qc_results.append({
            "id": check["id"],
            "name": check["name"],
            "status": status,
            "reason": reason,
            "section": check["section"],
            "drawing_present": None,
            "detected_present": None,
            "position_status": None,
            "evidence": [],
        })

    evidence_list, per_result_evidence = generate_evidence_images(
        inspection_id, image_path, component_results,
        missing_components, swapped_strings
    )
    for r in component_results:
        r["evidence"] = per_result_evidence.get(r["name"], [])

    combined_results = component_results + qc_results

    total = len(combined_results)
    passed = sum(1 for r in combined_results if r["status"] == "PASS")
    failed = sum(1 for r in combined_results if r["status"] == "FAIL")
    review = sum(1 for r in combined_results if r["status"] == "REVIEW")

    if failed > 0:
        overall = "FAIL"
        message = f"{failed} check(s) failed — review the evidence and correct before sign-off."
    elif review > 0:
        overall = "REVIEW"
        message = f"{review} check(s) require manual verification."
    elif total > 0:
        overall = "PASS"
        message = "All automated checks passed."
    else:
        overall = "REVIEW"
        message = "No checks could be executed — verify input files."

    ocr_results = {
        "image": {
            "detected_labels": image_data["detected_labels"],
            "detected_components": image_data["detected_components"],
            "raw_text_preview": image_data["ocr_text"][:2000],
            "image_size": [image_data["width"], image_data["height"]],
            "ocr_available": image_data["ocr_available"],
            "alignment_mode": image_data.get("alignment_mode"),
            "panel_confidence": image_data.get("panel_confidence"),
            "yolo_available": image_data.get("yolo_available", False),
            "yolo_detections": image_data.get("yolo_detections", []),
            "florence_caption": image_data.get("florence_caption"),
        },
        "drawing": {
            "detected_components": drawing_data["detected_components"],
            "drawing_number": drawing_data["drawing_number"],
            "revision": drawing_data["revision"],
            "text_extracted": drawing_data["text_extracted"],
            "page_count": drawing_data["page_count"],
        },
    }

    project_name = "Control Panel Inspection"
    inspection_ref = ""
    meta_path = folder / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            project_name = meta.get("project_name") or project_name
            inspection_ref = meta.get("inspection_reference", "")
        except Exception:
            pass

    report = {
        "inspection_id": inspection_id,
        "project_name": project_name,
        "inspection_reference": inspection_ref,
        "processed_at": datetime.now().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall,
        "message": message,
        "summary": {
            "total": total, "passed": passed, "failed": failed, "review": review,
            "pass": passed, "fail": failed, "total_checks": total,
            "components_checked": len(component_results),
            "qc_checks": len(qc_results),
        },
        "alignment": alignment,
        "extracted_data": {
            "image": image_data,
            "drawing": drawing_data,
            "image_components": image_data["detected_components"],
            "drawing_components": drawing_data["detected_components"],
        },
        "component_comparison": component_results,
        "position_comparison": position_checks,
        "missing_components": missing_components,
        "swapped_components": swapped_strings,
        "qc_results": qc_results,
        "results": combined_results,
        "evidence": evidence_list,
        "ocr_results": ocr_results,
        "errors": [],
    }

    with open(folder / "results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report

 
# PDF REPORT
 

def _wrap_text(text, max_chars):
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]

def create_pdf_report(data):
    path = REPORT_DIR / f"{data['inspection_id']}_report.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    y = H - 20 * mm

    def text_line(txt, size=9, bold=False, indent=15 * mm):
        nonlocal y
        if y < 30 * mm:
            c.showPage()
            y = H - 20 * mm
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        max_chars = max(40, int((W - 2 * indent) / (size * 0.52)))
        for line in _wrap_text(txt, max_chars):
            c.drawString(indent, y, line)
            y -= size * 1.4
            if y < 30 * mm:
                c.showPage()
                y = H - 20 * mm

    def gap(mm_=3):
        nonlocal y
        y -= mm_ * mm

    def add_image(img_path, caption=None, max_w=None, max_h=110 * mm):
        nonlocal y
        if not Path(img_path).exists():
            return
        try:
            reader = ImageReader(str(img_path))
            iw, ih = reader.getSize()
        except Exception:
            return
        mw = max_w or (W - 30 * mm)
        scale = min(mw / iw, max_h / ih)
        w, h = iw * scale, ih * scale
        if y - h - 12 * mm < 30 * mm:
            c.showPage()
            y = H - 20 * mm
        if caption:
            text_line(caption, size=8, bold=True)
        c.drawImage(reader, 15 * mm, y - h, width=w, height=h)
        y -= h + 5 * mm

    text_line("CabinSpect Inspection Report", size=18, bold=True)
    text_line(f"Inspection ID: {data['inspection_id']}", size=9)
    text_line(f"Project: {data.get('project_name', '—')}", size=9)
    if data.get("inspection_reference"):
        text_line(f"Reference: {data['inspection_reference']}", size=9)
    text_line(f"Processed: {data['processed_at']}", size=9)
    text_line(f"Overall Status: {data['overall_status']}", size=10, bold=True)
    gap(2)
    s = data["summary"]
    text_line(
        f"Total: {s['total']}  |  PASS: {s['passed']}  |  FAIL: {s['failed']}  |  REVIEW: {s['review']}",
        size=10, bold=True,
    )
    text_line(f"Summary: {data.get('message', '')}", size=9)
    gap(2)

    al = data.get("alignment") or {}
    if al:
        text_line("Panel Alignment", size=11, bold=True)
        text_line(f"Mode: {al.get('image_alignment_mode', '—')}  |  "
                  f"Confidence: {al.get('confidence', '—')}", size=9)
        if al.get("reason"):
            text_line(f"   {al['reason']}", size=8)
        gap(2)

    comps = data.get("component_comparison") or []
    if comps:
        text_line("Component Comparison", size=12, bold=True)
        gap(1)
        for c_row in comps:
            nm = c_row.get("name", "")
            st = c_row.get("status", "")
            pos = c_row.get("position_status", "")
            text_line(f"[{st}] {nm} — position: {pos}", size=9, bold=True)
            if c_row.get("reason"):
                text_line(f"   {c_row['reason']}", size=8)
        gap(3)

    if data.get("missing_components"):
        text_line("Missing Components", size=11, bold=True)
        for m in data["missing_components"]:
            text_line(f"• {m}", size=9)
        gap(2)
    if data.get("swapped_components"):
        text_line("Swapped Components", size=11, bold=True)
        for sp in data["swapped_components"]:
            text_line(f"• {sp}", size=9)
        gap(2)

    qc = data.get("qc_results") or []
    if qc:
        text_line("QC Checklist Results", size=12, bold=True)
        gap(1)
        for q in qc:
            text_line(f"[{q.get('status', '')}] {q.get('name', '')}", size=9, bold=True)
            if q.get("reason"):
                text_line(f"   {q['reason']}", size=8)
        gap(3)

    ev = data.get("evidence") or []
    if ev:
        c.showPage()
        y = H - 20 * mm
        text_line("Evidence Images", size=14, bold=True)
        gap(2)
        for item in ev:
            url = item.get("url", "")
            fname = url.rsplit("/", 1)[-1]
            fpath = UPLOAD_DIR / data["inspection_id"] / "evidence" / fname
            add_image(fpath, caption=item.get("label", ""))

    c.save()
    return path

 
# FLASK ROUTES
 

@app.route("/", methods=["GET"])
def home():
    frontend = BASE_DIR / "index.html"
    if not frontend.exists():
        return jsonify({"success": False, "error": "index.html not found"}), 404
    return send_file(frontend)

@app.route("/api/health")
def health():
    return jsonify({
        "success": True,
        "application": "CabinSpect",
        "version": "BEAST-SOTA-5.0",
        "status": "online",
        "ocr_available": EASYOCR_OK or TESSERACT_OK,
        "models": model_status(),
    })

@app.route("/api/dashboard")
def dashboard():
    total = passed = failed = review = 0
    for file in UPLOAD_DIR.glob("*/results.json"):
        try:
            with open(file, encoding="utf-8") as f:
                d = json.load(f)
            total += 1
            st = d.get("overall_status", "")
            passed += st == "PASS"
            failed += st == "FAIL"
            review += st == "REVIEW"
        except Exception:
            logger.exception("Cannot read %s", file)
    return jsonify({
        "success": True,
        "statistics": {
            "total_inspections": total,
            "passed": passed, "failed": failed, "review": review,
        },
        "system_status": "Online",
    })

@app.route("/api/config", methods=["GET"])
def get_config_route():
    return jsonify({"success": True, **get_config()})

@app.route("/api/config", methods=["POST"])
def save_config_route():
    try:
        body = request.get_json(force=True, silent=True) or {}
        if "config" in body and isinstance(body["config"], dict):
            body = body["config"]
        saved = save_config(body)
        return jsonify({"success": True, **saved})
    except Exception as e:
        logger.exception("Config save failed")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/inspections", methods=["GET"])
def list_inspections():
    items = []
    for folder in UPLOAD_DIR.glob("*/"):
        res = folder / "results.json"
        if not res.exists():
            continue
        try:
            with open(res, encoding="utf-8") as f:
                d = json.load(f)
            meta = {}
            mpath = folder / "metadata.json"
            if mpath.exists():
                try:
                    with open(mpath, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            items.append({
                "inspection_id": d.get("inspection_id", folder.name),
                "project_name": d.get("project_name") or meta.get("project_name") or "—",
                "timestamp": d.get("processed_at") or meta.get("uploaded_at"),
                "overall_status": d.get("overall_status", "REVIEW"),
                "summary": d.get("summary", {}),
            })
        except Exception:
            continue
    items.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return jsonify({"success": True, "inspections": items})

@app.route("/api/inspections/upload", methods=["POST"])
def upload_inspection():
    categories = list(ALLOWED_FILES.keys())
    files, errors = {}, []
    for cat in categories:
        f = request.files.get(cat)
        ok, msg = validate_file(f, cat)
        if not ok:
            errors.append({"input": cat, "error": msg})
        else:
            files[cat] = f
    if errors:
        return jsonify({"success": False, "message": "Input validation failed",
                        "errors": errors}), 400

    inspection_id = generate_inspection_id()
    try:
        folder = inspection_dir(inspection_id)
        folder.mkdir(parents=True, exist_ok=True)
        saved = {}
        for cat, file in files.items():
            sname = secure_filename(file.filename)
            ext = extension(sname)
            name = f"{cat}.{ext}"
            dest = folder / name
            file.save(str(dest))
            if dest.stat().st_size == 0:
                raise ValueError(f"{cat} file is empty")
            saved[cat] = {"filename": name, "size_bytes": dest.stat().st_size}

        meta = {
            "project_name": request.form.get("project_name", "Control Panel Inspection"),
            "inspection_reference": request.form.get("inspection_reference", ""),
            "uploaded_at": datetime.now().isoformat(),
            "files": saved,
        }
        with open(folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return jsonify({
            "success": True,
            "inspection_id": inspection_id,
            "status": "uploaded",
            "inputs_received": len(saved),
            "files": saved,
        }), 201
    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/inspections/<inspection_id>/inspect", methods=["POST"])
def inspect(inspection_id):
    try:
        data = process_inspection(inspection_id)
        return jsonify({
            "success": True,
            "message": data.get("message", "Inspection completed"),
            "inspection_id": inspection_id,
            "overall_status": data["overall_status"],
            "summary": data["summary"],
            "alignment": data.get("alignment", {}),
            "results": data["results"],
            "missing_components": data["missing_components"],
            "swapped_components": data["swapped_components"],
            "position_comparison": data["position_comparison"],
            "evidence": data["evidence"],
            "ocr_results": data["ocr_results"],
            "component_comparison": data["component_comparison"],
            "qc_results": data["qc_results"],
        })
    except Exception as e:
        logger.exception("Inspection failed")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Inspection processing failed",
            "error": str(e),
        }), 500

@app.route("/api/inspections/<inspection_id>/results")
def get_results(inspection_id):
    try:
        path = inspection_dir(inspection_id) / "results.json"
        if not path.exists():
            return jsonify({"success": False, "error": "Results not found"}), 404
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/inspections/<inspection_id>/report")
def download_report(inspection_id):
    try:
        path = inspection_dir(inspection_id) / "results.json"
        if not path.exists():
            return jsonify({"success": False, "error": "Run inspection first"}), 404
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        report = create_pdf_report(data)
        return send_file(report, as_attachment=True, download_name=report.name)
    except Exception as e:
        logger.exception("PDF generation failed")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/inspections/<inspection_id>/evidence/<filename>")
def get_evidence(inspection_id, filename):
    try:
        safe = secure_filename(filename)
        if not safe or safe != filename:
            return jsonify({"error": "Invalid filename"}), 400
        path = inspection_dir(inspection_id) / "evidence" / safe
        if not path.exists():
            return jsonify({"error": "Evidence not found"}), 404
        return send_file(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

 
# MAIN







if __name__ == "__main__":
    logger.info("Starting CabinSpect BEAST SOTA 5.0")
    logger.info("Upload folder: %s", UPLOAD_DIR)
    logger.info("Config file:   %s", CONFIG_FILE)
    logger.info("Dual OCR:      EasyOCR(%s) + Tesseract(%s)", EASYOCR_OK, TESSERACT_OK)
    logger.info("YOLO-World:    %s", _MODEL_STATE["yolo_world"]["enabled"])
    logger.info("Florence-2:    %s", _MODEL_STATE["florence2"]["enabled"])
    app.run(host="0.0.0.0", port=8000, debug=False)
