import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import io
import os
import re
import shutil
import cv2
import numpy as np


def configure_tesseract() -> str | None:
    """Resolve the Tesseract executable from env vars, PATH, or common Windows install paths."""
    candidates = [
        os.getenv("TESSERACT_CMD"),
        os.getenv("TESSERACT_PATH"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


TESSERACT_CMD = configure_tesseract()

def assess_quality(gray: np.ndarray) -> dict:
    brightness   = float(np.mean(gray))
    contrast     = float(np.std(gray))
    sharpness    = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "is_dark":        brightness < 80,
        "is_bright":      brightness > 200,
        "is_blurry":      sharpness < 100,
        "low_contrast":   contrast < 40,
    }

def auto_enhance(img: Image.Image) -> Image.Image:
    """
    Intelligently enhance image based on detected quality issues:
    handles dark scans, overexposed, blurry, low contrast, tilted.
    """
    img_np = np.array(img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Upscale if small
    h, w = gray.shape
    if w < 1200:
        scale = max(2, 1200 // w)
        gray  = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 2. Deskew — fix tilted scans
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = 90 + angle
        if abs(angle) > 0.5:
            M    = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 3. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Quality-aware contrast fix
    q = assess_quality(gray)
    if q["is_dark"] or q["low_contrast"]:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
    elif q["is_bright"]:
        gray = cv2.convertScaleAbs(gray, alpha=0.85, beta=-10)

    # 5. Sharpen blurry images
    if q["is_blurry"]:
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        gray   = cv2.filter2D(gray, -1, kernel)

    # 6. Adaptive threshold — clean black/white
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # 7. Remove tiny noise dots
    kernel2 = np.ones((1,1), np.uint8)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)

    return Image.fromarray(binary)

def is_noise_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 2: return True
    english    = len(re.findall(r'[a-zA-Z0-9]', line))
    tamil      = len(re.findall(r'[\u0B80-\u0BFF]', line))
    hindi      = len(re.findall(r'[\u0900-\u097F]', line))
    meaningful = english + tamil + hindi
    total      = len(line.replace(' ',''))
    if total > 3 and meaningful / total < 0.28: return True
    if re.match(r'^[\|\\\/_\-=\+\*\.\,\!\@\#\$\%\^\&\(\)\[\]\{\}~`\s"\']{2,}$', line): return True
    for p in [r'^QHAW',r'^QoAw',r'BDMLWUMEN',r'HAMSAFCU',r'^ATT\s+TT',
              r'^LS\s+Spa',r'^Cte\s+ay',r'^TCS\)\s+MHM',r'^TMH\)\s+MAM',
              r'^si\s+90\s+nn',r'^og\s+pnhap',r'achalasia',r'wee\s+avert',
              r'^ANMTOASM',r'BSED$',r'DPFMh',r'^use;\s+ser',r'^QL,\s*6n6oo']:
        if re.search(p, line): return True
    return False

def clean_text(text: str) -> str:
    cleaned = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or is_noise_line(line): continue
        line = re.sub(r'\s{3,}', '  ', line)
        line = re.sub(r'={3,}|~{2,}|\*{3,}', '', line).strip()
        if line and not is_noise_line(line):
            cleaned.append(line)
    return '\n'.join(cleaned)

def extract_tamil_lines(text: str) -> list:
    return [l.strip() for l in text.split('\n')
            if re.search(r'[\u0B80-\u0BFF]', l)
            and len(re.findall(r'[\u0B80-\u0BFF]', l)) >= 2]

def extract_hindi_lines(text: str) -> list:
    return [l.strip() for l in text.split('\n')
            if re.search(r'[\u0900-\u097F]', l)
            and len(re.findall(r'[\u0900-\u097F]', l)) >= 2]

def run_ocr(img: Image.Image) -> tuple:
    eng, tam, hin = "", "", ""
    best = ""
    for cfg in ["--psm 6 --oem 3","--psm 4 --oem 3","--psm 3 --oem 3"]:
        try:
            t = pytesseract.image_to_string(img, lang="eng", config=cfg)
            if len(t.strip()) > len(best.strip()): best = t
        except: pass
    eng = best
    try: tam = pytesseract.image_to_string(img, lang="tam+eng", config="--psm 6 --oem 3")
    except: pass
    try: hin = pytesseract.image_to_string(img, lang="hin+eng", config="--psm 6 --oem 3")
    except: pass
    return eng, tam, hin

def build_output(eng: str, tam: str, hin: str) -> str:
    parts = []
    e = clean_text(eng)
    if e.strip():
        parts.append("--- English ---")
        parts.append(e)
    t_lines = extract_tamil_lines(tam)
    if t_lines:
        parts.append("\n--- Tamil (தமிழ்) ---")
        parts.append('\n'.join(t_lines))
    h_lines = extract_hindi_lines(hin)
    if h_lines:
        parts.append("\n--- Hindi (हिन्दी) ---")
        parts.append('\n'.join(h_lines))
    return '\n'.join(parts)

def try_extract(img: Image.Image) -> str:
    """Try enhanced first, fall back to original if better."""
    enhanced    = auto_enhance(img)
    e, t, h     = run_ocr(enhanced)
    result      = build_output(e, t, h)
    if len(result.strip()) < 50:
        orig        = img.convert("L")
        e2, t2, h2  = run_ocr(orig)
        result2     = build_output(e2, t2, h2)
        if len(result2) > len(result):
            return result2
    return result

def extract_text_from_image(path: str) -> str:
    return try_extract(Image.open(path))

def extract_text_from_pdf(path: str) -> str:
    doc  = fitz.open(path)
    full = ""
    for n, page in enumerate(doc):
        native = page.get_text("text").strip()
        if len(native) > 80:
            full += f"\n--- Page {n+1} ---\n{clean_text(native)}"
        else:
            pix = page.get_pixmap(dpi=400)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            full += f"\n--- Page {n+1} ---\n{try_extract(img)}"
    return full.strip()

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if not TESSERACT_CMD:
            return "OCR Error: Tesseract executable not found. Set TESSERACT_CMD or install Tesseract OCR."
        if ext == ".pdf":   return extract_text_from_pdf(file_path)
        elif ext in [".png",".jpg",".jpeg",".bmp",".tiff",".webp"]:
            return extract_text_from_image(file_path)
        else: return f"Unsupported: {ext}"
    except Exception as e:
        return f"OCR Error: {str(e)}"
