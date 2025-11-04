import os
import cv2
import easyocr
import pandas as pd
import re
import json
import numpy as np
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import string
import re, torch
from ftfy import fix_text
from wordsegment import load as ws_load, segment as ws_segment
from spellchecker import SpellChecker
from transformers import pipeline
import wordninja
from wordfreq import zipf_frequency
import glob
import pathlib
import subprocess
import sys

warnings.filterwarnings('ignore')

# Configuration
# VIDEO_PATH = "/content/video.mp4"
OUTPUT_CSV = "video_text_outputs.csv"
OPENAI_API_KEY = "REPLACE_KEY"  # Replace with your key

import re
import os
import json
import cv2
import torch
import easyocr
import pandas as pd
from collections import defaultdict
from faster_whisper import WhisperModel
from openai import OpenAI
import numpy as np
from difflib import SequenceMatcher
import glob  # <-- make sure this is imported somewhere in your actual file

def _list_local_videos(root_dir):
    """Recursively list common video files under a local folder."""
    exts = ('.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
    return sorted(paths)

def _ensure_gdown():
    """Ensure gdown is available (for Drive folder downloads)."""
    try:
        import gdown  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])

def _fetch_videos_from_folder(folder_url_or_path, dest="/content/gdrive_videos"):
    """
    If given a Google Drive folder link, downloads it into `dest` and returns local video paths.
    If given a local path, just lists video files there.
    """
    if not folder_url_or_path or folder_url_or_path.strip() == "":
        raise ValueError("FOLDER_URL is empty. Please set it to a Drive folder link or a local folder path.")

    # If it's a Drive link, use gdown to download folder
    if "drive.google.com" in folder_url_or_path:
        os.makedirs(dest, exist_ok=True)
        _ensure_gdown()
        import gdown
        # Note: use_cookies=False is usually fine for public links; set True if needed
        gdown.download_folder(url=folder_url_or_path, output=dest, quiet=False, use_cookies=False)
        return _list_local_videos(dest)

    # Otherwise treat it as a local directory
    if os.path.isdir(folder_url_or_path):
        return _list_local_videos(folder_url_or_path)

    # If it's neither a Drive link nor an existing dir, raise
    raise ValueError(f"Not a valid Drive folder link or local directory: {folder_url_or_path}")

def _final_cleaned_phrase_list(merged_phrases):
    """
    Produce a simple list of final cleaned phrases (deduped by normalized text),
    ordered by earliest timestamp where available.
    """
    # Order by earliest timestamp first
    ordered = sorted(
        merged_phrases,
        key=lambda x: min(x['timestamps']) if x.get('timestamps') else float('inf')
    )

    seen = set()
    out = []
    for p in ordered:
        key = normalize_text(p.get('clean', ''))
        if key and key not in seen:
            out.append(p['clean'])
            seen.add(key)
    return out

def _process_one_video(video_path):
    """
    Run the *existing pipeline* on a single video WITHOUT changing any of your logic.
    Returns (asr_text, merged_phrases, final_text).
    """
    print(f"\n==============================")
    print(f"Processing video: {video_path}")
    print(f"==============================\n")

    # === STEP 1: ASR (unchanged) ===
    print("=== STEP 1: Audio Transcription ===")
    asr_text = extract_audio_with_whisper(video_path)
    print(f"ASR Result: {asr_text[:200]}{'...' if len(asr_text) > 200 else ''}\n")

    # === STEP 2: OCR (unchanged) ===
    print("=== STEP 2: OCR Extraction ===")
    ocr_data = extract_ocr_from_video(video_path, sample_rate_fps=1)
    print(f"Extracted {len(ocr_data)} unique text phrases\n")

    # === STEP 3: Clean OCR (unchanged) ===
    print("=== STEP 3: OCR Cleaning ===")
    cleaned_phrases = clean_ocr_with_openai(ocr_data, OPENAI_API_KEY)

    print("\nOCR Corrections (sample):")
    for orig, clean in list(zip(ocr_data.keys(), cleaned_phrases))[:10]:
        if orig != clean:
            print(f"  ✓ '{orig}' → '{clean}'")
    print()

    # === STEP 4: Dedup & Merge (unchanged) ===
    print("=== STEP 4: Deduplication & Merging ===")
    merged_phrases = smart_deduplicate_and_merge(ocr_data, cleaned_phrases)
    print(f"Consolidated to {len(merged_phrases)} unique phrases\n")

    # === STEP 5: Final Assembly (unchanged) ===
    print("=== STEP 5: Final Text Assembly ===")
    final_text = assemble_final_text(merged_phrases, OPENAI_API_KEY)
    print(f"Final Text: {final_text}\n")

    return asr_text, merged_phrases, final_text

def extract_audio_with_whisper(video_path):
    try:
        model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu",
                             compute_type="int8_float16" if torch.cuda.is_available() else "int8")
        segments, _ = model.transcribe(video_path, beam_size=5)
        return " ".join(s.text for s in segments).strip()
    except Exception as e:
        print(f"ASR Error: {e}")
        return ""

def is_valid_text(text):
    """Check if text contains meaningful content"""
    if not text or len(text.strip()) < 2:
        return False
    clean = re.sub(r'[^\w#@]', '', text)
    return len(clean) > 0

def preprocess_frame_for_ocr(frame):
    """Enhanced preprocessing for low-contrast overlaid text"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def extract_ocr_from_video(video_path, sample_rate_fps=1):
    """Extract OCR text from video frames with timestamps and positions"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(fps / max(0.1, sample_rate_fps))))

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    text_detections = defaultdict(lambda: {'count': 0, 'timestamps': [], 'positions': []})

    print("Processing video frames for OCR...")
    processed = 0

    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        max_w = 960
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        timestamp = frame_idx / fps

        try:
            frame_pp = preprocess_frame_for_ocr(frame)
            results = reader.readtext(frame_pp, detail=1, paragraph=False)
            for (bbox, text, confidence) in results:
                if confidence > 0.5 and is_valid_text(text):
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x_left = float(min(xs))
                    y_top = float(min(ys))

                    text_detections[text]['count'] += 1
                    if len(text_detections[text]['timestamps']) < 5:
                        text_detections[text]['timestamps'].append(round(timestamp, 2))
                    if len(text_detections[text]['positions']) < 5:
                        text_detections[text]['positions'].append((round(y_top, 2), round(x_left, 2)))

        except Exception as e:
            print(f"OCR error at frame {frame_idx}: {e}")

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed} sampled frames...")

    cap.release()
    print(f"OCR processing complete. Found {len(text_detections)} unique text phrases.")
    return dict(text_detections)

def normalize_text(text):
    """Normalize text for comparison (lowercase, collapse spaces, remove punctuation)"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s#@]', '', text)  # Keep hashtags and mentions
    return re.sub(r'\s+', ' ', text.strip())

def text_similarity(s1, s2):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def clean_ocr_with_openai(ocr_phrases, api_key, model="gpt-4o-mini"):
    """Clean OCR text using OpenAI API"""
    phrases = list(ocr_phrases.keys())
    if not phrases:
        return []

    print(f"Cleaning {len(phrases)} OCR phrases with {model}...")
    client = OpenAI(api_key=api_key)

    system_prompt = """You are an OCR error correction assistant. Fix only obvious OCR mistakes.

Common OCR errors:
- Character confusion: 'v' → 'y', 'rn' → 'm', '0' → 'O', 'i' → 'l', 'vv' → 'w', '@' → 'a', '@' → 'o'
- Missing spaces: 'helloworld' → 'hello world'
- Extra spaces: 'hel lo' → 'hello'

Rules:
1. ONLY fix clear OCR errors - do not rephrase or change meaning
2. Preserve hashtags (#) exactly
3. Keep original capitalization only for proper nouns, otherwise make everything lowercase.
4. Output ONLY the corrected text (no quotes, explanations, or extra words)
5. If a word has the letter 'v' in it and it looks misspelled, try swapping the 'v' with a 'y' to see if it makes more sense, and vice versa.
6. If a word other than "I" has the letter 'i' in it and it looks misspelled, try swapping the 'i' with a 'l' to see if it makes more sense, and vice versa.
7. Unless an acronym makes sense in the context, make it lowercase."""

    cleaned = []
    total_cost = 0.0

    for i, phrase in enumerate(phrases):
        if i % 10 == 0 and i > 0:
            print(f"  Cleaned {i}/{len(phrases)} (Cost: ${total_cost:.4f})")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Fix OCR errors: {phrase}"}
                ],
                temperature=0,
                max_tokens=100
            )

            result = response.choices[0].message.content.strip()

            # Track cost
            if hasattr(response, 'usage') and response.usage:
                input_tok = response.usage.prompt_tokens or 0
                output_tok = response.usage.completion_tokens or 0
                total_cost += (input_tok * 0.15 + output_tok * 0.60) / 1_000_000

            # Remove quotes if added
            if (result.startswith('"') and result.endswith('"')) or \
               (result.startswith("'") and result.endswith("'")):
                result = result[1:-1]

            # Fallback if result is empty or way too different
            if not result or len(result) > len(phrase) * 3:
                result = phrase

            cleaned.append(result.strip())

        except Exception as e:
            print(f"  Error cleaning '{phrase}': {e}")
            cleaned.append(phrase)

    print(f"Cleaning complete! Total cost: ${total_cost:.4f}")
    return cleaned

def smart_deduplicate_and_merge(ocr_data, cleaned_phrases):
    """
    Intelligently merge similar phrases and handle duplicates.
    Returns consolidated phrases with their metadata.
    """
    # Create phrase objects with metadata
    phrases = []
    for orig, clean in zip(ocr_data.keys(), cleaned_phrases):
        data = ocr_data[orig]
        phrases.append({
            'original': orig,
            'clean': clean,
            'normalized': normalize_text(clean),
            'count': data['count'],
            'timestamps': data.get('timestamps', []),
            'positions': data.get('positions', [])
        })

    # Sort by frequency (most common first) and timestamp (earliest first)
    phrases.sort(key=lambda x: (-x['count'], min(x['timestamps']) if x['timestamps'] else float('inf')))

    merged = []
    skip_indices = set()

    for i, phrase1 in enumerate(phrases):
        if i in skip_indices:
            continue

        # Start with this phrase as the canonical version
        canonical = phrase1.copy()

        # Check against remaining phrases
        for j in range(i + 1, len(phrases)):
            if j in skip_indices:
                continue

            phrase2 = phrases[j]

            # Calculate similarity
            similarity = text_similarity(phrase1['clean'], phrase2['clean'])

            # Merge if very similar (likely same text with OCR errors)
            if similarity > 0.85:
                # Choose the better version (longer, more common, or earlier)
                if len(phrase2['clean']) > len(canonical['clean']):
                    canonical['clean'] = phrase2['clean']
                    canonical['normalized'] = phrase2['normalized']

                # Merge metadata
                canonical['count'] += phrase2['count']
                canonical['timestamps'].extend(phrase2['timestamps'])
                canonical['positions'].extend(phrase2['positions'])

                skip_indices.add(j)

            # Check if one is substring of another
            elif canonical['normalized'] in phrase2['normalized']:
                # phrase1 is substring of phrase2, keep phrase2's text
                canonical['clean'] = phrase2['clean']
                canonical['normalized'] = phrase2['normalized']
                canonical['count'] += phrase2['count']
                canonical['timestamps'].extend(phrase2['timestamps'])
                canonical['positions'].extend(phrase2['positions'])
                skip_indices.add(j)

            elif phrase2['normalized'] in canonical['normalized']:
                # phrase2 is substring of phrase1, keep canonical and merge counts
                canonical['count'] += phrase2['count']
                canonical['timestamps'].extend(phrase2['timestamps'])
                canonical['positions'].extend(phrase2['positions'])
                skip_indices.add(j)

        # Clean up merged data
        canonical['timestamps'] = sorted(set(canonical['timestamps']))[:10]
        canonical['positions'] = list(set(map(tuple, canonical['positions'])))[:10]

        merged.append(canonical)

    return merged

def assemble_final_text(merged_phrases, api_key, model="gpt-4o-mini"):
    """
    Assemble phrases into coherent final text using LLM.
    Falls back to simple chronological assembly if LLM fails.
    """
    if not merged_phrases:
        return ""

    # Sort by timestamp (chronological order)
    sorted_phrases = sorted(merged_phrases,
                           key=lambda x: min(x['timestamps']) if x['timestamps'] else float('inf'))

    # Simple fallback assembly
    simple_assembly = " ".join(p['clean'] for p in sorted_phrases)

    # Try LLM assembly for better coherence
    try:
        client = OpenAI(api_key=api_key)

        phrases_list = [p['clean'] for p in sorted_phrases]

        system_prompt = """You are a Gen-Z person familiar with Tiktok trends assembling OCR text fragments into one coherent sentence or phrase.

Rules:
1. Arrange the fragments by timestamp; if it doesn't make sense, then you can rearrange it minimally.
2. Remove duplicate or very similar fragments
3. Add minimal punctuation ONLY where clearly needed
4. Do NOT add new words or rephrase or change existing words
5. Preserve all hashtags
6. Output ONE clean line of text"""

        user_prompt = f"""Assemble these OCR fragments in order into one coherent line:

{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(phrases_list))}

Assembled text:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=200
        )

        result = response.choices[0].message.content.strip()

        # Remove quotes if present
        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

        # Validate result isn't too different from source material
        if result and len(result) > 10 and len(result) < len(simple_assembly) * 2:
            print("Using LLM-assembled text")
            return result
        else:
            print("LLM assembly invalid, using simple assembly")
            return simple_assembly

    except Exception as e:
        print(f"LLM assembly failed ({e}), using simple assembly")
        return simple_assembly

def create_output_csv(asr_text, merged_phrases, final_text, output_csv):
    """Create output CSV with all extracted text"""
    rows = []

    # Add ASR
    if asr_text:
        rows.append({
            "source": "ASR",
            "text": asr_text,
            "count": 1,
            "timestamps": "[]",
            "original_text": ""
        })

    # Add individual OCR phrases
    for phrase in merged_phrases:
        rows.append({
            "source": "OCR_PHRASE",
            "text": phrase['clean'],
            "count": phrase['count'],
            "timestamps": json.dumps(phrase['timestamps'][:5]),
            "original_text": phrase['original']
        })

    # Add final assembled text
    if final_text:
        rows.append({
            "source": "OCR_FINAL",
            "text": final_text,
            "count": sum(p['count'] for p in merged_phrases),
            "timestamps": "[]",
            "original_text": ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

def main():
    """
    Batch main: loops through all videos in a Google Drive folder (or local folder),
    runs your existing pipeline per video, then writes ONE ROW PER VIDEO with 4 columns:
    [video, asr, ocr_final, cleaned_phrases]
    """
    # 1) Gather videos
    print("Discovering videos...\n")

    # CHANGE HERE: point to a LOCAL folder instead of a Drive URL
    # Example (Windows): r"C:\Users\YourName\Videos\my_clips"
    # Example (Mac/Linux): "/Users/yourname/Videos/my_clips"
    FOLDER_URL = r"/path/to/your/local/video/folder"

    video_paths = _fetch_videos_from_folder(FOLDER_URL)
    if not video_paths:
        print("No videos found. Please check FOLDER_URL.")
        return

    print(f"Found {len(video_paths)} video(s).")
    for v in video_paths:
        print(" -", v)
    print()

    # 2) Process each video and collect rows
    rows = []
    for vp in video_paths:
        asr_text, merged_phrases, final_text = _process_one_video(vp)
        phrases_list = _final_cleaned_phrase_list(merged_phrases)  # simple list of final cleaned phrases

        rows.append({
            "video": os.path.basename(vp),
            "asr": asr_text,
            "ocr_final": final_text,
            "cleaned_phrases": json.dumps(phrases_list, ensure_ascii=False)
        })

    # 3) Write the new 4-column CSV
    df = pd.DataFrame(rows, columns=["video", "asr", "ocr_final", "cleaned_phrases"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Batch results saved to: {OUTPUT_CSV}")
    print(f"Total videos processed: {len(df)}")
    return df

if __name__ == "__main__":
    result_df = main()
