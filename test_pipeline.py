# -*- coding: utf-8 -*-
# test_pipeline.py

import os
import time
import re
import base64
import requests
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import sys
import shutil
import urllib.parse
import json
import argparse # For command-line arguments

# FastAPI imports (Removed - not needed for standalone)
# from fastapi import ...

# Pydantic models (Removed - not needed for standalone)
# from pydantic import ...

# Text Processing & Embeddings
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util

# Qdrant
from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# Qwen-VL Model Imports
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login as hf_login # Optional login

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

# --- Load Environment Variables ---
# (Keep this section same as main.py)
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")
DATALAB_MARKER_URL = os.environ.get("DATALAB_MARKER_URL")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# --- Check Essential Variables ---
# (Keep this section same as main.py)
if not all([DATALAB_API_KEY, DATALAB_MARKER_URL, QDRANT_URL, GEMINI_API_KEY]):
    missing_vars = [var for var, val in {
        "DATALAB_API_KEY": DATALAB_API_KEY, "DATALAB_MARKER_URL": DATALAB_MARKER_URL,
        "QDRANT_URL": QDRANT_URL, "GEMINI_API_KEY": GEMINI_API_KEY
    }.items() if not val]
    logging.critical(f"FATAL ERROR: Missing essential environment variables: {', '.join(missing_vars)}")
    sys.exit("Missing essential environment variables.")

# --- Directories (Adjust TEMP paths if needed) ---
BASE_DIR = Path(__file__).parent # Directory where the script is located
TEMP_UPLOAD_DIR = BASE_DIR / "temp_standalone_uploads" # Temporary storage for this script's runs
FINAL_RESULTS_DIR = BASE_DIR / "final_standalone_results" # Output markdown
EXTRACTED_IMAGES_DIR = BASE_DIR / "extracted_standalone_images" # Output images
PROMPT_DIR = BASE_DIR / "content" # Directory for prompt templates

# --- Constants (Keep mostly same as main.py) ---
DATALAB_POST_TIMEOUT = 60
DATALAB_POLL_TIMEOUT = 30
MAX_POLLS = 300
POLL_INTERVAL = 3
GEMINI_TIMEOUT = 240
MAX_GEMINI_RETRIES = 3
GEMINI_RETRY_DELAY = 60
QDRANT_COLLECTION_NAME = "markdown_docs_v3_semantic" # Use the same collection
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:{action}?key={api_key}"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MAX_NEW_TOKENS = 256
QSTS_THRESHOLD = 0.5
QUALITATIVE_METRICS = ["Understandable", "TopicRelated", "Grammatical", "Clear", "Answerable", "Central"]
CRITICAL_QUALITATIVE_FAILURES = {"Understandable": False, "Grammatical": False, "Clear": False, "Answerable": False, "TopicRelated": False, "Central": False}
# Prompt File Paths (relative to BASE_DIR)
FINAL_USER_PROMPT_PATH = PROMPT_DIR / "final_user_prompt.txt"
HYPOTHETICAL_PROMPT_PATH = PROMPT_DIR / "hypothetical_prompt.txt"
QUALITATIVE_EVAL_PROMPT_PATH = PROMPT_DIR / "qualitative_eval_prompt.txt"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

# --- Ensure Base Directories Exist ---
for dir_path in [TEMP_UPLOAD_DIR, FINAL_RESULTS_DIR, EXTRACTED_IMAGES_DIR, PROMPT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Check for mandatory prompt files ---
mandatory_prompts = [FINAL_USER_PROMPT_PATH, HYPOTHETICAL_PROMPT_PATH, QUALITATIVE_EVAL_PROMPT_PATH]
missing_prompts = [p.name for p in mandatory_prompts if not p.exists()]
if missing_prompts:
    logger.critical(f"FATAL ERROR: Missing required prompt template files in '{PROMPT_DIR}': {', '.join(missing_prompts)}")
    sys.exit(f"Missing prompt files: {', '.join(missing_prompts)}")

# --- Initialize Models and Clients (Global Scope) ---
# (Keep this section same as main.py, assuming NLTK data is pre-downloaded)
model_st = None
qdrant_client = None
model_qwen = None
processor_qwen = None
device = None
stop_words = None
lemmatizer = None

try:
    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if not torch.cuda.is_available(): logger.warning("CUDA not available, Qwen-VL will run on CPU.")

    # Optional HF Login
    if HUGGINGFACE_TOKEN:
        try: hf_login(token=HUGGINGFACE_TOKEN); logger.info("Successfully logged into Hugging Face Hub.")
        except Exception as e: logger.warning(f"Hugging Face login failed: {e}")
    else: logger.info("No Hugging Face token provided.")

    # Qwen-VL Init
    logger.info(f"Initializing Qwen-VL model '{QWEN_MODEL_ID}'...")
    model_qwen = AutoModelForImageTextToText.from_pretrained(QWEN_MODEL_ID, torch_dtype="auto", trust_remote_code=True).to(device).eval()
    processor_qwen = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    logger.info("Qwen-VL model and processor loaded.")

    # Sentence Transformer Init
    logger.info(f"Initializing Sentence Transformer model '{EMBEDDING_MODEL_NAME}'...")
    model_st = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Sentence Transformer model loaded.")

    # Qdrant Client Init
    logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    # Ensure Qdrant Collection Exists
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception as e:
        if "Not found" in str(e) or "status_code=404" in str(e) or "Reason: Not Found" in str(e):
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
            qdrant_client.recreate_collection(collection_name=QDRANT_COLLECTION_NAME, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")
        else: raise e # Re-raise other Qdrant errors

    # NLTK Setup (Load pre-downloaded resources)
    logger.info("Loading NLTK resources (stopwords, wordnet, punkt)...")
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        logger.info("NLTK resources loaded successfully.")
    except LookupError as e:
        logger.critical(f"FATAL NLTK ERROR: Required resource not found: {e}. Please run the NLTK download steps first.", exc_info=True)
        sys.exit(f"NLTK resource missing: {e}")

except Exception as e:
    logger.critical(f"Fatal error during initialization: {e}", exc_info=True)
    sys.exit("Initialization failed.")


# --- Core Functions (Copied and potentially adapted slightly from main.py) ---

# generate_description_for_image (Same as in main.py)
def generate_description_for_image(image_path: Path, figure_caption: str = "") -> str:
    """Load an image and generate a description using the local Qwen-VL model."""
    global model_qwen, processor_qwen, device # Access globally loaded models/device
    if not model_qwen or not processor_qwen:
        logger.error(f"Qwen-VL model not initialized. Cannot generate description for {image_path.name}")
        return "Error: Qwen-VL model not available."
    try:
        image = Image.open(image_path).convert("RGB")
        prompt_text = (
            f"Describe the key technical findings in this figure/visualization "
            f"captioned: {figure_caption}. Illustrate and mention trends, "
            f"patterns, and numerical values that can be observed. Provide a scientific/academic styled short, "
            f"single paragraph summary that is highly insightful in context of the document."
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        text_prompt = processor_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor_qwen(text=[text_prompt], images=[image], return_tensors="pt").to(device)
        logger.info(f"Generating Qwen-VL description for {image_path.name} on device {device}...")
        with torch.inference_mode():
            output_ids = model_qwen.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False)
        logger.info(f"Finished Qwen-VL generation for {image_path.name}.")
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_token_len:]
        response = processor_qwen.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        description = response.replace('\n', ' ').strip()
        if not description:
            logger.warning(f"Qwen-VL generated an empty description for {image_path.name}")
            return "No description generated by model."
        logger.info(f"Successfully generated description for {image_path.name} (length: {len(description)}).")
        return description
    except FileNotFoundError:
         logger.error(f"Image file not found at {image_path}")
         return f"Error: Image file not found."
    except Exception as e:
        logger.error(f"Error generating Qwen-VL description for {image_path.name}: {e}", exc_info=True)
        if "CUDA out of memory" in str(e):
            logger.critical(f"CUDA OOM Error during Qwen-VL inference on {image_path.name}.")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return "Error: Ran out of GPU memory generating description."
        return f"Error generating description for this image due to an internal error."

# call_datalab_marker (Same as in main.py)
def call_datalab_marker(file_path: Path, job_id_for_log: str):
    logger.info(f"[{job_id_for_log}] Calling Datalab Marker API for {file_path.name}...")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        form_data = {"langs": (None, "English"), "output_format": (None, "markdown"), "disable_image_extraction": (None, False)}
        headers = {"X-Api-Key": DATALAB_API_KEY}
        try:
            response = requests.post(DATALAB_MARKER_URL, files=files, data=form_data, headers=headers, timeout=DATALAB_POST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.error(f"[{job_id_for_log}] Datalab API request timed out for {file_path.name}")
            raise TimeoutError("Datalab API request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"[{job_id_for_log}] Datalab API request failed for {file_path.name}: {e}")
            raise Exception(f"Datalab API request failed: {e}")
    if not data.get("success"):
        err_msg = data.get('error', 'Unknown Datalab error')
        logger.error(f"[{job_id_for_log}] Datalab API error for {file_path.name}: {err_msg}")
        raise Exception(f"Datalab API error: {err_msg}")
    check_url = data["request_check_url"]
    logger.info(f"[{job_id_for_log}] Polling Datalab result URL for {file_path.name}: {check_url}")
    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=DATALAB_POLL_TIMEOUT)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("status") == "complete":
                logger.info(f"[{job_id_for_log}] Datalab processing complete for {file_path.name}.")
                return poll_data
            elif poll_data.get("status") == "error":
                 err_msg = poll_data.get('error', 'Unknown Datalab processing error')
                 logger.error(f"[{job_id_for_log}] Datalab processing failed for {file_path.name}: {err_msg}")
                 raise Exception(f"Datalab processing failed: {err_msg}")
            if (i + 1) % 10 == 0: logger.info(f"[{job_id_for_log}] Polling Datalab for {file_path.name}... attempt {i+1}/{MAX_POLLS}")
        except requests.exceptions.Timeout: logger.warning(f"[{job_id_for_log}] Polling Datalab timed out on attempt {i+1} for {file_path.name}. Retrying...")
        except requests.exceptions.RequestException as e: logger.warning(f"[{job_id_for_log}] Polling error on attempt {i+1} for {file_path.name}: {e}. Retrying...")
    logger.error(f"[{job_id_for_log}] Polling timed out waiting for Datalab processing for {file_path.name}.")
    raise TimeoutError("Polling timed out waiting for Datalab processing.")

# save_extracted_images (Same as in main.py)
def save_extracted_images(images_dict, images_folder: Path, job_id_for_log: str) -> Dict[str, str]:
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    logger.info(f"[{job_id_for_log}] Saving {len(images_dict)} extracted images to {images_folder}...")
    for img_name, b64_data in images_dict.items():
        try:
            safe_img_name = "".join([c if c.isalnum() or c in ('-', '_', '.') else '_' for c in img_name])
            if not safe_img_name: safe_img_name = f"image_{uuid.uuid4().hex[:8]}.png"
            image_data = base64.b64decode(b64_data)
            image_path = images_folder / safe_img_name
            with open(image_path, "wb") as img_file: img_file.write(image_data)
            saved_files[img_name] = str(image_path)
        except Exception as e:
            logger.warning(f"[{job_id_for_log}] Could not decode/save image '{img_name}': {e}", exc_info=True)
    return saved_files

# process_markdown (Same as in main.py)
def process_markdown(markdown_text, saved_images: Dict[str, str], job_id_for_log: str):
    logger.info(f"[{job_id_for_log}] Processing markdown for image descriptions using Qwen-VL...")
    lines = markdown_text.splitlines()
    processed_lines = []
    i = 0; image_count = 0
    figure_pattern = re.compile(r"^!\[.*?\]\((.*?)\)$")
    caption_pattern = re.compile(r"^(Figure|Table|Chart)\s?(\d+[:.]?)\s?(.*)", re.IGNORECASE)
    while i < len(lines):
        line = lines[i]; stripped_line = line.strip()
        image_match = figure_pattern.match(stripped_line)
        if image_match:
            image_filename_encoded = image_match.group(1)
            try: image_filename_decoded = urllib.parse.unquote(image_filename_encoded)
            except Exception: image_filename_decoded = image_filename_encoded
            image_count += 1; caption = ""; caption_line_index = -1
            j = i + 1
            while j < len(lines) and lines[j].strip() == "": j += 1
            if j < len(lines):
                next_line_stripped = lines[j].strip()
                if caption_pattern.match(next_line_stripped): caption = next_line_stripped; caption_line_index = j
            image_path_str = saved_images.get(image_filename_decoded)
            if not image_path_str: image_path_str = saved_images.get(image_filename_encoded)
            description = ""
            if image_path_str:
                image_path = Path(image_path_str)
                if image_path.exists(): description = generate_description_for_image(image_path, caption)
                else: description = f"*Referenced image file '{image_path.name}' not found.*"; logger.warning(f"[{job_id_for_log}] {description}")
            else: description = f"*Referenced image '{image_filename_decoded}'/'{image_filename_encoded}' not found in saved images.*"; logger.warning(f"[{job_id_for_log}] {description}")
            title_text = caption if caption else f"Figure {image_count}"
            block_text = f"\n---\n### {title_text}\n\n**Figure Description (Generated by Qwen-VL):**\n{description}\n---\n"
            processed_lines.append(block_text)
            if caption_line_index != -1: i = caption_line_index
        else: processed_lines.append(line)
        i += 1
    logger.info(f"[{job_id_for_log}] Finished processing markdown with Qwen-VL. Processed {image_count} image references.")
    return "\n".join(processed_lines)

# clean_text_for_embedding (Same as in main.py)
def clean_text_for_embedding(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*Figure Description \(Generated by Qwen-VL\):\*\*', '', text)
    return text.strip()

# hierarchical_chunk_markdown (Same as in main.py)
def hierarchical_chunk_markdown(markdown_text, source_filename, job_id_for_log: str):
    logger.info(f"[{job_id_for_log}] Chunking markdown from source: {source_filename}")
    lines = markdown_text.splitlines(); chunks = []; current_chunk_lines = []; current_headers = {}; figure_title = None
    header_pattern = re.compile(r"^(#{1,6})\s+(.*)")
    figure_title_pattern = re.compile(r"^###\s+((?:Figure|Table|Chart).*)$", re.IGNORECASE)
    separator_pattern = re.compile(r"^---$"); inside_figure_block = False
    for line_num, line in enumerate(lines):
        stripped_line = line.strip(); header_match = header_pattern.match(stripped_line)
        figure_title_match = figure_title_pattern.match(stripped_line); separator_match = separator_pattern.match(stripped_line)
        if separator_match:
            if inside_figure_block:
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip(); cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                    if cleaned_chunk_text:
                        metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                        if figure_title: metadata["figure_title"] = figure_title
                        chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                    current_chunk_lines = []
                inside_figure_block = False; figure_title = None
            else:
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip(); cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                    if cleaned_chunk_text:
                        metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                        chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                    current_chunk_lines = []
                inside_figure_block = True
            continue
        if header_match:
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip(); cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                if cleaned_chunk_text:
                    metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                    if figure_title: metadata["figure_title"] = figure_title
                    chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                current_chunk_lines = []
            level, title = len(header_match.group(1)), header_match.group(2).strip()
            current_headers = {k: v for k, v in current_headers.items() if k < level}
            current_headers[level] = title; figure_title = None; inside_figure_block = False
            current_chunk_lines.append(line); continue
        if figure_title_match and inside_figure_block:
            figure_title = figure_title_match.group(1).strip()
            current_chunk_lines.append(line); continue
        current_chunk_lines.append(line)
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip(); cleaned_chunk_text = clean_text_for_embedding(chunk_text)
        if cleaned_chunk_text:
            metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
            if figure_title: metadata["figure_title"] = figure_title
            chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
    logger.info(f"[{job_id_for_log}] Generated {len(chunks)} hierarchical chunks for {source_filename}.")
    return chunks

# embed_chunks (Same as in main.py)
def embed_chunks(chunks_data, job_id_for_log: str):
    global model_st
    if not chunks_data: return []
    if not model_st: raise Exception("Embedding model not available.")
    logger.info(f"[{job_id_for_log}] Embedding {len(chunks_data)} text chunks...")
    texts_to_embed = [chunk['text'] for chunk in chunks_data]
    try:
        embeddings = model_st.encode(texts_to_embed, show_progress_bar=False).tolist()
        logger.info(f"[{job_id_for_log}] Embedding complete.")
        return embeddings
    except Exception as e: logger.error(f"[{job_id_for_log}] Error during embedding: {e}", exc_info=True); raise

# upsert_to_qdrant (Same as in main.py)
def upsert_to_qdrant(job_id_for_log: str, collection_name, embeddings, chunks_data, batch_size=100):
    global qdrant_client
    if not embeddings or not chunks_data: return 0
    if not qdrant_client: raise Exception("Qdrant client not available.")
    logger.info(f"[{job_id_for_log}] Upserting {len(embeddings)} points to Qdrant collection '{collection_name}'...")
    total_points_upserted = 0; points_to_upsert = []
    for embedding, chunk_data in zip(embeddings, chunks_data):
        if isinstance(chunk_data.get('metadata'), dict) and 'text' in chunk_data:
            payload = chunk_data['metadata'].copy(); payload["text"] = chunk_data['text']
            point_id = str(uuid.uuid4())
            points_to_upsert.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        else: logger.warning(f"[{job_id_for_log}] Skipping chunk due to invalid format: {chunk_data}")
    for i in range(0, len(points_to_upsert), batch_size):
        batch_points = points_to_upsert[i:i + batch_size]
        if not batch_points: continue
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch_points, wait=True)
            batch_count = len(batch_points); total_points_upserted += batch_count
            logger.info(f"[{job_id_for_log}] Upserted batch {i // batch_size + 1} ({batch_count} points) to Qdrant.")
        except Exception as e: logger.error(f"[{job_id_for_log}] Error upserting Qdrant batch {i // batch_size + 1}: {e}", exc_info=True); raise
    logger.info(f"[{job_id_for_log}] Finished upserting. Total points upserted: {total_points_upserted}")
    return total_points_upserted

# fill_placeholders (Same as in main.py)
def fill_placeholders(template_path: Path, output_path: Path, placeholders: Dict):
    try:
        if not template_path.exists(): raise FileNotFoundError(f"Template file not found: {template_path}")
        template = template_path.read_text(encoding='utf-8')
        for placeholder, value in placeholders.items(): template = template.replace(f"{{{placeholder}}}", str(value))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template, encoding='utf-8')
    except Exception as e: logger.error(f"Error filling placeholders for {template_path}: {e}", exc_info=True); raise

# get_gemini_response (Same as in main.py)
def get_gemini_response(system_prompt: str, user_prompt: str, is_json_output: bool = False, job_id_for_log: str = "standalone"):
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    api_url = GEMINI_API_URL_TEMPLATE.format(model_name=GEMINI_MODEL_NAME, action="generateContent", api_key=GEMINI_API_KEY)
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": user_prompt}]}], "systemInstruction": {"parts": [{"text": system_prompt}]},
         "generationConfig": {"temperature": 0.5 if is_json_output else 0.7, "maxOutputTokens": 8192, "topP": 0.95, "topK": 40}}
    if is_json_output: payload["generationConfig"]["responseMimeType"] = "application/json"
    last_error = None
    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            logger.info(f"[{job_id_for_log}] Calling Gemini API (Attempt {attempt+1}/{MAX_GEMINI_RETRIES})...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"[{job_id_for_log}] Gemini API call successful (Attempt {attempt+1}).")
            if response_data.get('candidates'):
                candidate = response_data['candidates'][0]
                if candidate.get('content', {}).get('parts'):
                    gemini_response_text = candidate['content']['parts'][0].get('text', '')
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    if finish_reason not in ['STOP', 'MAX_TOKENS']: logger.warning(f"[{job_id_for_log}] Gemini finished reason: {finish_reason}")
                    if gemini_response_text: return gemini_response_text.strip()
                    else: last_error = f"Error: Gemini empty response (finish reason: {finish_reason})."
                elif candidate.get('finishReason'):
                    reason = candidate['finishReason']; safety_ratings = candidate.get('safetyRatings', [])
                    blocked_by_safety = any(sr.get('probability') not in ['NEGLIGIBLE', 'LOW'] for sr in safety_ratings if sr.get('probability'))
                    if blocked_by_safety:
                         block_details = ", ".join([f"{sr['category']}: {sr['probability']}" for sr in safety_ratings if sr.get('probability') not in ['NEGLIGIBLE', 'LOW']])
                         last_error = f"Error: Gemini stopped - {reason} (Safety: {block_details})"
                    else: last_error = f"Error: Gemini stopped - {reason}"
                else: last_error = "Error: Cannot extract text from Gemini response."
            elif response_data.get('promptFeedback', {}).get('blockReason'):
                block_reason = response_data['promptFeedback']['blockReason']; safety_ratings = response_data['promptFeedback'].get('safetyRatings', [])
                block_details = ", ".join([f"{sr['category']}: {sr['probability']}" for sr in safety_ratings if sr.get('probability') not in ['NEGLIGIBLE', 'LOW']])
                if block_details: last_error = f"Error: Prompt blocked by Gemini - {block_reason} (Safety: {block_details})"
                else: last_error = f"Error: Prompt blocked by Gemini - {block_reason}"
            else: last_error = f"Error: Unexpected Gemini API response format."
            if last_error: logger.error(f"[{job_id_for_log}] {last_error}"); return last_error
        except requests.exceptions.Timeout:
            last_error = "Error: Gemini API request timed out."; logger.warning(f"[{job_id_for_log}] {last_error} Attempt {attempt + 1}.")
            if attempt + 1 < MAX_GEMINI_RETRIES: time.sleep(GEMINI_RETRY_DELAY); continue
        except requests.exceptions.RequestException as e:
            response_text = ""; status_code = None;
            if e.response is not None: response_text = e.response.text[:500]; status_code = e.response.status_code
            last_error = f"Error: Gemini API request failed - {e} (Status: {status_code})"; logger.warning(f"[{job_id_for_log}] {last_error} Attempt {attempt + 1}.")
            if status_code == 429: logger.warning(f"[{job_id_for_log}] Gemini rate limit hit. Waiting..."); time.sleep(GEMINI_RETRY_DELAY); continue
            elif status_code is not None and 500 <= status_code < 600: logger.warning(f"[{job_id_for_log}] Gemini server error. Waiting..."); time.sleep(GEMINI_RETRY_DELAY); continue
            else: logger.error(f"[{job_id_for_log}] {last_error}", exc_info=True); break
        except Exception as e: last_error = f"Error: Failed to process Gemini response - {e}"; logger.error(f"[{job_id_for_log}] {last_error}", exc_info=True); break
    logger.error(f"[{job_id_for_log}] Gemini API call failed after {MAX_GEMINI_RETRIES} attempts. Last error: {last_error}")
    return last_error if last_error else "Error: Gemini API call failed after multiple retries."

# find_topics_and_generate_hypothetical_text (Same as in main.py)
def find_topics_and_generate_hypothetical_text(job_id_for_log: str, academic_level, major, course_name, taxonomy_level, topics):
    logger.info(f"[{job_id_for_log}] Generating hypothetical text...")
    try:
        temp_path = TEMP_UPLOAD_DIR / f"{job_id_for_log}_hypothetical.txt"
        placeholders = {"course_name": course_name, "academic_level": academic_level, "topics": topics, "major": major, "taxonomy_level": taxonomy_level}
        fill_placeholders(HYPOTHETICAL_PROMPT_PATH, temp_path, placeholders)
        user_prompt = temp_path.read_text(encoding="utf8")
        system_prompt = f"You are an AI assistant for {major} at {academic_level}. Generate a concise, hypothetical student query for '{course_name}' on topics: {topics}, at Bloom's level: {taxonomy_level}."
        hypothetical_text = get_gemini_response(system_prompt, user_prompt, job_id_for_log=job_id_for_log)
        temp_path.unlink(missing_ok=True)
        if hypothetical_text.startswith("Error:"): raise Exception(f"Failed to generate hypothetical text: {hypothetical_text}")
        logger.info(f"[{job_id_for_log}] Successfully generated hypothetical text.")
        return hypothetical_text
    except FileNotFoundError as e: raise Exception(f"Hypothetical prompt template missing: {e}")
    except Exception as e: raise Exception(f"Error generating hypothetical text: {e}")

# search_results_from_qdrant (Same as in main.py)
def search_results_from_qdrant(job_id_for_log: str, collection_name, embedded_vector, limit=15, score_threshold: Optional[float] = None, session_id_filter=None, document_ids_filter=None):
    global qdrant_client
    if not qdrant_client: raise Exception("Qdrant client not available.")
    logger.info(f"[{job_id_for_log}] Searching Qdrant '{collection_name}' (limit={limit}, threshold={score_threshold})...")
    must_conditions = []
    if session_id_filter: must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id_filter)))
    if document_ids_filter:
        doc_ids = document_ids_filter if isinstance(document_ids_filter, list) else [document_ids_filter]
        if doc_ids: must_conditions.append(FieldCondition(key="document_id", match=MatchAny(any=doc_ids)))
    query_filter = Filter(must=must_conditions) if must_conditions else None
    try:
        query_vector_list = embedded_vector.tolist() if hasattr(embedded_vector, 'tolist') else list(map(float, embedded_vector))
        results = qdrant_client.search(collection_name=collection_name, query_vector=query_vector_list, query_filter=query_filter,
            limit=limit, score_threshold=score_threshold, with_payload=True, with_vectors=False)
        logger.info(f"[{job_id_for_log}] Qdrant search returned {len(results)} results.")
        if results: logger.info(f"[{job_id_for_log}] Top hit score: {results[0].score:.4f}")
        return results
    except Exception as e: logger.error(f"[{job_id_for_log}] Error searching Qdrant: {e}", exc_info=True); return []

# generate_initial_questions (Same as in main.py)
def generate_initial_questions(job_id_for_log: str, retrieved_context: str, params: Dict):
    logger.info(f"[{job_id_for_log}] Preparing to generate initial questions...")
    blooms = "Bloom's Levels: Remember, Understand, Apply, Analyze, Evaluate, Create."
    max_context_chars = 30000
    truncated_context = retrieved_context[:max_context_chars]
    if len(retrieved_context) > max_context_chars: logger.warning(f"[{job_id_for_log}] Truncating context to {max_context_chars} chars for LLM.")
    generate_diagrams_flag = params.get('generate_diagrams', False)
    logger.info(f"[{job_id_for_log}] generate_diagrams flag in generate_initial_questions: {generate_diagrams_flag}")
    diagram_instructions = ""
    if generate_diagrams_flag:
        logger.info(f"[{job_id_for_log}] PlantUML instructions included in prompt.")
        diagram_instructions = ("\n7. **PlantUML Diagram Generation (REQUIRED if applicable):** ... [Same instruction text as main.py] ...") # Keep full instruction text
    placeholders = {"content": truncated_context, "num_questions": params['num_questions'], "course_name": params['course_name'], "taxonomy": params['taxonomy_level'], "major": params['major'], "academic_level": params['academic_level'], "topics_list": params['topics_list'], "blooms_taxonomy_descriptions": blooms, "diagram_instructions": diagram_instructions}
    try:
        temp_path = TEMP_UPLOAD_DIR / f"{job_id_for_log}_initial_user_prompt.txt"
        fill_placeholders(FINAL_USER_PROMPT_PATH, temp_path, placeholders)
        user_prompt = temp_path.read_text(encoding="utf8")
        system_prompt_base = (f"You are an AI assistant creating educational questions for {params['academic_level']} {params['major']} course: '{params['course_name']}'. Generate {params['num_questions']} questions based ONLY on the provided context, aligned with Bloom's level: {params['taxonomy_level']}, topics: {params['topics_list']}.")
        plantuml_system_hint = " **If instructions require PlantUML, include it in ```plantuml ... ``` tags.**" if generate_diagrams_flag else ""
        output_format_instruction = " **Output ONLY the numbered list of questions (and PlantUML blocks if generated). NO introductions or summaries.**"
        system_prompt_final = system_prompt_base + plantuml_system_hint + output_format_instruction
        logger.info(f"[{job_id_for_log}] Generating initial questions via Gemini...")
        initial_questions = get_gemini_response(system_prompt_final, user_prompt, job_id_for_log=job_id_for_log)
        temp_path.unlink(missing_ok=True)
        if initial_questions.startswith("Error:"): raise Exception(f"Gemini Error: {initial_questions}")
        logger.info(f"[{job_id_for_log}] Successfully generated initial questions snippet: {initial_questions[:300]}...")
        plantuml_found = "```plantuml" in initial_questions
        if generate_diagrams_flag and not plantuml_found: logger.warning(f"[{job_id_for_log}] PlantUML requested, but '```plantuml' not found.")
        elif not generate_diagrams_flag and plantuml_found: logger.warning(f"[{job_id_for_log}] PlantUML *not* requested, but '```plantuml' *was* found.")
        return initial_questions
    except FileNotFoundError as e: raise Exception(f"Final user prompt template missing: {e}")
    except Exception as e: logger.error(f"[{job_id_for_log}] Initial question generation failed: {e}", exc_info=True); raise

# parse_questions (Same as in main.py)
def parse_questions(question_block: str, job_id_for_log: str = "standalone") -> List[str]:
    if not question_block: return []
    lines = question_block.splitlines(); questions = []; current_question_lines = []; in_plantuml_block = False
    question_start_pattern = re.compile(r"^\s*\d+\s*[\.\)\-:]\s+")
    for line in lines:
        stripped_line = line.strip(); is_plantuml_start = stripped_line.startswith("```plantuml"); is_plantuml_end = stripped_line == "```" and in_plantuml_block
        if is_plantuml_start: in_plantuml_block = True; current_question_lines.append(line); continue
        if is_plantuml_end: in_plantuml_block = False; current_question_lines.append(line); continue
        if in_plantuml_block: current_question_lines.append(line); continue
        if question_start_pattern.match(line):
            if current_question_lines: questions.append("\n".join(current_question_lines).strip())
            current_question_lines = [line]
        elif current_question_lines: current_question_lines.append(line)
    if current_question_lines: questions.append("\n".join(current_question_lines).strip())
    cleaned_questions = [q for q in questions if q]
    if not cleaned_questions and question_block.strip():
        logger.warning(f"[{job_id_for_log}] Could not parse numbered list. Falling back to simple line split.")
        cleaned_questions = [q.strip() for q in question_block.splitlines() if q.strip() and not re.fullmatch(r"\s*\d+[\.\)\-:]?\s*", q.strip())]
        if not cleaned_questions: logger.warning(f"[{job_id_for_log}] Fallback split yielded no questions."); return [question_block.strip()] if question_block.strip() else []
    return cleaned_questions

# evaluate_single_question_qsts (Same as in main.py)
def evaluate_single_question_qsts(job_id_for_log: str, question: str, context: str) -> float:
    global model_st
    if not model_st: logger.error(f"[{job_id_for_log}] Sentence Transformer model not initialized."); return 0.0
    if not question or not context: return 0.0
    question_text_only = re.sub(r"```plantuml.*?```", "", question, flags=re.DOTALL | re.MULTILINE)
    question_text_only = re.sub(r"^\s*\d+\s*[\.\)\-:]?\s+", "", question_text_only.strip()).strip()
    if not question_text_only: logger.warning(f"[{job_id_for_log}] No text found in question for QSTS eval: '{question[:50]}...'"); return 0.0
    try:
        q_emb = model_st.encode(question_text_only); c_emb = model_st.encode(context)
        if q_emb.ndim == 1: q_emb = q_emb.reshape(1, -1)
        if c_emb.ndim == 1: c_emb = c_emb.reshape(1, -1)
        score = sbert_util.pytorch_cos_sim(q_emb, c_emb).item()
        return round(max(-1.0, min(1.0, score)), 4)
    except Exception as e: logger.warning(f"[{job_id_for_log}] Error calculating QSTS for question '{question_text_only[:50]}...': {e}", exc_info=True); return 0.0

# evaluate_single_question_qualitative (Same as in main.py)
def evaluate_single_question_qualitative(job_id_for_log: str, question: str, context: str) -> Dict[str, bool]:
    results = {metric: False for metric in QUALITATIVE_METRICS}
    if not question or not context: return results
    full_question_block = question
    try:
        eval_context = context[:4000] + ("\n... [Context Truncated]" if len(context)>4000 else "")
        placeholders = {"question": full_question_block, "context": eval_context, "criteria_list_str": ", ".join(QUALITATIVE_METRICS)}
        temp_path = TEMP_UPLOAD_DIR / f"{job_id_for_log}_qualitative_eval_{uuid.uuid4().hex[:6]}.txt"
        fill_placeholders(QUALITATIVE_EVAL_PROMPT_PATH, temp_path, placeholders)
        eval_prompt = temp_path.read_text(encoding='utf-8')
        temp_path.unlink(missing_ok=True)
        eval_system_prompt = "You are an AI evaluating question quality. Evaluate the *entire* question block. Respond ONLY with a valid JSON object with boolean values (true/false) for criteria: " + ", ".join(QUALITATIVE_METRICS) + "."
        response_text = get_gemini_response(eval_system_prompt, eval_prompt, is_json_output=True, job_id_for_log=job_id_for_log)
        if response_text.startswith("Error:"): logger.error(f"[{job_id_for_log}] LLM qualitative evaluation failed: {response_text}"); return results
        try:
            cleaned_response = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
            eval_results = json.loads(cleaned_response)
            if not isinstance(eval_results, dict): raise ValueError("LLM response not a JSON object.")
            for metric in QUALITATIVE_METRICS:
                value = eval_results.get(metric)
                results[metric] = value if isinstance(value, bool) else False
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[{job_id_for_log}] Failed to parse/validate JSON from LLM qualitative eval: {e}. Response: {response_text}")
            return {metric: False for metric in QUALITATIVE_METRICS}
        return results
    except FileNotFoundError as e: logger.error(f"[{job_id_for_log}] Qualitative eval prompt template missing: {e}"); return results
    except Exception as e: logger.error(f"[{job_id_for_log}] Error during qualitative evaluation for '{full_question_block[:100]}...': {e}", exc_info=True); return results

# cleanup_job_files (Adapted for standalone script)
def cleanup_job_files(job_id_for_log: str, temp_dirs_to_remove: List[Path], temp_files_to_remove: List[Path]):
    logger.info(f"[{job_id_for_log}] Cleaning up temporary files and directories...")
    # Delete specific files (like temp prompts)
    for file_path in temp_files_to_remove:
        try:
            if file_path.exists(): file_path.unlink(); logger.info(f"[{job_id_for_log}] Deleted temp file: {file_path}")
        except Exception as e: logger.warning(f"[{job_id_for_log}] Error deleting temp file {file_path}: {e}")
    # Delete directories (like job-specific image dir)
    for dir_path in temp_dirs_to_remove:
        try:
            if dir_path.exists(): shutil.rmtree(dir_path); logger.info(f"[{job_id_for_log}] Removed temp directory: {dir_path}")
        except Exception as e: logger.warning(f"[{job_id_for_log}] Error deleting temp dir {dir_path}: {e}")
    logger.info(f"[{job_id_for_log}] Temporary file cleanup finished.")


# --- Main Execution Logic ---
def run_standalone_test(args):
    """Runs the main processing pipeline for a single PDF."""
    job_id = f"standalone_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{job_id}] Starting standalone test run for PDF: {args.pdf_path}")
    logger.info(f"[{job_id}] Parameters: {vars(args)}") # Log input parameters

    input_pdf_path = Path(args.pdf_path)
    if not input_pdf_path.exists() or not input_pdf_path.is_file():
        logger.error(f"[{job_id}] Input PDF file not found or is not a file: {input_pdf_path}")
        return

    # Define paths for this run's temporary data
    job_image_dir = EXTRACTED_IMAGES_DIR / job_id
    job_final_md_path = FINAL_RESULTS_DIR / f"{job_id}_processed.md"
    temp_files_for_cleanup = [] # Collect paths of temporary prompt files
    dirs_for_cleanup = [job_image_dir] # Collect directories to remove

    try:
        # 1. Call Datalab Marker
        logger.info(f"[{job_id}] --- Step 1: Calling Datalab Marker ---")
        datalab_result = call_datalab_marker(input_pdf_path, job_id)
        markdown_text = datalab_result.get("markdown", "")
        images_dict = datalab_result.get("images", {})
        if not markdown_text: logger.warning(f"[{job_id}] Datalab returned empty markdown content.")

        # 2. Save Images & Generate Descriptions
        logger.info(f"[{job_id}] --- Step 2: Saving Images & Generating Descriptions ---")
        # Note: Datalab sometimes uses document name in image keys, use a generic subfolder here
        doc_images_folder = job_image_dir / input_pdf_path.stem # Subfolder based on PDF name
        saved_images_map = save_extracted_images(images_dict, doc_images_folder, job_id)

        # 3. Process Markdown (Inject Descriptions)
        logger.info(f"[{job_id}] --- Step 3: Processing Markdown ---")
        final_markdown = process_markdown(markdown_text, saved_images_map, job_id)
        job_final_md_path.parent.mkdir(parents=True, exist_ok=True)
        job_final_md_path.write_text(final_markdown, encoding="utf-8")
        logger.info(f"[{job_id}] Saved processed markdown to: {job_final_md_path}")

        # 4. Chunk, Embed, Upsert to Qdrant
        processed_document_ids = []
        if final_markdown.strip():
            logger.info(f"[{job_id}] --- Step 4: Chunking, Embedding, Upserting ---")
            document_id = f"{job_id}_{input_pdf_path.stem}" # Unique ID for this doc in Qdrant
            chunks = hierarchical_chunk_markdown(final_markdown, input_pdf_path.name, job_id)
            if chunks:
                embeddings = embed_chunks(chunks, job_id)
                # Add metadata
                for chunk_data in chunks:
                    if 'metadata' not in chunk_data: chunk_data['metadata'] = {}
                    chunk_data['metadata']['document_id'] = document_id
                    chunk_data['metadata']['session_id'] = job_id # Use job_id as session_id
                upsert_to_qdrant(job_id, QDRANT_COLLECTION_NAME, embeddings, chunks)
                processed_document_ids.append(document_id)
                logger.info(f"[{job_id}] Successfully chunked, embedded, and upserted document ID: {document_id}")
            else:
                logger.warning(f"[{job_id}] No chunks generated from markdown.")
        else:
            logger.warning(f"[{job_id}] Skipping chunk/embed/upsert due to empty final markdown.")

        if not processed_document_ids:
            raise ValueError("Document processing did not result in any data being added to Qdrant.")

        # 5. Generate Hypothetical Text & Search
        logger.info(f"[{job_id}] --- Step 5: Generating Hypothetical Text & Searching ---")
        params_dict = vars(args) # Convert argparse Namespace to dict
        hypothetical_text = find_topics_and_generate_hypothetical_text(
            job_id, params_dict['academic_level'], params_dict['major'], params_dict['course_name'],
            params_dict['taxonomy_level'], params_dict['topics_list']
        )
        logger.info(f"[{job_id}] Hypothetical Text: {hypothetical_text}")
        query_embedding = model_st.encode(hypothetical_text)
        search_results = search_results_from_qdrant(
            job_id, QDRANT_COLLECTION_NAME, query_embedding,
            limit=args.retrieval_limit, score_threshold=args.similarity_threshold,
            session_id_filter=job_id, document_ids_filter=processed_document_ids
        )

        if not search_results:
            raise ValueError("No relevant context found in Qdrant for the generated query.")

        retrieved_context = "\n\n---\n\n".join([
            f"Context from: {r.payload.get('source', 'N/A')} (Score: {r.score:.4f}, DocID: {r.payload.get('document_id', 'N/A')})\n\n{r.payload.get('text', 'N/A')}"
            for r in search_results
        ])
        retrieved_context_preview = "\n\n".join([
            f"---\n**Context Snippet {i+1}** (Source: {r.payload.get('source', 'N/A')}, Score: {r.score:.3f})\n{r.payload.get('text', 'N/A')[:300]}...\n---"
            for i, r in enumerate(search_results[:3])
        ])
        print("\n" + "="*20 + " RETRIEVED CONTEXT PREVIEW " + "="*20)
        print(retrieved_context_preview)
        print("="*60 + "\n")

        # 6. Generate Initial Questions
        logger.info(f"[{job_id}] --- Step 6: Generating Questions ---")
        # Generate questions using the full params dict
        initial_questions = generate_initial_questions(job_id, retrieved_context, params_dict)

        print("\n" + "="*20 + " GENERATED QUESTIONS " + "="*20)
        print(initial_questions)
        print("="*60 + "\n")

        # 7. (Optional) Evaluate Generated Questions
        logger.info(f"[{job_id}] --- Step 7: Evaluating Questions ---")
        parsed_questions = parse_questions(initial_questions, job_id)
        evaluation_results = []
        if parsed_questions:
            for i, q_block in enumerate(parsed_questions):
                 q_eval = {"question_num": i + 1}
                 q_eval["qsts_score"] = evaluate_single_question_qsts(job_id, q_block, retrieved_context)
                 q_eval["qualitative"] = evaluate_single_question_qualitative(job_id, q_block, retrieved_context)
                 evaluation_results.append(q_eval)

            print("\n" + "="*20 + " EVALUATION RESULTS " + "="*20)
            passed_count = 0
            for res in evaluation_results:
                qsts_ok = res['qsts_score'] >= QSTS_THRESHOLD
                qual_ok = not any(not res['qualitative'].get(metric, True) for metric, must_be_true in CRITICAL_QUALITATIVE_FAILURES.items() if must_be_true is False)
                status = "PASS" if qsts_ok and qual_ok else "FAIL"
                if status == "PASS": passed_count += 1
                print(f"--- Question {res['question_num']} --- [Status: {status}]")
                print(f"  QSTS Score: {res['qsts_score']:.4f} ({'PASS' if qsts_ok else 'FAIL'})")
                print(f"  Qualitative:")
                for metric, passed in res['qualitative'].items():
                    critical_fail = metric in CRITICAL_QUALITATIVE_FAILURES and not passed
                    print(f"    - {metric}: {passed} {'(CRITICAL FAIL)' if critical_fail else ''}")
            print(f"\nOverall: {passed_count}/{len(evaluation_results)} questions passed all critical checks.")
            print("="*60 + "\n")
        else:
            logger.warning(f"[{job_id}] Could not parse generated questions for evaluation.")
            print("\n" + "="*20 + " EVALUATION RESULTS " + "="*20)
            print("Could not parse generated questions for evaluation.")
            print("="*60 + "\n")

        logger.info(f"[{job_id}] Standalone test run completed successfully.")

    except Exception as e:
        logger.exception(f"[{job_id}] Standalone test run failed: {e}")
        print(f"\nERROR during processing: {e}\n")

    finally:
        # Collect temporary prompt files for cleanup
        temp_files_for_cleanup.extend(list(TEMP_UPLOAD_DIR.glob(f"{job_id}_*.txt")))
        # 8. Cleanup
        logger.info(f"[{job_id}] --- Step 8: Cleaning Up ---")
        cleanup_job_files(job_id, dirs_for_cleanup, temp_files_for_cleanup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone script to test the PDF processing and question generation pipeline.")

    # Required Argument
    parser.add_argument("pdf_path", help="Path to the input PDF file.")

    # Processing Parameters (Match FastAPI Form)
    parser.add_argument("--course_name", default="Test Course", help="Name of the course.")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of questions to generate.")
    parser.add_argument("--academic_level", default="Undergraduate", help="Target academic level.")
    parser.add_argument("--taxonomy_level", default="Apply", help="Target Bloom's Taxonomy level.")
    parser.add_argument("--topics_list", default="Core Concepts, Applications", help="Comma-separated list of topics.")
    parser.add_argument("--major", default="Computer Science", help="Target major.")
    parser.add_argument("--retrieval_limit", type=int, default=15, help="Max context snippets to retrieve from Qdrant.")
    parser.add_argument("--similarity_threshold", type=float, default=0.3, help="Minimum similarity score for retrieved context.")
    parser.add_argument("--generate_diagrams", action='store_true', help="Flag to enable PlantUML diagram generation.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main test function
    run_standalone_test(args)