# -*- coding: utf-8 -*-
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
import asyncio
import urllib.parse
import json # For parsing LLM responses

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Text Processing & Embeddings
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util # Renamed to avoid confusion with general util

# Qdrant
from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# Moondream
import moondream as md

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv()

DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")
DATALAB_MARKER_URL = os.environ.get("DATALAB_MARKER_URL")
MOONDREAM_API_KEY = os.environ.get("MOONDREAM_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Directories
TEMP_UPLOAD_DIR = "temp_uploads"
RESULTS_DIR = "results"
FINAL_RESULTS_DIR = "final_results"
EXTRACTED_IMAGES_DIR = "extracted_images"

# API Timeouts and Polling
DATALAB_POST_TIMEOUT = 60
DATALAB_POLL_TIMEOUT = 30
MAX_POLLS = 300
POLL_INTERVAL = 3
GEMINI_TIMEOUT = 180

# Qdrant Configuration
QDRANT_COLLECTION_NAME = "markdown_docs_v3_semantic" # New collection name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Gemini Configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Model for generation and evaluation
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:{action}?key={api_key}"

# Evaluation Configuration
QSTS_THRESHOLD = 0.3 # Similarity threshold for a question vs context
# Qualitative metrics to check per question using LLM
QUALITATIVE_METRICS = [
    "Understandable", "TopicRelated", "Grammatical", "Clear", "Answerable", "Central"
    # Removed "WouldYouUseIt", "SkillLevel" as they are subjective/harder for LLM
]
# Critical qualitative metrics - if any fail, trigger regeneration
CRITICAL_QUALITATIVE_FAILURES = {"Grammatical": False, "Answerable": False, "Understandable": False, "Clear": False}

# Prompt File Paths
PROMPT_DIR = Path("content")
# Ensure these files exist in the 'content' directory
FINAL_USER_PROMPT_PATH = PROMPT_DIR / "final_user_prompt.txt" # Base for initial generation
HYPOTHETICAL_PROMPT_PATH = PROMPT_DIR / "hypothetical_prompt.txt" # Base for hypo text
QUALITATIVE_EVAL_PROMPT_PATH = PROMPT_DIR / "qualitative_eval_prompt.txt" # New prompt for LLM evaluation

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize Models and Clients (Global Scope) ---
try:
    logger.info("Initializing Moondream model...")
    model_md = md.vl(api_key=MOONDREAM_API_KEY)

    logger.info(f"Initializing Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
    model_st = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    # Ensure Qdrant Collection Exists
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception as e:
        if "404" in str(e) or "Not found" in str(e):
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")
        else:
             logger.error(f"Error checking/creating Qdrant collection: {e}")
             raise

    # NLTK Setup
    for resource in ['stopwords', 'wordnet', 'punkt']: # Add 'punkt' for sentence splitting
        try:
            nltk.data.find(f'corpora/{resource}') if resource != 'punkt' else nltk.data.find(f'tokenizers/{resource}')
        except (LookupError, nltk.downloader.DownloadError):
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

except Exception as e:
    logger.critical(f"Fatal error during initialization: {e}", exc_info=True)
    sys.exit(1)

# --- FastAPI App Setup ---
app = FastAPI(title="Semantic PDF Question Generator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure necessary directories exist
for dir_path in [TEMP_UPLOAD_DIR, RESULTS_DIR, FINAL_RESULTS_DIR, EXTRACTED_IMAGES_DIR, PROMPT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Check for mandatory prompt files
mandatory_prompts = [FINAL_USER_PROMPT_PATH, HYPOTHETICAL_PROMPT_PATH, QUALITATIVE_EVAL_PROMPT_PATH]
missing_prompts = [p.name for p in mandatory_prompts if not p.exists()]
if missing_prompts:
    logger.critical(f"FATAL ERROR: Missing required prompt template files in '{PROMPT_DIR}': {', '.join(missing_prompts)}")
    # Optionally create dummy files to allow startup for development
    # for p in mandatory_prompts: p.touch() # Uncomment to create empty files if missing
    sys.exit(1) # Exit if prompts are missing


# In-memory storage (Replace with persistent storage for production)
job_storage: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class JobResultData(BaseModel):
    generated_questions: Optional[str] = None # Final questions
    evaluation_feedback: Optional[str] = None # Text summary of evaluation and regeneration decision
    scores: Optional[Dict[str, float]] = None # DEPRECATED - will be removed or changed
    per_question_evaluation: Optional[List[Dict[str, Any]]] = None # Detailed eval per question
    retrieved_context_preview: Optional[str] = None
    extracted_markdown: Optional[str] = None
    initial_questions: Optional[str] = None

class Job(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    result: Optional[JobResultData] = None

class RegenerationRequest(BaseModel):
    feedback: str = Field(..., min_length=1, description="User feedback for question regeneration")

# --- Helper Functions ---

# >> Keep implementations of:
# generate_description_for_image
# call_datalab_marker
# save_extracted_images
# process_markdown
# clean_text_for_embedding
# hierarchical_chunk_markdown
# embed_chunks
# upsert_to_qdrant
# fill_placeholders
# find_topics_and_generate_hypothetical_text
# search_results_from_qdrant
# generate_initial_questions
# cleanup_job_files
# (Ensure they are robust and log with job_id)
# << (Paste the full function code here from previous steps, ensuring logging and error handling are correct)

def generate_description_for_image(image_path, figure_caption=""):
    """Load an image, encode it, and query Moondream for a description."""
    try:
        image = Image.open(image_path)
        encoded_image = model_md.encode_image(image)
        query_text = (
            f"Describe the key technical findings in this figure/visualization "
            f"captioned: {figure_caption} using natural language. Illustrate and mention trends, "
            f"patterns, and numerical values that can be observed. Provide a scientific/academic styled short, "
            f"single paragraph summary that is highly insightful in context of the document."
        )
        response = model_md.query(encoded_image, query_text)
        description = response.get("answer", "No description available.")
        description = description.replace('\n', ' ').strip()
        return description
    except FileNotFoundError:
         logger.error(f"Image file not found at {image_path}")
         return f"Error: Image file not found."
    except Exception as e:
        logger.error(f"Error generating description for {Path(image_path).name}: {e}")
        return f"Error generating description for this image."

def call_datalab_marker(file_path: Path):
    """Call Datalab marker endpoint."""
    logger.info(f"Calling Datalab Marker API for {file_path.name}...")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        form_data = {
            "langs": (None, "English"), "force_ocr": (None, False), "paginate": (None, False),
            "output_format": (None, "markdown"), "use_llm": (None, False),
            "strip_existing_ocr": (None, False), "disable_image_extraction": (None, False)
        }
        headers = {"X-Api-Key": DATALAB_API_KEY}
        try:
            response = requests.post(DATALAB_MARKER_URL, files=files, data=form_data, headers=headers, timeout=DATALAB_POST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Datalab API request timed out for {file_path.name}")
            raise TimeoutError("Datalab API request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Datalab API request failed for {file_path.name}: {e}")
            raise Exception(f"Datalab API request failed: {e}")

    if not data.get("success"):
        err_msg = data.get('error', 'Unknown Datalab error')
        logger.error(f"Datalab API error for {file_path.name}: {err_msg}")
        raise Exception(f"Datalab API error: {err_msg}")

    check_url = data["request_check_url"]
    logger.info(f"Polling Datalab result URL for {file_path.name}: {check_url}")
    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=DATALAB_POLL_TIMEOUT)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("status") == "complete":
                logger.info(f"Datalab processing complete for {file_path.name}.")
                return poll_data
            elif poll_data.get("status") == "error":
                 err_msg = poll_data.get('error', 'Unknown Datalab processing error')
                 logger.error(f"Datalab processing failed for {file_path.name}: {err_msg}")
                 raise Exception(f"Datalab processing failed: {err_msg}")
            if (i + 1) % 10 == 0:
                 logger.info(f"Polling Datalab for {file_path.name}... attempt {i+1}/{MAX_POLLS}")
        except requests.exceptions.Timeout:
             logger.warning(f"Polling Datalab timed out on attempt {i+1} for {file_path.name}. Retrying...")
        except requests.exceptions.RequestException as e:
             logger.warning(f"Polling error on attempt {i+1} for {file_path.name}: {e}. Retrying...")
             time.sleep(1)

    logger.error(f"Polling timed out waiting for Datalab processing for {file_path.name}.")
    raise TimeoutError("Polling timed out waiting for Datalab processing.")

def save_extracted_images(images_dict, images_folder: Path):
    """Decode and save base64 encoded images."""
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    logger.info(f"Saving {len(images_dict)} extracted images to {images_folder}...")
    for img_name, b64_data in images_dict.items():
        try:
            safe_img_name = "".join([c if c.isalnum() or c in ('-', '_', '.') else '_' for c in img_name])
            if not safe_img_name: safe_img_name = f"image_{uuid.uuid4().hex[:8]}.png"
            image_data = base64.b64decode(b64_data)
            image_path = images_folder / safe_img_name
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)
            saved_files[img_name] = str(image_path)
        except Exception as e:
            logger.warning(f"Could not decode/save image {img_name}: {e}")
    return saved_files

def process_markdown(markdown_text, saved_images: Dict[str, str], job_id: str):
    """Process markdown: replace image placeholders with descriptions."""
    logger.info(f"[{job_id}] Processing markdown for image descriptions...")
    lines = markdown_text.splitlines()
    processed_lines = []
    i = 0
    image_count = 0
    figure_pattern = re.compile(r"^!\[.*?\]\((.*?)\)$")
    caption_pattern = re.compile(r"^(Figure|Table|Chart)\s?(\d+[:.]?)\s?(.*)", re.IGNORECASE)

    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        image_match = figure_pattern.match(stripped_line)

        if image_match:
            image_filename_encoded = image_match.group(1)
            try:
                image_filename_decoded = urllib.parse.unquote(image_filename_encoded)
            except Exception:
                 image_filename_decoded = image_filename_encoded

            image_count += 1
            caption = ""
            caption_line_index = -1

            j = i + 1
            while j < len(lines) and lines[j].strip() == "": j += 1
            if j < len(lines):
                next_line_stripped = lines[j].strip()
                if caption_pattern.match(next_line_stripped):
                    caption = next_line_stripped
                    caption_line_index = j

            image_path = saved_images.get(image_filename_decoded)
            if not image_path: image_path = saved_images.get(image_filename_encoded)

            description = ""
            if image_path:
                description = generate_description_for_image(image_path, caption)
            else:
                description = f"*Referenced image '{image_filename_decoded}' not found.*"
                logger.warning(f"[{job_id}] {description}")

            title_text = caption if caption else f"Figure {image_count}"
            block_text = f"\n---\n### {title_text}\n\n**Figure Description:**\n{description}\n---\n"
            processed_lines.append(block_text)

            if caption_line_index != -1: i = caption_line_index
        else:
            processed_lines.append(line)
        i += 1

    logger.info(f"[{job_id}] Finished processing markdown. Processed {image_count} image references.")
    return "\n".join(processed_lines)

def clean_text_for_embedding(text):
    """Basic text cleaning for embedding."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    return text.strip()

def hierarchical_chunk_markdown(markdown_text, source_filename):
    """Chunks markdown text based on headers (H1-H6) and figure blocks."""
    logger.info(f"Chunking final markdown hierarchically from source: {source_filename}")
    lines = markdown_text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_headers = {}
    figure_title = None

    header_pattern = re.compile(r"^(#{1,6})\s+(.*)")
    figure_title_pattern = re.compile(r"^###\s+((?:Figure|Table|Chart).*)$", re.IGNORECASE)
    figure_desc_start_pattern = re.compile(r"^\*\*Figure Description:\*\*", re.IGNORECASE)
    separator_pattern = re.compile(r"^---$")

    for line_num, line in enumerate(lines):
        stripped_line = line.strip()
        header_match = header_pattern.match(stripped_line)
        figure_title_match = figure_title_pattern.match(stripped_line)
        separator_match = separator_pattern.match(stripped_line)

        if header_match or figure_title_match or separator_match:
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                if cleaned_chunk_text:
                    metadata = {
                        "source": source_filename,
                        **{f"h{level}": title for level, title in current_headers.items()},
                    }
                    if figure_title: metadata["figure_title"] = figure_title
                    chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                current_chunk_lines = []

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_headers = {k: v for k, v in current_headers.items() if k < level}
                current_headers[level] = title
                figure_title = None
                current_chunk_lines.append(line)
            elif figure_title_match:
                figure_title = figure_title_match.group(1).strip()
                current_chunk_lines.append(line)
            elif separator_match:
                 figure_title = None
        else:
            current_chunk_lines.append(line)

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        cleaned_chunk_text = clean_text_for_embedding(chunk_text)
        if cleaned_chunk_text:
            metadata = {
                "source": source_filename,
                **{f"h{level}": title for level, title in current_headers.items()},
            }
            if figure_title: metadata["figure_title"] = figure_title
            chunks.append({"text": cleaned_chunk_text, "metadata": metadata})

    logger.info(f"Generated {len(chunks)} hierarchical chunks for {source_filename}.")
    return chunks

def embed_chunks(chunks_data, model):
    """Embeds the 'text' field of each chunk dictionary."""
    if not chunks_data: return []
    logger.info(f"Embedding {len(chunks_data)} text chunks...")
    texts_to_embed = [chunk['text'] for chunk in chunks_data]
    try:
        embeddings = model.encode(texts_to_embed, show_progress_bar=False).tolist()
        logger.info("Embedding complete.")
        return embeddings
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        raise

def upsert_to_qdrant(job_id: str, collection_name, embeddings, chunks_data, batch_size=100):
    """Upserts chunks into Qdrant."""
    if not embeddings or not chunks_data: return 0
    logger.info(f"[{job_id}] Upserting {len(embeddings)} points to Qdrant collection '{collection_name}'...")
    total_points_upserted = 0
    points_to_upsert = []
    for embedding, chunk_data in zip(embeddings, chunks_data):
        if isinstance(chunk_data.get('metadata'), dict):
            payload = chunk_data['metadata'].copy()
            payload["text"] = chunk_data['text']
            point_id = str(uuid.uuid4())
            points_to_upsert.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        else:
             logger.warning(f"[{job_id}] Skipping chunk due to invalid metadata format: {chunk_data.get('metadata')}")

    for i in range(0, len(points_to_upsert), batch_size):
        batch_points = points_to_upsert[i:i + batch_size]
        if not batch_points: continue
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch_points, wait=True)
            batch_count = len(batch_points)
            total_points_upserted += batch_count
            logger.info(f"[{job_id}] Upserted Qdrant batch {i // batch_size + 1} ({batch_count} points). Total: {total_points_upserted}")
        except Exception as e:
            logger.error(f"[{job_id}] Error upserting Qdrant batch {i // batch_size + 1}: {e}")
            raise Exception(f"Failed to upsert batch to Qdrant: {e}")

    logger.info(f"[{job_id}] Finished upserting. Total points upserted: {total_points_upserted}")
    return total_points_upserted

def fill_placeholders(template_path: Path, output_path: Path, placeholders: Dict):
    """Fills placeholders in a template file."""
    try:
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        template = template_path.read_text(encoding='utf-8')
        for placeholder, value in placeholders.items():
            template = template.replace(f"{{{placeholder}}}", str(value))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template, encoding='utf-8')
        logger.info(f"Filled placeholders and saved updated prompt to {output_path}")
    except Exception as e:
        logger.error(f"Error filling placeholders for {template_path}: {e}")
        raise

def get_gemini_response(system_prompt: str, user_prompt: str, is_json_output: bool = False):
    """Gets a response from the Google Gemini API, optionally expecting JSON."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set.")
        raise ValueError("Gemini API Key not configured.")

    api_url = GEMINI_API_URL_TEMPLATE.format(model_name=GEMINI_MODEL_NAME, action="generateContent", api_key=GEMINI_API_KEY)
    headers = {'Content-Type': 'application/json'}

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
         "generationConfig": {
             "temperature": 0.5 if is_json_output else 0.7, # Lower temp for structured output
             "maxOutputTokens": 4096,
             # Add response_mime_type for Gemini models that support it
             "responseMimeType": "application/json" if is_json_output else "text/plain",
         }
    }
    # Remove responseMimeType if expecting plain text, as it might cause errors if model defaults to text
    if not is_json_output:
        payload["generationConfig"].pop("responseMimeType", None)


    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
        response.raise_for_status()
        response_data = response.json()

        if 'candidates' in response_data and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                gemini_response_text = candidate['content']['parts'][0].get('text', '')
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                if finish_reason not in ['STOP', 'MAX_TOKENS']:
                    logger.warning(f"Gemini generation finished reason: {finish_reason}")

                if gemini_response_text:
                    return gemini_response_text.strip()
                else:
                    logger.warning(f"Gemini response part was empty. Finish reason: {finish_reason}")
                    return f"Error: Gemini returned an empty response (finish reason: {finish_reason})."
            # ... (rest of error handling for finishReason, safetyRatings) ...
            elif 'finishReason' in candidate:
                 reason = candidate['finishReason']
                 logger.error(f"Gemini generation failed/stopped. Reason: {reason}")
                 # ... safety ratings check ...
                 return f"Error: Generation stopped by Gemini - {reason}"
            else:
                 logger.warning("Gemini response structure unexpected (no content/parts).")
                 return "Error: Could not extract text from Gemini response structure."

        # ... (rest of error handling for promptFeedback, unexpected format) ...
        elif 'promptFeedback' in response_data and 'blockReason' in response_data['promptFeedback']:
             block_reason = response_data['promptFeedback']['blockReason']
             logger.error(f"Gemini prompt blocked. Reason: {block_reason}")
             return f"Error: Prompt blocked by Gemini - {block_reason}"
        else:
            logger.error(f"Unexpected Gemini API response format: {response_data}")
            return "Error: Unexpected Gemini API response structure."


    except requests.exceptions.Timeout:
        logger.error("Gemini API request timed out.")
        return "Error: Gemini API request timed out."
    except requests.exceptions.RequestException as e:
        err_detail = str(e)
        if e.response is not None:
             try: err_detail = f"{err_detail} - Response: {e.response.text[:500]}"
             except Exception: pass
        logger.error(f"Error calling Gemini API: {err_detail}")
        return f"Error: Gemini API request failed - {e}"
    except Exception as e:
        logger.exception(f"Error processing Gemini response: {e}")
        return f"Error: Failed to process Gemini response - {e}"

def find_topics_and_generate_hypothetical_text(job_id: str, academic_level, major, course_name, taxonomy_level, topics):
    """Generates hypothetical text based on topics using Gemini."""
    logger.info(f"[{job_id}] Generating hypothetical text...")
    try:
        temp_updated_hypothetical_path = Path(TEMP_UPLOAD_DIR) / f"{job_id}_updated_hypothetical.txt"
        placeholders = {
            "course_name": course_name, "academic_level": academic_level, "topics": topics,
            "major": major, "taxonomy_level": taxonomy_level
        }
        fill_placeholders(HYPOTHETICAL_PROMPT_PATH, temp_updated_hypothetical_path, placeholders)
        updated_hypothetical_prompt = temp_updated_hypothetical_path.read_text(encoding="utf8")
        system_prompt = f"You are an AI assistant specializing in educational content for {major} at the {academic_level} level. Generate a concise, hypothetical query or student question related to the course '{course_name}' focusing on these topics: {topics}. The query should reflect the cognitive complexity of Bloom's taxonomy level: {taxonomy_level}."
        hypothetical_text = get_gemini_response(system_prompt, updated_hypothetical_prompt)
        temp_updated_hypothetical_path.unlink(missing_ok=True)

        if hypothetical_text.startswith("Error:"):
            raise Exception(f"Hypothetical text generation failed: {hypothetical_text}")
        logger.info(f"[{job_id}] Successfully generated hypothetical text.")
        return hypothetical_text
    except FileNotFoundError as e:
        logger.error(f"[{job_id}] Hypothetical prompt template not found: {e}")
        raise Exception("Hypothetical prompt template file missing.")
    except Exception as e:
        logger.error(f"[{job_id}] Error generating hypothetical text: {e}")
        raise Exception(f"Error generating hypothetical text: {e}")

def search_results_from_qdrant(job_id: str, collection_name, embedded_vector, limit=15, score_threshold: Optional[float] = None, session_id_filter=None, document_ids_filter=None):
    """Searches Qdrant using an embedded vector, with optional filters and threshold."""
    logger.info(f"[{job_id}] Searching Qdrant collection '{collection_name}' with limit={limit}, threshold={score_threshold}...")
    must_conditions = []
    if session_id_filter: must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id_filter)))
    if document_ids_filter:
        if not isinstance(document_ids_filter, list): document_ids_filter = [document_ids_filter]
        if document_ids_filter: must_conditions.append(FieldCondition(key="document_id", match=MatchAny(any=document_ids_filter)))
    query_filter = Filter(must=must_conditions) if must_conditions else None

    try:
        query_vector_list = embedded_vector.tolist() if hasattr(embedded_vector, 'tolist') else list(map(float, embedded_vector))
        search_results = qdrant_client.search(
            collection_name=collection_name, query_vector=query_vector_list, query_filter=query_filter,
            limit=limit, score_threshold=score_threshold, with_payload=True, with_vectors=False
        )
        count = len(search_results)
        logger.info(f"[{job_id}] Qdrant search returned {count} results.")
        if count > 0: logger.info(f"[{job_id}] Top hit score: {search_results[0].score:.4f}")
        return search_results
    except Exception as e:
        logger.error(f"[{job_id}] Error searching Qdrant: {e}")
        return []

def generate_initial_questions(job_id: str, retrieved_context: str, params: Dict):
    """Generates the initial set of questions using Gemini."""
    logger.info(f"[{job_id}] Preparing to generate initial questions...")
    blooms_taxonomy_descriptions = """
    Bloom's Taxonomy Levels:
    - Remember: Recall facts and basic concepts (define, list, memorize).
    - Understand: Explain ideas or concepts (classify, describe, discuss).
    - Apply: Use information in new situations (execute, implement, solve).
    - Analyze: Draw connections among ideas (differentiate, organize, relate).
    - Evaluate: Justify a stand or decision (appraise, argue, defend).
    - Create: Produce new or original work (design, assemble, construct).
    """
    max_context_chars = 25000
    truncated_context = retrieved_context[:max_context_chars]
    if len(retrieved_context) > max_context_chars:
        logger.warning(f"[{job_id}] Truncating retrieved context from {len(retrieved_context)} to {max_context_chars} characters for LLM prompt.")

    placeholders_final = {
        "content": truncated_context, "num_questions": params['num_questions'], "course_name": params['course_name'],
        "taxonomy": params['taxonomy_level'], "major": params['major'], "academic_level": params['academic_level'],
        "topics_list": params['topics_list'], "blooms_taxonomy_descriptions": blooms_taxonomy_descriptions,
    }

    try:
        temp_updated_final_user_path = Path(TEMP_UPLOAD_DIR) / f"{job_id}_initial_user_prompt.txt"
        if not FINAL_USER_PROMPT_PATH.exists(): raise FileNotFoundError(f"Prompt template not found: {FINAL_USER_PROMPT_PATH}")
        fill_placeholders(FINAL_USER_PROMPT_PATH, temp_updated_final_user_path, placeholders_final)
        final_user_prompt_content = temp_updated_final_user_path.read_text(encoding="utf8")

        system_prompt = f"You are an AI assistant specialized in creating educational assessment questions for a {params['academic_level']}-level {params['major']} course titled '{params['course_name']}'. Generate exactly {params['num_questions']} high-quality questions based *strictly* on the provided context, aligned with Bloom's taxonomy level: {params['taxonomy_level']}. Focus on the topics: {params['topics_list']}. Ensure questions are clear, unambiguous, and directly answerable from the context. Output the questions as a numbered list."
        job_storage[job_id]["generation_prompts"] = {
            "user_prompt_content": final_user_prompt_content,
            "system_prompt": system_prompt
        }

        logger.info(f"[{job_id}] Generating initial questions via Gemini...")
        initial_questions = get_gemini_response(system_prompt, final_user_prompt_content)
        temp_updated_final_user_path.unlink(missing_ok=True)

        if initial_questions.startswith("Error:"):
            raise Exception(f"Gemini Error: {initial_questions}")

        logger.info(f"[{job_id}] Successfully generated initial questions.")
        return initial_questions

    except FileNotFoundError as e:
         logger.error(f"[{job_id}] Final user prompt template not found: {e}")
         raise Exception("Core prompt template file missing.")
    except Exception as e:
        logger.exception(f"[{job_id}] Unexpected error during initial question generation: {e}")
        raise Exception(f"Initial generation failed unexpectedly: {e}")

def cleanup_job_files(job_id: str, original_file_paths: List[str]):
    """Cleans up temporary files associated with a job."""
    logger.info(f"[{job_id}] Cleaning up temporary files...")
    for file_path_str in original_file_paths:
        try: Path(file_path_str).unlink(missing_ok=True)
        except Exception as e: logger.warning(f"[{job_id}] Error deleting temp file {file_path_str}: {e}")

    job_image_dir = Path(EXTRACTED_IMAGES_DIR) / job_id
    if job_image_dir.exists():
        try:
            shutil.rmtree(job_image_dir)
            logger.info(f"[{job_id}] Removed temp image directory: {job_image_dir}")
        except Exception as e: logger.warning(f"[{job_id}] Error deleting temp image dir {job_image_dir}: {e}")

    for prompt_file in Path(TEMP_UPLOAD_DIR).glob(f"{job_id}_*.txt"):
         try: prompt_file.unlink(missing_ok=True)
         except Exception as e: logger.warning(f"[{job_id}] Error deleting temp prompt file {prompt_file}: {e}")
    logger.info(f"[{job_id}] Temporary file cleanup finished.")


# --- New Evaluation Helper Functions ---

def parse_questions(question_block: str) -> List[str]:
    """Splits a block of text into individual questions, assuming numbered list format."""
    if not question_block: return []
    # Regex to find lines starting with number, period, optional space
    # Handles potential variations like "1.", "1)", "1 -" etc. followed by text
    questions = re.findall(r"^\s*\d+[\.\)\-]\s*(.*)", question_block, re.MULTILINE)
    if not questions:
        # Fallback: Split by newline if no numbered list found, filter empty lines
        questions = [q.strip() for q in question_block.splitlines() if q.strip()]
    return [q.strip() for q in questions if q.strip()] # Ensure no empty strings

def evaluate_single_question_qsts(job_id: str, question: str, context: str) -> float:
    """Calculates QSTS score between a single question and the context."""
    if not question or not context: return 0.0
    try:
        q_emb = model_st.encode(question)
        c_emb = model_st.encode(context) # Consider pre-embedding context once?
        score = sbert_util.pytorch_cos_sim(q_emb, c_emb).item()
        qsts_score = round(max(-1.0, min(1.0, score)), 4)
        # logger.debug(f"[{job_id}] QSTS for question '{question[:50]}...': {qsts_score}")
        return qsts_score
    except Exception as e:
        logger.warning(f"[{job_id}] Error calculating QSTS for question '{question[:50]}...': {e}")
        return 0.0

def evaluate_single_question_qualitative(job_id: str, question: str, context: str) -> Dict[str, bool]:
    """Uses LLM to evaluate qualitative aspects of a single question."""
    results = {metric: False for metric in QUALITATIVE_METRICS} # Default to False
    if not question or not context:
        logger.warning(f"[{job_id}] Cannot evaluate qualitative metrics for empty question/context.")
        return results

    try:
        # Limit context for the evaluation prompt
        eval_context_limit = 2000
        eval_context = context[:eval_context_limit]
        if len(context) > eval_context_limit:
            eval_context += "\n... [Context Truncated]"

        # Prepare prompt using the template
        placeholders = {
            "question": question,
            "context": eval_context,
            "criteria_list_str": ", ".join(QUALITATIVE_METRICS)
        }
        temp_eval_prompt_path = Path(TEMP_UPLOAD_DIR) / f"{job_id}_qualitative_eval_prompt_filled.txt"
        fill_placeholders(QUALITATIVE_EVAL_PROMPT_PATH, temp_eval_prompt_path, placeholders)
        eval_prompt = temp_eval_prompt_path.read_text(encoding='utf-8')
        temp_eval_prompt_path.unlink(missing_ok=True) # Clean up filled prompt

        # Define system prompt for the evaluator LLM
        eval_system_prompt = "You are an AI assistant evaluating the quality of an educational question based on provided context and specific criteria. Respond ONLY with a valid JSON object containing boolean values (true/false) for each criterion."

        # logger.debug(f"[{job_id}] Sending qualitative eval prompt for question: {question[:50]}...")
        response_text = get_gemini_response(eval_system_prompt, eval_prompt, is_json_output=True)

        if response_text.startswith("Error:"):
            logger.error(f"[{job_id}] LLM qualitative evaluation failed: {response_text}")
            return results # Return defaults on error

        # Attempt to parse the JSON response
        try:
            # Clean the response text - remove potential markdown code blocks
            cleaned_response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            eval_results = json.loads(cleaned_response_text)
            if not isinstance(eval_results, dict):
                raise ValueError("LLM response is not a JSON object.")

            # Update results, ensuring keys match and values are boolean
            for metric in QUALITATIVE_METRICS:
                value = eval_results.get(metric)
                if isinstance(value, bool):
                    results[metric] = value
                else:
                     logger.warning(f"[{job_id}] Invalid or missing value for metric '{metric}' in LLM eval response: {value}. Defaulting to False.")
                     results[metric] = False # Default if type is wrong or missing

        except json.JSONDecodeError as e:
            logger.error(f"[{job_id}] Failed to decode JSON response from LLM qualitative evaluation: {e}. Response was: {response_text}")
            return results # Return defaults if JSON parsing fails
        except ValueError as e:
             logger.error(f"[{job_id}] Invalid format in LLM qualitative evaluation response: {e}. Response was: {response_text}")
             return results

        # logger.debug(f"[{job_id}] Qualitative results for '{question[:50]}...': {results}")
        return results

    except FileNotFoundError as e:
        logger.error(f"[{job_id}] Qualitative evaluation prompt template not found: {e}")
        # Don't raise, just return default False values
        return results
    except Exception as e:
        logger.exception(f"[{job_id}] Unexpected error during qualitative evaluation for question '{question[:50]}...': {e}")
        return results # Return defaults on other errors


# --- Background Task Functions (Modified) ---

def run_processing_job(job_id: str, file_paths: List[str], params: Dict):
    """Main background task: Process docs, generate initial Qs, await feedback."""
    logger.info(f"[{job_id}] Background job started with params: {params}")
    job_storage[job_id]["status"] = "processing"
    job_storage[job_id]["message"] = "Starting document processing..."

    processed_document_ids = []
    session_id = job_id
    all_final_markdown = ""
    retrieved_context = ""
    error_occurred = False
    error_message = ""

    try:
        # STEP 1: Process PDFs (Datalab, Images, Markdown, Qdrant)
        # ... (Keep the loop from previous version, ensure it populates all_final_markdown and processed_document_ids) ...
        for i, file_path_str in enumerate(file_paths):
            file_path = Path(file_path_str)
            if not file_path.exists(): continue
            job_storage[job_id]["message"] = f"Processing file {i+1}/{len(file_paths)}: {file_path.name}..."
            base_name, _ = os.path.splitext(file_path.name)
            document_id = str(uuid.uuid4())
            try:
                data = call_datalab_marker(file_path)
                markdown_text = data.get("markdown", "")
                images_dict = data.get("images", {})
                images_folder = Path(EXTRACTED_IMAGES_DIR) / job_id / base_name
                saved_images = save_extracted_images(images_dict, images_folder)
                final_markdown = process_markdown(markdown_text, saved_images, job_id)
                all_final_markdown += f"\n\n## --- Document: {file_path.name} ---\n\n" + final_markdown
                # Optional: Save individual final markdown
                output_markdown_path = Path(FINAL_RESULTS_DIR) / f"{job_id}_{base_name}_final.md"
                output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
                output_markdown_path.write_text(final_markdown, encoding="utf-8")

                if final_markdown.strip():
                    chunks = hierarchical_chunk_markdown(final_markdown, file_path.name)
                    if chunks:
                        embeddings = embed_chunks(chunks, model_st)
                        for chunk_data in chunks:
                            chunk_data['metadata']['document_id'] = document_id
                            chunk_data['metadata']['session_id'] = session_id
                        upsert_to_qdrant(job_id, QDRANT_COLLECTION_NAME, embeddings, chunks)
                        processed_document_ids.append(document_id)
                    else: logger.warning(f"[{job_id}][{file_path.name}] No chunks generated.")
                else: logger.warning(f"[{job_id}][{file_path.name}] Final markdown empty.")
                logger.info(f"[{job_id}] Finished processing {file_path.name}.")
            except TimeoutError as e: error_message = f"Timeout processing {file_path.name}: {e}"; error_occurred = True; break
            except Exception as e: error_message = f"Error processing {file_path.name}: {e}"; logger.exception(f"[{job_id}] Error detail"); error_occurred = True; break

        # Check for errors after loop
        if error_occurred: raise Exception(f"Job failed during document processing. Last error: {error_message}")
        if not processed_document_ids: raise ValueError("No documents were successfully processed and indexed.")

        # STEP 2: Generate Hypothetical Text & Search Qdrant
        job_storage[job_id]["message"] = "Generating hypothetical text..."
        hypothetical_text = find_topics_and_generate_hypothetical_text(
            job_id, params['academic_level'], params['major'], params['course_name'], params['taxonomy_level'], params['topics_list']
        )
        job_storage[job_id]["message"] = "Searching context..."
        query_embedding = model_st.encode(hypothetical_text)
        search_results = search_results_from_qdrant(
            job_id, QDRANT_COLLECTION_NAME, query_embedding,
            limit=params['retrieval_limit'], score_threshold=params['similarity_threshold'],
            session_id_filter=session_id, document_ids_filter=processed_document_ids
        )
        if not search_results: raise ValueError("No relevant context found in vector database.")
        context_parts = [f"--- Context Snippet {i+1} (Source: {r.payload.get('source', 'N/A')}, Score: {r.score:.3f}) ---\n{r.payload.get('text', 'N/A')}" for i, r in enumerate(search_results)]
        retrieved_context = "\n\n".join(context_parts)
        job_storage[job_id]["retrieved_context"] = retrieved_context # Store full context
        logger.info(f"[{job_id}] Retrieved {len(search_results)} context snippets.")

        # STEP 3: Generate Initial Questions
        job_storage[job_id]["message"] = "Generating initial questions..."
        initial_questions = generate_initial_questions(job_id, retrieved_context, params)

        # STEP 4: Update Status and Result (Awaiting Feedback)
        job_storage[job_id]["status"] = "awaiting_feedback"
        job_storage[job_id]["message"] = "Initial questions generated. Please review and provide feedback."
        job_storage[job_id]["result"] = {
            "extracted_markdown": all_final_markdown.strip(),
            "initial_questions": initial_questions,
            "retrieved_context_preview": retrieved_context[:1000] + ("..." if len(retrieved_context) > 1000 else ""),
            "generated_questions": None, # Final values initially null
            "evaluation_feedback": None,
            "per_question_evaluation": None,
        }
        logger.info(f"[{job_id}] Job awaiting user feedback.")

    except Exception as e:
        logger.exception(f"[{job_id}] Job failed during initial processing: {e}")
        job_storage[job_id]["status"] = "error"
        job_storage[job_id]["message"] = f"An error occurred: {e}"
        if "result" in job_storage[job_id]: job_storage[job_id]["result"] = None
        # Ensure cleanup is called on error during this phase
        cleanup_job_files(job_id, job_storage[job_id].get("temp_file_paths", file_paths))


def run_regeneration_task(job_id: str, user_feedback: str):
    """Performs question evaluation, potential regeneration, and final evaluation."""
    logger.info(f"[{job_id}] Starting evaluation and regeneration task.")
    job_data = job_storage.get(job_id)
    if not job_data:
         logger.error(f"[{job_id}] Regeneration failed: Job data not found.")
         return

    file_paths_to_clean = job_data.get("temp_file_paths", [])

    try:
        job_data["status"] = "processing_feedback"
        job_data["message"] = "Evaluating initial questions..."

        # Retrieve necessary data
        retrieved_context = job_data.get("retrieved_context")
        initial_questions_block = job_data.get("result", {}).get("initial_questions")
        prompts = job_data.get("generation_prompts", {})
        original_user_prompt_filled = prompts.get("user_prompt_content")
        system_prompt = prompts.get("system_prompt")
        params = job_data.get("params", {}) # Get original params

        if not all([retrieved_context, initial_questions_block, original_user_prompt_filled, system_prompt, params]):
             raise ValueError("Missing necessary data stored from initial stage for evaluation/regeneration.")

        # Parse initial questions
        parsed_questions = parse_questions(initial_questions_block)
        if not parsed_questions:
            raise ValueError("Could not parse any questions from the initial generation.")
        logger.info(f"[{job_id}] Parsed {len(parsed_questions)} initial questions for evaluation.")

        # Evaluate each initial question
        evaluation_results = []
        needs_regeneration = False
        failed_question_details = []

        for i, question in enumerate(parsed_questions):
            q_eval = {"question_num": i + 1, "question_text": question}
            # 1. QSTS Score
            q_eval["qsts_score"] = evaluate_single_question_qsts(job_id, question, retrieved_context)
            # 2. Qualitative Metrics
            q_eval["qualitative"] = evaluate_single_question_qualitative(job_id, question, retrieved_context)

            evaluation_results.append(q_eval)

            # Check if this question fails criteria
            qsts_failed = q_eval["qsts_score"] < QSTS_THRESHOLD
            qualitative_failed = any(
                not q_eval["qualitative"].get(metric, True) # Fail if metric is False
                for metric, must_be_true in CRITICAL_QUALITATIVE_FAILURES.items() if must_be_true is False # Check only critical failures
            )

            if qsts_failed or qualitative_failed:
                needs_regeneration = True
                fail_reasons = []
                if qsts_failed: fail_reasons.append(f"QSTS below threshold ({q_eval['qsts_score']:.2f} < {QSTS_THRESHOLD})")
                if qualitative_failed:
                     failed_metrics = [m for m, passed in q_eval["qualitative"].items() if m in CRITICAL_QUALITATIVE_FAILURES and not passed]
                     fail_reasons.append(f"Failed qualitative checks: {', '.join(failed_metrics)}")
                failed_question_details.append(f"  - Question {i+1}: {'; '.join(fail_reasons)}")

        # Store per-question evaluation details
        job_data["result"]["per_question_evaluation"] = evaluation_results

        # --- Regeneration Decision ---
        final_questions_block = initial_questions_block
        final_evaluation_feedback = f"Initial evaluation summary ({len(parsed_questions)} questions):\n"
        final_evaluation_feedback += "\n".join([
            f"- Q{res['question_num']}: QSTS={res['qsts_score']:.2f}, Qual={ {k: ('Pass' if v else 'FAIL') for k,v in res['qualitative'].items()} }"
            for res in evaluation_results
        ]) + "\n"


        if needs_regeneration or user_feedback: # Regenerate if auto-eval failed OR user provided feedback
            logger.info(f"[{job_id}] Regeneration triggered. Needs_Regen_Flag={needs_regeneration}, UserFeedbackProvided={bool(user_feedback)}")
            job_data["message"] = "Regenerating questions based on evaluation and user feedback..."

            # Construct detailed feedback for the LLM
            llm_feedback = "The following issues were identified in the previous attempt:\n"
            if failed_question_details:
                llm_feedback += "Automatic Evaluation Failures:\n" + "\n".join(failed_question_details) + "\n"
            if user_feedback:
                llm_feedback += "User Provided Feedback:\n" + user_feedback + "\n"
            llm_feedback += "\nPlease regenerate the questions, addressing these points while adhering to all original instructions (context, Bloom's level, number of questions, topics)."

            # Create regeneration prompt
            regeneration_prompt = (
                f"{original_user_prompt_filled}\n\n" # Original filled prompt has context etc.
                f"--- FEEDBACK ON PREVIOUS ATTEMPT ---\n"
                f"{llm_feedback}\n\n"
                f"--- REGENERATION INSTRUCTIONS ---\n"
                f"Regenerate exactly {params['num_questions']} questions based on the original context and instructions, incorporating the feedback above."
            )

            # Call Gemini for Regeneration
            regenerated_questions_block = get_gemini_response(system_prompt, regeneration_prompt)

            if regenerated_questions_block.startswith("Error:"):
                 # Failed regeneration, keep initial questions but report error
                 logger.error(f"[{job_id}] Regeneration attempt failed: {regenerated_questions_block}")
                 final_evaluation_feedback += f"\n\nERROR: Regeneration failed ({regenerated_questions_block}). Displaying initial questions."
                 # Keep final_questions_block = initial_questions_block
            else:
                 # Regeneration successful, use the new questions
                 logger.info(f"[{job_id}] Successfully regenerated questions.")
                 final_questions_block = regenerated_questions_block
                 # Add note about regeneration to feedback
                 final_evaluation_feedback += "\nRegeneration was performed based on evaluation and user feedback."
                 # Optionally: Could re-run evaluation on the *regenerated* questions here for a final check

        else:
            # No regeneration needed and no user feedback
            logger.info(f"[{job_id}] No regeneration needed based on automatic evaluation, and no user feedback provided.")
            final_evaluation_feedback += "\nNo regeneration was triggered based on automatic checks, and no user feedback was provided."


        # --- Final Update to Job Storage ---
        job_data["status"] = "completed"
        job_data["message"] = "Processing complete."
        job_data["result"]["generated_questions"] = final_questions_block # Store final questions
        job_data["result"]["evaluation_feedback"] = final_evaluation_feedback.strip() # Store feedback text
        # job_data["result"]["scores"] = None # Deprecated field
        # per_question_evaluation is already stored

        logger.info(f"[{job_id}] Evaluation/Regeneration task completed successfully.")

    except Exception as e:
         logger.exception(f"[{job_id}] Evaluation/Regeneration task failed: {e}")
         job_data["status"] = "error"
         job_data["message"] = f"Processing failed during evaluation/regeneration: {e}"
         # Keep existing results but mark job as error

    finally:
         # Final cleanup for the job occurs here
         cleanup_job_files(job_id, file_paths_to_clean)
         logger.info(f"[{job_id}] Regeneration task finished (Status: {job_data.get('status', 'unknown')}). Final cleanup executed.")


# --- FastAPI Endpoints (Modified) ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    # Check for prompt files already done at startup
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-processing", response_model=Job)
async def start_processing_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF documents to process"),
    course_name: str = Form(...),
    num_questions: str = Form(...),
    academic_level: str = Form(...),
    taxonomy_level: str = Form(...),
    topics_list: str = Form(...),
    major: str = Form(...),
    retrieval_limit: int = Form(15, ge=1, le=100),
    similarity_threshold: float = Form(0.3, ge=0.0, le=1.0)
):
    """ Starts the PDF processing and initial question generation job. """
    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Received request to start job")
    temp_file_paths = []
    job_storage[job_id] = {
        "status": "pending", "message": "Validating inputs...",
        "params": {"course_name": course_name, "num_questions": num_questions, "academic_level": academic_level, "taxonomy_level": taxonomy_level, "topics_list": topics_list, "major": major, "retrieval_limit": retrieval_limit, "similarity_threshold": similarity_threshold},
        "result": {}, "temp_file_paths": []
    }
    try:
        if not files: raise HTTPException(status_code=400, detail="No files uploaded.")
        try:
            num_q_int = int(num_questions); assert 1 <= num_q_int <= 100
        except (ValueError, AssertionError): raise HTTPException(status_code=400, detail="Num questions must be int between 1-100.")

        upload_dir = Path(TEMP_UPLOAD_DIR)
        valid_files_saved = 0
        for file in files:
            if file.filename and file.filename.lower().endswith(".pdf"):
                safe_filename = "".join([c if c.isalnum() or c in ('-', '_', '.') else '_' for c in file.filename])
                temp_file_path = upload_dir / f"{job_id}_{uuid.uuid4().hex}_{safe_filename}"
                try:
                    with open(temp_file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
                    temp_file_paths.append(str(temp_file_path))
                    valid_files_saved += 1
                    logger.info(f"[{job_id}] Saved temp file: {temp_file_path}")
                except Exception as e: logger.error(f"[{job_id}] Failed to save {file.filename}: {e}"); raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}.")
                finally: await file.close()
            else: logger.warning(f"[{job_id}] Skipping invalid/non-PDF file: {file.filename}")

        if valid_files_saved == 0: raise HTTPException(status_code=400, detail="No valid PDF files provided.")
        job_storage[job_id]["temp_file_paths"] = temp_file_paths

        background_tasks.add_task(run_processing_job, job_id=job_id, file_paths=temp_file_paths, params=job_storage[job_id]["params"])
        job_storage[job_id]["status"] = "queued"
        job_storage[job_id]["message"] = f"Processing job queued for {valid_files_saved} PDF file(s)."
        logger.info(f"[{job_id}] Job queued.")
        return Job(job_id=job_id, status="queued", message=job_storage[job_id]["message"])

    except HTTPException as http_exc:
        cleanup_job_files(job_id, temp_file_paths)
        job_storage.pop(job_id, None)
        logger.error(f"[{job_id}] Validation error starting job: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        cleanup_job_files(job_id, temp_file_paths)
        job_storage.pop(job_id, None)
        logger.exception(f"[{job_id}] Failed unexpectedly while starting job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error starting job.")


@app.post("/regenerate-questions/{job_id}", response_model=Job)
async def regenerate_questions_endpoint(
    job_id: str, request: RegenerationRequest, background_tasks: BackgroundTasks
):
    """ Triggers the evaluation and potential regeneration based on user feedback. """
    logger.info(f"[{job_id}] Received request to regenerate/finalize questions.")
    job_data = job_storage.get(job_id)
    if not job_data: raise HTTPException(status_code=404, detail="Job not found")
    current_status = job_data.get("status")
    if current_status != "awaiting_feedback": raise HTTPException(status_code=400, detail=f"Job not awaiting feedback (status: {current_status})")

    job_data["status"] = "processing_feedback" # Tentative status
    job_data["message"] = "Evaluating questions and processing feedback..."
    logger.info(f"[{job_id}] Queuing evaluation/regeneration task.")
    background_tasks.add_task(run_regeneration_task, job_id=job_id, user_feedback=request.feedback)

    result_model = JobResultData(**job_data.get("result", {})) if job_data.get("result") else None
    return Job(job_id=job_id, status=job_data["status"], message=job_data["message"], result=result_model)


@app.get("/status/{job_id}", response_model=Job)
async def get_job_status(job_id: str):
    """ Endpoint to check the status and result of a processing job. """
    job = job_storage.get(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    result_data = job.get("result")
    job_result_model = None
    if isinstance(result_data, dict):
        try: job_result_model = JobResultData(**result_data)
        except Exception as e: logger.error(f"[{job_id}] Error parsing result data: {e}. Data: {result_data}")
    return Job(job_id=job_id, status=job.get("status", "unknown"), message=job.get("message"), result=job_result_model)


@app.get("/health")
async def health_check():
    """ Basic health check endpoint. """
    # Add Qdrant check?
    return {"status": "ok"}

# --- Run with Uvicorn ---
# Command: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000