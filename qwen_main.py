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
# Ensure required NLTK data is downloaded (run this once if needed)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util # Renamed

# Qdrant
from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# --- Qwen-VL Model Imports ---
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText # Using the class from notebook example
from huggingface_hub import login as hf_login # Optional login

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv()

# --- Load Environment Variables ---
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")
DATALAB_MARKER_URL = os.environ.get("DATALAB_MARKER_URL")
# MOONDREAM_API_KEY = os.environ.get("MOONDREAM_API_KEY") # REMOVED
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") # Optional, handle if None later
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") # Optional token for private models/gated access

# --- Check Essential Variables ---
# Removed MOONDREAM_API_KEY check
if not all([DATALAB_API_KEY, DATALAB_MARKER_URL, QDRANT_URL, GEMINI_API_KEY]):
    missing_vars = [var for var, val in {
        "DATALAB_API_KEY": DATALAB_API_KEY, "DATALAB_MARKER_URL": DATALAB_MARKER_URL,
        # "MOONDREAM_API_KEY": MOONDREAM_API_KEY, # REMOVED
        "QDRANT_URL": QDRANT_URL,
        "GEMINI_API_KEY": GEMINI_API_KEY
    }.items() if not val]
    logging.critical(f"FATAL ERROR: Missing essential environment variables: {', '.join(missing_vars)}")
    sys.exit("Missing essential environment variables.")


# Directories
TEMP_UPLOAD_DIR = Path("temp_uploads")
RESULTS_DIR = Path("results") # Keep for potential future use
FINAL_RESULTS_DIR = Path("final_results") # Use this for final markdown outputs
EXTRACTED_IMAGES_DIR = Path("extracted_images")
STATIC_DIR = Path("static") # Define static directory

# API Timeouts and Polling
DATALAB_POST_TIMEOUT = 60
DATALAB_POLL_TIMEOUT = 30
MAX_POLLS = 300
POLL_INTERVAL = 3
GEMINI_TIMEOUT = 240 # Increased timeout further for complex generations
MAX_GEMINI_RETRIES = 3
GEMINI_RETRY_DELAY = 60

# Qdrant Configuration
QDRANT_COLLECTION_NAME = "markdown_docs_v3_semantic"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Gemini Configuration
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Using latest flash model
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:{action}?key={api_key}"

# Qwen-VL Configuration
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MAX_NEW_TOKENS = 256 # Max tokens for image description

# Evaluation Configuration
QSTS_THRESHOLD = 0.5 # *** USER REQUESTED THRESHOLD ***
# Qualitative metrics to check per question using LLM
QUALITATIVE_METRICS = [
    "Understandable", "TopicRelated", "Grammatical", "Clear", "Answerable", "Central"
]
# Critical qualitative metrics - if any of these evaluate to False, it's a failure
CRITICAL_QUALITATIVE_FAILURES = {
    "Understandable": False,
    "Grammatical": False,
    "Clear": False,
    "Answerable": False,
    "TopicRelated": False, # Marking TopicRelated as critical too
    "Central": False # Marking Central as potentially critical
 }
MAX_REGENERATION_ATTEMPTS = 3 # Maximum times to try regeneration

# Prompt File Paths
PROMPT_DIR = Path("content")
FINAL_USER_PROMPT_PATH = PROMPT_DIR / "final_user_prompt.txt"
HYPOTHETICAL_PROMPT_PATH = PROMPT_DIR / "hypothetical_prompt.txt"
QUALITATIVE_EVAL_PROMPT_PATH = PROMPT_DIR / "qualitative_eval_prompt.txt" # For LLM evaluation

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize Models and Clients (Global Scope) ---
model_st = None
qdrant_client = None
model_qwen = None
processor_qwen = None
device = None
stop_words = None
lemmatizer = None

try:
    # --- Device Setup (CPU or GPU for Qwen) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, Qwen-VL will run on CPU (this may be very slow).")

    # --- Optional: Hugging Face Login ---
    if HUGGINGFACE_TOKEN:
        try:
            hf_login(token=HUGGINGFACE_TOKEN)
            logger.info("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning(f"Hugging Face login failed (token provided but error occurred): {e}")
    else:
        logger.info("No Hugging Face token provided. Assuming public model access.")

    # --- Initialize Qwen-VL Model and Processor ---
    logger.info(f"Initializing Qwen-VL model and processor from '{QWEN_MODEL_ID}'...")
    # Using AutoModelForImageTextToText as shown in the notebook
    # Using torch_dtype="auto" for potential optimizations like bfloat16 if available
    model_qwen = AutoModelForImageTextToText.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True # Qwen models often require this
    ).to(device).eval() # Set to evaluation mode
    processor_qwen = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True
    )
    logger.info(f"Qwen-VL model '{QWEN_MODEL_ID}' and processor loaded successfully on {device}.")

    # --- Initialize Sentence Transformer ---
    logger.info(f"Initializing Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
    model_st = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- Initialize Qdrant Client ---
    logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    # --- Ensure Qdrant Collection Exists ---
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
    except Exception as e:
        if "Not found" in str(e) or "status_code=404" in str(e) or "Reason: Not Found" in str(e):
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")
        else:
            logger.error(f"Unexpected error checking/creating Qdrant collection: {e}", exc_info=True)
            raise

    # --- NLTK Setup ---
    logger.info("Setting up NLTK resources (stopwords, wordnet, punkt)...")
    for resource in ['stopwords', 'wordnet', 'punkt']:
        try:
            if resource == 'punkt': nltk.data.find(f'tokenizers/{resource}')
            else: nltk.data.find(f'corpora/{resource}')
        except (LookupError, nltk.downloader.DownloadError):
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK setup complete.")

except Exception as e:
    logger.critical(f"Fatal error during initialization: {e}", exc_info=True)
    # Attempt to clean up partially loaded models if possible? Maybe not safe.
    model_qwen = None
    processor_qwen = None
    model_st = None
    qdrant_client = None
    sys.exit("Initialization failed.")


# --- FastAPI App Setup ---
app = FastAPI(title="Semantic PDF Question Generator")

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount extracted images
app.mount("/extracted_images", StaticFiles(directory=EXTRACTED_IMAGES_DIR), name="extracted_images")

templates = Jinja2Templates(directory="templates")

# Ensure necessary directories exist
for dir_path in [TEMP_UPLOAD_DIR, RESULTS_DIR, FINAL_RESULTS_DIR, EXTRACTED_IMAGES_DIR, PROMPT_DIR, STATIC_DIR / "css", STATIC_DIR / "js"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Check for mandatory prompt files
mandatory_prompts = [FINAL_USER_PROMPT_PATH, HYPOTHETICAL_PROMPT_PATH, QUALITATIVE_EVAL_PROMPT_PATH]
missing_prompts = [p.name for p in mandatory_prompts if not p.exists()]
if missing_prompts:
    logger.critical(f"FATAL ERROR: Missing required prompt template files in '{PROMPT_DIR}': {', '.join(missing_prompts)}")
    sys.exit(f"Missing prompt files: {', '.join(missing_prompts)}")


# In-memory storage (Replace with persistent storage for production)
job_storage: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class JobResultData(BaseModel):
    generated_questions: Optional[str] = None # Final questions
    evaluation_feedback: Optional[str] = None # Text summary of evaluation
    per_question_evaluation: Optional[List[Dict[str, Any]]] = None # Detailed eval per question
    retrieved_context_preview: Optional[str] = None
    extracted_markdown: Optional[str] = None
    initial_questions: Optional[str] = None # Questions generated before feedback
    image_paths: Optional[List[str]] = None # Image URLs

class Job(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    result: Optional[JobResultData] = None

class RegenerationRequest(BaseModel):
    feedback: str = Field(..., description="User feedback for question regeneration (can be empty)")


# --- Core Functions ---

def generate_description_for_image(image_path: Path, figure_caption: str = "") -> str:
    """Load an image and generate a description using the local Qwen-VL model."""
    global model_qwen, processor_qwen, device # Access globally loaded models/device

    if not model_qwen or not processor_qwen:
        logger.error(f"Qwen-VL model not initialized. Cannot generate description for {image_path.name}")
        return "Error: Qwen-VL model not available."

    try:
        image = Image.open(image_path).convert("RGB")

        # Construct the prompt using the message format expected by Qwen-VL processor
        # Adapted from the notebook's describe_figure function prompt
        prompt_text = (
            f"Describe the key technical findings in this figure/visualization "
            f"captioned: {figure_caption}. Illustrate and mention trends, "
            f"patterns, and numerical values that can be observed. Provide a scientific/academic styled short, "
            f"single paragraph summary that is highly insightful in context of the document."
        )
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, # Placeholder for the image, processor handles linking
                {"type": "text", "text": prompt_text}
            ]}
        ]

        # Prepare inputs using the processor
        # The processor combines text and image based on the messages format
        text_prompt = processor_qwen.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs (text and image)
        inputs = processor_qwen(
            text=[text_prompt],
            images=[image], # Pass the PIL image directly
            return_tensors="pt"
        ).to(device) # Move inputs to the correct device (GPU/CPU)

        # Generate description using the model
        # Use torch.inference_mode() for efficiency during inference
        logger.info(f"Generating Qwen-VL description for {image_path.name} on device {device}...")
        with torch.inference_mode():
            output_ids = model_qwen.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False # Use greedy decoding for more deterministic output
                # Add other generation params if needed: temperature, top_p, etc.
                # For example: num_beams=4, early_stopping=True for beam search
            )
        logger.info(f"Finished Qwen-VL generation for {image_path.name}.")

        # Decode the generated tokens, skipping the prompt part
        # We need to slice the output_ids correctly
        # The processor usually pads on the left, so prompt tokens are at the start
        input_token_len = inputs['input_ids'].shape[1]
        # Slice output IDs to get only the newly generated tokens
        generated_ids = output_ids[:, input_token_len:]

        response = processor_qwen.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

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
        # Check for common issues like OOM
        if "CUDA out of memory" in str(e):
            logger.critical(f"CUDA OOM Error during Qwen-VL inference on {image_path.name}. Suggest reducing image size/resolution if possible, or using a smaller model/CPU.")
            # Clean up CUDA cache if possible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Error: Ran out of GPU memory generating description."
        # Add checks for other specific errors if needed
        return f"Error generating description for this image due to an internal error."


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
            # Only log polling message periodically
            if (i + 1) % 10 == 0: # Log every 30 seconds or so
                 logger.info(f"Polling Datalab for {file_path.name}... attempt {i+1}/{MAX_POLLS}")
        except requests.exceptions.Timeout:
             logger.warning(f"Polling Datalab timed out on attempt {i+1} for {file_path.name}. Retrying...")
        except requests.exceptions.RequestException as e:
             logger.warning(f"Polling error on attempt {i+1} for {file_path.name}: {e}. Retrying...")
             time.sleep(1)

    logger.error(f"Polling timed out waiting for Datalab processing for {file_path.name}.")
    raise TimeoutError("Polling timed out waiting for Datalab processing.")

def save_extracted_images(images_dict, images_folder: Path) -> Dict[str, str]:
    """Decode and save base64 encoded images. Returns dict {original_name: saved_path_str}"""
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    logger.info(f"Saving {len(images_dict)} extracted images to {images_folder}...")
    for img_name, b64_data in images_dict.items():
        try:
            # Keep original name for key, sanitize filename for saving
            safe_img_name = "".join([c if c.isalnum() or c in ('-', '_', '.') else '_' for c in img_name])
            if not safe_img_name: safe_img_name = f"image_{uuid.uuid4().hex[:8]}.png" # fallback

            image_data = base64.b64decode(b64_data)
            image_path = images_folder / safe_img_name
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)
            # Use the *original* name Datalab provided as the key
            saved_files[img_name] = str(image_path)
        except Exception as e:
            logger.warning(f"Could not decode/save image '{img_name}': {e}", exc_info=True)
    return saved_files

def process_markdown(markdown_text, saved_images: Dict[str, str], job_id: str):
    """Process markdown: replace image placeholders with Qwen-VL descriptions."""
    logger.info(f"[{job_id}] Processing markdown for image descriptions using Qwen-VL...")
    lines = markdown_text.splitlines()
    processed_lines = []
    i = 0
    image_count = 0
    figure_pattern = re.compile(r"^!\[.*?\]\((.*?)\)$") # Image tag: ![alt text](path/filename.ext)
    caption_pattern = re.compile(r"^(Figure|Table|Chart)\s?(\d+[:.]?)\s?(.*)", re.IGNORECASE) # Figure caption

    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        image_match = figure_pattern.match(stripped_line)

        if image_match:
            image_filename_encoded = image_match.group(1)
            try:
                image_filename_decoded = urllib.parse.unquote(image_filename_encoded)
            except Exception:
                 logger.warning(f"[{job_id}] Could not URL-decode image filename: {image_filename_encoded}")
                 image_filename_decoded = image_filename_encoded # Fallback

            image_count += 1
            caption = ""
            caption_line_index = -1

            # Look ahead for a caption line
            j = i + 1
            while j < len(lines) and lines[j].strip() == "": j += 1 # Skip blank lines
            if j < len(lines):
                next_line_stripped = lines[j].strip()
                if caption_pattern.match(next_line_stripped):
                    caption = next_line_stripped
                    caption_line_index = j

            # Find the image path using the *original* name from Datalab keys
            image_path_str = saved_images.get(image_filename_decoded) # Try decoded first
            if not image_path_str: image_path_str = saved_images.get(image_filename_encoded) # Try encoded as fallback

            description = ""
            if image_path_str:
                image_path = Path(image_path_str) # Convert to Path object
                if image_path.exists():
                    # Call the NEW function using Qwen-VL
                    description = generate_description_for_image(image_path, caption)
                else:
                    description = f"*Referenced image file '{image_path.name}' not found at expected path.*"
                    logger.warning(f"[{job_id}] {description}")
            else:
                description = f"*Referenced image '{image_filename_decoded}' (or '{image_filename_encoded}') was not found in extracted images dictionary.*"
                logger.warning(f"[{job_id}] {description}")

            title_text = caption if caption else f"Figure {image_count}"
            # Format the description block in markdown
            block_text = f"\n---\n### {title_text}\n\n**Figure Description (Generated by Qwen-VL):**\n{description}\n---\n"
            processed_lines.append(block_text)

            if caption_line_index != -1: i = caption_line_index # Skip the original caption line if found and used
        else:
            processed_lines.append(line) # Keep non-image lines

        i += 1 # Move to the next line

    logger.info(f"[{job_id}] Finished processing markdown with Qwen-VL. Processed {image_count} image references.")
    return "\n".join(processed_lines)


def clean_text_for_embedding(text):
    """Basic text cleaning for embedding."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE) # Remove standalone '---' lines
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE) # Remove markdown headers
    text = re.sub(r'\*\*Figure Description \(Generated by Qwen-VL\):\*\*', '', text) # Remove added description label
    return text.strip()

def hierarchical_chunk_markdown(markdown_text, source_filename):
    """Chunks markdown text based on headers and figure blocks."""
    logger.info(f"Chunking markdown from source: {source_filename}")
    lines = markdown_text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_headers = {}
    figure_title = None

    header_pattern = re.compile(r"^(#{1,6})\s+(.*)")
    # Updated pattern to capture the title after '###' potentially followed by Figure/Table/Chart info
    figure_title_pattern = re.compile(r"^###\s+((?:Figure|Table|Chart).*)$", re.IGNORECASE)
    separator_pattern = re.compile(r"^---$") # Separator for figure blocks

    inside_figure_block = False

    for line_num, line in enumerate(lines):
        stripped_line = line.strip()
        header_match = header_pattern.match(stripped_line)
        figure_title_match = figure_title_pattern.match(stripped_line)
        separator_match = separator_pattern.match(stripped_line)

        # Start or end of a figure block based on separator
        if separator_match:
            if inside_figure_block:
                # End of figure block - process the collected chunk
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    # Include the separator line itself? Maybe not for embedding.
                    cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                    if cleaned_chunk_text:
                        metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                        if figure_title: metadata["figure_title"] = figure_title
                        chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                    current_chunk_lines = []
                inside_figure_block = False
                figure_title = None
            else:
                # Start of figure block - process preceding text chunk if any
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                    if cleaned_chunk_text:
                        metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                        # Don't associate figure title with the text *before* the figure block
                        chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                    current_chunk_lines = []
                inside_figure_block = True
            continue # Don't add the separator itself to the chunk lines

        # New Header - process previous chunk
        if header_match:
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                if cleaned_chunk_text:
                    metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                    if figure_title: metadata["figure_title"] = figure_title
                    chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                current_chunk_lines = []

            level, title = len(header_match.group(1)), header_match.group(2).strip()
            current_headers = {k: v for k, v in current_headers.items() if k < level} # Keep parent headers
            current_headers[level] = title
            figure_title = None # Reset figure context on new header
            inside_figure_block = False # Headers reset figure block context
            current_chunk_lines.append(line) # Include header in the new chunk's *lines* (cleaned later)
            continue

        # Figure Title Line (inside a potential figure block)
        if figure_title_match and inside_figure_block:
            figure_title = figure_title_match.group(1).strip()
            # Add line to chunk, it might contain the description below it
            current_chunk_lines.append(line)
            continue

        # Regular line - add to current chunk
        current_chunk_lines.append(line)

    # Add the last chunk if any content remains
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        cleaned_chunk_text = clean_text_for_embedding(chunk_text)
        if cleaned_chunk_text:
            metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
            if figure_title: metadata["figure_title"] = figure_title
            chunks.append({"text": cleaned_chunk_text, "metadata": metadata})

    logger.info(f"Generated {len(chunks)} hierarchical chunks for {source_filename}.")
    # Debug: print first few chunk texts
    # for i, chunk in enumerate(chunks[:3]):
    #      logger.debug(f"Chunk {i} text sample: {chunk['text'][:100]}...")
    return chunks

def embed_chunks(chunks_data, model):
    """Embeds the 'text' field of each chunk dictionary."""
    global model_st # Access global embedding model
    if not chunks_data: return []
    if not model_st:
        logger.error("Sentence Transformer model not initialized. Cannot embed chunks.")
        raise Exception("Embedding model not available.")

    logger.info(f"Embedding {len(chunks_data)} text chunks...")
    texts_to_embed = [chunk['text'] for chunk in chunks_data]
    try:
        embeddings = model_st.encode(texts_to_embed, show_progress_bar=False).tolist()
        logger.info("Embedding complete.")
        return embeddings
    except Exception as e:
        logger.error(f"Error during embedding: {e}", exc_info=True)
        raise

def upsert_to_qdrant(job_id: str, collection_name, embeddings, chunks_data, batch_size=100):
    """Upserts chunks into Qdrant."""
    global qdrant_client # Access global client
    if not embeddings or not chunks_data: return 0
    if not qdrant_client:
         logger.error(f"[{job_id}] Qdrant client not initialized. Cannot upsert.")
         raise Exception("Qdrant client not available.")

    logger.info(f"[{job_id}] Upserting {len(embeddings)} points to Qdrant collection '{collection_name}'...")
    total_points_upserted = 0
    points_to_upsert = []
    for embedding, chunk_data in zip(embeddings, chunks_data):
        if isinstance(chunk_data.get('metadata'), dict) and 'text' in chunk_data:
            payload = chunk_data['metadata'].copy()
            payload["text"] = chunk_data['text'] # Make sure text is in payload
            point_id = str(uuid.uuid4())
            points_to_upsert.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        else:
            logger.warning(f"[{job_id}] Skipping chunk due to invalid format (missing text or metadata dict): {chunk_data}")

    for i in range(0, len(points_to_upsert), batch_size):
        batch_points = points_to_upsert[i:i + batch_size]
        if not batch_points: continue
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch_points, wait=True)
            batch_count = len(batch_points)
            total_points_upserted += batch_count
            logger.info(f"[{job_id}] Upserted batch {i // batch_size + 1} ({batch_count} points) to Qdrant.")
        except Exception as e:
            logger.error(f"[{job_id}] Error upserting Qdrant batch {i // batch_size + 1}: {e}", exc_info=True)
            # Consider whether to raise immediately or try other batches
            raise Exception(f"Failed to upsert batch to Qdrant: {e}")

    logger.info(f"[{job_id}] Finished upserting. Total points upserted: {total_points_upserted}")
    return total_points_upserted

def fill_placeholders(template_path: Path, output_path: Path, placeholders: Dict):
    """Fills placeholders in a template file."""
    try:
        if not template_path.exists(): raise FileNotFoundError(f"Template file not found: {template_path}")
        template = template_path.read_text(encoding='utf-8')
        for placeholder, value in placeholders.items():
            template = template.replace(f"{{{placeholder}}}", str(value)) # Ensure value is string
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template, encoding='utf-8')
    except Exception as e:
        logger.error(f"Error filling placeholders for {template_path}: {e}", exc_info=True)
        raise

def get_gemini_response(system_prompt: str, user_prompt: str, is_json_output: bool = False):
    """Gets a response from the Google Gemini API, optionally expecting JSON, with retries."""
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")

    api_url = GEMINI_API_URL_TEMPLATE.format(model_name=GEMINI_MODEL_NAME, action="generateContent", api_key=GEMINI_API_KEY)
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
         "generationConfig": {
             "temperature": 0.5 if is_json_output else 0.7,
             "maxOutputTokens": 8192,
             "topP": 0.95,
             "topK": 40,
         }
    }
    if is_json_output:
         payload["generationConfig"]["responseMimeType"] = "application/json"

    last_error = None
    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            # logger.debug(f"Gemini Request Payload (partial user): {str(payload)[:500]}...") # Uncomment for deep debug
            logger.info(f"Calling Gemini API (Attempt {attempt+1}/{MAX_GEMINI_RETRIES})...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Gemini API call successful (Attempt {attempt+1}).")

            # --- Robust response extraction (unchanged from previous version) ---
            if response_data.get('candidates'):
                candidate = response_data['candidates'][0]
                if candidate.get('content', {}).get('parts'):
                    gemini_response_text = candidate['content']['parts'][0].get('text', '')
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    if finish_reason not in ['STOP', 'MAX_TOKENS']:
                        logger.warning(f"Gemini generation finished with reason: {finish_reason}. Safety: {candidate.get('safetyRatings')}")
                    if gemini_response_text:
                        return gemini_response_text.strip()
                    else:
                        logger.warning(f"Gemini returned an empty response. Finish reason: {finish_reason}. Candidate: {candidate}")
                        last_error = f"Error: Gemini returned an empty response (finish reason: {finish_reason})."
                        # Consider retry? Maybe not if it finished normally empty.

                elif candidate.get('finishReason'):
                    reason = candidate['finishReason']
                    safety_ratings = candidate.get('safetyRatings', [])
                    blocked_by_safety = any(sr.get('probability') not in ['NEGLIGIBLE', 'LOW'] for sr in safety_ratings if sr.get('probability')) # Added check for probability existence
                    if blocked_by_safety:
                         block_details = ", ".join([f"{sr['category']}: {sr['probability']}" for sr in safety_ratings if sr.get('probability') not in ['NEGLIGIBLE', 'LOW']])
                         error_msg = f"Error: Generation stopped by Gemini - {reason} (Safety concerns: {block_details})"
                         logger.error(error_msg)
                         last_error = error_msg
                    else:
                         error_msg = f"Error: Generation stopped by Gemini - {reason}"
                         logger.error(error_msg)
                         last_error = error_msg
                else:
                    error_msg = f"Error: Could not extract text from Gemini response structure. Candidate: {candidate}"
                    logger.error(error_msg)
                    last_error = error_msg

            elif response_data.get('promptFeedback', {}).get('blockReason'):
                block_reason = response_data['promptFeedback']['blockReason']
                safety_ratings = response_data['promptFeedback'].get('safetyRatings', [])
                block_details = ", ".join([f"{sr['category']}: {sr['probability']}" for sr in safety_ratings if sr.get('probability') not in ['NEGLIGIBLE', 'LOW']])
                if block_details:
                    error_msg = f"Error: Prompt blocked by Gemini - {block_reason} (Safety concerns: {block_details})"
                else:
                    error_msg = f"Error: Prompt blocked by Gemini - {block_reason}"
                logger.error(error_msg)
                last_error = error_msg
            else:
                error_msg = f"Error: Unexpected Gemini API response format: {response_data}"
                logger.error(error_msg)
                last_error = error_msg

            # If we got an error message from response parsing, return it without retry
            if last_error: return last_error

        except requests.exceptions.Timeout:
            last_error = "Error: Gemini API request timed out."
            logger.warning(f"{last_error} Attempt {attempt + 1}/{MAX_GEMINI_RETRIES}.")
            if attempt + 1 < MAX_GEMINI_RETRIES: time.sleep(GEMINI_RETRY_DELAY)
            continue # Retry on timeout

        except requests.exceptions.RequestException as e:
            response_text = ""; status_code = None
            if e.response is not None:
                response_text = e.response.text[:500]; status_code = e.response.status_code
            last_error = f"Error: Gemini API request failed - {e} (Status: {status_code}, Response: {response_text})"
            logger.warning(f"{last_error} Attempt {attempt + 1}/{MAX_GEMINI_RETRIES}.")
            if status_code == 429: # Rate limit
                logger.warning(f"Gemini rate limit hit (HTTP 429). Waiting {GEMINI_RETRY_DELAY}s...")
                if attempt + 1 < MAX_GEMINI_RETRIES: time.sleep(GEMINI_RETRY_DELAY); continue # Retry on rate limit
                else: last_error = "Error: Gemini rate limit exceeded after retries."; break
            elif status_code is not None and 500 <= status_code < 600: # Server errors
                logger.warning(f"Gemini server error (HTTP {status_code}). Waiting {GEMINI_RETRY_DELAY}s...")
                if attempt + 1 < MAX_GEMINI_RETRIES: time.sleep(GEMINI_RETRY_DELAY); continue # Retry on server error
                else: last_error = f"Error: Gemini server error persisted after retries (HTTP {status_code})."; break
            else: # Other request errors (e.g., 400 Bad Request) - likely won't succeed on retry
                logger.error(last_error, exc_info=True)
                break # Don't retry client errors like 400

        except Exception as e:
            last_error = f"Error: Failed to process Gemini response - {e}"
            logger.error(last_error, exc_info=True)
            break # Don't retry unknown processing errors

    # If loop finishes without returning, it means all retries failed
    logger.error(f"Gemini API call failed after {MAX_GEMINI_RETRIES} attempts. Last error: {last_error}")
    # Return the last error encountered
    return last_error if last_error else "Error: Gemini API call failed after multiple retries."


def find_topics_and_generate_hypothetical_text(job_id: str, academic_level, major, course_name, taxonomy_level, topics):
    """Generates hypothetical text based on topics using Gemini."""
    logger.info(f"[{job_id}] Generating hypothetical text...")
    try:
        temp_path = TEMP_UPLOAD_DIR / f"{job_id}_updated_hypothetical.txt"
        placeholders = {"course_name": course_name, "academic_level": academic_level, "topics": topics, "major": major, "taxonomy_level": taxonomy_level}
        fill_placeholders(HYPOTHETICAL_PROMPT_PATH, temp_path, placeholders)
        user_prompt = temp_path.read_text(encoding="utf8")
        system_prompt = f"You are an AI assistant for {major} at the {academic_level} level. Generate a concise, hypothetical student query for the course '{course_name}' on topics: {topics}, reflecting Bloom's level: {taxonomy_level}."
        hypothetical_text = get_gemini_response(system_prompt, user_prompt)
        temp_path.unlink(missing_ok=True)
        if hypothetical_text.startswith("Error:"):
            raise Exception(f"Failed to generate hypothetical text: {hypothetical_text}")
        logger.info(f"[{job_id}] Successfully generated hypothetical text.")
        return hypothetical_text
    except FileNotFoundError as e: raise Exception(f"Hypothetical prompt template file missing: {e}")
    except Exception as e: raise Exception(f"Error generating hypothetical text: {e}")


def search_results_from_qdrant(job_id: str, collection_name, embedded_vector, limit=15, score_threshold: Optional[float] = None, session_id_filter=None, document_ids_filter=None):
    """Searches Qdrant."""
    global qdrant_client # Access global client
    if not qdrant_client:
        logger.error(f"[{job_id}] Qdrant client not initialized. Cannot search.")
        raise Exception("Qdrant client not available.")

    logger.info(f"[{job_id}] Searching Qdrant '{collection_name}' (limit={limit}, threshold={score_threshold})...")
    must_conditions = []
    if session_id_filter: must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id_filter)))
    if document_ids_filter:
        doc_ids = document_ids_filter if isinstance(document_ids_filter, list) else [document_ids_filter]
        if doc_ids: must_conditions.append(FieldCondition(key="document_id", match=MatchAny(any=doc_ids)))
    query_filter = Filter(must=must_conditions) if must_conditions else None
    try:
        query_vector_list = embedded_vector.tolist() if hasattr(embedded_vector, 'tolist') else list(map(float, embedded_vector))
        results = qdrant_client.search(
            collection_name=collection_name, query_vector=query_vector_list, query_filter=query_filter,
            limit=limit, score_threshold=score_threshold, with_payload=True, with_vectors=False
        )
        logger.info(f"[{job_id}] Qdrant search returned {len(results)} results.")
        if results: logger.info(f"[{job_id}] Top hit score: {results[0].score:.4f}")
        return results
    except Exception as e:
        logger.error(f"[{job_id}] Error searching Qdrant: {e}", exc_info=True)
        return []


def generate_initial_questions(job_id: str, retrieved_context: str, params: Dict):
    """Generates the initial set of questions using Gemini."""
    logger.info(f"[{job_id}] Preparing to generate initial questions...")
    blooms = "Bloom's Levels: Remember, Understand, Apply, Analyze, Evaluate, Create." # Simpler
    max_context_chars = 30000 # Limit context size for LLM
    truncated_context = retrieved_context[:max_context_chars]
    if len(retrieved_context) > max_context_chars:
        logger.warning(f"[{job_id}] Truncating retrieved context from {len(retrieved_context)} to {max_context_chars} chars for initial question generation.")
        truncated_context += "\n... [Context Truncated]"

    generate_diagrams_flag = params.get('generate_diagrams', False)
    logger.info(f"[{job_id}] generate_diagrams flag in generate_initial_questions: {generate_diagrams_flag}")

    # PlantUML Instructions (same as before, included if flag is True)
    diagram_instructions = ""
    if generate_diagrams_flag:
        logger.info(f"[{job_id}] PlantUML diagram generation instructions included in prompt.")
        diagram_instructions = (
            "\n7. **PlantUML Diagram Generation (REQUIRED if applicable):** "
            "For questions involving graph structures, tree traversals, algorithm steps (like flowcharts or state changes), or finite state machines described in the context, "
            "you **MUST** generate a relevant PlantUML diagram to visually aid the question. "
            "The diagram must be directly derivable from the provided context. "
            "Enclose the PlantUML code **strictly** within ```plantuml ... ``` tags immediately after the question text it relates to. "
            "Example format:\n"
            "   ```plantuml\n"
            "   @startuml\n"
            "   (*) --> State1\n"
            "   State1 --> State2 : Event\n"
            "   State2 --> (*)\n"
            "   @enduml\n"
            "   ```\n"
            "Generate diagrams **only** when the context provides sufficient detail to create a meaningful and accurate visual representation relevant to the question. Do not invent details not present in the context."
        )
    else:
         logger.info(f"[{job_id}] PlantUML diagram generation instructions *excluded* from prompt.")

    placeholders = {
        "content": truncated_context,
        "num_questions": params['num_questions'],
        "course_name": params['course_name'],
        "taxonomy": params['taxonomy_level'],
        "major": params['major'],
        "academic_level": params['academic_level'],
        "topics_list": params['topics_list'],
        "blooms_taxonomy_descriptions": blooms,
        "diagram_instructions": diagram_instructions # Pass the instructions (or empty string)
    }

    try:
        temp_path = TEMP_UPLOAD_DIR / f"{job_id}_initial_user_prompt.txt"
        fill_placeholders(FINAL_USER_PROMPT_PATH, temp_path, placeholders)
        user_prompt = temp_path.read_text(encoding="utf8")

        # Construct system prompt (same as before)
        system_prompt_base = (
            f"You are an AI assistant specialized in creating high-quality educational questions for a {params['academic_level']} {params['major']} course: '{params['course_name']}'. "
            f"Generate exactly {params['num_questions']} questions based ONLY on the provided context, strictly aligned with Bloom's level: {params['taxonomy_level']}, focusing on topics: {params['topics_list']}. "
            "Ensure questions are clear, unambiguous, directly answerable *only* from the given context, and suitable for the specified academic standard."
        )
        plantuml_system_hint = ""
        if generate_diagrams_flag:
            plantuml_system_hint = " **If the instructions require a PlantUML diagram for a question, you MUST include it formatted within ```plantuml ... ``` tags.**"

        output_format_instruction = (
            " **Your final output MUST consist ONLY of the numbered list of questions. Each question may optionally be followed by its relevant PlantUML code block if diagrams were requested and deemed necessary based on context. Do not add any other text, introductions, or summaries.**"
        )
        system_prompt_final = system_prompt_base + plantuml_system_hint + output_format_instruction

        # Store the final, filled prompts used for generation
        if job_id in job_storage: # Ensure job exists
             job_storage[job_id]["generation_prompts"] = {"user_prompt_content": user_prompt, "system_prompt": system_prompt_final}
             logger.info(f"[{job_id}] Stored prompts for potential regeneration.")
        else: logger.warning(f"[{job_id}] Job storage not found when trying to store prompts.")

        logger.info(f"[{job_id}] Generating initial questions via Gemini...")
        initial_questions = get_gemini_response(system_prompt_final, user_prompt)
        temp_path.unlink(missing_ok=True) # Clean up temporary prompt file

        # Log the raw response for debugging
        # print("-" * 80)
        # print(f"[{job_id}] RAW GEMINI RESPONSE for initial questions:")
        # print(initial_questions)
        # print("-" * 80)

        if initial_questions.startswith("Error:"):
            raise Exception(f"Gemini Error during initial question generation: {initial_questions}")

        logger.info(f"[{job_id}] Successfully generated initial questions snippet: {initial_questions[:300]}...")
        # Check if PlantUML was expected vs generated
        plantuml_found = "```plantuml" in initial_questions
        if generate_diagrams_flag and not plantuml_found:
             logger.warning(f"[{job_id}] PlantUML was requested, but '```plantuml' not found in the initial response. Context might not have supported diagram generation.")
        elif not generate_diagrams_flag and plantuml_found:
             logger.warning(f"[{job_id}] PlantUML was *not* requested, but '```plantuml' *was* found in the initial response.")

        return initial_questions
    except FileNotFoundError as e:
        raise Exception(f"Final user prompt template missing: {e}")
    except Exception as e:
        logger.error(f"[{job_id}] Initial question generation failed: {e}", exc_info=True)
        # Re-raise to be caught by the background task runner
        raise Exception(f"Initial question generation failed unexpectedly: {e}")


def parse_questions(question_block: str) -> List[str]:
    """Splits text into questions, handling potential PlantUML blocks."""
    if not question_block: return []
    lines = question_block.splitlines()
    questions = []
    current_question_lines = []
    in_plantuml_block = False
    # More robust pattern: optional whitespace, digit(s), optional ., ), -, :, required whitespace
    question_start_pattern = re.compile(r"^\s*\d+\s*[\.\)\-:]\s+")

    for line in lines:
        stripped_line = line.strip()
        is_plantuml_start = stripped_line.startswith("```plantuml")
        is_plantuml_end = stripped_line == "```" and in_plantuml_block

        if is_plantuml_start:
            in_plantuml_block = True
            if current_question_lines: current_question_lines.append(line) # Add start tag
            continue
        if is_plantuml_end:
            in_plantuml_block = False
            if current_question_lines: current_question_lines.append(line) # Add end tag
            continue
        if in_plantuml_block:
             if current_question_lines: current_question_lines.append(line) # Add lines inside block
             continue

        # Check if the line starts a new question *outside* a PlantUML block
        if question_start_pattern.match(line):
            # If we have a previous question collected, add it
            if current_question_lines:
                questions.append("\n".join(current_question_lines).strip())
            # Start collecting the new question
            current_question_lines = [line]
        elif current_question_lines:
            # Append continuation lines (or blank lines) to the current question
            current_question_lines.append(line)
        # else: Ignore lines before the first question number is found

    # Add the last collected question block
    if current_question_lines: questions.append("\n".join(current_question_lines).strip())

    # Final cleanup: remove empty strings potentially added
    cleaned_questions = [q for q in questions if q]

    # Fallback if parsing fails completely but there's text
    if not cleaned_questions and question_block.strip():
        logger.warning(f"Could not parse numbered list with potential PlantUML blocks using regex. Falling back to simple non-empty line split for questions: {question_block[:200]}...")
        # Basic fallback: split by lines, keep non-empty ones, maybe filter standalone numbers?
        cleaned_questions = [q.strip() for q in question_block.splitlines() if q.strip() and not re.fullmatch(r"\s*\d+[\.\)\-:]?\s*", q.strip())]
        if not cleaned_questions: # If even fallback yields nothing, return the original block as one item
             logger.warning("Fallback line split also yielded no questions. Returning original block.")
             return [question_block.strip()]
    return cleaned_questions


def evaluate_single_question_qsts(job_id: str, question: str, context: str) -> float:
    """Calculates QSTS score between a single question (text part only) and the context."""
    global model_st # Access global embedding model
    if not model_st:
        logger.error(f"[{job_id}] Sentence Transformer model not initialized. Cannot calculate QSTS.")
        return 0.0
    if not question or not context: return 0.0

    # Extract text part, removing PlantUML and leading number
    question_text_only = re.sub(r"```plantuml.*?```", "", question, flags=re.DOTALL | re.MULTILINE)
    question_text_only = re.sub(r"^\s*\d+\s*[\.\)\-:]?\s+", "", question_text_only.strip()).strip()

    if not question_text_only:
         logger.warning(f"[{job_id}] No text found in question after removing PlantUML/number for QSTS eval: '{question[:50]}...'")
         return 0.0
    try:
        q_emb = model_st.encode(question_text_only)
        c_emb = model_st.encode(context)
        # Ensure embeddings are 2D for pytorch_cos_sim
        if q_emb.ndim == 1: q_emb = q_emb.reshape(1, -1)
        if c_emb.ndim == 1: c_emb = c_emb.reshape(1, -1)
        score = sbert_util.pytorch_cos_sim(q_emb, c_emb).item()
        # Clamp score between -1 and 1 just in case
        return round(max(-1.0, min(1.0, score)), 4)
    except Exception as e:
        logger.warning(f"[{job_id}] Error calculating QSTS for question '{question_text_only[:50]}...': {e}", exc_info=True)
        return 0.0


def evaluate_single_question_qualitative(job_id: str, question: str, context: str) -> Dict[str, bool]:
    """Uses LLM to evaluate qualitative aspects of a single question (including PlantUML if present)."""
    results = {metric: False for metric in QUALITATIVE_METRICS}
    if not question or not context: return results

    full_question_block = question # Evaluate the entire block including potential diagrams
    try:
        # Truncate context slightly more aggressively for this eval if needed
        eval_context = context[:4000] + ("\n... [Context Truncated for Qualitative Eval]" if len(context)>4000 else "")
        placeholders = {"question": full_question_block, "context": eval_context, "criteria_list_str": ", ".join(QUALITATIVE_METRICS)}
        temp_path = TEMP_UPLOAD_DIR / f"{job_id}_qualitative_eval_prompt_{uuid.uuid4().hex[:6]}.txt"
        fill_placeholders(QUALITATIVE_EVAL_PROMPT_PATH, temp_path, placeholders)
        eval_prompt = temp_path.read_text(encoding='utf-8')
        temp_path.unlink(missing_ok=True) # Clean up temporary prompt file

        eval_system_prompt = "You are an AI assistant evaluating educational question quality based on provided context and criteria. The question might include PlantUML code for a diagram; evaluate the *entire* question block (text and diagram code if present). Respond ONLY with a single, valid JSON object containing boolean values (true/false) for each of the following criteria: " + ", ".join(QUALITATIVE_METRICS) + "."

        response_text = get_gemini_response(eval_system_prompt, eval_prompt, is_json_output=True)

        if response_text.startswith("Error:"):
            logger.error(f"[{job_id}] LLM qualitative evaluation failed: {response_text}")
            return results # Return default False values

        try:
            # Clean potential markdown code fences around JSON
            cleaned_response = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
            eval_results = json.loads(cleaned_response)

            if not isinstance(eval_results, dict):
                raise ValueError("LLM response for qualitative eval was not a JSON object.")

            # Validate and populate results, defaulting to False if missing or invalid type
            for metric in QUALITATIVE_METRICS:
                value = eval_results.get(metric)
                if isinstance(value, bool):
                    results[metric] = value
                else:
                    logger.warning(f"[{job_id}] Invalid or missing value for metric '{metric}' in LLM qualitative eval response: {value}. Defaulting to False.")
                    results[metric] = False # Explicitly set to False

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[{job_id}] Failed to parse/validate JSON from LLM qualitative evaluation: {e}. Response: {response_text}")
            # Return default False values if parsing fails
            return {metric: False for metric in QUALITATIVE_METRICS}

        return results
    except FileNotFoundError as e:
        logger.error(f"[{job_id}] Qualitative eval prompt template file missing: {e}")
        return results # Return default False values
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error during qualitative evaluation for question block starting with: '{full_question_block[:100]}...': {e}", exc_info=True)
        return results # Return default False values


def cleanup_job_files(job_id: str):
    """Cleans up temporary files and directories associated with a job."""
    logger.info(f"[{job_id}] Cleaning up temporary files and directories...")
    job_data = job_storage.get(job_id, {})

    # Delete temporary uploaded PDF files
    original_file_paths = job_data.get("temp_file_paths", [])
    for file_path_str in original_file_paths:
        try:
            file_path = Path(file_path_str)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"[{job_id}] Deleted temp file: {file_path}")
            else:
                logger.warning(f"[{job_id}] Temp file not found for deletion: {file_path}")
        except Exception as e:
            logger.warning(f"[{job_id}] Error deleting temp file {file_path_str}: {e}")

    # Delete temporary extracted images directory for the job
    job_image_dir = EXTRACTED_IMAGES_DIR / job_id
    if job_image_dir.exists():
        try:
            shutil.rmtree(job_image_dir)
            logger.info(f"[{job_id}] Removed temp image directory: {job_image_dir}")
        except Exception as e:
            logger.warning(f"[{job_id}] Error deleting temp image dir {job_image_dir}: {e}")

    # Delete temporary prompt files generated during the process
    for prompt_file in TEMP_UPLOAD_DIR.glob(f"{job_id}_*.txt"):
         try:
             if prompt_file.exists():
                 prompt_file.unlink()
                 logger.info(f"[{job_id}] Deleted temp prompt file: {prompt_file}")
         except Exception as e:
             logger.warning(f"[{job_id}] Error deleting temp prompt file {prompt_file}: {e}")

    logger.info(f"[{job_id}] Temporary file cleanup finished.")


# --- Background Task Functions (Main Logic) ---

def run_processing_job(job_id: str, file_paths: List[str], params: Dict):
    """Main background task: Process docs, generate initial Qs, await feedback."""
    global job_storage # Allow modification of global storage

    logger.info(f"[{job_id}] Background job started with params: {params}")
    job_storage[job_id]["status"] = "processing"
    job_storage[job_id]["message"] = "Starting document processing..."

    processed_document_ids = []
    session_id = job_id # Use job_id as session_id for Qdrant filtering
    all_final_markdown = ""
    retrieved_context = ""
    saved_image_paths_urls: List[str] = [] # Store URLs for frontend

    try:
        # STEP 1: Process PDFs, Extract Text/Images, Generate Descriptions, Chunk & Embed
        all_saved_images_map: Dict[str, str] = {} # Maps original filename to saved path string
        job_image_dir = Path(EXTRACTED_IMAGES_DIR) / job_id # Job-specific image folder

        for i, file_path_str in enumerate(file_paths):
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"[{job_id}] Input file path does not exist: {file_path_str}. Skipping.")
                continue

            job_storage[job_id]["message"] = f"Processing file {i+1}/{len(file_paths)}: {file_path.name}..."
            logger.info(f"[{job_id}] Processing file {i+1}/{len(file_paths)}: {file_path.name}")

            # Create a unique ID for this document within the job/session
            safe_base_name = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in file_path.stem])
            if not safe_base_name: safe_base_name = f"doc_{i+1}"
            document_id = f"{job_id}_{safe_base_name}" # e.g., jobuuid_mypdffile
            logger.info(f"[{job_id}] Assigned document ID: {document_id}")

            try:
                # Call Datalab Marker API
                data = call_datalab_marker(file_path)
                markdown_text = data.get("markdown", "")
                images_dict = data.get("images", {}) # {original_name: base64_data}

                # Save extracted images and get map {original_name: saved_path_str}
                doc_images_folder = job_image_dir / safe_base_name # Subfolder per document
                saved_images_map_doc = save_extracted_images(images_dict, doc_images_folder)
                all_saved_images_map.update(saved_images_map_doc) # Add to overall map for this job

                # Store relative URLs for frontend display
                for original_name, saved_path_str in saved_images_map_doc.items():
                    saved_path = Path(saved_path_str)
                    # Construct URL like /extracted_images/job_id/safe_doc_name/image.png
                    relative_path_url = f"{job_id}/{safe_base_name}/{saved_path.name}"
                    url = f"/extracted_images/{relative_path_url.replace(os.sep, '/')}"
                    saved_image_paths_urls.append(url)

                # Process markdown: Generate image descriptions using Qwen-VL
                # Pass the map of saved images for this specific document
                final_markdown = process_markdown(markdown_text, saved_images_map_doc, job_id)
                all_final_markdown += f"\n\n## --- Document: {file_path.name} ---\n\n" + final_markdown

                # Save the final processed markdown (with descriptions)
                output_markdown_path = FINAL_RESULTS_DIR / f"{document_id}_processed.md"
                output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
                output_markdown_path.write_text(final_markdown, encoding="utf-8")
                logger.info(f"[{job_id}] Saved processed markdown for {file_path.name} to {output_markdown_path}")

                # Chunk, Embed, and Upsert to Qdrant
                if final_markdown.strip():
                    logger.info(f"[{job_id}] Chunking and embedding markdown for {document_id}...")
                    chunks = hierarchical_chunk_markdown(final_markdown, file_path.name) # Pass original filename as source
                    if chunks:
                        embeddings = embed_chunks(chunks, model_st)
                        # Add document_id and session_id to metadata before upserting
                        for chunk_data in chunks:
                            if 'metadata' not in chunk_data: chunk_data['metadata'] = {}
                            chunk_data['metadata']['document_id'] = document_id
                            chunk_data['metadata']['session_id'] = session_id # Use job_id as session_id
                        upsert_to_qdrant(job_id, QDRANT_COLLECTION_NAME, embeddings, chunks)
                        processed_document_ids.append(document_id) # Track successfully processed docs
                    else:
                        logger.warning(f"[{job_id}] No chunks generated from markdown for {document_id}.")
                else:
                    logger.warning(f"[{job_id}] Final markdown for {document_id} was empty after processing. Skipping embedding.")

                logger.info(f"[{job_id}] Finished processing steps for {file_path.name}.")

            except Exception as e:
                # Log error for the specific file but continue if possible
                error_message = f"Error during processing of {file_path.name}: {e}"
                logger.error(error_message, exc_info=True)
                # Optionally update job status with partial error?
                # For now, we let the overall job fail if any file fails critically
                raise Exception(error_message) # Re-raise to fail the job

        if not processed_document_ids:
            raise ValueError("No documents were successfully processed and embedded.")

        # Store image URLs in job storage for later retrieval by status endpoint
        job_storage[job_id]["image_paths"] = saved_image_paths_urls

        # STEP 2: Generate Hypothetical Text & Search Qdrant for Relevant Context
        job_storage[job_id]["message"] = "Generating hypothetical text for search..."
        hypothetical_text = find_topics_and_generate_hypothetical_text(
            job_id, params['academic_level'], params['major'], params['course_name'], params['taxonomy_level'], params['topics_list']
        )

        job_storage[job_id]["message"] = "Embedding hypothetical text and searching context..."
        query_embedding = model_st.encode(hypothetical_text)

        search_results = search_results_from_qdrant(
            job_id, QDRANT_COLLECTION_NAME, query_embedding,
            limit=params['retrieval_limit'],
            score_threshold=params['similarity_threshold'],
            session_id_filter=session_id, # Filter by current job/session
            document_ids_filter=processed_document_ids # Filter by successfully processed docs in this job
        )

        if not search_results:
            logger.warning(f"[{job_id}] No relevant context found in vector database for the generated hypothetical text and specified documents.")
            # Decide how to proceed: fail, or try to generate questions without context?
            # For now, let's fail the job if no context is retrieved.
            raise ValueError("No relevant context found in vector database matching the query and processed documents.")

        # Compile retrieved context
        retrieved_context = "\n\n---\n\n".join([
            f"Context from: {r.payload.get('source', 'Unknown Source')} "
            f"(Score: {r.score:.4f}, DocID: {r.payload.get('document_id', 'N/A')})\n\n"
            f"{r.payload.get('text', 'N/A')}"
            for r in search_results
        ])
        # Generate a shorter preview for the UI
        retrieved_context_preview = "\n\n".join([
             f"---\n**Context Snippet {i+1}** (Source: {r.payload.get('source', 'N/A')}, "
             f"Score: {r.score:.3f}, DocID: {r.payload.get('document_id', 'N/A')})\n"
             f"{r.payload.get('text', 'N/A')[:300]}...\n---"
             for i, r in enumerate(search_results[:3]) # Show preview of top 3
        ])
        job_storage[job_id]["retrieved_context"] = retrieved_context # Store full context for generation

        # STEP 3: Generate Initial Questions based on Retrieved Context
        job_storage[job_id]["message"] = "Generating initial questions..."
        initial_questions = generate_initial_questions(job_id, retrieved_context, params)

        # STEP 4: Update Status and Result - Now Awaiting Feedback
        job_storage[job_id]["status"] = "awaiting_feedback"
        job_storage[job_id]["message"] = "Initial questions generated. Please review and provide feedback if needed."
        job_storage[job_id]["result"] = {
            "extracted_markdown": all_final_markdown.strip(), # Combined markdown from all docs
            "initial_questions": initial_questions,
            "retrieved_context_preview": retrieved_context_preview,
            "image_paths": saved_image_paths_urls, # Pass URLs to frontend
            # Initialize other result fields to None
            "generated_questions": None,
            "evaluation_feedback": None,
            "per_question_evaluation": None,
        }
        logger.info(f"[{job_id}] Job status updated to 'awaiting_feedback'.")

    except Exception as e:
        logger.exception(f"[{job_id}] Job failed during initial processing: {e}")
        job_storage[job_id]["status"] = "error"
        job_storage[job_id]["message"] = f"An error occurred during processing: {str(e)}"
        # Clear potentially incomplete results, keep image paths if generated
        if "result" in job_storage[job_id]:
             job_storage[job_id]["result"] = {
                 "image_paths": job_storage[job_id].get("image_paths", []) # Keep image paths if available
             }
        # Ensure cleanup runs even on failure
        cleanup_job_files(job_id)

    # Note: Cleanup is now called *only* in the regeneration task or if the initial task fails.
    # This preserves intermediate files while awaiting feedback.

def run_regeneration_task(job_id: str, user_feedback: str):
    """Performs question evaluation, potential regeneration, and final evaluation."""
    global job_storage # Allow modification of global storage

    logger.info(f"[{job_id}] Starting evaluation and regeneration task with user feedback: '{user_feedback[:100]}...'")
    job_data = job_storage.get(job_id)
    if not job_data:
        logger.error(f"[{job_id}] Task failed: Job data not found for evaluation/regeneration.")
        return
    if job_data.get("status") not in ["queued_feedback", "processing_feedback"]: # Check if it's ready
        logger.warning(f"[{job_id}] Regeneration task called but job status is '{job_data.get('status')}'. Aborting.")
        return

    try:
        job_data["status"] = "processing_feedback"
        job_data["message"] = "Evaluating initial questions..."

        # Retrieve necessary data stored from the initial processing stage
        retrieved_context = job_data.get("retrieved_context")
        initial_questions_block = job_data.get("result", {}).get("initial_questions")
        prompts = job_data.get("generation_prompts", {}) # Prompts used for initial generation
        original_user_prompt_filled = prompts.get("user_prompt_content")
        system_prompt = prompts.get("system_prompt") # System prompt (includes diagram hint if needed)
        params = job_data.get("params", {}) # Original parameters
        image_paths = job_data.get("result", {}).get("image_paths", []) # Image URLs
        generate_diagrams_flag = params.get('generate_diagrams', False) # Get flag again for logging/logic

        # Validate required data
        if not all([retrieved_context, initial_questions_block, original_user_prompt_filled, system_prompt, params]):
             missing_data_keys = [k for k, v in {
                 "retrieved_context": retrieved_context, "initial_questions_block": initial_questions_block,
                 "original_user_prompt_filled": original_user_prompt_filled, "system_prompt": system_prompt,
                 "params": params
             }.items() if not v]
             raise ValueError(f"Missing necessary data from initial stage for evaluation/regeneration: {', '.join(missing_data_keys)}")

        logger.info(f"[{job_id}] Regeneration task using generate_diagrams flag: {generate_diagrams_flag}")
        logger.info(f"[{job_id}] Regen task using stored system prompt (first 100 chars): {system_prompt[:100]}...")

        # --- Regeneration Loop ---
        current_questions_block = initial_questions_block
        final_questions_block = initial_questions_block # Default to initial if no regen happens
        regeneration_attempts = 0
        regeneration_performed = False
        final_evaluation_results = [] # Store final eval details
        loop_exit_reason = "Evaluation loop started."

        while regeneration_attempts < MAX_REGENERATION_ATTEMPTS:
            current_attempt = regeneration_attempts + 1
            logger.info(f"[{job_id}] Starting evaluation cycle (Attempt {current_attempt}/{MAX_REGENERATION_ATTEMPTS}).")
            job_data["message"] = f"Evaluating questions (Attempt {current_attempt})..."

            # Parse the *current* set of questions for evaluation
            parsed_current_questions = parse_questions(current_questions_block)
            if not parsed_current_questions:
                logger.error(f"[{job_id}] Failed to parse current questions block in attempt {current_attempt}. Block starts with: {current_questions_block[:200]}...")
                loop_exit_reason = f"Failed to parse questions during regeneration cycle {current_attempt}. Using previous version."
                # Keep the 'current_questions_block' from *before* this failed parse attempt
                # The final_questions_block should already hold the last successfully parsed version.
                # We need to evaluate the *previous* set if parsing the current one failed.
                # Let's re-evaluate the `final_questions_block` if parsing fails here.
                if regeneration_performed: # Only re-evaluate if we actually tried regenerating
                    logger.info(f"[{job_id}] Re-evaluating last successful set ({final_questions_block[:100]}...) due to parsing failure.")
                    parsed_final_questions = parse_questions(final_questions_block)
                    final_evaluation_results = [] # Recalculate final eval
                    for i, q_block in enumerate(parsed_final_questions):
                         q_eval = {"question_num": i + 1, "question_text": q_block}
                         q_eval["qsts_score"] = evaluate_single_question_qsts(job_id, q_block, retrieved_context)
                         q_eval["qualitative"] = evaluate_single_question_qualitative(job_id, q_block, retrieved_context)
                         final_evaluation_results.append(q_eval)
                else: # If parsing failed on the *initial* questions
                     final_evaluation_results = [] # No valid evaluation possible
                break # Exit loop

            current_evaluation_results = []
            needs_regeneration_auto = False # Based on automatic checks
            failed_question_details = [] # Collect reasons for regeneration feedback

            # --- Evaluate current questions (parsed list) ---
            logger.info(f"[{job_id}] Evaluating {len(parsed_current_questions)} parsed questions (Attempt {current_attempt})...")
            for i, question_block_item in enumerate(parsed_current_questions):
                q_eval = {"question_num": i + 1, "question_text": question_block_item} # Store full block text
                # Calculate QSTS score (on text part)
                q_eval["qsts_score"] = evaluate_single_question_qsts(job_id, question_block_item, retrieved_context)
                # Perform qualitative evaluation (on full block)
                q_eval["qualitative"] = evaluate_single_question_qualitative(job_id, question_block_item, retrieved_context)
                current_evaluation_results.append(q_eval)

                # Check if this question fails critical criteria
                qsts_failed = q_eval["qsts_score"] < QSTS_THRESHOLD
                qualitative_failed = any(not q_eval["qualitative"].get(metric, True) # Check if failed (False)
                                         for metric, should_pass in CRITICAL_QUALITATIVE_FAILURES.items()
                                         if should_pass is False) # Only check metrics that MUST be True

                if qsts_failed or qualitative_failed:
                    needs_regeneration_auto = True
                    # Get clean text for message (remove PlantUML, number)
                    question_text_for_msg = re.sub(r"```plantuml.*?```", "", question_block_item, flags=re.DOTALL | re.MULTILINE).strip()
                    question_text_for_msg = re.sub(r"^\s*\d+\s*[\.\)\-:]?\s+", "", question_text_for_msg).strip()
                    fail_reasons = []
                    if qsts_failed: fail_reasons.append(f"QSTS below threshold ({q_eval['qsts_score']:.2f} < {QSTS_THRESHOLD})")
                    if qualitative_failed:
                        failed_metrics = [m for m, passed in q_eval["qualitative"].items()
                                          if m in CRITICAL_QUALITATIVE_FAILURES and not passed]
                        if failed_metrics: fail_reasons.append(f"Failed critical checks: {', '.join(failed_metrics)}")
                    if fail_reasons: # Add details only if a specific reason was found
                        failed_question_details.append(f"  - Question {i+1} ('{question_text_for_msg[:50]}...'): {'; '.join(fail_reasons)}")

            # --- Decide on Regeneration ---
            # Regenerate if auto-checks fail OR if it's the first cycle and user provided feedback
            trigger_regeneration = needs_regeneration_auto or (regeneration_attempts == 0 and user_feedback.strip())

            if trigger_regeneration:
                logger.info(f"[{job_id}] Regeneration triggered (Attempt {current_attempt}). AutoFail={needs_regeneration_auto}, UserFeedbackProvidedOnFirstAttempt={bool(user_feedback.strip()) and regeneration_attempts == 0}")
                job_data["message"] = f"Regenerating questions based on feedback/evaluation (Attempt {current_attempt})..."
                regeneration_performed = True # Mark that we tried at least once

                # --- Construct Insightful Feedback for LLM ---
                llm_feedback = f"--- FEEDBACK ON PREVIOUS ATTEMPT (Attempt {regeneration_attempts + 1}) ---\n" # Use attempt number for clarity
                llm_feedback += "Issues identified:\n"
                if failed_question_details:
                     llm_feedback += "Automatic Evaluation Failures:\n" + "\n".join(failed_question_details) + "\n"
                     llm_feedback += "Focus on improving these specific aspects: "
                     # Add specific guidance based on failure types
                     if any("QSTS" in reason for reason in failed_question_details): llm_feedback += "Ensure question text is more semantically aligned with the core context provided. "
                     if any("Failed critical checks" in reason for reason in failed_question_details): llm_feedback += "Improve clarity, grammar, direct answerability from context, and topic relevance as indicated. "
                     llm_feedback += "\n"
                else:
                     # This case happens if only user feedback triggered regen on the first attempt
                     llm_feedback += "Automatic evaluation found no critical issues based on thresholds, but user feedback requires regeneration.\n"

                if user_feedback.strip():
                     llm_feedback += "User Provided Feedback:\n" + user_feedback.strip() + "\n"
                     llm_feedback += "Incorporate this user feedback directly into the regenerated questions.\n"
                elif not failed_question_details: # Neither auto-fail nor user feedback? Should not happen based on trigger logic, but as safety:
                     llm_feedback += "Regeneration requested without specific automatic failures or user feedback. Please review and improve overall quality based on original instructions.\n"


                # Add the reminder about diagrams if needed
                diagram_reminder = ""
                if generate_diagrams_flag:
                    diagram_reminder = (", including appropriate PlantUML diagrams (in ```plantuml ... ``` blocks) where relevant and supported by the context. "
                                        "Ensure diagrams are accurate based *only* on the provided text.")

                llm_feedback += (
                     f"\nPlease regenerate EXACTLY {params['num_questions']} questions, addressing these points while strictly adhering to all original instructions "
                     f"(Use ONLY the provided context, target Bloom's level: {params['taxonomy_level']}, course: '{params['course_name']}', topics: {params['topics_list']}"
                     f"{diagram_reminder}). Strive for high-quality, insightful questions directly answerable from the context. "
                     "Output ONLY the numbered list of questions (and associated PlantUML code blocks if generated), with NO other text, introductions, explanations, or summaries." # Reiterate output format strongly
                 )

                # Combine the original *filled* user prompt with the new feedback for regeneration
                # This ensures the LLM still has the context and original task details
                regeneration_user_prompt = f"{original_user_prompt_filled}\n\n{llm_feedback}"

                # --- Call Gemini for Regeneration ---
                # Use the *same system prompt* stored earlier, which already contains output format and diagram reqs
                logger.info(f"[{job_id}] Calling Gemini for regeneration (Attempt {current_attempt})...")
                regenerated_questions_block_attempt = get_gemini_response(system_prompt, regeneration_user_prompt)

                if regenerated_questions_block_attempt.startswith("Error:"):
                    logger.error(f"[{job_id}] Regeneration attempt {current_attempt} failed during LLM call: {regenerated_questions_block_attempt}")
                    loop_exit_reason = f"Regeneration failed during LLM call in attempt {current_attempt}. Keeping previous version."
                    # Keep the 'current_questions_block' from *before* this failed attempt
                    final_questions_block = current_questions_block
                    final_evaluation_results = current_evaluation_results # Use eval results from *before* failed regen
                    job_data["regeneration_error"] = f"Regeneration failed ({regenerated_questions_block_attempt})."
                    break # Exit loop after LLM failure
                else:
                    # Regeneration successful, update current block for next evaluation cycle
                    logger.info(f"[{job_id}] Successfully regenerated questions (Attempt {current_attempt}). Snippet: {regenerated_questions_block_attempt[:300]}...")
                    # Log PlantUML presence check for regenerated content
                    plantuml_found_regen = "```plantuml" in regenerated_questions_block_attempt
                    if generate_diagrams_flag and not plantuml_found_regen:
                        logger.warning(f"[{job_id}] PlantUML requested, but '```plantuml' not found in *regenerated* response (Attempt {current_attempt}).")
                    elif not generate_diagrams_flag and plantuml_found_regen:
                        logger.warning(f"[{job_id}] PlantUML *not* requested, but '```plantuml' *was* found in *regenerated* response (Attempt {current_attempt}).")

                    # Update the block to be evaluated in the next loop iteration
                    current_questions_block = regenerated_questions_block_attempt
                    final_questions_block = current_questions_block # Keep track of the latest successful generation
                    regeneration_attempts += 1 # Increment successful attempt counter
                    # We will evaluate this new block at the start of the next loop iteration

            else:
                # No regeneration needed (passed auto checks, no user feedback on first try)
                logger.info(f"[{job_id}] No regeneration needed after evaluation attempt {current_attempt}.")
                final_questions_block = current_questions_block # This set passed
                final_evaluation_results = current_evaluation_results # Use the results from this successful evaluation
                loop_exit_reason = f"Questions met criteria in evaluation attempt {current_attempt}, or user provided no actionable feedback on first attempt."
                break # Exit loop as criteria are met

        # --- End of Regeneration Loop ---

        # Handle case where max attempts are reached
        if regeneration_attempts == MAX_REGENERATION_ATTEMPTS and loop_exit_reason.startswith("Evaluation loop started."): # Check if loop completed due to max attempts
            loop_exit_reason = f"Reached maximum regeneration attempts ({MAX_REGENERATION_ATTEMPTS}). Using last generated set."
            final_questions_block = current_questions_block # Use the very last set generated
            logger.warning(f"[{job_id}] {loop_exit_reason}")
            # Perform final evaluation on the last generated set
            logger.info(f"[{job_id}] Performing final evaluation on questions from max attempt ({MAX_REGENERATION_ATTEMPTS})...")
            final_parsed_questions = parse_questions(final_questions_block)
            final_evaluation_results = [] # Reset and re-evaluate
            if not final_parsed_questions:
                 logger.error(f"[{job_id}] Failed to parse final questions after max attempts! Block: {final_questions_block[:200]}...")
                 loop_exit_reason += " (Failed to parse final block)"
            else:
                 for i, q_block in enumerate(final_parsed_questions):
                     q_eval = {"question_num": i + 1, "question_text": q_block}
                     q_eval["qsts_score"] = evaluate_single_question_qsts(job_id, q_block, retrieved_context)
                     q_eval["qualitative"] = evaluate_single_question_qualitative(job_id, q_block, retrieved_context)
                     final_evaluation_results.append(q_eval)

        # --- Construct Final Feedback Summary for UI ---
        job_data["message"] = "Finalizing results..."
        logger.info(f"[{job_id}] Constructing final report. Loop exit reason: {loop_exit_reason}")

        # Re-parse the absolute final block just to be sure of the count
        final_parsed_questions_for_summary = parse_questions(final_questions_block)
        num_final_questions = len(final_parsed_questions_for_summary)
        final_feedback_summary = f"Processing finished. {num_final_questions} question blocks generated.\n"
        final_feedback_summary += f"Processing Summary: {loop_exit_reason}\n"
        if regeneration_performed:
            final_feedback_summary += f"Regeneration was attempted {regeneration_attempts} time(s).\n" # Use actual attempts count
        else:
            final_feedback_summary += "No regeneration cycles were performed.\n"

        if job_data.get("regeneration_error"):
             final_feedback_summary += f"Note on Regeneration: {job_data['regeneration_error']}\n"

        # Summarize the final evaluation results
        passed_count = 0
        failed_details_summary = []
        if final_evaluation_results:
             for i, res in enumerate(final_evaluation_results):
                # Check pass criteria based on the final evaluation
                qsts_ok = res.get('qsts_score', 0) >= QSTS_THRESHOLD
                qual_metrics = res.get('qualitative', {})
                # Check if any critical metric that *should* be True is False
                qual_ok = not any(not qual_metrics.get(metric, True) for metric, must_be_true in CRITICAL_QUALITATIVE_FAILURES.items() if must_be_true is False)

                if qsts_ok and qual_ok:
                    passed_count += 1
                else:
                    # Collect details of failures for the summary
                    fail_reasons_summary = []
                    if not qsts_ok: fail_reasons_summary.append(f"QSTS {res.get('qsts_score', 0):.2f} < {QSTS_THRESHOLD}")
                    if not qual_ok:
                         failed_metrics_summary = [m for m, passed in qual_metrics.items() if m in CRITICAL_QUALITATIVE_FAILURES and not passed]
                         if failed_metrics_summary: fail_reasons_summary.append(f"CritFail({','.join(failed_metrics_summary)})")
                    failed_details_summary.append(f"  - Q{i+1}: {'; '.join(fail_reasons_summary)}")

             final_feedback_summary += f"\nFinal Evaluation Summary:\n"
             final_feedback_summary += f"- Passed Checks: {passed_count}/{num_final_questions} " \
                                       f"(QSTS >= {QSTS_THRESHOLD} and critical qualitative metrics met).\n"
             if failed_details_summary:
                 final_feedback_summary += f"- Failed Checks Details:\n" + "\n".join(failed_details_summary) + "\n"

        elif not final_parsed_questions_for_summary and final_questions_block:
             final_feedback_summary += "\nFinal evaluation could not be fully reported because the final question block could not be parsed.\n"
        else: # No evaluation results likely due to earlier error
             final_feedback_summary += "\nFinal evaluation results are unavailable.\n"

        # --- Update Job Storage with Final Results ---
        job_data["status"] = "completed"
        job_data["message"] = "Processing complete."
        # Ensure result dict exists
        if "result" not in job_data or not isinstance(job_data["result"], dict):
            job_data["result"] = {}
        # Update relevant fields in the result dict
        job_data["result"]["generated_questions"] = final_questions_block # Store the final raw block
        job_data["result"]["evaluation_feedback"] = final_feedback_summary.strip() # UI summary
        job_data["result"]["per_question_evaluation"] = final_evaluation_results # Detailed eval list
        job_data["result"]["image_paths"] = image_paths # Ensure image paths are still there

        logger.info(f"[{job_id}] Evaluation/Regeneration task completed successfully. Final status: completed.")

    except Exception as e:
         # Catch unexpected errors during the regen/eval process
         logger.exception(f"[{job_id}] Evaluation/Regeneration task failed unexpectedly: {e}")
         job_data["status"] = "error"
         job_data["message"] = f"Processing failed during evaluation/regeneration: {e}"
         # Preserve results generated so far if possible? Maybe too risky.
         if "result" in job_data:
             job_data["result"]["evaluation_feedback"] = job_data["result"].get("evaluation_feedback", "") + f"\nERROR during final processing: {e}"

    finally:
         # --- IMPORTANT: Cleanup temporary files AFTER regeneration task is complete ---
         cleanup_job_files(job_id)
         logger.info(f"[{job_id}] Regeneration task finished (Final Status: {job_data.get('status', 'unknown')}). Cleanup executed.")


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    # Ensure models are loaded before serving requests that might depend on them
    if model_st is None or qdrant_client is None or model_qwen is None or processor_qwen is None:
         raise HTTPException(status_code=503, detail="Service Unavailable: Models are not ready.")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start-processing", response_model=Job)
async def start_processing_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF documents to process"),
    course_name: str = Form(...),
    num_questions: str = Form(...), # Receive as string for initial validation
    academic_level: str = Form(...),
    taxonomy_level: str = Form(...),
    topics_list: str = Form(...),
    major: str = Form(...),
    retrieval_limit: int = Form(15, ge=1, le=100),
    similarity_threshold: float = Form(0.3, ge=0.0, le=1.0),
    generate_diagrams: bool = Form(False, description="Check if PlantUML diagrams should be generated")
):
    """ Starts the PDF processing and initial question generation job. """
    global job_storage # Allow modification of global storage

    # Check if models are ready before accepting job
    if model_st is None or qdrant_client is None or model_qwen is None or processor_qwen is None:
         logger.error("Attempted to start job, but models are not initialized.")
         raise HTTPException(status_code=503, detail="Service Unavailable: Models are initializing or failed to load. Please try again later.")

    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Received request to start job. generate_diagrams: {generate_diagrams}")
    temp_file_paths = []

    # Validate num_questions robustly
    try:
        num_q_int = int(num_questions)
        if not (1 <= num_q_int <= 100):
             raise ValueError("Number of questions must be between 1 and 100.")
    except (ValueError, TypeError):
         logger.error(f"[{job_id}] Invalid num_questions received: '{num_questions}'")
         raise HTTPException(status_code=400, detail="Invalid input: 'Number of questions' must be an integer between 1 and 100.")

    # Initialize job storage entry
    job_storage[job_id] = {
        "job_id": job_id, # Store job_id itself
        "status": "pending",
        "message": "Validating inputs and saving files...",
        "params": {
            "course_name": course_name,
            "num_questions": num_q_int, # Store the validated integer
            "academic_level": academic_level,
            "taxonomy_level": taxonomy_level,
            "topics_list": topics_list,
            "major": major,
            "retrieval_limit": retrieval_limit,
            "similarity_threshold": similarity_threshold,
            "generate_diagrams": generate_diagrams # Store the boolean
            },
        "result": {}, # Initialize result structure
        "temp_file_paths": [], # To store paths of saved uploads for cleanup
        "image_paths": [], # To store URLs of extracted images for UI
        "generation_prompts": {} # To store prompts for regeneration
    }
    logger.info(f"[{job_id}] Initialized job storage. Params: {job_storage[job_id]['params']}")

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        upload_dir = TEMP_UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True) # Ensure upload dir exists
        valid_files_saved = 0

        # --- File Saving Logic ---
        for file in files:
            # Basic validation
            if not file.filename:
                logger.warning(f"[{job_id}] Skipping file with no filename.")
                continue
            if not file.filename.lower().endswith(".pdf"):
                logger.warning(f"[{job_id}] Skipping non-PDF file: {file.filename}")
                continue
            if file.size == 0:
                logger.warning(f"[{job_id}] Skipping empty file: {file.filename}")
                continue
            # Limit file size? e.g., if file.size > MAX_FILE_SIZE: continue

            # Sanitize filename (more robustly)
            safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename) # Replace invalid chars with underscore
            if not safe_filename: safe_filename = f"file_{valid_files_saved+1}.pdf" # Fallback

            temp_file_path = upload_dir / f"{job_id}_{uuid.uuid4().hex[:8]}_{safe_filename}"

            try:
                # Save the file chunk by chunk to handle large files potentially
                with temp_file_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                # Add path string to list for processing and later cleanup
                temp_file_paths.append(str(temp_file_path))
                valid_files_saved += 1
                logger.info(f"[{job_id}] Successfully saved uploaded file: {file.filename} to {temp_file_path}")
            except Exception as e:
                logger.error(f"[{job_id}] Failed to save uploaded file {file.filename}: {e}", exc_info=True)
                # Clean up already saved files for this job if saving fails mid-way
                for p_str in temp_file_paths: Path(p_str).unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=f"Failed to save file '{file.filename}'.")
            finally:
                # Ensure file handle is closed
                await file.close()
        # --- End File Saving ---

        if valid_files_saved == 0:
            # No valid files were processed, maybe only invalid types were uploaded
            raise HTTPException(status_code=400, detail="No valid PDF files provided or saved.")

        # Store the paths of the successfully saved files in job storage
        job_storage[job_id]["temp_file_paths"] = temp_file_paths

        # Add the main processing task to background tasks
        background_tasks.add_task(run_processing_job, job_id=job_id, file_paths=temp_file_paths, params=job_storage[job_id]["params"])

        # Update status to queued
        job_storage[job_id]["status"] = "queued"
        job_storage[job_id]["message"] = f"Processing job queued for {valid_files_saved} valid PDF file(s)."
        logger.info(f"[{job_id}] Job successfully queued.")

        # Return the initial job status
        return Job(job_id=job_id, status="queued", message=job_storage[job_id]["message"])

    except HTTPException as http_exc:
        # Handle exceptions raised intentionally (like validation errors)
        logger.error(f"[{job_id}] HTTP exception occurred while starting job: {http_exc.detail}")
        cleanup_job_files(job_id) # Clean up any files potentially saved before error
        job_storage.pop(job_id, None) # Remove job entry
        raise http_exc # Re-raise the exception for FastAPI to handle
    except Exception as e:
        # Handle unexpected errors during setup/file saving
        logger.exception(f"[{job_id}] Unexpected error occurred while starting job: {e}")
        cleanup_job_files(job_id) # Clean up
        job_storage.pop(job_id, None) # Remove job entry
        raise HTTPException(status_code=500, detail="Internal server error occurred while starting the job.")


@app.post("/regenerate-questions/{job_id}", response_model=Job)
async def regenerate_questions_endpoint(
    job_id: str, request: RegenerationRequest, background_tasks: BackgroundTasks
):
    """ Triggers the evaluation and potential regeneration based on user feedback. """
    global job_storage # Allow modification of global storage

    logger.info(f"[{job_id}] Received request to regenerate/finalize questions.")
    job_data = job_storage.get(job_id)

    # --- Job Validation ---
    if not job_data:
        logger.error(f"[{job_id}] Regeneration request failed: Job ID not found.")
        raise HTTPException(status_code=404, detail="Job not found")

    current_status = job_data.get("status")
    # Allow regeneration only if awaiting feedback
    if current_status != "awaiting_feedback":
        logger.warning(f"[{job_id}] Regeneration request rejected: Job status is '{current_status}', not 'awaiting_feedback'.")
        raise HTTPException(status_code=400, detail=f"Job is not in a state awaiting feedback (current status: {current_status}). Cannot regenerate.")

    # --- Queue Regeneration Task ---
    try:
        # Update status to indicate feedback processing is queued
        job_data["status"] = "queued_feedback"
        job_data["message"] = "Queued for evaluation and potential regeneration based on feedback..."
        logger.info(f"[{job_id}] Queuing evaluation/regeneration task.")

        # Add the regeneration task to the background
        background_tasks.add_task(run_regeneration_task, job_id=job_id, user_feedback=request.feedback or "") # Pass feedback

        # Return the current state (queued for feedback)
        result_model = JobResultData(**job_data.get("result", {})) if job_data.get("result") else None
        return Job(job_id=job_id, status=job_data["status"], message=job_data["message"], result=result_model)

    except Exception as e:
        # Handle unexpected errors during the queuing process itself
        logger.exception(f"[{job_id}] Unexpected error occurred while queuing regeneration task: {e}")
        # Revert status if possible? Maybe set to error.
        job_data["status"] = "error"
        job_data["message"] = f"Failed to queue regeneration task: {e}"
        raise HTTPException(status_code=500, detail="Internal server error occurred while queuing the regeneration task.")


@app.get("/status/{job_id}", response_model=Job)
async def get_job_status(job_id: str):
    """ Endpoint to check the status and result of a processing job. """
    global job_storage # Access global storage

    logger.debug(f"Status request received for job_id: {job_id}")
    job = job_storage.get(job_id)

    if not job:
        logger.warning(f"Status request for non-existent job_id: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")

    # Prepare the result data safely
    result_data = job.get("result")
    job_result_model = None
    parsing_error_occurred = False

    if isinstance(result_data, dict):
        try:
            # Attempt to parse the result dictionary into the Pydantic model
            job_result_model = JobResultData(**result_data)
        except Exception as e:
            parsing_error_occurred = True
            logger.error(f"[{job_id}] Error parsing result data into JobResultData model: {e}. Result Data: {result_data}", exc_info=True)
            # If parsing fails, don't include the broken result in the response
            # Update the job message to reflect the error, but don't change status necessarily unless it's already error
            job["message"] = job.get("message", "") + " [Warning: Result data parsing error]"
            # Optionally set status to error if not already?
            # if job.get("status") != "error": job["status"] = "error" # Be cautious with this

    # Return the Job model including potentially parsed results
    # If parsing failed, job_result_model will be None
    return Job(
        job_id=job_id,
        status=job.get("status", "unknown"),
        message=job.get("message"),
        result=job_result_model # This will be None if result_data was not a dict or if parsing failed
    )


@app.get("/health")
async def health_check():
    """ Basic health check endpoint. """
    # Check essential components if possible
    status = "ok"
    details = {"message": "Service is running."}
    try:
        # Check if Qdrant client is connected (simple check)
        if qdrant_client:
             qdrant_client.get_collections() # Throws error if connection fails
             details["qdrant"] = "connected"
        else: details["qdrant"] = "not initialized"; status = "error"

        # Check if embedding model loaded
        if model_st: details["embedding_model"] = "loaded"
        else: details["embedding_model"] = "not loaded"; status = "error"

        # Check if Qwen model loaded
        if model_qwen and processor_qwen: details["vision_language_model"] = "loaded"
        else: details["vision_language_model"] = "not loaded"; status = "error"

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        status = "error"
        details["error"] = str(e)

    if status == "error":
        return JSONResponse(status_code=503, content={"status": status, **details})
    else:
        return {"status": status, **details}