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
import urllib.parse
import json
import argparse # For command-line arguments

# FastAPI imports (Removed - not needed for standalone)
# Pydantic models (Removed - not needed for standalone)

# Text Processing & Embeddings
from PIL import Image
import nltk
try:
    nltk.data.find('corpora/stopwords')
    # nltk.data.find('corpora/wordnet') # Still commented out as per original
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK data not found. Downloading...")
    nltk.download('stopwords')
    # nltk.download('wordnet') # Still commented out as per original
    nltk.download('punkt')
    print("NLTK data downloaded.")

from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer # Removed as WordNet not used
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
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")
DATALAB_MARKER_URL = os.environ.get("DATALAB_MARKER_URL")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# --- Check Essential Variables ---
if not all([DATALAB_API_KEY, DATALAB_MARKER_URL, QDRANT_URL, GEMINI_API_KEY]):
    missing_vars = [var for var, val in {
        "DATALAB_API_KEY": DATALAB_API_KEY, "DATALAB_MARKER_URL": DATALAB_MARKER_URL,
        "QDRANT_URL": QDRANT_URL, "GEMINI_API_KEY": GEMINI_API_KEY
    }.items() if not val]
    logging.critical(f"FATAL ERROR: Missing essential environment variables: {', '.join(missing_vars)}")
    sys.exit("Missing essential environment variables.")

# --- Directories ---
BASE_DIR = Path(__file__).parent # Directory where the script is located
TEMP_UPLOAD_DIR = BASE_DIR / "temp_standalone_uploads" # Temporary storage for this script's runs
FINAL_RESULTS_DIR = BASE_DIR / "final_standalone_results" # Output markdown
EXTRACTED_IMAGES_DIR = BASE_DIR / "extracted_standalone_images" # Output images
PROMPT_DIR = BASE_DIR / "content" # Directory for prompt templates

# --- Constants ---
DATALAB_POST_TIMEOUT = 60
DATALAB_POLL_TIMEOUT = 30
MAX_POLLS = 300
POLL_INTERVAL = 3
GEMINI_TIMEOUT = 300 # Increased slightly
MAX_GEMINI_RETRIES = 3
GEMINI_RETRY_DELAY = 60
QDRANT_COLLECTION_NAME = "markdown_docs_v3_semantic" # Use the same collection
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
# Choose your desired Gemini model
GEMINI_MODEL_NAME = "gemini-2.0-flash"

GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:{action}?key={api_key}"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 256
QSTS_THRESHOLD = 0.5 # Question-Semantic Text Similarity threshold for evaluation
QUALITATIVE_METRICS = ["Understandable", "TopicRelated", "Grammatical", "Clear", "Answerable", "Central"]
CRITICAL_QUALITATIVE_FAILURES = {"Understandable": False, "Grammatical": False, "Clear": False, "Answerable": False, "TopicRelated": False, "Central": False} # If a metric listed here is False, it's a critical failure
MAX_REGENERATION_ATTEMPTS = 5 # Limit feedback loop iterations
MAX_HISTORY_TURNS = 10 # Max user+model turns in history (5 pairs) to keep context manageable

# Prompt File Paths (relative to BASE_DIR)
FINAL_USER_PROMPT_PATH = PROMPT_DIR / "final_user_prompt.txt"
HYPOTHETICAL_PROMPT_PATH = PROMPT_DIR / "hypothetical_prompt.txt"
QUALITATIVE_EVAL_PROMPT_PATH = PROMPT_DIR / "qualitative_eval_prompt.txt"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
# Suppress noisy PIL logging for truncated images unless it's a warning or error
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Ensure Base Directories Exist ---
for dir_path in [TEMP_UPLOAD_DIR, FINAL_RESULTS_DIR, EXTRACTED_IMAGES_DIR, PROMPT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Check for mandatory prompt files ---
mandatory_prompts = [FINAL_USER_PROMPT_PATH, HYPOTHETICAL_PROMPT_PATH, QUALITATIVE_EVAL_PROMPT_PATH]
missing_prompts = [p.name for p in mandatory_prompts if not p.exists()]
if missing_prompts:
    logger.critical(f"FATAL ERROR: Missing required prompt template files in '{PROMPT_DIR}': {', '.join(missing_prompts)}. Make sure '{FINAL_USER_PROMPT_PATH.name}' contains the updated structure.")
    sys.exit(f"Missing prompt files: {', '.join(missing_prompts)}")

# --- Initialize Models and Clients (Global Scope) ---
model_st = None
qdrant_client = None
model_qwen = None
processor_qwen = None
device = None
stop_words = None
# lemmatizer = None # Removed as WordNet not used

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
        # Handle potential variations in Qdrant error messages for "Not Found"
        # Handle case where Qdrant client might raise different exception types
        err_str = str(e).lower()
        is_not_found = ("not found" in err_str or "status_code=404" in err_str or "reason: not found" in err_str or "collection not found" in err_str)

        if is_not_found:
            logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
            try:
                # Explicitly use recreate_collection for simplicity, ensuring it's reset if needed
                qdrant_client.recreate_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
                logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")
            except Exception as create_e:
                logger.critical(f"FATAL QDRANT ERROR: Failed to create collection '{QDRANT_COLLECTION_NAME}': {create_e}", exc_info=True)
                sys.exit(f"Failed to create Qdrant collection: {create_e}")
        else:
             logger.critical(f"FATAL QDRANT ERROR: Could not connect or access collection '{QDRANT_COLLECTION_NAME}': {e}", exc_info=True)
             sys.exit(f"Qdrant connection/access error: {e}")


    # NLTK Setup (Load pre-downloaded resources)
    logger.info("Loading NLTK resources (stopwords, punkt)...")
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        stop_words = set(stopwords.words('english'))
        # lemmatizer = WordNetLemmatizer() # Removed
        logger.info("NLTK resources loaded successfully.")
    except LookupError as e:
        logger.critical(f"FATAL NLTK ERROR: Required resource not found: {e}. Please run the NLTK download steps first (see comments in script).", exc_info=True)
        sys.exit(f"NLTK resource missing: {e}")

except Exception as e:
    logger.critical(f"Fatal error during initialization: {e}", exc_info=True)
    sys.exit("Initialization failed.")


# --- Bloom's Taxonomy Guidance Function ---
def get_bloom_guidance(level: str) -> str:
    """Provides specific instructions based on the target Bloom's level."""
    level = level.lower()
    guidance = {
        "remember": """
        - **Focus:** Recalling facts, terms, basic concepts, and answers from the context.
        - **Action Verbs:** Define, list, state, identify, label, name, recall, recognize.
        - **Instructions:** Ask questions that require retrieving specific information directly stated in the text. Avoid interpretation or analysis. Example: "List the steps involved in X as described in the context."
        """,
        "understand": """
        - **Focus:** Explaining ideas or concepts; interpreting information.
        - **Action Verbs:** Explain, describe, summarize, paraphrase, interpret, classify, compare (basic), contrast (basic).
        - **Instructions:** Ask questions that require processing information, putting it in one's own words, or identifying similarities/differences explicitly mentioned. Avoid questions requiring judgment or creation. Example: "Explain the main purpose of Y according to the provided text."
        """,
        "apply": """
        - **Focus:** Using information in a new situation; solving problems using learned knowledge or rules.
        - **Action Verbs:** Apply, use, demonstrate, solve, calculate, implement, execute, illustrate (with an example from context).
        - **Instructions:** Ask questions that require applying a concept, method, or rule mentioned in the context to a specific (potentially hypothetical but context-grounded) scenario. The scenario should be solvable using only context information. Example: "Based on the method described, how would you calculate Z for the given parameters A and B found in the context?"
        """,
        "analyze": """
        - **Focus:** Breaking down information into parts; identifying patterns, causes, and motives; making inferences and finding evidence.
        - **Action Verbs:** Analyze, compare (in-depth), contrast (in-depth), differentiate, examine, break down, categorize, investigate, deduce.
        - **Instructions:** Ask questions that require identifying underlying structures, relationships, or biases within the context. Require justification based *only* on evidence present in the snippets. Avoid simple comparisons; focus on *how* or *why* things are related or different based on the text. Example: "Analyze the relationship between component A and component B as presented in the context, highlighting the key interactions described."
        """,
        "evaluate": """
        - **Focus:** Making judgments about information, validity of ideas, or quality of work based on a set of criteria (which should be derivable *from the context* or generally accepted principles relevant to the context).
        - **Action Verbs:** Evaluate, judge, critique, justify, recommend, assess, defend, rate, argue for/against based on context.
        - **Instructions:** Ask questions that require forming an opinion or making a decision supported *explicitly* by evidence or criteria found within the context snippets. The judgment should not rely on external knowledge. Emphasize justification using the text. Example: "Based *only* on the advantages and disadvantages listed in the context, evaluate the suitability of method X compared to method Y for the specific scenario described in the text. Justify your reasoning using the provided information." **AVOID simple recall.** Require critical assessment.
        """,
        "create": """
        - **Focus:** Combining elements in a new pattern or proposing alternative solutions; generating, designing, or constructing something new based *integratively* on the context.
        - **Action Verbs:** Create, design, develop, propose, formulate, generate, invent, hypothesize, construct, synthesize.
        - **Instructions:** Ask questions that require synthesizing information from *multiple parts* of the context to produce a novel idea, plan, or product grounded in the provided text. The creation should be a logical extension or combination of concepts within the context, not pure invention. Example: "Based on the principles of A and the components of B described in the context, propose a basic structure for a system that achieves outcome C. Describe the key elements and their relationship, using only concepts mentioned in the text." **AVOID simple recall.** Require generation or synthesis.
        """
    }
    default_guidance = """
    - **Focus:** General question generation based on context.
    - **Instructions:** Ensure the question aligns with the specified Bloom's level, is clear, answerable from the context, and relevant to the topics.
    """
    return guidance.get(level, default_guidance).strip()


# --- Core Functions ---

# generate_description_for_image (No changes needed)
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
             # Explicitly setting temperature is redundant with do_sample=False, but harmless.
             # Setting it low reinforces deterministic intent if defaults were different.
            output_ids = model_qwen.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False, temperature=0.001, top_p=None, top_k=None)
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
    except Image.DecompressionBombError:
         logger.error(f"Image file is too large or corrupt (DecompressionBombError): {image_path}")
         return f"Error: Image file too large or corrupt."
    except Exception as e:
        logger.error(f"Error generating Qwen-VL description for {image_path.name}: {e}", exc_info=True)
        if "CUDA out of memory" in str(e):
            logger.critical(f"CUDA OOM Error during Qwen-VL inference on {image_path.name}.")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return "Error: Ran out of GPU memory generating description."
        return f"Error generating description for this image due to an internal error."

# call_datalab_marker (No changes needed)
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
            status_code = e.response.status_code if e.response is not None else "N/A"
            response_text = e.response.text if e.response is not None else "N/A"
            logger.error(f"[{job_id_for_log}] Datalab API request failed for {file_path.name} (Status: {status_code}): {e}. Response: {response_text[:500]}")
            raise Exception(f"Datalab API request failed (Status: {status_code}): {e}")
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
            elif poll_data.get("status") == "processing" and (i + 1) % 10 == 0:
                 logger.info(f"[{job_id_for_log}] Datalab still processing {file_path.name}... attempt {i+1}/{MAX_POLLS}")
            elif poll_data.get("status") not in ["processing", "complete", "error"]:
                 logger.warning(f"[{job_id_for_log}] Unknown Datalab status '{poll_data.get('status')}' for {file_path.name} on attempt {i+1}.")

        except requests.exceptions.Timeout: logger.warning(f"[{job_id_for_log}] Polling Datalab timed out on attempt {i+1} for {file_path.name}. Retrying...")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            logger.warning(f"[{job_id_for_log}] Polling error on attempt {i+1} for {file_path.name} (Status: {status_code}): {e}. Retrying...")
    logger.error(f"[{job_id_for_log}] Polling timed out waiting for Datalab processing for {file_path.name}.")
    raise TimeoutError("Polling timed out waiting for Datalab processing.")

# save_extracted_images (No changes needed)
def save_extracted_images(images_dict, images_folder: Path, job_id_for_log: str) -> Dict[str, str]:
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    logger.info(f"[{job_id_for_log}] Saving {len(images_dict)} extracted images to {images_folder}...")
    for img_name, b64_data in images_dict.items():
        try:
            # More robust safe name generation
            safe_img_name = re.sub(r'[^\w\-.]', '_', img_name)
            # Prevent names starting with '.' or overly long names
            if safe_img_name.startswith('.'): safe_img_name = '_' + safe_img_name
            safe_img_name = safe_img_name[:200] # Limit filename length

            if not safe_img_name: safe_img_name = f"image_{uuid.uuid4().hex[:8]}.png" # Fallback if name becomes empty

            image_data = base64.b64decode(b64_data)
            image_path = images_folder / safe_img_name
            with open(image_path, "wb") as img_file: img_file.write(image_data)
            saved_files[img_name] = str(image_path) # Store original name mapping to saved path
        except (base64.binascii.Error, ValueError) as decode_err:
             logger.warning(f"[{job_id_for_log}] Could not decode base64 for image '{img_name}': {decode_err}")
        except Exception as e:
            logger.warning(f"[{job_id_for_log}] Could not decode/save image '{img_name}': {e}", exc_info=True)
    return saved_files

# process_markdown (No changes needed)
def process_markdown(markdown_text, saved_images: Dict[str, str], job_id_for_log: str):
    logger.info(f"[{job_id_for_log}] Processing markdown for image descriptions using Qwen-VL...")
    lines = markdown_text.splitlines()
    processed_lines = []
    i = 0; image_count = 0
    figure_pattern = re.compile(r"^!\[.*?\]\((.*?)\)$") # Match image tags like ![alt text](filename.png)
    caption_pattern = re.compile(r"^(Figure|Table|Chart)\s?(\d+[:.]?)\s?(.*)", re.IGNORECASE) # Match captions like "Figure 1: ..."

    while i < len(lines):
        line = lines[i]; stripped_line = line.strip()
        image_match = figure_pattern.match(stripped_line)

        if image_match:
            image_filename_encoded = image_match.group(1)
            try:
                # Handle URL encoding in filenames (e.g., spaces become %20)
                image_filename_decoded = urllib.parse.unquote(image_filename_encoded)
            except Exception as decode_err:
                logger.warning(f"[{job_id_for_log}] Could not URL-decode image filename '{image_filename_encoded}': {decode_err}. Using original.")
                image_filename_decoded = image_filename_encoded

            image_count += 1
            caption = ""; caption_line_index = -1

            # Look ahead for a potential caption immediately following the image tag (allowing for blank lines)
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1 # Skip blank lines
            if j < len(lines):
                next_line_stripped = lines[j].strip()
                if caption_pattern.match(next_line_stripped):
                    caption = next_line_stripped
                    caption_line_index = j # Remember the index of the caption line

            # Find the actual saved image path using the mapping
            # Try both decoded and encoded names as keys, as Datalab might use either
            image_path_str = saved_images.get(image_filename_decoded)
            if not image_path_str:
                 image_path_str = saved_images.get(image_filename_encoded)

            description = ""
            if image_path_str:
                image_path = Path(image_path_str)
                if image_path.exists():
                    description = generate_description_for_image(image_path, caption)
                else:
                    description = f"*Referenced image file '{image_path.name}' not found locally.*"
                    logger.warning(f"[{job_id_for_log}] {description}")
            else:
                description = f"*Referenced image filename '{image_filename_decoded}' (or '{image_filename_encoded}') not found in the saved images dictionary.*"
                logger.warning(f"[{job_id_for_log}] {description}")

            # Construct the block to insert
            title_text = caption if caption else f"Figure {image_count}"
            # Using --- for thematic breaks and ### for the title consistently
            block_text = f"\n---\n### {title_text}\n\n**Figure Description (Generated by Qwen-VL):**\n{description}\n---\n"
            processed_lines.append(block_text) # Add the generated block

            # If we found and used a caption, skip the original caption line in the input
            if caption_line_index != -1:
                i = caption_line_index # Advance the main loop index past the caption

        else:
            # If it's not an image line, just append the original line
            processed_lines.append(line)

        i += 1 # Move to the next line in the original markdown

    logger.info(f"[{job_id_for_log}] Finished processing markdown with Qwen-VL. Processed {image_count} image references.")
    return "\n".join(processed_lines)


# clean_text_for_embedding (No changes needed)
def clean_text_for_embedding(text):
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text) # Reduce multiple newlines to max 2
    # Remove markdown horizontal rules (---, ***, etc.)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
     # Remove markdown headers (## Header)
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)
    # Remove specific injected text like figure descriptions for embedding clarity
    text = re.sub(r'\*\*Figure Description \(Generated by Qwen-VL\):\*\*', '', text, flags=re.IGNORECASE)
    # Remove potential leftover figure titles if they weren't cleaned by header removal
    text = re.sub(r'^###\s+(?:Figure|Table|Chart).*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove list markers? Optional, might remove useful info.
    # text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    # Remove blockquotes? Optional.
    # text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)

    return text.strip()

# hierarchical_chunk_markdown (No changes needed)
def hierarchical_chunk_markdown(markdown_text, source_filename, job_id_for_log: str):
    logger.info(f"[{job_id_for_log}] Chunking markdown from source: {source_filename}")
    lines = markdown_text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_headers = {} # Store headers like {1: "Main Title", 2: "Subtitle"}
    figure_title = None # Store title of the current figure block, if any
    inside_figure_block = False # Flag to track if we are inside a --- delimited block

    header_pattern = re.compile(r"^(#{1,6})\s+(.*)") # Matches lines like ### Title
    figure_title_pattern = re.compile(r"^###\s+((?:Figure|Table|Chart).*)$", re.IGNORECASE) # Matches ### Figure 1...
    separator_pattern = re.compile(r"^\s*---\s*$") # Matches --- separators

    for line_num, line in enumerate(lines):
        stripped_line = line.strip()
        header_match = header_pattern.match(stripped_line)
        figure_title_match = figure_title_pattern.match(stripped_line)
        separator_match = separator_pattern.match(stripped_line)

        # Handle separators (---) which delimit figure/description blocks
        if separator_match:
            if inside_figure_block:
                # End of a figure block
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                    if cleaned_chunk_text: # Only add if there's content after cleaning
                        metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                        if figure_title: metadata["figure_title"] = figure_title
                        chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                    current_chunk_lines = [] # Reset for next chunk
                inside_figure_block = False
                figure_title = None # Reset figure title
            else:
                # Start of a figure block (or just a separator)
                # First, save the chunk *before* the separator if it exists
                if current_chunk_lines:
                     chunk_text = "\n".join(current_chunk_lines).strip()
                     cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                     if cleaned_chunk_text:
                        metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                        chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                     current_chunk_lines = []
                inside_figure_block = True # Mark that we are now inside a figure block
            continue # Move to next line after processing separator

        # Handle Headers (# Title)
        if header_match:
            # Save the previous chunk before starting a new section
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                cleaned_chunk_text = clean_text_for_embedding(chunk_text)
                if cleaned_chunk_text:
                    metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
                    # If the previous chunk was part of a figure block, add its title
                    if figure_title and inside_figure_block: metadata["figure_title"] = figure_title
                    chunks.append({"text": cleaned_chunk_text, "metadata": metadata})
                current_chunk_lines = [] # Reset for the new section

            # Update header hierarchy
            level, title = len(header_match.group(1)), header_match.group(2).strip()
            # Remove deeper headers as we encounter a shallower one
            current_headers = {k: v for k, v in current_headers.items() if k < level}
            current_headers[level] = title
            figure_title = None # Reset figure title when a new header is found
            inside_figure_block = False # Headers usually end figure blocks implicitly
            # current_chunk_lines.append(line) # Include header line in the new chunk? DECIDED AGAINST
            continue # Move to next line

        # Capture Figure Titles (### Figure X) if inside a figure block
        if figure_title_match and inside_figure_block:
             figure_title = figure_title_match.group(1).strip()
             # Don't add the title line itself to the chunk text if cleaning removes headers anyway
             # current_chunk_lines.append(line) # Decide if you want the "### Figure.." line in the chunk text
             continue # Move to next line

        # Regular content line
        current_chunk_lines.append(line)

    # Add the last remaining chunk after the loop ends
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        cleaned_chunk_text = clean_text_for_embedding(chunk_text)
        if cleaned_chunk_text:
            metadata = {"source": source_filename, **{f"h{level}": title for level, title in current_headers.items()}}
            if figure_title and inside_figure_block: metadata["figure_title"] = figure_title # Capture title if ended inside block
            chunks.append({"text": cleaned_chunk_text, "metadata": metadata})

    logger.info(f"[{job_id_for_log}] Generated {len(chunks)} hierarchical chunks for {source_filename}.")
    return chunks

# embed_chunks (No changes needed)
def embed_chunks(chunks_data, job_id_for_log: str):
    global model_st
    if not chunks_data: return []
    if not model_st: raise Exception("Embedding model not available.")
    logger.info(f"[{job_id_for_log}] Embedding {len(chunks_data)} text chunks...")
    texts_to_embed = [chunk['text'] for chunk in chunks_data]
    try:
        # Consider adding batch_size for very large documents if memory becomes an issue
        embeddings = model_st.encode(texts_to_embed, show_progress_bar=False).tolist()
        logger.info(f"[{job_id_for_log}] Embedding complete.")
        return embeddings
    except Exception as e: logger.error(f"[{job_id_for_log}] Error during embedding: {e}", exc_info=True); raise

# upsert_to_qdrant (No changes needed)
def upsert_to_qdrant(job_id_for_log: str, collection_name, embeddings, chunks_data, batch_size=100):
    global qdrant_client
    if not embeddings or not chunks_data: return 0
    if not qdrant_client: raise Exception("Qdrant client not available.")
    logger.info(f"[{job_id_for_log}] Upserting {len(embeddings)} points to Qdrant collection '{collection_name}'...")
    total_points_upserted = 0
    points_to_upsert = []

    # Prepare points with unique IDs and payloads
    for i, (embedding, chunk_data) in enumerate(zip(embeddings, chunks_data)):
        if isinstance(chunk_data.get('metadata'), dict) and 'text' in chunk_data:
            # Ensure payload values are JSON serializable (strings, numbers, bools, lists, dicts)
            payload = {k: v for k, v in chunk_data['metadata'].items() if isinstance(v, (str, int, float, bool, list, dict))}
            payload["text"] = chunk_data['text'] # Add the chunk text itself to the payload

            # Generate a deterministic ID based on job and chunk index? Or stick with UUID? UUID is safer for potential overlaps.
            point_id = str(uuid.uuid4())
            # Ensure the embedding is a list of floats
            vector = [float(x) for x in embedding]

            points_to_upsert.append(PointStruct(id=point_id, vector=vector, payload=payload))
        else:
            logger.warning(f"[{job_id_for_log}] Skipping chunk index {i} due to invalid format (missing text or non-dict metadata): {chunk_data}")

    # Upsert in batches
    num_batches = (len(points_to_upsert) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_points = points_to_upsert[batch_start:batch_end]

        if not batch_points: continue # Should not happen with correct batch logic, but safe check

        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch_points, wait=True) # wait=True ensures completion
            batch_count = len(batch_points)
            total_points_upserted += batch_count
            logger.info(f"[{job_id_for_log}] Upserted batch {i + 1}/{num_batches} ({batch_count} points) to Qdrant.")
        except Exception as e:
            logger.error(f"[{job_id_for_log}] Error upserting Qdrant batch {i + 1}/{num_batches}: {e}", exc_info=True)
            # Decide: raise immediately or try subsequent batches? Raising is safer.
            raise
    logger.info(f"[{job_id_for_log}] Finished upserting. Total points upserted: {total_points_upserted}")
    return total_points_upserted

# fill_placeholders (No changes needed)
def fill_placeholders(template_path: Path, output_path: Path, placeholders: Dict):
    try:
        if not template_path.exists(): raise FileNotFoundError(f"Template file not found: {template_path}")
        template = template_path.read_text(encoding='utf-8')
        for placeholder, value in placeholders.items():
             # Ensure value is a string before replacement
             template = template.replace(f"{{{placeholder}}}", str(value))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template, encoding='utf-8')
        logger.debug(f"Filled template '{template_path.name}' and saved to '{output_path.name}'")
    except Exception as e: logger.error(f"Error filling placeholders for {template_path} -> {output_path}: {e}", exc_info=True); raise

# get_gemini_response (Includes conversation_history parameter)
def get_gemini_response(
    system_prompt: str,
    user_prompt: str,
    conversation_history: Optional[List[Dict]] = None,
    is_json_output: bool = False,
    job_id_for_log: str = "standalone"
):
    """
    Calls the Gemini API, optionally including conversation history.

    Args:
        system_prompt: The system instruction for the LLM.
        user_prompt: The latest user message.
        conversation_history: A list of previous turns (user/model dicts).
                              Example: [{"role": "user", "parts": [...]}, {"role": "model", "parts": [...]}]
        is_json_output: Whether to request JSON output format.
        job_id_for_log: Identifier for logging.

    Returns:
        The generated text content from the model, or an error string starting with "Error:".
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    api_url = GEMINI_API_URL_TEMPLATE.format(model_name=GEMINI_MODEL_NAME, action="generateContent", api_key=GEMINI_API_KEY)
    headers = {'Content-Type': 'application/json'}

    # --- Prepare conversation content ---
    content_payload = []
    if conversation_history:
        # Truncate history if it exceeds the limit
        if len(conversation_history) > MAX_HISTORY_TURNS * 2 : # Multiply by 2 because each turn has user+model
            num_turns_to_keep = MAX_HISTORY_TURNS * 2
            history_to_send = conversation_history[-num_turns_to_keep:]
            logger.debug(f"[{job_id_for_log}] Truncating conversation history from {len(conversation_history)} to {len(history_to_send)} entries ({MAX_HISTORY_TURNS} turns).")
        else:
            history_to_send = conversation_history
        content_payload.extend(history_to_send)

    # Add the current user prompt
    content_payload.append({"role": "user", "parts": [{"text": user_prompt}]})
    # --- End Prepare conversation content ---


    # Construct payload
    payload = {
        "contents": content_payload,
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.5 if is_json_output else 0.7,
            "maxOutputTokens": 8192,
            "topP": 0.95,
            "topK": 40
        }
    }
    # Safety Settings (Optional but recommended) - Adjust levels as needed
    payload["safetySettings"] = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    if is_json_output: payload["generationConfig"]["responseMimeType"] = "application/json"

    last_error = None
    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            log_history_len = len(conversation_history) if conversation_history else 0
            logger.info(f"[{job_id_for_log}] Calling Gemini API ({GEMINI_MODEL_NAME}, Attempt {attempt+1}/{MAX_GEMINI_RETRIES}, History Entries: {log_history_len}). JSON Mode: {is_json_output}")
            # Log snippet of *current* prompts for debugging
            logger.debug(f"[{job_id_for_log}] System Prompt Snippet: {system_prompt[:200]}...")
            logger.debug(f"[{job_id_for_log}] Current User Prompt Snippet: {user_prompt[:300]}...")
            # Optionally log full payload for deep debugging (can be large)
            # logger.debug(f"[{job_id_for_log}] Full Gemini Payload: {json.dumps(payload, indent=2)}")


            response = requests.post(api_url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            logger.info(f"[{job_id_for_log}] Gemini API call successful (Attempt {attempt+1}).")

            # --- Enhanced Error Checking ---
            # 1. Check for promptFeedback block first (indicates issue with input)
            if response_data.get('promptFeedback', {}).get('blockReason'):
                 block_reason = response_data['promptFeedback']['blockReason']
                 safety_ratings = response_data['promptFeedback'].get('safetyRatings', [])
                 block_details = ", ".join([f"{sr['category']}: {sr['probability']}" for sr in safety_ratings if sr.get('probability') not in ['NEGLIGIBLE', 'LOW', 'HARM_BLOCK_THRESHOLD_UNSPECIFIED']])
                 last_error = f"Error: Prompt blocked by Gemini - {block_reason}" + (f" (Safety: {block_details})" if block_details else "")
                 logger.error(f"[{job_id_for_log}] {last_error}")
                 # This is usually non-retryable as the prompt itself is the issue. Break loop.
                 break

            # 2. Check if candidates array is missing or empty
            if not response_data.get('candidates'):
                # Check finishReason in promptFeedback if available (sometimes indicates issues like recitation)
                finish_reason_pf = response_data.get('promptFeedback', {}).get('finishReason', 'UNKNOWN')
                if finish_reason_pf != 'UNKNOWN' and finish_reason_pf != 'FINISH_REASON_UNSPECIFIED':
                     last_error = f"Error: Gemini API response missing 'candidates'. Prompt Feedback Finish Reason: {finish_reason_pf}."
                else:
                     last_error = "Error: Unexpected Gemini API response format (no candidates array)."
                logger.error(f"[{job_id_for_log}] {last_error}. Response: {response.text[:500]}")
                # Let retry logic handle it unless it's the last attempt
                if attempt + 1 >= MAX_GEMINI_RETRIES: break
                else: continue

            # 3. Process the first candidate
            candidate = response_data['candidates'][0]
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            safety_ratings = candidate.get('safetyRatings', [])

            # Check for blocking reasons within the candidate itself
            if finish_reason not in ['STOP', 'MAX_TOKENS', 'FINISH_REASON_UNSPECIFIED']:
                # Reasons like SAFETY, RECITATION, OTHER indicate a problem with the *generated* content
                block_details = ", ".join([f"{sr['category']}: {sr['probability']}" for sr in safety_ratings if sr.get('probability') not in ['NEGLIGIBLE', 'LOW', 'HARM_BLOCK_THRESHOLD_UNSPECIFIED']])
                last_error = f"Error: Gemini response generation stopped - {finish_reason}" + (f" (Safety: {block_details})" if block_details else "")
                logger.error(f"[{job_id_for_log}] {last_error}")
                # Let retry logic handle this unless it's the last attempt
                if attempt + 1 >= MAX_GEMINI_RETRIES: break
                else: continue

            # 4. Extract content if everything seems okay so far
            if candidate.get('content', {}).get('parts'):
                gemini_response_text = candidate['content']['parts'][0].get('text', '')
                if finish_reason == 'MAX_TOKENS':
                    logger.warning(f"[{job_id_for_log}] Gemini finished due to MAX_TOKENS. Output might be truncated.")
                if gemini_response_text:
                    logger.debug(f"[{job_id_for_log}] Gemini Raw Response Snippet: {gemini_response_text[:300]}...")
                    return gemini_response_text.strip() # SUCCESS! Return the response
                else:
                    # It's unusual to have STOP/MAX_TOKENS but empty text
                    last_error = f"Error: Gemini returned empty response text despite finish reason '{finish_reason}'."
                    logger.error(f"[{job_id_for_log}] {last_error}")
                    # Let retry logic handle it unless it's the last attempt
            else:
                # Missing content/parts but finish reason was OK? Unexpected.
                last_error = f"Error: Gemini candidate missing 'content' or 'parts' despite finish reason '{finish_reason}'."
                logger.error(f"[{job_id_for_log}] {last_error}")
                # Let retry logic handle it unless it's the last attempt
            # --- End Enhanced Error Checking ---


        # --- Exception Handling (Retry Logic) ---
        except requests.exceptions.Timeout:
            last_error = f"Error: Gemini API request timed out after {GEMINI_TIMEOUT}s."; logger.warning(f"[{job_id_for_log}] {last_error} Attempt {attempt + 1}/{MAX_GEMINI_RETRIES}.")
        except requests.exceptions.RequestException as e:
            response_text = ""; status_code = None;
            if e.response is not None:
                response_text = e.response.text[:500] # Log beginning of error response
                status_code = e.response.status_code
            last_error = f"Error: Gemini API request failed - {e} (Status: {status_code})"; logger.warning(f"[{job_id_for_log}] {last_error} Attempt {attempt + 1}/{MAX_GEMINI_RETRIES}. Response: {response_text}")
            # Decide if non-retryable
            if status_code is not None and not (status_code == 429 or (500 <= status_code < 600)):
                logger.error(f"[{job_id_for_log}] Non-retryable API error encountered ({status_code}). Stopping retries.", exc_info=False); break # Break loop
        except Exception as e:
            last_error = f"Error: Failed to process Gemini response - {e}"; logger.error(f"[{job_id_for_log}] {last_error}", exc_info=True); break # Break on unexpected errors

        # If we reach here, it means an error occurred and we might retry
        if attempt + 1 < MAX_GEMINI_RETRIES:
            logger.info(f"[{job_id_for_log}] Retrying Gemini call in {GEMINI_RETRY_DELAY} seconds...")
            time.sleep(GEMINI_RETRY_DELAY)
        else:
             logger.error(f"[{job_id_for_log}] Gemini API call failed after {MAX_GEMINI_RETRIES} attempts. Last error: {last_error}")
             break # Max retries reached, exit loop

    # After loop finishes (either success returned earlier or max retries hit)
    # If success didn't return, return the last error encountered
    return last_error if last_error else "Error: Gemini call failed after retries without specific error capture."


# find_topics_and_generate_hypothetical_text (No history needed here)
def find_topics_and_generate_hypothetical_text(job_id_for_log: str, academic_level, major, course_name, taxonomy_level, topics):
    logger.info(f"[{job_id_for_log}] Generating hypothetical text for RAG query...")
    temp_prompt_files = []
    try:
        # Ensure TEMP_UPLOAD_DIR exists
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = TEMP_UPLOAD_DIR / f"{job_id_for_log}_hypothetical_prompt_{uuid.uuid4().hex[:6]}.txt"
        temp_prompt_files.append(temp_path)

        placeholders = {"course_name": course_name, "academic_level": academic_level, "topics": topics, "major": major, "taxonomy_level": taxonomy_level}
        fill_placeholders(HYPOTHETICAL_PROMPT_PATH, temp_path, placeholders)
        user_prompt = temp_path.read_text(encoding="utf8")

        # System prompt tailored for generating a student query
        system_prompt = (f"You are an AI assistant simulating a student query. "
                         f"Generate a concise, plausible, hypothetical student question or statement related to a '{course_name}' course "
                         f"for a {academic_level} {major} student. "
                         f"The query should touch upon the topics: {topics}. "
                         f"It should reflect the cognitive level of Bloom's: {taxonomy_level}. "
                         f"Output ONLY the query itself, without any preamble like 'Here is a hypothetical query:'.")

        # Call get_gemini_response *without* history for this task
        hypothetical_text = get_gemini_response(
            system_prompt,
            user_prompt,
            conversation_history=None, # Explicitly no history
            is_json_output=False,
            job_id_for_log=job_id_for_log
        )

        if hypothetical_text.startswith("Error:"):
            raise Exception(f"Failed to generate hypothetical text: {hypothetical_text}")

        logger.info(f"[{job_id_for_log}] Successfully generated hypothetical text: {hypothetical_text[:200]}...")
        return hypothetical_text

    except FileNotFoundError as e:
        logger.error(f"[{job_id_for_log}] Hypothetical prompt template missing: {HYPOTHETICAL_PROMPT_PATH}")
        raise Exception(f"Hypothetical prompt template missing: {e}")
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error generating hypothetical text: {e}", exc_info=True)
        raise Exception(f"Error generating hypothetical text: {e}")
    finally:
        # Cleanup temp file
        for fpath in temp_prompt_files:
            try:
                if fpath.exists(): fpath.unlink()
            except Exception as clean_e:
                logger.warning(f"[{job_id_for_log}] Could not clean up temp hypothetical prompt file {fpath}: {clean_e}")


# search_results_from_qdrant (No changes needed)
def search_results_from_qdrant(job_id_for_log: str, collection_name, embedded_vector, limit=15, score_threshold: Optional[float] = None, session_id_filter=None, document_ids_filter=None):
    global qdrant_client
    if not qdrant_client: raise Exception("Qdrant client not available.")
    logger.info(f"[{job_id_for_log}] Searching Qdrant '{collection_name}' (limit={limit}, threshold={score_threshold}, session='{session_id_filter}', docs='{document_ids_filter}')...")

    must_conditions = []
    # Add session ID filter if provided
    if session_id_filter:
         must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id_filter)))
         logger.debug(f"[{job_id_for_log}] Added session_id filter: {session_id_filter}")

    # Add document ID filter if provided
    if document_ids_filter:
        # Ensure it's a list, even if only one ID is passed
        doc_ids = document_ids_filter if isinstance(document_ids_filter, list) else [document_ids_filter]
        if doc_ids: # Only add filter if list is not empty
             must_conditions.append(FieldCondition(key="document_id", match=MatchAny(any=doc_ids)))
             logger.debug(f"[{job_id_for_log}] Added document_id filter: {doc_ids}")

    # Construct the final filter only if there are conditions
    query_filter = Filter(must=must_conditions) if must_conditions else None
    if query_filter: logger.debug(f"[{job_id_for_log}] Using Qdrant filter: {query_filter}")
    else: logger.debug(f"[{job_id_for_log}] No Qdrant filter applied.")

    try:
        # Ensure the query vector is a list of floats
        if hasattr(embedded_vector, 'tolist'): # Handle numpy arrays
            query_vector_list = embedded_vector.tolist()
        else: # Assume it's already list-like, convert elements to float
            query_vector_list = list(map(float, embedded_vector))

        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector_list,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold, # Can be None
            with_payload=True, # We need payload for context and metadata
            with_vectors=False # Don't need vectors in the search results
        )
        logger.info(f"[{job_id_for_log}] Qdrant search returned {len(results)} results.")
        if results:
            top_score = results[0].score
            logger.info(f"[{job_id_for_log}] Top hit score: {top_score:.4f}")
            # Log if top score is below threshold (if threshold is set)
            if score_threshold is not None and top_score < score_threshold:
                 logger.warning(f"[{job_id_for_log}] Top score {top_score:.4f} is below threshold {score_threshold}.")
        return results
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error searching Qdrant collection '{collection_name}': {e}", exc_info=True)
        return [] # Return empty list on error


# parse_questions (No changes needed)
def parse_questions(question_block: str, job_id_for_log: str = "standalone") -> List[str]:
    """Parses the generated text, expecting a single question block."""
    if not question_block or not question_block.strip():
        logger.warning(f"[{job_id_for_log}] Received empty block for parsing.")
        return []

    # Assume basic cleaning (numbering, etc.) might be needed if LLM doesn't follow instructions perfectly
    # Handle potential preamble/postamble if strict output instructions failed
    lines = question_block.strip().splitlines()

    # Filter common unwanted introductory/concluding lines more aggressively
    filtered_lines = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"): # Handle code blocks (like PlantUML)
            in_code_block = not in_code_block
            filtered_lines.append(line) # Keep code block fences
            continue
        if in_code_block: # Keep lines inside code blocks as is
             filtered_lines.append(line)
             continue

        # Filter common preambles/postambles outside code blocks
        if any(stripped.lower().startswith(prefix) for prefix in [
            "here is the question:", "okay, here's the question:", "based on the context:",
            "the generated question is:", "sure, here is a question:", "certainly, the question:",
            "i hope this helps", "let me know if you need", "this question requires",
            "note:", "evaluation:", "rationale:", "response:"
        ]):
            continue
        # Filter potential numbering/bullets that might have been missed
        if re.match(r"^\s*[\d\.\-\*\)\:]+\s*", stripped):
             stripped = re.sub(r"^\s*[\d\.\-\*\)\:]+\s*", "", stripped).strip()

        if stripped: # Only add non-empty lines after filtering
            # Use the original line if filtering didn't change it significantly, otherwise use stripped
            # This preserves original formatting within the main block
            filtered_lines.append(line if line.strip() == stripped else stripped)


    cleaned_block = "\n".join(filtered_lines).strip()

    if not cleaned_block:
        logger.warning(f"[{job_id_for_log}] Parsing resulted in empty question after filtering. Original block: {question_block[:300]}...")
        return []

    logger.info(f"[{job_id_for_log}] Parsed single question block (length {len(cleaned_block)}).")
    # Return as a list containing one item, as expected by the calling loop
    return [cleaned_block]


# evaluate_single_question_qsts (No changes needed)
def evaluate_single_question_qsts(job_id_for_log: str, question: str, context: str) -> float:
    global model_st
    if not model_st: logger.error(f"[{job_id_for_log}] Sentence Transformer model not initialized for QSTS."); return 0.0
    if not question or not context: return 0.0

    # Prepare question text for embedding: remove PlantUML, leading numbers/bullets
    question_text_only = re.sub(r"```plantuml.*?```", "", question, flags=re.DOTALL | re.MULTILINE)
    question_text_only = re.sub(r"^\s*[\d\.\-\*\)\:]+\s*", "", question_text_only.strip()).strip()

    if not question_text_only:
        logger.warning(f"[{job_id_for_log}] No text found in question for QSTS eval after cleaning: '{question[:50]}...'")
        return 0.0

    try:
        # Embed question and context
        q_emb = model_st.encode(question_text_only)
        c_emb = model_st.encode(context) # Embed the full (potentially truncated) context used for generation

        # Ensure embeddings are 2D arrays for cosine similarity calculation
        if q_emb.ndim == 1: q_emb = q_emb.reshape(1, -1)
        if c_emb.ndim == 1: c_emb = c_emb.reshape(1, -1)

        # Calculate cosine similarity
        score = sbert_util.pytorch_cos_sim(q_emb, c_emb).item()

        # Clamp score between -1.0 and 1.0 (cosine similarity range) and round
        return round(max(-1.0, min(1.0, score)), 4)
    except Exception as e:
        logger.warning(f"[{job_id_for_log}] Error calculating QSTS for question '{question_text_only[:50]}...': {e}", exc_info=True)
        return 0.0 # Return 0.0 on error


# evaluate_single_question_qualitative (No changes needed - DOES NOT USE HISTORY)
def evaluate_single_question_qualitative(job_id_for_log: str, question: str, context: str) -> Dict[str, bool]:
    results = {metric: False for metric in QUALITATIVE_METRICS}
    if not question or not context: return results

    full_question_block = question # Evaluate the entire block including PlantUML if present
    temp_prompt_files = [] # Keep track of temp files
    try:
        # Truncate context for the evaluation prompt as well
        eval_context = context[:4000] + ("\n... [Context Truncated]" if len(context)>4000 else "")

        placeholders = {
            "question": full_question_block,
            "context": eval_context,
            "criteria_list_str": ", ".join(QUALITATIVE_METRICS)
        }

        # Ensure TEMP_UPLOAD_DIR exists
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = TEMP_UPLOAD_DIR / f"{job_id_for_log}_qualitative_eval_{uuid.uuid4().hex[:6]}.txt"
        temp_prompt_files.append(temp_path)

        fill_placeholders(QUALITATIVE_EVAL_PROMPT_PATH, temp_path, placeholders)
        eval_prompt = temp_path.read_text(encoding='utf-8')

        # System prompt for the evaluator LLM
        eval_system_prompt = (f"You are an AI assistant acting as a strict quality evaluator for educational questions. "
                              f"Evaluate the provided 'Question' based *only* on the given 'Context Snippets'. "
                              f"Assess the question against these criteria: {', '.join(QUALITATIVE_METRICS)}. "
                              f"Respond ONLY with a valid JSON object containing boolean values (true/false) for each criterion. "
                              f"Be critical and objective. 'Answerable' means answerable *solely* from the provided context. 'Central' means it addresses a core concept presented in the context, not a minor detail.")

        # Call Gemini WITHOUT history for evaluation
        response_text = get_gemini_response(
            eval_system_prompt,
            eval_prompt,
            conversation_history=None, # No history for eval
            is_json_output=True,
            job_id_for_log=job_id_for_log
        )


        if response_text.startswith("Error:"):
            logger.error(f"[{job_id_for_log}] LLM qualitative evaluation failed: {response_text}")
            return results # Return default False results on error

        try:
            # Attempt to clean potential markdown code block fences before parsing JSON
            cleaned_response = re.sub(r"^\s*```json\s*|\s*```\s*$", "", response_text.strip(), flags=re.MULTILINE | re.IGNORECASE)
            eval_results = json.loads(cleaned_response)

            if not isinstance(eval_results, dict):
                raise ValueError(f"LLM response was not a JSON object. Got: {type(eval_results)}")

            # Validate and populate results, default to False if key missing or not boolean
            for metric in QUALITATIVE_METRICS:
                value = eval_results.get(metric)
                if isinstance(value, bool):
                    results[metric] = value
                else:
                    logger.warning(f"[{job_id_for_log}] Qualitative metric '{metric}' missing or not boolean in LLM response. Setting to False. Response: {eval_results}")
                    results[metric] = False # Default to False if key missing or wrong type

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[{job_id_for_log}] Failed to parse/validate JSON from LLM qualitative eval: {e}. Raw Response: {response_text[:500]}...")
            return {metric: False for metric in QUALITATIVE_METRICS} # Return all False on parse failure

        return results
    except FileNotFoundError as e:
        logger.error(f"[{job_id_for_log}] Qualitative eval prompt template missing: {QUALITATIVE_EVAL_PROMPT_PATH}")
        return results # Return default False
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error during qualitative evaluation for question '{full_question_block[:100]}...': {e}", exc_info=True)
        return results # Return default False
    finally:
        # Ensure cleanup even on errors
        for fpath in temp_prompt_files:
            try:
                if fpath.exists(): fpath.unlink()
            except Exception as clean_e:
                 logger.warning(f"[{job_id_for_log}] Could not clean up temp qualitative eval file {fpath}: {clean_e}")


# cleanup_job_files (No changes needed)
def cleanup_job_files(job_id_for_log: str, temp_dirs_to_remove: List[Path], temp_files_to_remove: List[Path]):
    logger.info(f"[{job_id_for_log}] Cleaning up temporary files and directories...")

    # Delete specific temporary files first
    for file_path in temp_files_to_remove:
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                logger.info(f"[{job_id_for_log}] Deleted temp file: {file_path}")
            elif file_path.exists():
                 logger.warning(f"[{job_id_for_log}] Expected file for cleanup but found something else: {file_path}")

        except Exception as e:
            logger.warning(f"[{job_id_for_log}] Error deleting temp file {file_path}: {e}")

    # Delete temporary directories (like job-specific image dir, temp prompt dir if used)
    for dir_path in temp_dirs_to_remove:
        try:
            if dir_path.exists() and dir_path.is_dir():
                 shutil.rmtree(dir_path)
                 logger.info(f"[{job_id_for_log}] Removed temp directory: {dir_path}")
            elif dir_path.exists(): # It exists but isn't a directory? Log a warning.
                logger.warning(f"[{job_id_for_log}] Expected directory for cleanup but found file: {dir_path}")
        except Exception as e:
            logger.warning(f"[{job_id_for_log}] Error deleting temp dir {dir_path}: {e}")
    logger.info(f"[{job_id_for_log}] Temporary file cleanup finished.")


# --- Main Execution Logic (CORRECTED `job_id` variable use) ---
def run_standalone_test(args):
    """Runs the main processing pipeline for a single PDF with iterative question refinement."""
    job_id = f"standalone_{uuid.uuid4().hex[:8]}" # Define job_id here
    logger.info(f"==> [{job_id}] Starting standalone test run for PDF: {args.pdf_path} <==")
    logger.info(f"[{job_id}] Parameters: Level='{args.academic_level}', Major='{args.major}', Course='{args.course_name}', Bloom='{args.taxonomy_level}', Topics='{args.topics_list}', Diagrams='{args.generate_diagrams}', Limit='{args.retrieval_limit}', Threshold='{args.similarity_threshold}'")
    logger.info(f"[{job_id}] Using Gemini Model: {GEMINI_MODEL_NAME}")

    input_pdf_path = Path(args.pdf_path)
    if not input_pdf_path.exists() or not input_pdf_path.is_file():
        logger.critical(f"[{job_id}] FATAL ERROR: Input PDF file not found or is not a file: {input_pdf_path}")
        return # Exit if PDF not found

    # Define paths for this run's temporary data
    job_image_dir = EXTRACTED_IMAGES_DIR / job_id
    job_final_md_path = FINAL_RESULTS_DIR / f"{job_id}_{input_pdf_path.stem}_processed.md"
    # Ensure parent dirs exist for outputs
    job_final_md_path.parent.mkdir(parents=True, exist_ok=True)

    temp_files_for_cleanup = [] # Collect paths of temporary prompt files created directly
    dirs_for_cleanup = [job_image_dir] # Collect directories to remove at the end


    final_question = None
    final_eval_results = None
    processed_document_ids = [] # Keep track of docs upserted in this run
    params_dict = vars(args) # Convert argparse Namespace to dict

    try:
        # 1. Call Datalab Marker
        logger.info(f"[{job_id}] --- Step 1: Calling Datalab Marker ---")
        datalab_result = call_datalab_marker(input_pdf_path, job_id)
        markdown_text = datalab_result.get("markdown", "")
        images_dict = datalab_result.get("images", {})
        if not markdown_text:
             logger.warning(f"[{job_id}] Datalab returned empty markdown content. Proceeding without text content.")

        # 2. Save Images & Generate Descriptions
        logger.info(f"[{job_id}] --- Step 2: Saving Images & Generating Descriptions ---")
        doc_images_folder = job_image_dir / input_pdf_path.stem
        doc_images_folder.mkdir(parents=True, exist_ok=True)
        saved_images_map = save_extracted_images(images_dict, doc_images_folder, job_id)
        logger.info(f"[{job_id}] Saved {len(saved_images_map)} images references to {doc_images_folder}")


        # 3. Process Markdown (Inject Descriptions)
        logger.info(f"[{job_id}] --- Step 3: Processing Markdown (Injecting Image Descriptions) ---")
        final_markdown = process_markdown(markdown_text, saved_images_map, job_id) if markdown_text else ""
        if final_markdown:
             job_final_md_path.write_text(final_markdown, encoding="utf-8")
             logger.info(f"[{job_id}] Saved processed markdown with descriptions to: {job_final_md_path}")
        else:
             logger.warning(f"[{job_id}] No markdown content to process or save after Datalab/description step.")


        # 4. Chunk, Embed, Upsert to Qdrant
        if final_markdown and final_markdown.strip():
            logger.info(f"[{job_id}] --- Step 4: Chunking, Embedding, Upserting ---")
            document_id = f"{job_id}_{input_pdf_path.stem}"
            chunks = hierarchical_chunk_markdown(final_markdown, input_pdf_path.name, job_id)
            if chunks:
                embeddings = embed_chunks(chunks, job_id)
                for chunk_data in chunks:
                    if 'metadata' not in chunk_data: chunk_data['metadata'] = {}
                    chunk_data['metadata']['document_id'] = document_id
                    chunk_data['metadata']['session_id'] = job_id
                    chunk_data['metadata']['source_file'] = input_pdf_path.name

                upsert_to_qdrant(job_id, QDRANT_COLLECTION_NAME, embeddings, chunks)
                processed_document_ids.append(document_id)
                logger.info(f"[{job_id}] Successfully chunked, embedded, and upserted document. Document ID: {document_id}")
            else:
                logger.warning(f"[{job_id}] Markdown processed, but no usable chunks generated after cleaning. Skipping Qdrant upsert.")
        else:
            logger.warning(f"[{job_id}] Skipping chunk/embed/upsert step due to empty or missing markdown content.")

        if not processed_document_ids:
            raise ValueError("Document processing did not result in any content being added to Qdrant. Cannot proceed with question generation.")

        # 5. Generate Hypothetical Text & Search
        logger.info(f"[{job_id}] --- Step 5: Generating Hypothetical Text & Searching Qdrant ---")
        hypothetical_text = find_topics_and_generate_hypothetical_text(
            job_id, params_dict['academic_level'], params_dict['major'], params_dict['course_name'],
            params_dict['taxonomy_level'], params_dict['topics_list']
        )
        logger.info(f"[{job_id}] Generated Hypothetical Text for Search: {hypothetical_text}")
        query_embedding = model_st.encode(hypothetical_text)
        search_results = search_results_from_qdrant(
            job_id, QDRANT_COLLECTION_NAME, query_embedding,
            limit=args.retrieval_limit,
            score_threshold=args.similarity_threshold,
            document_ids_filter=processed_document_ids
        )

        if not search_results:
            logger.error(f"[{job_id}] No relevant context snippets found in Qdrant for the generated query and document ID(s) '{processed_document_ids}'. Try adjusting topics, taxonomy, or similarity threshold.")
            raise ValueError("No relevant context found in Qdrant. Cannot generate question.")

        # Compile retrieved context
        retrieved_context_parts = []
        for i, r in enumerate(search_results):
             source_info = r.payload.get('source_file', r.payload.get('source', 'N/A'))
             doc_id_info = r.payload.get('document_id', 'N/A')
             figure_title_info = f", Figure: {r.payload.get('figure_title')}" if r.payload.get('figure_title') else ""
             header_info = ", ".join([f"H{k}={v}" for k, v in r.payload.items() if k.startswith('h') and isinstance(k,str)])
             header_info_str = f", Section: {header_info}" if header_info else ""
             retrieved_context_parts.append(
                 f"--- Context Snippet {i+1} ---\n"
                 f"Source: {source_info} (Score: {r.score:.4f}, DocID: {doc_id_info}{figure_title_info}{header_info_str})\n\n"
                 f"{r.payload.get('text', 'N/A')}\n"
                 f"--- End Snippet {i+1} ---"
             )
        retrieved_context = "\n\n".join(retrieved_context_parts)

        # Display context preview
        retrieved_context_preview = "\n".join([
            f"---\n**Context Snippet {i+1}** (Source: {r.payload.get('source_file', 'N/A')}, Score: {r.score:.3f})\n{r.payload.get('text', 'N/A')[:300]}...\n---"
            for i, r in enumerate(search_results[:3])
        ])
        print("\n" + "="*20 + " RETRIEVED CONTEXT PREVIEW (Top 3) " + "="*20)
        print(retrieved_context_preview)
        print(f"(Total {len(search_results)} snippets retrieved)")
        print("="*70 + "\n")


        # 6. Iterative Question Generation & Evaluation (with History)
        logger.info(f"[{job_id}] --- Step 6: Iterative Question Generation & Evaluation (with History) ---")

        conversation_history = [] # Initialize conversation history for this session
        current_question = None
        feedback_for_llm = "" # Start with no feedback for the first attempt
        iteration = 0
        accepted_question_found = False
        # Ensure TEMP_UPLOAD_DIR exists for prompt files
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


        while iteration < MAX_REGENERATION_ATTEMPTS:
            iteration += 1
            print(f"\n--- Generating Question: Attempt {iteration}/{MAX_REGENERATION_ATTEMPTS} ---")
            try:
                # --- Prepare Prompts for this iteration ---
                # Get Bloom's guidance
                bloom_guidance_text = get_bloom_guidance(params_dict['taxonomy_level'])

                # Truncate context if needed
                max_context_chars = 30000 # Adjust based on model limits if necessary
                truncated_context = retrieved_context[:max_context_chars]
                if len(retrieved_context) > max_context_chars:
                    logger.warning(f"[{job_id}] Truncating context from {len(retrieved_context)} to {max_context_chars} chars for LLM prompt.")
                    truncated_context += "\n... [Context Truncated]"

                # Prepare Diagram instructions
                generate_diagrams_flag = params_dict.get('generate_diagrams', False)
                diagram_instructions = ""
                if generate_diagrams_flag:
                    diagram_instructions = (
                         "\n7. **PlantUML Diagram Generation (Optional but Recommended if Applicable):** For questions that benefit from a visual explanation (e.g., processes, structures, relationships), consider generating a concise PlantUML diagram using ```plantuml ... ``` tags. Ensure the diagram directly relates to and clarifies the question asked, using standard PlantUML syntax based *only* on context information. Focus on clarity and relevance."
                    )

                # Prepare Feedback section (using feedback_for_llm from previous iteration)
                optional_feedback_section = ""
                if feedback_for_llm:
                    optional_feedback_section = (f"\n\n!! IMPORTANT FEEDBACK ON PREVIOUS ATTEMPT (See History) !!\n"
                                                 f"Feedback / Issues Identified:\n{feedback_for_llm}\n\n"
                                                 f"Action Required: Generate a *new*, improved question strictly following all original instructions AND addressing this feedback.\n"
                                                 f"--- End Feedback ---")

                # Fill placeholders for the user prompt template
                placeholders = {
                    "content": truncated_context,
                    "num_questions": 1,
                    "course_name": params_dict['course_name'],
                    "taxonomy": params_dict['taxonomy_level'],
                    "major": params_dict['major'],
                    "academic_level": params_dict['academic_level'],
                    "topics_list": params_dict['topics_list'],
                    "bloom_guidance": bloom_guidance_text,
                    "diagram_instructions": diagram_instructions,
                    "optional_feedback_section": optional_feedback_section
                }

                # Create a temporary file for the filled user prompt
                temp_user_prompt_path = TEMP_UPLOAD_DIR / f"{job_id}_user_prompt_iter_{iteration}_{uuid.uuid4().hex[:6]}.txt"
                temp_files_for_cleanup.append(temp_user_prompt_path) # Ensure cleanup
                fill_placeholders(FINAL_USER_PROMPT_PATH, temp_user_prompt_path, placeholders)
                current_user_prompt_text = temp_user_prompt_path.read_text(encoding="utf8")

                # Define the system prompt for question generation
                system_prompt_base = (f"You are an expert AI assistant specializing in creating high-quality educational questions. "
                                     f"Your task is to generate exactly ONE question for a {params_dict['academic_level']} {params_dict['major']} student "
                                     f"in the course '{params_dict['course_name']}'.")
                system_prompt_constraints = (f"The question MUST be based *only* on the provided context snippets, "
                                             f"strictly adhere to the cognitive complexity of Bloom's level '{params_dict['taxonomy_level']}', "
                                             f"focus on the topics '{params_dict['topics_list']}', "
                                             f"and meet all quality criteria (clear, answerable from context, grammatical, etc.).")
                plantuml_system_hint = " Include PlantUML in ```plantuml ... ``` tags if generated." if generate_diagrams_flag else ""
                output_format_instruction = (" Respond ONLY with the single question text (and PlantUML block, if generated). "
                                            "Do NOT use numbering (e.g., '1.'). Do NOT include introductions, summaries, or apologies.")
                feedback_instruction = ""
                if optional_feedback_section:
                    feedback_instruction = " **Crucially, pay close attention to the provided feedback on the previous attempt (detailed in the prompt and potentially in history) and ensure the new question corrects those specific issues.**"

                generation_system_prompt = f"{system_prompt_base} {system_prompt_constraints}{plantuml_system_hint}{output_format_instruction}{feedback_instruction}"
                # --- End Prompt Preparation ---


                # --- Call Gemini with History ---
                logger.info(f"[{job_id}] Generating question via Gemini ({GEMINI_MODEL_NAME}, Iteration {iteration})...")
                model_response_text = get_gemini_response(
                    generation_system_prompt,
                    current_user_prompt_text,
                    conversation_history=conversation_history, # Pass the current history
                    is_json_output=False,
                    job_id_for_log=job_id # *** CORRECTED VARIABLE NAME HERE ***
                )
                # --- End Gemini Call ---


                # --- Update History ---
                conversation_history.append({"role": "user", "parts": [{"text": current_user_prompt_text}]})
                if not model_response_text.startswith("Error:"):
                    conversation_history.append({"role": "model", "parts": [{"text": model_response_text}]})
                else:
                    # Log error and prepare feedback for retry
                    logger.error(f"[{job_id}] Gemini error in iteration {iteration}: {model_response_text}. Not adding error response to history.")
                    feedback_for_llm = f"Feedback: A system error occurred during the previous attempt ({model_response_text}). Please try generating the question again, carefully following all original instructions."
                    # Reset current_question as this attempt failed
                    current_question = None
                    # Skip evaluation and proceed to the next attempt (or exit loop if max retries hit)
                    continue
                # --- End History Update ---


                # Log more of the generated text for debugging
                logger.info(f"[{job_id}] Successfully generated question block (length {len(model_response_text)}). Snippet: {model_response_text[:500]}...")


                # Check for PlantUML presence vs. request (logging only)
                plantuml_found = "```plantuml" in model_response_text
                if generate_diagrams_flag and not plantuml_found:
                    logger.warning(f"[{job_id}] PlantUML was requested, but '```plantuml' tags were not found in the generated text.")
                elif not generate_diagrams_flag and plantuml_found:
                    logger.warning(f"[{job_id}] PlantUML was *not* requested, but '```plantuml' tags *were* found in the generated text.")


                # Parse the result (expecting one question)
                parsed_list = parse_questions(model_response_text, job_id)
                if not parsed_list:
                    logger.error(f"[{job_id}] Failed to parse question from LLM output in iteration {iteration}. Raw output: {model_response_text[:500]}...")
                    print("Error: Could not parse the generated question structure. Asking LLM to retry.")
                    # Provide feedback specifically about parsing/format failure
                    feedback_for_llm = ("Feedback: Critical Error - The previous response could not be parsed. "
                                        "Ensure the output contains ONLY the question text (and optional PlantUML block) "
                                        "with no extra introductions, numbering, or explanations.")
                    current_question = None # Reset current question
                    # Remove the last model response from history as it was unparseable
                    if conversation_history and conversation_history[-1]["role"] == "model":
                        conversation_history.pop()
                    continue # Skip evaluation, try regenerating immediately

                current_question = parsed_list[0] # Get the single parsed question

                # --- Evaluate the current question ---
                print(f"\n--- Evaluating Question: Attempt {iteration} ---")
                current_eval_results = {"question_num": iteration} # Store iteration number
                # Calculate QSTS Score
                current_eval_results["qsts_score"] = evaluate_single_question_qsts(job_id, current_question, retrieved_context)
                # Perform Qualitative Evaluation (Does NOT use history)
                current_eval_results["qualitative"] = evaluate_single_question_qualitative(job_id, current_question, retrieved_context)
                # --- End Evaluation ---


                # --- Display Question and Evaluation ---
                print("\n" + "="*20 + f" GENERATED QUESTION (Attempt {iteration}) " + "="*20)
                print(current_question)
                print("\n" + "="*20 + f" EVALUATION RESULTS (Attempt {iteration}) " + "="*20)

                qsts_ok = current_eval_results['qsts_score'] >= QSTS_THRESHOLD
                critical_qual_failures_dict = {
                    metric: passed
                    for metric, passed in current_eval_results['qualitative'].items()
                    if metric in CRITICAL_QUALITATIVE_FAILURES and not passed
                }
                qual_ok = not bool(critical_qual_failures_dict)
                status = "PASS (Critically)" if qsts_ok and qual_ok else "FAIL (Critically)"

                print(f"Overall Status: {status}")
                print(f"  QSTS Score: {current_eval_results['qsts_score']:.4f} (Threshold: {QSTS_THRESHOLD}) -> {'PASS' if qsts_ok else 'FAIL'}")
                print(f"  Qualitative Checks:")
                for metric in QUALITATIVE_METRICS:
                    passed = current_eval_results['qualitative'].get(metric, False)
                    critical_fail_marker = " (CRITICAL FAIL)" if metric in CRITICAL_QUALITATIVE_FAILURES and not passed else ""
                    critical_pass_marker= " (Critical PASS)" if metric in CRITICAL_QUALITATIVE_FAILURES and passed else ""
                    print(f"    - {metric}: {passed}{critical_fail_marker}{critical_pass_marker}")
                print("="*70 + "\n")
                # --- End Display ---


                # --- Prompt user for action ---
                user_input = input(f"Accept this question? (y/n) [Or type feedback to guide regeneration, then press Enter]: ").strip()

                if user_input.lower() in ['y', 'yes']:
                    final_question = current_question
                    final_eval_results = current_eval_results
                    accepted_question_found = True
                    logger.info(f"[{job_id}] User accepted question on iteration {iteration}.")
                    break # Exit the regeneration loop

                # --- Prepare feedback for the *next* iteration's prompt ---
                # Store the question text that needs regeneration for context, although feedback_for_llm is primary
                # previous_question_for_feedback = current_question # Replaced by history

                if user_input.lower() in ['n', 'no', '']: # Treat empty input, 'n', 'no' as regenerate based on metrics
                     feedback_parts = []
                     if not qsts_ok:
                         feedback_parts.append(f"QSTS score {current_eval_results['qsts_score']:.3f} is below threshold {QSTS_THRESHOLD} (needs better semantic relevance to context).")
                     if not qual_ok:
                         failed_metrics_str = ', '.join(critical_qual_failures_dict.keys())
                         feedback_parts.append(f"Failed critical qualitative checks: [{failed_metrics_str}].")

                     if feedback_parts:
                        feedback_for_llm = ("The previous question needs improvement. Specific issues: " + " ".join(feedback_parts) +
                                            f" Please generate a new question addressing these points while strictly adhering to the '{params_dict['taxonomy_level']}' Bloom's level and all other original instructions.")
                     else:
                         feedback_for_llm = (f"User requested regeneration without specific metric failures noted. Please generate a *different* question that strictly adheres to the '{params_dict['taxonomy_level']}' Bloom's level "
                                             f"and all quality requirements (clear, answerable from context, etc.).")

                     logger.info(f"[{job_id}] User requested regeneration based on metrics for iteration {iteration+1}. Feedback prepared: {feedback_for_llm}")

                else:
                     # User provided specific feedback text
                     feedback_for_llm = (f"User Feedback: {user_input}. "
                                         f"In addition to this feedback, ensure the new question strictly adheres to the '{params_dict['taxonomy_level']}' Bloom's level "
                                         f"and all other original instructions (clarity, answerability from context, etc.).")
                     logger.info(f"[{job_id}] User provided specific feedback for regeneration for iteration {iteration+1}: '{user_input}'")
                 # --- End Feedback Preparation ---


            except Exception as gen_e:
                # Catch unexpected errors during the loop processing (not API errors handled by get_gemini_response)
                logger.error(f"[{job_id}] Error during question generation/evaluation loop (Iteration {iteration}): {gen_e}", exc_info=True)
                print(f"\nAn error occurred during generation/evaluation: {gen_e}")
                # Add the error to conversation history? Maybe not useful for LLM.
                try:
                    retry_input = input("Failed to generate/evaluate this attempt. Retry? (y/n): ").strip().lower()
                except EOFError: # Handle case where input stream is closed
                    retry_input = 'n'
                    print("Non-interactive mode detected or input stream closed. Exiting loop.")

                if retry_input != 'y':
                     print("Exiting generation loop due to error.")
                     break # Exit the while loop
                else:
                    # Provide generic feedback about the error for the retry
                    feedback_for_llm = f"Feedback: A system error occurred during the previous attempt ({str(gen_e)[:100]}...). Please try generating the question again, carefully following all original instructions for Bloom's level '{params_dict['taxonomy_level']}' and quality."
                    current_question = None # Reset current question

        # --- End of regeneration loop ---

        if not accepted_question_found:
             if current_question: # Check if the last attempt produced a question
                 logger.warning(f"[{job_id}] Max regeneration attempts ({MAX_REGENERATION_ATTEMPTS}) reached or loop exited without explicit acceptance. Using the last generated question (Attempt {iteration}).")
                 print(f"\nWARNING: Max regeneration attempts reached or loop exited. Using the last generated question (Attempt {iteration}) as the final result.")
                 final_question = current_question
                 final_eval_results = current_eval_results
             else:
                 # This means the loop finished (max attempts or error) AND the last attempt failed to produce a question
                 logger.error(f"[{job_id}] Could not generate or parse a valid question after {iteration} attempts and potential errors.")
                 print("\nERROR: Failed to generate a final question after multiple attempts and potential errors.")
                 raise ValueError("Failed to produce a final question after regeneration attempts.")


        # 7. Display Final Result
        logger.info(f"[{job_id}] --- Step 7: Final Result ---")
        print("\n" + "#"*30 + " FINAL ACCEPTED/SELECTED QUESTION & EVALUATION " + "#"*30)
        if final_question:
            print("\nFinal Generated Question:")
            print("-" * 25)
            print(final_question)
            print("-" * 25)

            if final_eval_results:
                print("\nFinal Evaluation Metrics (for the selected question):")
                print("-" * 25)
                qsts_ok = final_eval_results['qsts_score'] >= QSTS_THRESHOLD
                critical_qual_failures_dict = {
                    m: passed for m, passed in final_eval_results['qualitative'].items()
                    if m in CRITICAL_QUALITATIVE_FAILURES and not passed
                }
                qual_ok = not bool(critical_qual_failures_dict)
                status = "PASS (Critically)" if qsts_ok and qual_ok else "FAIL (Critically)"

                print(f"Overall Status: {status}")
                print(f"  QSTS Score: {final_eval_results['qsts_score']:.4f} (Threshold: {QSTS_THRESHOLD}) -> {'PASS' if qsts_ok else 'FAIL'}")
                print(f"  Qualitative Checks:")
                for metric in QUALITATIVE_METRICS:
                    passed = final_eval_results['qualitative'].get(metric, False)
                    critical_fail_marker = " (CRITICAL FAIL)" if metric in CRITICAL_QUALITATIVE_FAILURES and not passed else ""
                    critical_pass_marker= " (Critical PASS)" if metric in CRITICAL_QUALITATIVE_FAILURES and passed else ""
                    print(f"    - {metric}: {passed}{critical_fail_marker}{critical_pass_marker}")
                print("-" * 25)
            else:
                 print("\nFinal Evaluation Metrics: Not Available (Evaluation might have failed on the final attempt).")
        else:
             print("\nNo final question was generated or selected.")

        # Optionally print the full conversation history for debugging
        # print("\n" + "#"*30 + " FULL CONVERSATION HISTORY " + "#"*30)
        # print(json.dumps(conversation_history, indent=2))
        # print("#"*80 + "\n")


        print("#"*80 + "\n")
        logger.info(f"[{job_id}] Standalone test run processing completed.")

    except ValueError as ve:
        logger.critical(f"[{job_id}] Standalone test run failed due to a configuration or data issue: {ve}", exc_info=False) # Less verbose for expected errors
        print(f"\nCRITICAL ERROR: {ve}\n")
    except Exception as e:
        logger.exception(f"[{job_id}] An unexpected error occurred during the standalone test run: {e}")
        print(f"\nUNEXPECTED ERROR during processing: {e}\n")

    finally:
        # --- Cleanup ---
        # Add any temp files created directly in the main loop/sub-functions if not already added
        temp_files_for_cleanup.extend(list(TEMP_UPLOAD_DIR.glob(f"{job_id}_*.txt")))
        # Remove duplicates just in case
        temp_files_for_cleanup = list(set(temp_files_for_cleanup))

        logger.info(f"[{job_id}] --- Step 8: Cleaning Up Temporary Files ---")
        cleanup_job_files(job_id, dirs_for_cleanup, temp_files_for_cleanup)
        logger.info(f"==> [{job_id}] Standalone test run finished. <==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bloom's taxonomy based question generation from PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument("--course_name", default="Data Structures and Algorithms", help="Name of the course.")
    parser.add_argument("--academic_level", default="Undergraduate", help="Target academic level (e.g., Undergraduate, Graduate).")
    parser.add_argument("--taxonomy_level", default="Evaluate", choices=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"], help="Target Bloom's Taxonomy level.")
    parser.add_argument("--topics_list", default="Breadth First Search , Shortest path", help="Comma-separated list of relevant topics.")
    parser.add_argument("--major", default="Computer Science", help="Target major (e.g., Computer Science, Biology).")
    parser.add_argument("--retrieval_limit", type=int, default=15, help="Max context snippets to retrieve from Qdrant.")
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="Minimum similarity score for retrieved context (e.g., 0.0 to 1.0).")
    parser.add_argument("--generate_diagrams", action='store_true', default=False, help="Flag to enable PlantUML diagram generation if applicable.")

    args = parser.parse_args()

    if not Path(args.pdf_path).is_file():
        print(f"Error: PDF file not found at '{args.pdf_path}'")
        sys.exit(1)
    if args.retrieval_limit <= 0:
         print(f"Error: Retrieval limit (--retrieval_limit) must be positive.")
         sys.exit(1)
    if not (0.0 <= args.similarity_threshold <= 1.0):
        print(f"Error: Similarity threshold (--similarity_threshold) must be between 0.0 and 1.0.")
        sys.exit(1)

    run_standalone_test(args)