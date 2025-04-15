document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const form = document.getElementById('upload-form');
    const submitButton = document.getElementById('submit-button');
    const statusArea = document.getElementById('status-area');
    const statusMessage = document.getElementById('status-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorArea = document.getElementById('error-area');
    const errorMessage = document.getElementById('error-message');

    // Feedback Stage Elements
    const feedbackStageArea = document.getElementById('feedback-stage-area');
    const markdownContentEl = document.getElementById('markdown-content'); // In feedback stage
    const initialQuestionsContentEl = document.getElementById('initial-questions-content');
    const feedbackInput = document.getElementById('feedback-input');
    const regenerateButton = document.getElementById('regenerate-button');

    // Final Results Elements
    const resultsArea = document.getElementById('results-area');
    const resultsTitle = document.getElementById('results-title');
    const finalMarkdownContentEl = document.getElementById('final-markdown-content'); // <<< Added
    const finalQuestionsContentEl = document.getElementById('final-questions-content');
    const finalEvaluationFeedbackEl = document.getElementById('final-evaluation-feedback');
    const finalPerQuestionEvaluationEl = document.getElementById('final-per-question-evaluation');
    const finalContextPreviewEl = document.getElementById('final-context-preview');
    const imageSlideshowInnerEl = document.getElementById('image-slideshow-inner'); // <<< Added
    const noImagesMessageEl = document.getElementById('no-images-message'); // <<< Added


    // --- State Variables ---
    let currentJobId = null;
    let pollInterval = null;
    const POLLING_INTERVAL_MS = 5000; // 5 seconds

    // --- Event Listeners ---
    form.addEventListener('submit', handleFormSubmit);
    regenerateButton.addEventListener('click', handleRegenerationSubmit);

    // --- Functions ---

    async function handleFormSubmit(event) {
        event.preventDefault();
        console.log("[handleFormSubmit] Form submitted");
        resetUI();
        showStatus('Initiating request...');
        disableButton(submitButton, 'Processing...');
        const formData = new FormData(form);
        if (!validateForm(formData)) { resetSubmitButton(); hideStatus(); return; }
        try {
            const response = await fetch('/start-processing', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) { throw new Error(data.detail || `HTTP error! status: ${response.status}`); }
            currentJobId = data.job_id;
            console.log("[handleFormSubmit] Job started with ID:", currentJobId);
            if (currentJobId) { showStatus('Job started. Processing documents...'); startPolling(); }
            else { throw new Error('Failed to get Job ID from server.'); }
        } catch (error) {
            console.error('[handleFormSubmit] Error submitting form:', error);
            showError(`Submission failed: ${error.message}`); resetSubmitButton(); hideStatus();
        }
    }

    async function handleRegenerationSubmit() {
        console.log("[handleRegenerationSubmit] Regenerate button clicked for job:", currentJobId);
        if (!currentJobId) { showError("No active job found for regeneration."); return; }
        const feedback = feedbackInput.value.trim();
        feedbackInput.classList.remove('is-invalid');
        showStatus('Submitting feedback and finalizing/regenerating questions...');
        disableButton(regenerateButton, 'Processing...');
        feedbackStageArea.style.display = 'none';
        try {
            const response = await fetch(`/regenerate-questions/${currentJobId}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ feedback: feedback }),
            });
            const data = await response.json();
            if (!response.ok) { throw new Error(data.detail || `Regeneration request failed: ${response.status}`); }
             console.log("[handleRegenerationSubmit] Regeneration request response status:", data.status);
             if (data.status === 'processing_feedback' || data.status === 'queued') {
                 currentJobId = data.job_id; showStatus(data.message || 'Evaluation/Regeneration in progress...'); startPolling();
             } else {
                  console.warn("[handleRegenerationSubmit] Unexpected status from /regenerate endpoint:", data.status);
                   if (data.status === 'completed') { displayFinalResults(data.result); hideStatus(); }
                   else if (data.status === 'error') { showError(data.message || 'Regeneration failed on server.'); hideStatus(); }
                   else { showStatus(`Unexpected status: ${data.status}. Polling anyway...`); startPolling(); }
             }
        } catch (error) {
            console.error('[handleRegenerationSubmit] Error during regeneration:', error);
            showError(`Regeneration failed: ${error.message}`);
            enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
        }
    }

    function startPolling() {
        stopPolling(); if (!currentJobId) { console.error("[startPolling] No job ID."); return; }
        console.log(`[startPolling] Polling started for job ${currentJobId}`);
        checkJobStatus(); pollInterval = setInterval(checkJobStatus, POLLING_INTERVAL_MS);
    }

    function stopPolling() {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; console.log(`[stopPolling] Polling stopped for job ${currentJobId}`); }
    }

    async function checkJobStatus() {
        console.log(`[checkJobStatus] Checking status for job ${currentJobId}...`);
        if (!currentJobId) { console.warn("[checkJobStatus] No currentJobId."); stopPolling(); return; };
        try {
            console.log(`[checkJobStatus] Fetching /status/${currentJobId}`);
            const response = await fetch(`/status/${currentJobId}`);
            console.log(`[checkJobStatus] Fetch response status: ${response.status}`);
             if (!response.ok) {
                 let errorMsg = `Error checking status ${response.status}.`; if (response.status === 404) errorMsg = `Job ID ${currentJobId} not found.`;
                 console.error(`[checkJobStatus] Fetch failed: ${errorMsg}`); showError(errorMsg); stopPolling(); resetSubmitButton(); enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); return;
             }
            console.log(`[checkJobStatus] Parsing JSON...`);
            const data = await response.json();
            console.log("[checkJobStatus] Received status data:", data);
            updateStatus(data.status, data.message || '...');

            switch (data.status) {
                case 'awaiting_feedback':
                    console.log("[checkJobStatus] Status is awaiting_feedback."); stopPolling(); displayFeedbackStage(data.result); hideStatus(); resetSubmitButton(); enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); break;
                case 'completed':
                    console.log("[checkJobStatus] Status is completed."); stopPolling(); displayFinalResults(data.result); hideStatus(); resetSubmitButton(); enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); feedbackStageArea.style.display = 'none'; break;
                case 'error':
                    console.error("[checkJobStatus] Job status is error:", data.message); stopPolling(); showError(data.message || 'An unknown error occurred.'); hideStatus(); resetSubmitButton(); enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); feedbackStageArea.style.display = 'none'; break;
                case 'pending': case 'queued': case 'processing': case 'processing_feedback':
                    console.log(`[checkJobStatus] Status is ${data.status}, continuing poll.`); break;
                default: console.warn(`[checkJobStatus] Unknown job status: ${data.status}.`);
            }
        } catch (error) {
            console.error('[checkJobStatus] Error during fetch/processing status:', error); showError(`Error checking job status: ${error.message}.`); stopPolling(); resetSubmitButton(); enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
        }
    }

    function displayFeedbackStage(result) {
        hideError();
        if (!result) { showError("Result data missing for feedback stage."); console.error("displayFeedbackStage no result"); feedbackStageArea.style.display = 'none'; return; }
        console.log("[displayFeedbackStage] Displaying feedback stage");

        if (markdownContentEl) { markdownContentEl.textContent = result.extracted_markdown || 'Markdown not available.'; } else { console.error("Markdown element missing"); }
        if (initialQuestionsContentEl) { initialQuestionsContentEl.textContent = result.initial_questions || 'Initial questions not available.'; } else { console.error("Initial questions element missing"); }

        feedbackInput.value = ''; feedbackInput.classList.remove('is-invalid');
        hideResults(); feedbackStageArea.style.display = 'block'; enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
        console.log("[displayFeedbackStage] Feedback stage visible.");
    }

    function displayFinalResults(result) {
        hideError();
        if (!result) { showError("Final result data missing."); console.error("displayFinalResults no result"); resultsArea.style.display = 'none'; return; }
        console.log("[displayFinalResults] Displaying final results");

        resultsArea.style.display = 'block'; resultsTitle.textContent = "Final Generated Questions & Evaluation"; feedbackStageArea.style.display = 'none';

        // --- Populate Final Results Elements ---
        // Final Markdown
        if(finalMarkdownContentEl) { finalMarkdownContentEl.textContent = result.extracted_markdown || 'Markdown not available.'; }
        else { console.error("[displayFinalResults] Final markdown element not found"); }

        // Final Questions
        if(finalQuestionsContentEl) { finalQuestionsContentEl.textContent = result.generated_questions || 'No final questions generated.'; }
        else { console.error("[displayFinalResults] Final questions element not found"); }

        // Eval Summary
        if (finalEvaluationFeedbackEl) { finalEvaluationFeedbackEl.textContent = result.evaluation_feedback || 'No evaluation feedback available.'; }
        else { console.error("[displayFinalResults] Final evaluation feedback element not found"); }

        // Per-Question Details
        if (finalPerQuestionEvaluationEl) {
             finalPerQuestionEvaluationEl.innerHTML = ''; // Clear previous
             if (result.per_question_evaluation && Array.isArray(result.per_question_evaluation) && result.per_question_evaluation.length > 0) {
                 const list = document.createElement('ul'); list.className = 'evaluation-list';
                 result.per_question_evaluation.forEach(q_eval => {
                     const item = document.createElement('li');
                     let metricsHtml = `<span class="metric-item">QSTS: ${q_eval.qsts_score?.toFixed(3) ?? 'N/A'}</span>`;
                     if (q_eval.qualitative && typeof q_eval.qualitative === 'object') {
                         for (const [metric, passed] of Object.entries(q_eval.qualitative)) {
                              const statusClass = passed ? 'pass' : 'fail';
                              metricsHtml += `<span class="metric-item">${metric}: <span class="${statusClass}">${passed ? 'Pass' : 'FAIL'}</span></span>`;
                         }
                     } else { metricsHtml += `<span class="metric-item">Qualitative: N/A</span>`; }
                     // Ensure question text exists
                     const qText = q_eval.question_text || '[Question text missing]';
                     item.innerHTML = `
                         <span class="question-text">Q${q_eval.question_num}: ${qText}</span>
                         <div style="width: 100%; margin-top: 5px;">${metricsHtml}</div>
                     `;
                     list.appendChild(item);
                 });
                 finalPerQuestionEvaluationEl.appendChild(list);
             } else { finalPerQuestionEvaluationEl.textContent = 'No per-question evaluation details available.'; }
        } else { console.error("[displayFinalResults] Final per-question evaluation element not found"); }

        // Image Slideshow
        if (imageSlideshowInnerEl) {
            imageSlideshowInnerEl.innerHTML = ''; // Clear previous items or 'no images' message
            const imagePaths = result.image_paths; // Get paths from result
            if (imagePaths && Array.isArray(imagePaths) && imagePaths.length > 0) {
                imagePaths.forEach((imgPath, index) => {
                    const div = document.createElement('div');
                    div.className = `carousel-item${index === 0 ? ' active' : ''}`; // Mark first as active
                    const img = document.createElement('img');
                    // IMPORTANT: Construct the URL based on how static files are served
                    // The backend stores URLs like "/static/images/JOB_ID/REL_PATH"
                    img.src = imgPath; // Use the URL directly from backend
                    img.className = 'd-block w-100'; // Bootstrap class
                    img.alt = `Extracted Image ${index + 1}`;
                    img.style.objectFit = 'contain'; // Ensure image fits without distortion
                    img.style.maxHeight = '50vh';   // Limit height
                    img.onerror = () => { // Handle cases where the image might not load
                        img.alt = `Error loading image: ${imgPath}`;
                        // Optionally display placeholder or error message
                        div.innerHTML = `<div class="d-flex justify-content-center align-items-center" style="height: 200px; color: red;">Error loading image: ${imgPath.split('/').pop()}</div>`;
                    };
                    div.appendChild(img);
                    imageSlideshowInnerEl.appendChild(div);
                });
                 // Ensure carousel controls are visible only if there are multiple images
                 const carouselElement = document.getElementById('image-slideshow');
                 const controls = carouselElement.querySelectorAll('.carousel-control-prev, .carousel-control-next');
                 controls.forEach(control => {
                     control.style.display = imagePaths.length > 1 ? 'flex' : 'none';
                 });

            } else {
                // Display the 'no images' message if the element exists
                if (noImagesMessageEl) {
                     imageSlideshowInnerEl.innerHTML = '<div class="carousel-item active"><span id="no-images-message">No images were extracted from the document(s).</span></div>';
                } else { // Fallback if specific message element isn't found
                     imageSlideshowInnerEl.innerHTML = '<div class="carousel-item active"><span>No images extracted.</span></div>';
                }
                 // Hide controls if no images
                 const carouselElement = document.getElementById('image-slideshow');
                 const controls = carouselElement.querySelectorAll('.carousel-control-prev, .carousel-control-next');
                 controls.forEach(control => { control.style.display = 'none'; });
            }
        } else { console.error("[displayFinalResults] Image slideshow inner element not found"); }


        // Context Preview
        if(finalContextPreviewEl) { finalContextPreviewEl.textContent = result.retrieved_context_preview || 'No context preview available.'; }
        else { console.error("[displayFinalResults] Final context preview element not found"); }

        console.log("[displayFinalResults] Final results display updated.");
   }


    // --- UI Helper Functions ---
    function resetUI() {
        console.log("[resetUI] Resetting UI state");
        hideError(); hideStatus(); hideResults();
        feedbackStageArea.style.display = 'none';
        resetSubmitButton(); enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
        feedbackInput.value = '';
        // Clear content
        if (markdownContentEl) markdownContentEl.textContent = '';
        if (initialQuestionsContentEl) initialQuestionsContentEl.textContent = '';
        if (finalMarkdownContentEl) finalMarkdownContentEl.textContent = ''; // <<< Added
        if (finalQuestionsContentEl) finalQuestionsContentEl.textContent = '';
        if (finalEvaluationFeedbackEl) finalEvaluationFeedbackEl.textContent = '';
        if (finalPerQuestionEvaluationEl) finalPerQuestionEvaluationEl.innerHTML = '';
        if (finalContextPreviewEl) finalContextPreviewEl.textContent = '';
        if (imageSlideshowInnerEl) imageSlideshowInnerEl.innerHTML = '<div class="carousel-item active"><span id="no-images-message">No images extracted or available.</span></div>'; // <<< Reset Slideshow

        form.querySelectorAll('.is-invalid').forEach(el => el.classList.remove('is-invalid'));
    }

     function validateForm(formData) { /* ... keep validation logic ... */
        console.log("[validateForm] Validating form data");
        let isValid = true;
        const files = formData.getAll('files');
        if (!files || files.length === 0 || files.some(f => typeof f !== 'object' || !f.name) ) {
            showError('Please select at least one valid PDF file.');
            isValid = false; document.getElementById('files')?.classList.add('is-invalid');
        } else { document.getElementById('files')?.classList.remove('is-invalid'); }
        const requiredFields = ['course_name', 'major', 'academic_level', 'taxonomy_level', 'num_questions', 'topics_list'];
        for (const fieldName of requiredFields) {
            const field = document.getElementById(fieldName);
            if (!formData.get(fieldName)) {
                showError(`Please fill in the '${field?.previousElementSibling?.textContent || fieldName}' field.`);
                field?.classList.add('is-invalid'); isValid = false;
            } else { field?.classList.remove('is-invalid'); }
        }
        console.log("[validateForm] Validation result:", isValid);
        return isValid;
    }
    function disableButton(button, text) { if (!button) return; button.disabled = true; button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${text}`; }
    function enableButton(button, text) { if (!button) return; button.disabled = false; button.innerHTML = text; }
    function resetSubmitButton() { enableButton(submitButton, 'Generate Questions'); }
    function showStatus(message) { statusMessage.textContent = message; statusArea.style.display = 'block'; loadingSpinner.style.display = 'inline-block'; }
    function hideStatus() { statusArea.style.display = 'none'; }
    function showError(message) { errorMessage.textContent = message; errorArea.style.display = 'block'; hideStatus(); }
    function hideError() { errorArea.style.display = 'none'; errorMessage.textContent = ''; }
    function hideResults() {
        resultsArea.style.display = 'none';
        // Clear results content when hiding
        if (finalMarkdownContentEl) finalMarkdownContentEl.textContent = ''; // <<< Added
        if (finalQuestionsContentEl) finalQuestionsContentEl.textContent = '';
        if (finalEvaluationFeedbackEl) finalEvaluationFeedbackEl.textContent = '';
        if (finalPerQuestionEvaluationEl) finalPerQuestionEvaluationEl.innerHTML = '';
        if (finalContextPreviewEl) finalContextPreviewEl.textContent = '';
        if (imageSlideshowInnerEl) imageSlideshowInnerEl.innerHTML = '<div class="carousel-item active"><span id="no-images-message">No images extracted or available.</span></div>'; // <<< Reset Slideshow
    }
    function updateStatus(status, message) {
        const statusText = status ? `${status.toUpperCase()}: ${message}` : message;
        statusMessage.textContent = statusText;
        if (['completed', 'error', 'awaiting_feedback'].includes(status)) { loadingSpinner.style.display = 'none'; }
        else { loadingSpinner.style.display = 'inline-block'; }
    }

}); // End DOMContentLoaded