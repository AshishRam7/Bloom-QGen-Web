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
    const markdownContentEl = document.getElementById('markdown-content'); // Renamed variable
    const initialQuestionsContentEl = document.getElementById('initial-questions-content'); // Renamed variable
    const feedbackInput = document.getElementById('feedback-input');
    const regenerateButton = document.getElementById('regenerate-button');

    // Final Results Elements
    const resultsArea = document.getElementById('results-area');
    const resultsTitle = document.getElementById('results-title');
    // Specific elements within final results for clarity
    const finalQuestionsContentEl = document.getElementById('final-questions-content');
    const finalEvaluationFeedbackEl = document.getElementById('final-evaluation-feedback');
    const finalPerQuestionEvaluationEl = document.getElementById('final-per-question-evaluation');
    const finalContextPreviewEl = document.getElementById('final-context-preview');


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
        console.log("Form submitted");
        resetUI();

        showStatus('Initiating request...');
        disableButton(submitButton, 'Processing...');

        const formData = new FormData(form);
        if (!validateForm(formData)) {
            resetSubmitButton();
            hideStatus();
            return;
        }

        try {
            const response = await fetch('/start-processing', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `HTTP error! status: ${response.status}`);
            }

            currentJobId = data.job_id;
            console.log("Job started with ID:", currentJobId);
            if (currentJobId) {
                showStatus('Job started. Processing documents... (This may take several minutes)');
                startPolling();
            } else {
                throw new Error('Failed to get Job ID from server.');
            }

        } catch (error) {
            console.error('Error submitting form:', error);
            showError(`Submission failed: ${error.message}`);
            resetSubmitButton();
            hideStatus();
        }
    }

    async function handleRegenerationSubmit() {
        console.log("Regenerate button clicked for job:", currentJobId);
        if (!currentJobId) {
            showError("No active job found for regeneration.");
            return;
        }

        const feedback = feedbackInput.value.trim();
        // Allow regeneration even without feedback (to finalize/run evaluation)
        // if (!feedback) {
        //     showError("Please provide feedback before regenerating.");
        //     feedbackInput.classList.add('is-invalid');
        //     return;
        // }
        feedbackInput.classList.remove('is-invalid');

        showStatus('Submitting feedback and finalizing/regenerating questions...');
        disableButton(regenerateButton, 'Processing...');
        feedbackStageArea.style.display = 'none'; // Hide feedback section during processing

        try {
            const response = await fetch(`/regenerate-questions/${currentJobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ feedback: feedback }), // Send feedback (can be empty)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `Regeneration request failed: ${response.status}`);
            }

             console.log("Regeneration request response status:", data.status);
             // Status should now be processing_feedback or queued for the background task
             if (data.status === 'processing_feedback' || data.status === 'queued') {
                 currentJobId = data.job_id; // Re-confirm job ID
                 showStatus(data.message || 'Evaluation/Regeneration in progress...');
                 startPolling(); // Restart polling for the final result
             } else {
                  // Handle unexpected immediate completion or error from endpoint directly
                  console.warn("Unexpected status from /regenerate endpoint:", data.status);
                   if (data.status === 'completed') {
                       displayFinalResults(data.result);
                       hideStatus();
                   } else if (data.status === 'error') {
                       showError(data.message || 'Regeneration failed on server.');
                       hideStatus();
                   } else {
                        showStatus(`Unexpected status: ${data.status}. Polling anyway...`);
                        startPolling(); // Fallback to polling
                   }
             }

        } catch (error) {
            console.error('Error during regeneration:', error);
            showError(`Regeneration failed: ${error.message}`);
            enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); // Re-enable button on error
            // Optionally re-show feedback area? Maybe not, let user retry.
            // feedbackStageArea.style.display = 'block';
        }
    }


    function startPolling() {
        stopPolling(); // Clear any existing interval
        if (!currentJobId) {
            console.error("Cannot start polling without a job ID.");
            return;
        }
        console.log(`Polling started for job ${currentJobId} (interval: ${POLLING_INTERVAL_MS}ms)`);
        // Run immediately first time
        checkJobStatus();
        // Then set interval
        pollInterval = setInterval(checkJobStatus, POLLING_INTERVAL_MS);
    }

    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
             console.log(`Polling stopped for job ${currentJobId}`);
        }
    }

    async function checkJobStatus() {
        if (!currentJobId) {
             console.warn("checkJobStatus called without currentJobId, stopping poll.");
             stopPolling();
             return;
        };

        // console.log(`Checking status for job ${currentJobId}`); // Reduce log verbosity
        try {
            const response = await fetch(`/status/${currentJobId}`);

             if (!response.ok) {
                 let errorMsg = `Error checking status: ${response.status} ${response.statusText}. Stopping polling.`;
                 if (response.status === 404) { errorMsg = `Job ID ${currentJobId} not found on server. Stopping polling.`; }
                 showError(errorMsg);
                 console.error(errorMsg);
                 stopPolling();
                 resetSubmitButton(); // Also enable main submit button
                 enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
                 return;
             }

            const data = await response.json();
            // console.log("Received status data:", data); // Log received data for debugging

            updateStatus(data.status, data.message || '...'); // Update status bar message

            // --- Handle different statuses ---
            switch (data.status) {
                case 'awaiting_feedback':
                    console.log("Status is awaiting_feedback, updating UI.");
                    stopPolling(); // Stop polling once we reach this intermediate state
                    displayFeedbackStage(data.result);
                    hideStatus();
                    resetSubmitButton(); // Allow new submission
                    enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); // Ensure regenerate button is active
                    break;
                case 'completed':
                    console.log("Status is completed, updating UI.");
                    stopPolling();
                    displayFinalResults(data.result);
                    hideStatus();
                    resetSubmitButton();
                    enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); // Re-enable just in case
                    feedbackStageArea.style.display = 'none'; // Ensure feedback stage is hidden
                    break;
                case 'error':
                    console.error("Job status is error:", data.message);
                    stopPolling();
                    showError(data.message || 'An unknown error occurred during processing.');
                    hideStatus();
                    resetSubmitButton();
                    enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
                    feedbackStageArea.style.display = 'none'; // Hide feedback stage on error
                    break;
                case 'pending':
                case 'queued':
                case 'processing':
                case 'processing_feedback':
                    // Continue polling, status message already updated
                    // console.log(`Status is ${data.status}, continuing poll.`); // Reduce verbosity
                    break;
                default:
                     console.warn(`Unknown job status received: ${data.status}. Continuing poll.`);
                     // Continue polling, maybe log message
            }

        } catch (error) {
            console.error('Error parsing status response or during polling:', error);
            showError(`Error checking job status: ${error.message}. Stopping polling.`);
            stopPolling();
            resetSubmitButton();
            enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
        }
    }

     function displayFeedbackStage(result) {
        hideError(); // Clear previous errors
        if (!result) {
            showError("No result data received for feedback stage. Cannot display.");
            console.error("displayFeedbackStage called with null or undefined result");
            feedbackStageArea.style.display = 'none';
            return;
        }
        console.log("Displaying feedback stage for job:", currentJobId);

        // Check if elements exist before setting content
        if (markdownContentEl) {
            markdownContentEl.textContent = result.extracted_markdown || 'No markdown content available.';
        } else { console.error("Markdown content element not found"); }

        if (initialQuestionsContentEl) {
             initialQuestionsContentEl.textContent = result.initial_questions || 'No initial questions available.';
        } else { console.error("Initial questions content element not found"); }


        feedbackInput.value = ''; // Clear previous feedback
        feedbackInput.classList.remove('is-invalid'); // Clear error state

        hideResults(); // Hide final results area
        feedbackStageArea.style.display = 'block'; // Show the feedback section
        enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate'); // Ensure button is enabled
    }

    function displayFinalResults(result) {
        hideError(); // Clear previous errors
        if (!result) {
           showError("No final result data received.");
           console.error("displayFinalResults called with null or undefined result");
           resultsArea.style.display = 'none';
           return;
       }
        console.log("Displaying final results for job:", currentJobId);

       resultsArea.style.display = 'block'; // Show final results area
       resultsTitle.textContent = "Final Generated Questions & Evaluation"; // Set title
       feedbackStageArea.style.display = 'none'; // Hide feedback area

        // --- Populate Final Results Elements ---
        if(finalQuestionsContentEl) {
             finalQuestionsContentEl.textContent = result.generated_questions || 'No final questions generated.';
        } else { console.error("Final questions element not found"); }

        if (finalEvaluationFeedbackEl) {
            finalEvaluationFeedbackEl.textContent = result.evaluation_feedback || 'No evaluation feedback available.';
        } else { console.error("Final evaluation feedback element not found"); }

        // Display Per-Question Evaluation
        if (finalPerQuestionEvaluationEl) {
             finalPerQuestionEvaluationEl.innerHTML = ''; // Clear previous
             if (result.per_question_evaluation && Array.isArray(result.per_question_evaluation)) {
                 const list = document.createElement('ul');
                 list.className = 'evaluation-list';
                 result.per_question_evaluation.forEach(q_eval => {
                     const item = document.createElement('li');
                     let metricsHtml = `<span class="metric-item">QSTS: ${q_eval.qsts_score?.toFixed(3) ?? 'N/A'}</span>`;
                     if (q_eval.qualitative && typeof q_eval.qualitative === 'object') {
                         for (const [metric, passed] of Object.entries(q_eval.qualitative)) {
                              const statusClass = passed ? 'pass' : 'fail';
                              metricsHtml += `<span class="metric-item">${metric}: <span class="${statusClass}">${passed ? 'Pass' : 'FAIL'}</span></span>`;
                         }
                     }
                     item.innerHTML = `
                         <span class="question-text">Q${q_eval.question_num}: ${q_eval.question_text}</span>
                         <div style="width: 100%; margin-top: 5px;">${metricsHtml}</div>
                     `;
                     list.appendChild(item);
                 });
                 finalPerQuestionEvaluationEl.appendChild(list);
             } else {
                 finalPerQuestionEvaluationEl.textContent = 'No per-question evaluation details available.';
             }
        } else { console.error("Final per-question evaluation element not found"); }

        // Display Context Preview
        if(finalContextPreviewEl) {
            finalContextPreviewEl.textContent = result.retrieved_context_preview || 'No context preview available.';
        } else { console.error("Final context preview element not found"); }
   }


    // --- UI Helper Functions ---
    function resetUI() {
        console.log("Resetting UI");
        hideError();
        hideStatus();
        hideResults();
        feedbackStageArea.style.display = 'none'; // Ensure feedback stage is hidden
        resetSubmitButton();
        enableButton(regenerateButton, 'Evaluate & Finalize / Regenerate');
        feedbackInput.value = '';
        // Clear content of display areas
        if (markdownContentEl) markdownContentEl.textContent = '';
        if (initialQuestionsContentEl) initialQuestionsContentEl.textContent = '';
        if (finalQuestionsContentEl) finalQuestionsContentEl.textContent = '';
        if (finalEvaluationFeedbackEl) finalEvaluationFeedbackEl.textContent = '';
        if (finalPerQuestionEvaluationEl) finalPerQuestionEvaluationEl.innerHTML = '';
        if (finalContextPreviewEl) finalContextPreviewEl.textContent = '';

        // Clear validation errors on form inputs
         form.querySelectorAll('.is-invalid').forEach(el => el.classList.remove('is-invalid'));
         // Optionally reset form fields completely:
         // form.reset();
    }

     function validateForm(formData) {
         let isValid = true;
         const files = formData.getAll('files');
         // Check if files array exists and contains actual file objects (name check isn't sufficient)
         if (!files || files.length === 0 || files.some(f => typeof f !== 'object' || !f.name)) {
             showError('Please select at least one valid PDF file.');
             isValid = false;
             document.getElementById('files')?.classList.add('is-invalid');
         } else {
             document.getElementById('files')?.classList.remove('is-invalid');
         }

         const requiredFields = ['course_name', 'major', 'academic_level', 'taxonomy_level', 'num_questions', 'topics_list'];
         for (const fieldName of requiredFields) {
             const field = document.getElementById(fieldName);
             if (!formData.get(fieldName)) {
                 showError(`Please fill in the '${field?.previousElementSibling?.textContent || fieldName}' field.`);
                 field?.classList.add('is-invalid');
                 isValid = false;
             } else {
                  field?.classList.remove('is-invalid');
             }
         }
         // Add validation for number range? Handled by HTML5 min/max mostly.

         return isValid;
     }

    function disableButton(button, text) {
        if (!button) return;
        button.disabled = true;
        button.innerHTML = `
            <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
            ${text}
        `;
    }

     function enableButton(button, text) {
        if (!button) return;
        button.disabled = false;
        button.innerHTML = text;
    }


    function resetSubmitButton() {
        enableButton(submitButton, 'Generate Questions');
    }

    function showStatus(message) {
        statusMessage.textContent = message;
        statusArea.style.display = 'block';
        loadingSpinner.style.display = 'inline-block';
    }

    function updateStatus(status, message) {
        const statusText = status ? `${status.toUpperCase()}: ${message}` : message;
        statusMessage.textContent = statusText;
        // Hide spinner only on final states or intermediate feedback state
        if (['completed', 'error', 'awaiting_feedback'].includes(status)) {
            loadingSpinner.style.display = 'none';
        } else {
            loadingSpinner.style.display = 'inline-block';
        }
    }

    function hideStatus() {
        statusArea.style.display = 'none';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorArea.style.display = 'block';
        hideStatus(); // Hide loading status when error occurs
    }

    function hideError() {
        errorArea.style.display = 'none';
        errorMessage.textContent = '';
    }

    function hideResults() {
        resultsArea.style.display = 'none';
        // Clear results content when hiding
        if (finalQuestionsContentEl) finalQuestionsContentEl.textContent = '';
        if (finalEvaluationFeedbackEl) finalEvaluationFeedbackEl.textContent = '';
        if (finalPerQuestionEvaluationEl) finalPerQuestionEvaluationEl.innerHTML = '';
        if (finalContextPreviewEl) finalContextPreviewEl.textContent = '';
    }

}); // End DOMContentLoaded