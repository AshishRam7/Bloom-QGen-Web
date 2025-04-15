document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const statusMessage = document.getElementById('status-message');
    const resultsSection = document.getElementById('results-section');
    const initialQuestionsSection = document.getElementById('initial-questions-section');
    const finalResultsSection = document.getElementById('final-results-section');
    const feedbackSection = document.getElementById('feedback-section');
    const regenerateButton = document.getElementById('regenerate-button');
    const feedbackInput = document.getElementById('feedback-input');
    const jobIdDisplay = document.getElementById('job-id-display');
    const generatedQuestionsPre = document.getElementById('generated-questions');
    const evaluationFeedbackPre = document.getElementById('evaluation-feedback');
    const perQuestionEvaluationDiv = document.getElementById('per-question-evaluation');
    const initialQuestionsContentPre = document.getElementById('initial-questions-content');
    const contextPreviewContentPre = document.getElementById('context-preview-content');
    const markdownSection = document.getElementById('markdown-section');
    const markdownContentPre = document.getElementById('markdown-content');
    const errorSection = document.getElementById('error-section');
    const errorMessagePre = document.getElementById('error-message');
    const startNewJobButton = document.getElementById('start-new-job-button');
    const startNewJobButtonError = document.getElementById('start-new-job-button-error');
    const progressBarContainer = document.getElementById('progress-bar-container');
    const progressBar = document.getElementById('progress-bar');
    const submitButton = document.getElementById('submit-button');

    // Image Slideshow Elements
    const imageSlideshowSection = document.getElementById('image-slideshow-section');
    const imageContainer = document.getElementById('image-container');
    const imageNav = document.getElementById('image-nav');
    const prevImageButton = document.getElementById('prev-image');
    const nextImageButton = document.getElementById('next-image');
    const imageCounterSpan = document.getElementById('image-counter');
    const noImagesMessage = document.getElementById('no-images-message');

    let currentJobId = null;
    let pollInterval = null;
    let imagePaths = [];
    let currentImageIndex = 0;

    // --- Theme Toggle ---
    const themeToggleButton = document.getElementById('theme-toggle');
    const body = document.body;

    // Function to apply theme based on localStorage
    function applyTheme() {
      const currentTheme = localStorage.getItem('theme');
      if (currentTheme === 'dark') {
        body.classList.add('dark-mode');
        themeToggleButton.textContent = '☀️'; // Sun icon for dark mode
        themeToggleButton.setAttribute('aria-label', 'Toggle light mode');

      } else {
        body.classList.remove('dark-mode');
        themeToggleButton.textContent = '🌙'; // Moon icon for light mode
        themeToggleButton.setAttribute('aria-label', 'Toggle dark mode');
      }
    }

    // Function to toggle theme
    function toggleTheme() {
      body.classList.toggle('dark-mode');
      let theme = 'light';
      if (body.classList.contains('dark-mode')) {
        theme = 'dark';
      }
      localStorage.setItem('theme', theme);
      applyTheme(); // Update button icon/label
    }

    // Event listener for theme toggle button
    themeToggleButton.addEventListener('click', toggleTheme);

    // Apply theme on initial load
    applyTheme();
    // --- End Theme Toggle ---


    // --- Image Slideshow Logic ---
    function updateImageDisplay() {
        if (!imagePaths || imagePaths.length === 0) {
             imageSlideshowSection.style.display = 'block'; // Show section
             imageContainer.innerHTML = ''; // Clear container
             imageNav.style.display = 'none'; // Hide nav
             noImagesMessage.style.display = 'block'; // Show no images message
             return;
        }

        imageSlideshowSection.style.display = 'block';
        noImagesMessage.style.display = 'none';

        if (imagePaths.length > 1) {
             imageNav.style.display = 'flex'; // Show nav only if multiple images
             imageNav.style.justifyContent = 'center';
             imageNav.style.alignItems = 'center';
        } else {
             imageNav.style.display = 'none';
        }


        // Sanitize URL before setting it as src
        const imageUrl = imagePaths[currentImageIndex];
        // Basic check: ensure it starts with /extracted_images/
        if (imageUrl && imageUrl.startsWith('/extracted_images/')) {
            imageContainer.innerHTML = `<img src="${imageUrl}" alt="Extracted Image ${currentImageIndex + 1}">`;
        } else {
             console.error("Invalid image URL detected:", imageUrl);
             imageContainer.innerHTML = `<p style="color: red;">Error loading image: Invalid URL</p>`;
        }

        imageCounterSpan.textContent = `${currentImageIndex + 1} / ${imagePaths.length}`;
        prevImageButton.disabled = currentImageIndex === 0;
        nextImageButton.disabled = currentImageIndex === imagePaths.length - 1;
    }

    prevImageButton.addEventListener('click', () => {
        if (currentImageIndex > 0) {
            currentImageIndex--;
            updateImageDisplay();
        }
    });

    nextImageButton.addEventListener('click', () => {
        if (currentImageIndex < imagePaths.length - 1) {
            currentImageIndex++;
            updateImageDisplay();
        }
    });
    // --- End Image Slideshow Logic ---

    // --- Form Submission & Polling ---
    uploadForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        resetUI();
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';
        statusMessage.textContent = 'Uploading files...';
        progressBarContainer.style.display = 'block';
        progressBar.style.width = '10%'; // Initial small progress
        progressBar.textContent = '10%';

        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/start-processing', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            currentJobId = data.job_id;
            statusMessage.textContent = `Job started with ID: ${currentJobId}. Status: ${data.status}. ${data.message}`;
            jobIdDisplay.textContent = `Job ID: ${currentJobId}`;
            resultsSection.style.display = 'block';
            progressBar.style.width = '25%';
             progressBar.textContent = '25%';

            // Start polling
            pollInterval = setInterval(checkJobStatus, 3000); // Poll every 3 seconds

        } catch (error) {
            console.error('Error starting job:', error);
            displayError(`Error starting job: ${error.message}`);
            submitButton.disabled = false;
            submitButton.textContent = 'Start Processing';
        }
    });

    async function checkJobStatus() {
        if (!currentJobId) return;

        try {
            const response = await fetch(`/status/${currentJobId}`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();

            statusMessage.textContent = `Job ID: ${currentJobId}. Status: ${data.status}. ${data.message}`;

            // Update progress bar based on status
            updateProgressBar(data.status);


            if (data.status === 'awaiting_feedback') {
                clearInterval(pollInterval);
                pollInterval = null;
                displayInitialResults(data.result);
                submitButton.disabled = false; // Re-enable for potential next job
                submitButton.textContent = 'Start Processing';
            } else if (data.status === 'completed') {
                clearInterval(pollInterval);
                pollInterval = null;
                displayFinalResults(data.result);
                submitButton.disabled = false;
                 submitButton.textContent = 'Start Processing';

            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                pollInterval = null;
                displayError(data.message || 'An unknown error occurred.');
                 submitButton.disabled = false;
                 submitButton.textContent = 'Start Processing';
            }
            // Continue polling if status is 'processing', 'queued', 'processing_feedback' etc.

        } catch (error) {
            console.error('Error polling status:', error);
            statusMessage.textContent = `Error checking status: ${error.message}. Retrying...`;
            // Optionally stop polling after too many errors
        }
    }

     function updateProgressBar(status) {
        let progress = 0;
        let text = '';
        switch (status) {
            case 'queued':
                progress = 15;
                text = 'Queued (15%)';
                break;
            case 'processing':
                progress = 40; // Example intermediate value
                text = 'Processing Docs (40%)';
                break;
             case 'searching_context': // Add hypothetical intermediate steps if desired
                 progress = 60;
                 text = 'Searching Context (60%)';
                 break;
             case 'generating_initial':
                 progress = 75;
                 text = 'Generating Questions (75%)';
                 break;
            case 'awaiting_feedback':
                progress = 90;
                 text = 'Awaiting Feedback (90%)';
                break;
            case 'processing_feedback':
                 progress = 95;
                 text = 'Processing Feedback (95%)';
                break;
            case 'completed':
                progress = 100;
                text = 'Completed (100%)';
                break;
            case 'error':
                progress = 100; // Or maybe keep previous progress?
                text = 'Error';
                 progressBar.style.backgroundColor = 'var(--error-color)';
                break;
            default:
                progress = 5; // Default small progress for unknown states
                text = 'Starting...';
        }
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = text;

         if(status === 'completed' || status === 'error') {
            // Optionally hide progress bar after a delay
            setTimeout(() => {
               // progressBarContainer.style.display = 'none';
            }, 2000);
         } else {
            progressBarContainer.style.display = 'block';
              if (status !== 'error') progressBar.style.backgroundColor = 'var(--progress-bar-fill)'; // Reset color if not error
         }
    }


    function displayInitialResults(result) {
         progressBar.style.width = '90%'; // Update progress
         progressBar.textContent = 'Awaiting Feedback (90%)';
        if (!result) return;
        initialQuestionsContentPre.textContent = result.initial_questions || 'No initial questions generated.';
        contextPreviewContentPre.textContent = result.retrieved_context_preview || 'No context preview available.';
        markdownContentPre.textContent = result.extracted_markdown || 'No markdown extracted.';
        initialQuestionsSection.style.display = 'block';
        markdownSection.style.display = 'block';
        feedbackSection.style.display = 'block'; // Show feedback input and button
        finalResultsSection.style.display = 'none';
        errorSection.style.display = 'none';

        // Handle Images
        imagePaths = result.image_paths || [];
        currentImageIndex = 0;
        updateImageDisplay(); // This will show/hide the slideshow section and message appropriately
    }

    function displayFinalResults(result) {
         progressBar.style.width = '100%'; // Ensure progress is 100%
         progressBar.textContent = 'Completed (100%)';
         if (!result) return;
        generatedQuestionsPre.textContent = result.generated_questions || 'No final questions available.';
        evaluationFeedbackPre.textContent = result.evaluation_feedback || 'No evaluation feedback.';
        markdownContentPre.textContent = result.extracted_markdown || 'No markdown extracted.'; // Keep markdown visible

        perQuestionEvaluationDiv.innerHTML = ''; // Clear previous
        if (result.per_question_evaluation && result.per_question_evaluation.length > 0) {
            result.per_question_evaluation.forEach(evalItem => {
                const itemDiv = document.createElement('div');
                let qualitativeHtml = '<ul>';
                for (const [metric, value] of Object.entries(evalItem.qualitative || {})) {
                    qualitativeHtml += `<li>${metric}: ${value ? '✅ Pass' : '❌ Fail'}</li>`;
                }
                qualitativeHtml += '</ul>';

                itemDiv.innerHTML = `
                    <strong>Question ${evalItem.question_num}:</strong> ${evalItem.question_text} <br>
                    QSTS Score: ${evalItem.qsts_score !== null ? evalItem.qsts_score.toFixed(3) : 'N/A'} <br>
                    Qualitative: ${qualitativeHtml}
                `;
                perQuestionEvaluationDiv.appendChild(itemDiv);
            });
        } else {
            perQuestionEvaluationDiv.textContent = 'No per-question evaluation data.';
        }

        initialQuestionsSection.style.display = 'none'; // Hide initial section
        finalResultsSection.style.display = 'block';
        markdownSection.style.display = 'block'; // Keep markdown visible
        errorSection.style.display = 'none';

         // Keep image slideshow visible if images exist
         imagePaths = result.image_paths || [];
         currentImageIndex = 0;
         updateImageDisplay();

         // Hide progress bar after a short delay
         setTimeout(() => { progressBarContainer.style.display = 'none'; }, 1500);
    }

    function displayError(message) {
        errorMessagePre.textContent = message;
        resultsSection.style.display = 'block';
        initialQuestionsSection.style.display = 'none';
        finalResultsSection.style.display = 'none';
        markdownSection.style.display = 'none';
        imageSlideshowSection.style.display = 'none';
        errorSection.style.display = 'block';
         submitButton.disabled = false;
         submitButton.textContent = 'Start Processing';
         progressBar.style.width = '100%';
         progressBar.textContent = 'Error';
         progressBar.style.backgroundColor = 'var(--error-color)';
    }

    function resetUI() {
        statusMessage.textContent = 'Upload a PDF to start.';
        resultsSection.style.display = 'none';
        initialQuestionsSection.style.display = 'none';
        finalResultsSection.style.display = 'none';
        markdownSection.style.display = 'none';
        errorSection.style.display = 'none';
        imageSlideshowSection.style.display = 'none'; // Hide image section on reset
        imageContainer.innerHTML = '';
        feedbackInput.value = '';
        jobIdDisplay.textContent = '';
        progressBarContainer.style.display = 'none';
        progressBar.style.width = '0%';
        progressBar.textContent = '';
        progressBar.style.backgroundColor = 'var(--progress-bar-fill)'; // Reset color
        currentJobId = null;
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
        imagePaths = [];
        currentImageIndex = 0;
         submitButton.disabled = false;
         submitButton.textContent = 'Start Processing';
    }

    regenerateButton.addEventListener('click', async () => {
        if (!currentJobId) return;

        const feedback = feedbackInput.value.trim();
        regenerateButton.disabled = true;
        regenerateButton.textContent = 'Processing...';
        statusMessage.textContent = `Job ID: ${currentJobId}. Status: processing_feedback. Processing feedback...`;
        updateProgressBar('processing_feedback');

        try {
            const response = await fetch(`/regenerate-questions/${currentJobId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ feedback: feedback }),
            });

            if (!response.ok) {
                 const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            statusMessage.textContent = `Job ID: ${currentJobId}. Status: ${data.status}. ${data.message}`;

            // Restart polling to wait for completion or error after regeneration request
            if (!pollInterval) {
                 pollInterval = setInterval(checkJobStatus, 3000);
            }


        } catch (error) {
            console.error('Error regenerating questions:', error);
             displayError(`Error regenerating questions: ${error.message}`);
        } finally {
            regenerateButton.disabled = false;
            regenerateButton.textContent = 'Evaluate & Regenerate (if needed)';
        }
    });

     startNewJobButton.addEventListener('click', resetUI);
     startNewJobButtonError.addEventListener('click', resetUI);

});