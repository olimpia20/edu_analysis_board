// train_regression.js - Linear Regression model training only
// This file handles the training of Linear Regression models

console.log("Linear Regression training script loaded");

// Wait for the DOM to be fully loaded before running any code
document.addEventListener('DOMContentLoaded', function () {
    console.log("DOM fully loaded - starting Linear Regression training setup");
    
    // ========================================
    // STEP 1: GET ALL THE HTML ELEMENTS WE NEED
    // ========================================
    
    // Get the algorithm dropdown (now fixed to Linear Regression)
    const regressionAlgorithm = document.getElementById('regressionAlgorithm');
    
    // Get the target column dropdown (what we want to predict)
    const targetColumn = document.getElementById('targetColumn');
    
    // Get the feature columns multi-select (input variables)
    const featureColumns = document.getElementById('featureColumns');
    
    // Get the train button
    const trainModelBtn = document.getElementById('trainModelBtn');
    
    // Get the feedback area where we show success/error messages
    const trainFeedback = document.getElementById('trainFeedback');
    
    // Get the spinner that shows during training
    const trainSpinner = document.getElementById('trainSpinner');
    
    // Get the prediction section (hidden until model is trained)
    const predictionSection = document.getElementById('predictionSection');
    
    // Get the prediction form and result areas
    const predictForm = document.getElementById('predictForm');
    const predictResult = document.getElementById('predictResult');
    
    console.log("All DOM elements found:", {
        algorithm: regressionAlgorithm,
        target: targetColumn,
        features: featureColumns,
        trainBtn: trainModelBtn
    });
    
    // ========================================
    // STEP 2: FUNCTION TO CHECK IF FORM IS VALID
    // ========================================
    
    function isFormValid() {
        // Algorithm is always selected (Linear Regression is the only option)
        const algorithmSelected = true;
        
        // Check if target column is selected
        const targetSelected = targetColumn.value !== '';
        
        // Check if at least one feature column is selected
        const featuresSelected = featureColumns.selectedOptions.length > 0;
        
        // Check if target is not selected as a feature
        const targetNotInFeatures = !Array.from(featureColumns.selectedOptions)
            .map(option => option.value)
            .includes(targetColumn.value);
        
        return algorithmSelected && targetSelected && featuresSelected && targetNotInFeatures;
    }
    
    // ========================================
    // STEP 3: FUNCTION TO UPDATE TRAIN BUTTON STATE
    // ========================================
    
    function updateTrainButtonState() {
        const isValid = isFormValid();
        trainModelBtn.disabled = !isValid;
        
        if (isValid) {
            trainModelBtn.classList.remove('btn-secondary');
            trainModelBtn.classList.add('btn-primary');
        } else {
            trainModelBtn.classList.remove('btn-primary');
            trainModelBtn.classList.add('btn-secondary');
        }
        
        console.log("Train button state updated. Enabled:", isValid);
    }
    
    // ========================================
    // STEP 4: ADD EVENT LISTENERS TO FORM ELEMENTS
    // ========================================
    
    // When target column changes, update button state
    targetColumn.addEventListener('change', function() {
        console.log("Target column selected:", this.value);
        updateTrainButtonState();
    });
    
    // When feature columns change, update button state
    featureColumns.addEventListener('change', function() {
        console.log("Feature columns changed. Selected:", 
            Array.from(this.selectedOptions).map(opt => opt.value));
        updateTrainButtonState();
    });
    
    // ========================================
    // STEP 5: FUNCTION TO SHOW TRAINING FEEDBACK
    // ========================================
    
    function showFeedback(message, type = 'info') {
        // Clear previous feedback
        trainFeedback.innerHTML = '';
        
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = message;
        
        // Add to feedback area
        trainFeedback.appendChild(alertDiv);
        
        console.log("Feedback shown:", { message, type });
    }
    
    // ========================================
    // STEP 6: FUNCTION TO PREPARE TRAINING DATA
    // ========================================
    
    function prepareTrainingData() {
        // Algorithm is always Linear Regression
        const algorithm = 'LinearRegression';
        
        // Clean column names to avoid encoding issues
        const features = Array.from(featureColumns.selectedOptions)
            .map(option => option.value);
        const target = targetColumn.value;
        
        // Create clean versions of column names
        const cleanFeatures = features.map(f => f.replace(/[ăășț]/g, function(match) {
            const replacements = {
                'ă': 'a', 'ă': 'a', 'ș': 's', 'ț': 't',
                'Ă': 'A', 'Ă': 'A', 'Ș': 'S', 'Ț': 'T'
            };
            return replacements[match] || match;
        }));
        const cleanTarget = target.replace(/[ăășț]/g, function(match) {
            const replacements = {
                'ă': 'a', 'ă': 'a', 'ș': 's', 'ț': 't',
                'Ă': 'A', 'Ă': 'A', 'Ș': 'S', 'Ț': 'T'
            };
            return replacements[match] || match;
        });
        
        const payload = {
            algorithm: algorithm,
            features: cleanFeatures,
            target: cleanTarget,
            original_features: features,  // Keep original for reference
            original_target: target
        };
        
        // Debug: Log the exact payload being sent
        console.log("=== TRAINING PAYLOAD DEBUG ===");
        console.log("Original features:", features);
        console.log("Clean features:", cleanFeatures);
        console.log("Original target:", target);
        console.log("Clean target:", cleanTarget);
        console.log("Raw payload:", payload);
        console.log("JSON stringified:", JSON.stringify(payload));
        console.log("=============================");
        
        return payload;
    }
    
    // ========================================
    // STEP 7: FUNCTION TO SEND TRAINING REQUEST
    // ========================================
    
    function sendTrainingRequest(payload) {
        console.log("Sending Linear Regression training request to server...");
        
        // Show spinner and disable button
        trainSpinner.style.display = 'inline-block';
        trainModelBtn.disabled = true;
        
        const csrfMetaTag = document.querySelector('meta[name="csrf-token"]');
        if (!csrfMetaTag) {
            throw new Error('CSRF token meta tag not found');
        }
        const csrfToken = csrfMetaTag.getAttribute('content');
        
        // Send POST request to /train_model endpoint
        return fetch('/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            console.log("Server response received:", response.status);
            return response.json();
        })
        .then(data => {
            console.log("Training response data:", data);
            return data;
        })
        .catch(error => {
            console.error("Training request failed:", error);
            throw error;
        });
    }
    
    // ========================================
    // STEP 8: FUNCTION TO HANDLE TRAINING SUCCESS
    // ========================================
    
    function handleTrainingSuccess(data) {
        console.log("Linear Regression training successful, handling response...");
        
        // Hide spinner and re-enable button
        trainSpinner.style.display = 'none';
        trainModelBtn.disabled = false;
        
        // Show success message
        showFeedback(
            `<i class="fas fa-check-circle mr-2"></i>
             <strong>Success!</strong> Linear Regression model trained successfully!
             <br><small class="text-muted">You can now make predictions below.</small>`,
            'success'
        );
        
        // Show prediction section
        predictionSection.style.display = 'block';
        
        // Generate prediction form
        generatePredictionForm();
        
        console.log("Training success handled");
    }
    
    // ========================================
    // STEP 9: FUNCTION TO HANDLE TRAINING ERROR
    // ========================================
    
    function handleTrainingError(error) {
        console.error("Linear Regression training error occurred:", error);
        
        // Hide spinner and re-enable button
        trainSpinner.style.display = 'none';
        trainModelBtn.disabled = false;
        
        // Show error message
        let errorMessage = "An unexpected error occurred during Linear Regression training.";
        
        if (error.error) {
            errorMessage = error.error;
        } else if (error.message) {
            errorMessage = error.message;
        }
        
        showFeedback(
            `<i class="fas fa-exclamation-triangle mr-2"></i>
             <strong>Training Failed:</strong> ${errorMessage}`,
            'danger'
        );
    }
    
    // ========================================
    // STEP 10: FUNCTION TO GENERATE PREDICTION FORM
    // ========================================
    
    function generatePredictionForm() {
        console.log("Generating prediction form for Linear Regression...");
        
        // Clear previous form
        predictForm.innerHTML = '';
        
        // Get selected features
        const features = Array.from(featureColumns.selectedOptions)
            .map(option => option.value);
        
        // Create form title
        const formTitle = document.createElement('h6');
        formTitle.textContent = 'Enter values to predict ' + targetColumn.value + ':';
        formTitle.className = 'mb-3';
        predictForm.appendChild(formTitle);
        
        // Create input field for each feature
        features.forEach(feature => {
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';
            
            const label = document.createElement('label');
            label.textContent = feature;
            label.setAttribute('for', `predict_${feature}`);
            
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control';
            input.id = `predict_${feature}`;
            input.name = feature;
            input.required = true;
            input.step = 'any'; // Allow decimal numbers
            input.placeholder = `Enter value for ${feature}`;
            
            const helpText = document.createElement('small');
            helpText.className = 'form-text text-muted';
            helpText.textContent = `Enter a numeric value for ${feature}`;
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
            formGroup.appendChild(helpText);
            predictForm.appendChild(formGroup);
        });
        
        // Create predict button
        const predictBtn = document.createElement('button');
        predictBtn.type = 'submit';
        predictBtn.className = 'btn btn-success mt-3';
        predictBtn.innerHTML = '<i class="fas fa-magic mr-2"></i>Predict ' + targetColumn.value;
        predictForm.appendChild(predictBtn);
        
        console.log("Prediction form generated for features:", features);
    }
    
    // ========================================
    // STEP 11: ADD TRAIN BUTTON CLICK HANDLER
    // ========================================
    
    trainModelBtn.addEventListener('click', function(e) {
        e.preventDefault(); // Prevent form submission
        
        console.log("Train Linear Regression button clicked");
        
        // Check if form is valid
        if (!isFormValid()) {
            showFeedback(
                `<i class="fas fa-exclamation-triangle mr-2"></i>
                 <strong>Please complete all fields:</strong>
                 <ul class="mb-0 mt-2">
                     <li>Select a target column (what you want to predict)</li>
                     <li>Select at least one feature column (input variables)</li>
                     <li>Make sure target column is not selected as a feature</li>
                 </ul>`,
                'warning'
            );
            return;
        }
        
        // Prepare training data
        const payload = prepareTrainingData();
        
        // Send training request
        sendTrainingRequest(payload)
            .then(data => {
                if (data.success) {
                    handleTrainingSuccess(data);
                } else {
                    handleTrainingError(data);
                }
            })
            .catch(error => {
                handleTrainingError(error);
            });
    });
    
    // ========================================
    // STEP 12: ADD PREDICTION FORM SUBMIT HANDLER
    // ========================================
    
    predictForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        console.log("Prediction form submitted");
        
        // Get all input values
        const inputs = predictForm.querySelectorAll('input[type="number"]');
        const predictionData = {};
        
        inputs.forEach(input => {
            predictionData[input.name] = parseFloat(input.value);
        });
        
        console.log("Prediction data:", predictionData);
        
        // Get CSRF token from meta tag
        const csrfMetaTag = document.querySelector('meta[name="csrf-token"]');
        if (!csrfMetaTag) {
            throw new Error('CSRF token meta tag not found');
        }
        const csrfToken = csrfMetaTag.getAttribute('content');
        
        // Send prediction request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(predictionData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Show prediction result
                predictResult.innerHTML = `
                    <div class="alert alert-success">
                        <h6><i class="fas fa-chart-line mr-2"></i>Linear Regression Prediction:</h6>
                        <p class="mb-0"><strong>Predicted ${targetColumn.value}:</strong> ${data.prediction[0]}</p>
                    </div>
                `;
            } else {
                // Show error
                predictResult.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle mr-2"></i>
                        <strong>Prediction Failed:</strong> ${data.error}
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error("Prediction error:", error);
            predictResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle mr-2"></i>
                    <strong>Prediction Error:</strong> An unexpected error occurred.
                </div>
            `;
        });
    });
    
    // ========================================
    // STEP 13: INITIALIZE THE FORM
    // ========================================
    
    console.log("Initializing Linear Regression training form...");
    updateTrainButtonState();
    

    
    console.log("Linear Regression training setup complete!");
}); 