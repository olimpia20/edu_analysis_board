// train.js - Comprehensive AI Model Training Interface
// Supports Regression, Classification, and Clustering algorithms

console.log("Comprehensive AI training script loaded");

document.addEventListener('DOMContentLoaded', function () {
    console.log("DOM fully loaded - starting AI training setup");
    
    // ========================================
    // STEP 1: GET ALL THE HTML ELEMENTS WE NEED
    // ========================================
    
    // Task and algorithm selection
        const taskType = document.getElementById('taskType');
        const algorithm = document.getElementById('algorithm');
    const algorithmDescription = document.getElementById('algorithmDescription');
    
    // Data selection
        const targetColDiv = document.getElementById('targetColDiv');
        const targetCol = document.getElementById('targetCol');
        const featureCols = document.getElementById('featureCols');
    
    // Training controls
    const trainForm = document.getElementById('trainForm');
        const trainBtn = document.getElementById('trainBtn');
        const trainSpinner = document.getElementById('trainSpinner');
        const trainFeedback = document.getElementById('trainFeedback');
    
    // Prediction section
        const predictionSection = document.getElementById('predictionSection');
        const predictForm = document.getElementById('predictForm');
        const predictResult = document.getElementById('predictResult');
        const predictionHistory = document.getElementById('predictionHistory');

    // Warning
    const normalizeWarning = document.getElementById('normalizeWarning');
    
    console.log("All DOM elements found:", {
        taskType, algorithm, targetCol, featureCols, trainBtn
    });
    
    // ========================================
    // STEP 2: ALGORITHM DEFINITIONS
    // ========================================
    
    const algorithms = {
        regression: [
            { 
                value: 'LinearRegression', 
                label: 'Linear Regression',
                description: 'Best for linear relationships between features and target. Simple, interpretable, and fast.'
            },
            { 
                value: 'RandomForestRegressor', 
                label: 'Random Forest Regressor',
                description: 'Powerful ensemble method that handles non-linear relationships and provides feature importance.'
            }
        ],
        classification: [
            { 
                value: 'LogisticRegression', 
                label: 'Logistic Regression',
                description: 'Best for binary classification problems. Provides probability scores and is interpretable.'
            },
            { 
                value: 'RandomForestClassifier', 
                label: 'Random Forest Classifier',
                description: 'Robust ensemble method for classification with feature importance and handles multiple classes.'
            }
        ],
        clustering: [
            { 
                value: 'KMeans', 
                label: 'K-Means Clustering',
                description: 'Groups similar data points into clusters. No target variable needed - discovers patterns in your data.'
            }
        ]
    };
    
    // ========================================
    // STEP 3: HELPER FUNCTIONS
    // ========================================
    
    // Update algorithm dropdown based on task type
        function updateAlgorithmOptions() {
            const task = taskType.value;
            console.log("Task selected:", task);
        
        algorithm.innerHTML = '<option value="">-- Select Algorithm --</option>';
        algorithmDescription.innerHTML = '';
        
            if (algorithms[task]) {
                algorithms[task].forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt.value;
                    option.textContent = opt.label;
                    algorithm.appendChild(option);
                });
                algorithm.disabled = false;
                console.log("Algorithm dropdown enabled");
            } else {
                algorithm.disabled = true;
                console.log("Algorithm dropdown disabled");
            }
        }

    // Show/hide target column based on task type
        function updateTargetColVisibility() {
            if (taskType.value === 'clustering') {
                if (targetColDiv) {
                    targetColDiv.style.display = 'none';
                    console.log("Target column hidden (clustering)");
                }
            } else {
                if (targetColDiv) {
                    targetColDiv.style.display = '';
                    console.log("Target column shown");
                }
            }
        }

    // Update algorithm description
    function updateAlgorithmDescription() {
        const task = taskType.value;
        const algo = algorithm.value;
        
        if (task && algo && algorithms[task]) {
            const selectedAlgo = algorithms[task].find(a => a.value === algo);
            if (selectedAlgo) {
                algorithmDescription.innerHTML = `
                    <div class="alert alert-info">
                        <strong>${selectedAlgo.label}:</strong> ${selectedAlgo.description}
                    </div>
                `;
            }
        } else {
            algorithmDescription.innerHTML = '';
        }
    }
    
    // Enable/disable train button based on form validity
        function updateTrainBtnState() {
            const task = taskType.value;
            const algo = algorithm.value;
            const features = Array.from(featureCols.selectedOptions).map(opt => opt.value);
            const target = targetCol.value;
        
            let valid = task && algo && features.length > 0;
        
        // For clustering, no target needed
        if (task !== 'clustering') {
            valid = valid && target;
        }
        
        // Check if target is not selected as a feature
        if (task !== 'clustering' && target && features.includes(target)) {
            valid = false;
        }
        
            trainBtn.disabled = !valid;
            console.log("Train button state updated. Enabled:", valid);
        }

    // Show feedback message
    function showFeedback(message, type = 'info') {
        trainFeedback.innerHTML = '';
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = message;
        trainFeedback.appendChild(alertDiv);
        console.log("Feedback shown:", { message, type });
    }
    
    // Generate prediction form based on features
    function generatePredictionForm(features) {
        predictForm.innerHTML = '';
        
        features.forEach(f => {
            const div = document.createElement('div');
            div.className = 'form-group';
            div.innerHTML = `
                <label for="${f}"><strong>${f}:</strong></label>
                <input type="text" class="form-control" id="${f}" name="${f}" required>
                <small class="form-text text-muted">Enter a value for ${f}</small>
            `;
            predictForm.appendChild(div);
        });
        
        const btn = document.createElement('button');
        btn.type = 'submit';
        btn.className = 'btn btn-success';
        btn.innerHTML = '<i class="fas fa-magic mr-2"></i>Make Prediction';
        predictForm.appendChild(btn);
        
        console.log("Prediction form generated for features:", features);
    }
    
    // Load prediction history
    function loadPredictionHistory() {
        // This could be implemented to show previous predictions
        predictionHistory.innerHTML = '<h6>Recent Predictions</h6><p>No predictions yet.</p>';
    }
    
    // ========================================
    // STEP 4: EVENT LISTENERS
    // ========================================
    
    // Task type change
        taskType.addEventListener('change', function () {
        console.log("Task type changed to:", this.value);
            updateAlgorithmOptions();
            updateTargetColVisibility();
            updateTrainBtnState();
        });
    
    // Algorithm change
    algorithm.addEventListener('change', function () {
        console.log("Algorithm changed to:", this.value);
        updateAlgorithmDescription();
        updateTrainBtnState();
    });
    
    // Target column change
    targetCol.addEventListener('change', function () {
        console.log("Target column changed to:", this.value);
        updateTrainBtnState();
    });
    
    // Feature columns change
    featureCols.addEventListener('change', function () {
        console.log("Feature columns changed. Selected:", 
            Array.from(this.selectedOptions).map(opt => opt.value));
        updateTrainBtnState();
    });
    
    // ========================================
    // STEP 5: TRAINING FORM SUBMISSION
    // ========================================
    
        trainForm.addEventListener('submit', function (e) {
            e.preventDefault();
        console.log("Training form submitted");
        
            const task = taskType.value;
        const algo = algorithm.value;
            const features = Array.from(featureCols.selectedOptions).map(opt => opt.value);
        const target = targetCol.value;
        
        // Validate clustering features are numeric
        if (task === 'clustering') {
            // For now, we'll assume all features are numeric
            // In a real implementation, you'd check data types
        }
        
        // Prepare training data
        const payload = {
            algorithm: algo,
            features: features,
            original_features: features
        };
        
        // Add task-specific data
        if (task === 'clustering') {
            payload.n_clusters = 3; // Default, could be made configurable
        } else {
                payload.target = target;
            payload.original_target = target;
        }
        
        console.log("Training payload:", payload);
        
        // Show spinner and disable button
        trainBtn.disabled = true;
        trainSpinner.style.display = 'inline-block';
        trainFeedback.innerHTML = '';
        predictionSection.style.display = 'none';
        
        // Get CSRF token
        const csrfMetaTag = document.querySelector('meta[name="csrf-token"]');
        if (!csrfMetaTag) {
            showFeedback('CSRF token not found. Please refresh the page.', 'danger');
            trainBtn.disabled = false;
            trainSpinner.style.display = 'none';
            return;
        }
        const csrfToken = csrfMetaTag.getAttribute('content');
        
        // Send training request
            fetch('/train_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
                },
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                trainSpinner.style.display = 'none';
            
            if (data.success) {
                showFeedback(data.message, 'success');
                predictionSection.style.display = 'block';
                generatePredictionForm(features);
                loadPredictionHistory();
                console.log("Model trained successfully");
            } else {
                showFeedback(data.error, 'danger');
                console.error("Training failed:", data.error);
            }
            
                updateTrainBtnState();
            })
            .catch(error => {
                trainSpinner.style.display = 'none';
            showFeedback(`Unexpected error: ${error.message}`, 'danger');
                updateTrainBtnState();
            console.error("Training request failed:", error);
        });
    });
    
    // ========================================
    // STEP 6: PREDICTION FORM SUBMISSION
    // ========================================
    
    predictForm.addEventListener('submit', function (e) {
        e.preventDefault();
        console.log("Prediction form submitted");
        
        const formData = new FormData(predictForm);
        const inputData = {};
        
        // Collect form data
        for (let [key, value] of formData.entries()) {
            inputData[key] = value;
        }
        
        console.log("Prediction input:", inputData);
        
        // Get CSRF token
        const csrfMetaTag = document.querySelector('meta[name="csrf-token"]');
        if (!csrfMetaTag) {
            alert('CSRF token not found. Please refresh the page.');
            return;
        }
        const csrfToken = csrfMetaTag.getAttribute('content');
        
        // Send prediction request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(inputData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayPredictionResult(data);
                console.log("Prediction successful:", data);
            } else {
                predictResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                console.error("Prediction failed:", data.error);
            }
        })
        .catch(error => {
            predictResult.innerHTML = `<div class="alert alert-danger">Unexpected error: ${error.message}</div>`;
            console.error("Prediction request failed:", error);
        });
    });
    
    // Display prediction result
    function displayPredictionResult(data) {
        let resultHtml = '<div class="alert alert-success">';
        
        if (data.task === 'regression') {
            resultHtml += `
                <h5><i class="fas fa-chart-line mr-2"></i>Regression Prediction</h5>
                <p><strong>Predicted ${data.target}:</strong> ${data.prediction[0].toFixed(4)}</p>
            `;
            if (data.model_equation) {
                resultHtml += `<p><strong>Model Equation:</strong> ${data.model_equation}</p>`;
            }
        } else if (data.task === 'classification') {
            resultHtml += `
                <h5><i class="fas fa-tags mr-2"></i>Classification Prediction</h5>
                <p><strong>Predicted Class:</strong> ${data.prediction[0]}</p>
            `;
            if (data.probabilities) {
                resultHtml += '<p><strong>Class Probabilities:</strong></p><ul>';
                Object.entries(data.probabilities).forEach(([class_name, prob]) => {
                    resultHtml += `<li>${class_name}: ${(prob * 100).toFixed(2)}%</li>`;
                });
                resultHtml += '</ul>';
            }
        } else if (data.task === 'clustering') {
            resultHtml += `
                <h5><i class="fas fa-object-group mr-2"></i>Clustering Result</h5>
                <p><strong>Assigned Cluster:</strong> ${data.prediction[0]}</p>
                <p><strong>Cluster Number:</strong> ${data.cluster}</p>
            `;
        }
        
        resultHtml += `
            <hr>
            <p><strong>Input Values:</strong></p>
            <ul>
                ${Object.entries(data.input_values).map(([key, value]) => `<li>${key}: ${value}</li>`).join('')}
            </ul>
        </div>`;
        
        predictResult.innerHTML = resultHtml;
    }
    
    // ========================================
    // STEP 7: INITIALIZATION
    // ========================================
    
    // Check normalization status on page load
    console.log("Checking normalization status...");
            fetch('/is_normalized_status')
              .then(response => response.json())
              .then(data => {
            console.log("Normalization status:", data);
                if (!data.is_normalized) {
                trainBtn.disabled = true;
                if (normalizeWarning) {
                    normalizeWarning.classList.remove('d-none');
                }
                } else {
                if (normalizeWarning) {
                    normalizeWarning.classList.add('d-none');
                }
            }
        })
        .catch(error => {
            console.error("Error checking normalization:", error);
        });
    
    // Initialize form state
    updateAlgorithmOptions();
    updateTargetColVisibility();
    updateTrainBtnState();
    
    console.log("AI training interface initialized successfully");
}); 