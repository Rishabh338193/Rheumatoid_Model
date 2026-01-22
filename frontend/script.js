// Rheumatoid Arthritis Prediction - JavaScript

const API_BASE_URL = 'http://127.0.0.1:5001';

// DOM Elements
const form = document.getElementById('predictionForm');
const resultCard = document.getElementById('resultCard');
const loadingOverlay = document.getElementById('loadingOverlay');
const resetBtn = document.getElementById('resetBtn');
const sampleBtn = document.getElementById('sampleBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    form.addEventListener('submit', handleSubmit);
    resetBtn.addEventListener('click', resetForm);
    sampleBtn.addEventListener('click', loadSampleData);
}

// Load Model Information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('modelAccuracy').textContent = 
                `${(data.accuracy * 100).toFixed(2)}%`;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelAccuracy').textContent = 'N/A';
    }
}

// Handle Form Submit
async function handleSubmit(e) {
    e.preventDefault();
    
    // Show loading
    showLoading(true);
    
    // Get form data
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        // Convert numeric fields to numbers
        if (['Age', 'Morning_Stiffness_Duration', 'Joint_Pain_Score', 
             'Swollen_Joint_Count', 'Rheumatoid_Factor', 'Anti_CCP', 
             'ESR', 'CRP', 'Fatigue_Score'].includes(key)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }
    
    try {
        // Make prediction
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// Display Results
function displayResults(result) {
    // Show result card
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Main Result
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    const resultConfidence = document.getElementById('resultConfidence');
    
    if (result.prediction === 1) {
        resultIcon.textContent = '⚠️';
        resultText.textContent = 'RA Positive';
        resultText.className = 'result-text positive';
    } else {
        resultIcon.textContent = '✅';
        resultText.textContent = 'No RA Detected';
        resultText.className = 'result-text negative';
    }
    
    resultConfidence.textContent = 
        `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    
    // Risk Level
    displayRiskLevel(result);
    
    // Explanation
    displayExplanation(result.explanation);
    
    // Feature Importance
    displayFeatureImportance(result.explanation.top_features);
}

// Display Risk Level
function displayRiskLevel(result) {
    const riskLevel = document.getElementById('riskLevel');
    const riskBadge = document.getElementById('riskBadge');
    const riskMeterFill = document.getElementById('riskMeterFill');
    const probabilityText = document.getElementById('probabilityText');
    
    riskLevel.style.display = 'block';
    
    const probability = result.probability.RA_Positive;
    const riskLevelText = result.risk_level;
    
    // Update badge
    riskBadge.textContent = `${riskLevelText} Risk`;
    riskBadge.className = `risk-badge ${riskLevelText.toLowerCase()}`;
    
    // Update meter
    riskMeterFill.style.width = `${probability * 100}%`;
    riskMeterFill.className = `risk-meter-fill ${riskLevelText.toLowerCase()}`;
    
    // Update probability text
    probabilityText.textContent = 
        `RA Probability: ${(probability * 100).toFixed(1)}% | ` +
        `No RA: ${(result.probability.No_RA * 100).toFixed(1)}%`;
}

// Display Explanation
function displayExplanation(explanation) {
    const explanationDiv = document.getElementById('explanation');
    const keyFactors = document.getElementById('keyFactors');
    const recommendations = document.getElementById('recommendations');
    
    explanationDiv.style.display = 'block';
    
    // Key Factors
    if (explanation.key_factors && explanation.key_factors.length > 0) {
        keyFactors.innerHTML = `
            <ul>
                ${explanation.key_factors.map(factor => 
                    `<li>🔸 ${factor}</li>`
                ).join('')}
            </ul>
        `;
    } else {
        keyFactors.innerHTML = '<p>No significant risk factors detected.</p>';
    }
    
    // Recommendations
    if (explanation.recommendations && explanation.recommendations.length > 0) {
        recommendations.innerHTML = `
            <ul>
                ${explanation.recommendations.map(rec => 
                    `<li>💡 ${rec}</li>`
                ).join('')}
            </ul>
        `;
    }
}

// Display Feature Importance
function displayFeatureImportance(topFeatures) {
    const featureImportance = document.getElementById('featureImportance');
    const featuresList = document.getElementById('featuresList');
    
    featureImportance.style.display = 'block';
    
    if (topFeatures && topFeatures.length > 0) {
        featuresList.innerHTML = topFeatures.map(feature => `
            <div class="feature-item">
                <span class="feature-name">${formatFeatureName(feature.feature)}</span>
                <span class="feature-value">${(feature.importance * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }
}

// Format Feature Name
function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/Encoded/g, '')
        .trim()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Load Sample Data
function loadSampleData() {
    // Sample RA positive patient
    const sampleData = {
        Age: 52,
        Gender: 'Female',
        Morning_Stiffness_Duration: 90,
        Joint_Pain_Score: 8,
        Swollen_Joint_Count: 12,
        Rheumatoid_Factor: 95.5,
        Anti_CCP: 150.3,
        ESR: 55.2,
        CRP: 25.8,
        Fatigue_Score: 9,
        Family_History: 'Yes',
        Smoking_Status: 'Yes'
    };
    
    // Fill form
    document.getElementById('age').value = sampleData.Age;
    document.getElementById('gender').value = sampleData.Gender;
    document.getElementById('stiffness').value = sampleData.Morning_Stiffness_Duration;
    document.getElementById('jointPain').value = sampleData.Joint_Pain_Score;
    document.getElementById('swollenJoints').value = sampleData.Swollen_Joint_Count;
    document.getElementById('rf').value = sampleData.Rheumatoid_Factor;
    document.getElementById('antiCCP').value = sampleData.Anti_CCP;
    document.getElementById('esr').value = sampleData.ESR;
    document.getElementById('crp').value = sampleData.CRP;
    document.getElementById('fatigue').value = sampleData.Fatigue_Score;
    document.getElementById('familyHistory').value = sampleData.Family_History;
    document.getElementById('smoking').value = sampleData.Smoking_Status;
    
    // Scroll to top of form
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Reset Form
function resetForm() {
    form.reset();
    resultCard.style.display = 'none';
    
    // Hide all result sections
    document.getElementById('riskLevel').style.display = 'none';
    document.getElementById('explanation').style.display = 'none';
    document.getElementById('featureImportance').style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show Loading
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

// Show Error
function showError(message) {
    const resultMain = document.getElementById('resultMain');
    
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    document.getElementById('resultIcon').textContent = '❌';
    document.getElementById('resultText').textContent = 'Error';
    document.getElementById('resultText').className = 'result-text';
    document.getElementById('resultConfidence').textContent = message;
    
    // Hide other sections
    document.getElementById('riskLevel').style.display = 'none';
    document.getElementById('explanation').style.display = 'none';
    document.getElementById('featureImportance').style.display = 'none';
}

// Check API Connection on Load
async function checkAPIConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('Cannot connect to API:', error);
        alert('Warning: Cannot connect to prediction API. Please ensure the backend server is running.');
    }
}

// Check connection when page loads
checkAPIConnection();
