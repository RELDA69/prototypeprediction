document.getElementById('recommendationForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    // Collect form data
    const formData = new FormData(event.target);
    const data = {};
    for (let [key, value] of formData.entries()) {
        if (key === 'strongest_subjects') {
            if (!data[key]) data[key] = [];
            data[key].push(value);
        } else {
            data[key] = value;
        }
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('error').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    
    try {
        const response = await fetch('http://localhost:5000/predict', {  // Backend URL
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Prediction failed');
        
        const result = await response.json();
        document.getElementById('prediction').textContent = result.prediction;
        document.getElementById('results').style.display = 'block';
    } catch (err) {
        document.getElementById('error').textContent = 'Error: ' + err.message;
        document.getElementById('error').style.display = 'block';
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});