const form = document.getElementById('recommendation-form');
const resultContainer = document.getElementById('result-container');
const loadingIndicator = document.getElementById('loading-indicator');

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    
    // Show loading indicator
    loadingIndicator.style.display = 'block';
    resultContainer.innerHTML = '';

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        resultContainer.innerHTML = `<h3>Recommended Major: ${result.major}</h3>`;
    } catch (error) {
        resultContainer.innerHTML = `<p>Error: ${error.message}</p>`;
    } finally {
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
    }
});