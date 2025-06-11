/**
 * Converts a base64 encoded image string to an HTMLImageElement.
 * @param {string} base64Str - The base64 encoded image string.
 * @returns {HTMLImageElement} - The image element displaying the decoded image.
 */
function base64ToImage(base64Str) {
    const img = new Image();
    img.src = "data:image/png;base64," + base64Str;
    return img;
}

/**
 * Fetches prediction data from the /predict endpoint for a given ticker,
 * extracts the base64 encoded image, converts it to an image element,
 * and appends it to the specified container element.
 * @param {string} ticker - The stock ticker symbol.
 * @param {HTMLElement} container - The container element to append the image to.
 */
async function fetchAndDisplayPredictionImage(ticker, container) {
    try {
        const response = await fetch(`/predict?ticker=${ticker}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.prediction_plot) {
            const img = base64ToImage(data.prediction_plot);
            container.innerHTML = ''; // Clear previous content
            container.appendChild(img);
        } else {
            container.textContent = 'No image data available.';
        }
    } catch (error) {
        container.textContent = 'Error fetching prediction image: ' + error.message;
    }
}
