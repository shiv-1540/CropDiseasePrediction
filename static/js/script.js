document.getElementById('predict-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('file', document.getElementById('image-upload').files[0]);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.predicted_disease) {
            document.getElementById('disease-name').textContent = `Predicted Disease: ${data.predicted_disease}`;
            document.getElementById('result').style.display = 'block';
        } else {
            document.getElementById('disease-name').textContent = 'Error: Could not predict disease';
        }
    })
    .catch(error => {
        document.getElementById('disease-name').textContent = 'Error: Could not predict disease';
    });
});