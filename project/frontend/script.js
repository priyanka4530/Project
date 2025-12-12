function uploadImage() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    // Show preview
    let preview = document.getElementById("preview");
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    // Prepare form data
    let formData = new FormData();
    formData.append("image", file);

    // Send to backend
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        let resultBox = document.getElementById("result");

        if (data.prediction === "real") {
            resultBox.innerHTML = "✔ This is a REAL image";
            resultBox.className = "real";
        }
        else if (data.prediction === "ai") {
            resultBox.innerHTML = "❌ This is an AI-generated image";
            resultBox.className = "fake";
        }
        else {
            resultBox.innerHTML = "Error processing image!";
        }
    })
    .catch(err => {
        console.error(err);
        alert("Something went wrong with the server.");
    });
}
