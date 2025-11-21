// =======================
// 🌙 Theme Toggle
// =======================
const themeToggle = document.getElementById('themeToggle');
const icon = themeToggle.querySelector('i');

themeToggle.addEventListener('click', () => {
    document.body.dataset.theme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    icon.className = document.body.dataset.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
});

// =======================
// 🖼️ Single File Upload
// =======================
const singleUploadBox = document.getElementById('singleUploadBox');
const singleFileInput = document.getElementById('singleFileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultContainer = document.getElementById('resultContainer');
const annotatedImage = document.getElementById('annotatedImage');
const downloadBtn = document.getElementById('downloadBtn');
const jsonViewer = document.getElementById('jsonViewer');
const loadingSpinner = document.getElementById('loadingSpinner');

// 🧲 Drag & Drop (Single)
singleUploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    singleUploadBox.style.borderColor = getComputedStyle(document.documentElement)
        .getPropertyValue('--accent-color');
});

singleUploadBox.addEventListener('dragleave', () => {
    singleUploadBox.style.borderColor = '';
});

singleUploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    singleUploadBox.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleSingleFile(file);
});

singleUploadBox.addEventListener('click', () => singleFileInput.click());
singleFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleSingleFile(e.target.files[0]);
});

function handleSingleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        resultContainer.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// =======================
// 🤖 Single File Analysis
// =======================
analyzeBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    const file = singleFileInput.files[0];
    if (!file) return alert('Please upload a file first.');

    loadingSpinner.style.display = 'flex';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('https://pylord-layout.hf.space/predict/single', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        annotatedImage.src = `data:image/png;base64,${data.annotated_image_base64}`;
        jsonViewer.textContent = JSON.stringify(data.annotations, null, 2);
        resultContainer.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        alert(`An error occurred during analysis: ${error.message}`);
    } finally {
        loadingSpinner.style.display = 'none';
    }
});

// 💾 Download Annotated Image
downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.href = annotatedImage.src;
    link.download = 'annotated_image.png';
    link.click();
});

// =======================
// 📦 Bulk Upload Handling
// =======================
const bulkUploadBox = document.getElementById('bulkUploadBox');
const bulkFileInput = document.getElementById('bulkFileInput');
const fileList = document.getElementById('fileList');
const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
const bulkResults = document.getElementById('bulkResults');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');

bulkUploadBox.addEventListener('click', () => bulkFileInput.click());

bulkFileInput.addEventListener('change', (e) => {
    fileList.innerHTML = '';
    bulkResults.innerHTML = '';
    bulkResults.style.display = 'none';
    
    Array.from(e.target.files).forEach(file => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = file.name;
        fileList.appendChild(li);
    });
    batchAnalyzeBtn.style.display = e.target.files.length > 0 ? 'block' : 'none';
});

// =======================
// 🚀 NEW APPROACH: Process Files One by One
// =======================
batchAnalyzeBtn.addEventListener('click', async () => {
    const files = Array.from(bulkFileInput.files);
    if (files.length === 0) {
        alert('Please select at least one file.');
        return;
    }

    // Show progress
    document.getElementById('bulkProgress').style.display = 'block';
    bulkResults.innerHTML = '';
    bulkResults.style.display = 'block';
    batchAnalyzeBtn.disabled = true;

    const results = [];
    let completed = 0;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        progressText.textContent = `Processing ${i + 1} of ${files.length}: ${file.name}`;
        progressBar.style.width = `${((i) / files.length) * 100}%`;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('https://pylord-layout.hf.space/predict/single', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Failed to process ${file.name}`);

            const data = await response.json();
            results.push({
                filename: file.name,
                success: true,
                data: data
            });

            // Add result card
            addResultCard(file.name, data, true);
            completed++;

        } catch (error) {
            console.error(`Error processing ${file.name}:`, error);
            results.push({
                filename: file.name,
                success: false,
                error: error.message
            });
            addResultCard(file.name, null, false, error.message);
        }

        progressBar.style.width = `${((i + 1) / files.length) * 100}%`;
    }

    progressText.textContent = `✅ Completed! Processed ${completed} of ${files.length} files`;
    batchAnalyzeBtn.disabled = false;

    // Add download all button
    if (completed > 0) {
        addDownloadAllButton(results.filter(r => r.success));
    }
});

// Add result card to display
function addResultCard(filename, data, success, errorMsg = '') {
    const card = document.createElement('div');
    card.className = 'result-card mb-3';
    
    if (success) {
        card.innerHTML = `
            <div class="card">
                <div class="card-header bg-success text-white">
                    <i class="fas fa-check-circle"></i> ${filename}
                </div>
                <div class="card-body">
                    <img src="data:image/png;base64,${data.annotated_image_base64}" 
                         class="img-fluid mb-2" 
                         style="max-height: 300px; object-fit: contain;">
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-primary download-image" data-filename="${filename}">
                            <i class="fas fa-download"></i> Image
                        </button>
                        <button class="btn btn-sm btn-secondary download-json" data-filename="${filename}">
                            <i class="fas fa-file-code"></i> JSON
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Add download handlers
        const imgBtn = card.querySelector('.download-image');
        const jsonBtn = card.querySelector('.download-json');

        imgBtn.addEventListener('click', () => {
            const link = document.createElement('a');
            link.href = `data:image/png;base64,${data.annotated_image_base64}`;
            link.download = `${filename.split('.')[0]}_annotated.png`;
            link.click();
        });

        jsonBtn.addEventListener('click', () => {
            const json = JSON.stringify(data.annotations, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${filename.split('.')[0]}_annotations.json`;
            link.click();
            URL.revokeObjectURL(url);
        });

    } else {
        card.innerHTML = `
            <div class="card border-danger">
                <div class="card-header bg-danger text-white">
                    <i class="fas fa-times-circle"></i> ${filename}
                </div>
                <div class="card-body">
                    <p class="text-danger mb-0">Error: ${errorMsg}</p>
                </div>
            </div>
        `;
    }

    bulkResults.appendChild(card);
}

// Add download all button
function addDownloadAllButton(successResults) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-success btn-lg mt-3';
    btn.innerHTML = '<i class="fas fa-download"></i> Download All Results';
    
    btn.addEventListener('click', () => {
        successResults.forEach(result => {
            // Download image
            const imgLink = document.createElement('a');
            imgLink.href = `data:image/png;base64,${result.data.annotated_image_base64}`;
            imgLink.download = `${result.filename.split('.')[0]}_annotated.png`;
            imgLink.click();

            // Download JSON
            setTimeout(() => {
                const json = JSON.stringify(result.data.annotations, null, 2);
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const jsonLink = document.createElement('a');
                jsonLink.href = url;
                jsonLink.download = `${result.filename.split('.')[0]}_annotations.json`;
                jsonLink.click();
                URL.revokeObjectURL(url);
            }, 200);
        });
    });

    bulkResults.appendChild(btn);
}

// =======================
// 🧭 Smooth Scroll
// =======================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});