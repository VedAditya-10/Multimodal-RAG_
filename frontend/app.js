/**
 * CHAKRAVYUH - Frontend Application
 * Multimodal RAG with Universal Evidence Citing
 */

const API_URL = 'http://localhost:8000';

// Global state
let currentCitations = [];
let currentResponse = null; // Store last query response for export
let currentQueryId = null;

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');
const evidencePanel = document.getElementById('evidence-panel');
const evidenceList = document.getElementById('evidence-list');
const viewerModal = document.getElementById('viewer-modal');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');
const closeModal = document.getElementById('close-modal');
const closePanel = document.getElementById('close-panel');
const uploadProgress = document.getElementById('upload-progress');
const progressText = document.getElementById('progress-text');

// Event Listeners
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);
fileInput.addEventListener('change', handleFileUpload);
closeModal.addEventListener('click', () => viewerModal.classList.add('hidden'));
closePanel.addEventListener('click', () => evidencePanel.classList.add('hidden'));

// Click outside modal to close
viewerModal.addEventListener('click', (e) => {
    if (e.target === viewerModal) {
        viewerModal.classList.add('hidden');
    }
});

/**
 * Send a query message
 */
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Clear input
    chatInput.value = '';

    // Add user message
    addMessage(query, 'user');

    // Show loading
    const loadingId = addMessage('Searching evidence...', 'assistant', true);

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, max_results: 5 })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);

        // Display response
        displayResponse(data);

    } catch (error) {
        console.error('Query error:', error);
        removeMessage(loadingId);
        addMessage('Error: Failed to process query. Please try again.', 'assistant');
    }
}

/**
 * Display query response with citations
 */
function displayResponse(data) {
    const { answer, confidence, citations, conflicts, refused, refusal_reason } = data;

    // Store response for export
    currentResponse = data;

    // Store citations for reference
    currentCitations = citations;

    // Build answer with citation badges
    let formattedAnswer = answer;
    citations.forEach((_, i) => {
        const badge = `<span class="citation-badge" onclick="viewCitation(${i})">[${i + 1}]</span>`;
        formattedAnswer = formattedAnswer.replace(`[${i + 1}]`, badge);
    });

    // Build confidence badge
    let confidenceClass = 'confidence-high';
    if (confidence < 0.6) confidenceClass = 'confidence-low';
    else if (confidence < 0.8) confidenceClass = 'confidence-medium';

    // Build message HTML
    let html = `<div class="message-content">${formattedAnswer}</div>`;
    html += `<span class="confidence-badge ${confidenceClass}">Confidence: ${(confidence * 100).toFixed(0)}%</span>`;

    // Add conflict warning if any
    if (conflicts && conflicts.length > 0) {
        html += `<div class="conflict-warning">
            <strong>⚠️ Conflict Detected:</strong> 
            ${conflicts.map(c => c.reason || 'Sources contain contradictory information').join('; ')}
        </div>`;
    }

    // Add refusal notice if applicable
    if (refused) {
        html = `<div class="message-content" style="color: var(--warning)">
            ⚠️ ${answer}
        </div>`;
    }

    addMessage(html, 'assistant', false, true);

    // Update evidence panel
    updateEvidencePanel(citations, conflicts);
}

/**
 * Add message to chat
 */
function addMessage(content, role, loading = false, isHtml = false) {
    // Remove welcome message if exists
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const id = 'msg-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = `message ${role}${loading ? ' loading' : ''}`;
    
    if (isHtml) {
        div.innerHTML = content;
    } else {
        div.innerHTML = `<div class="message-content">${escapeHtml(content)}</div>`;
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return id;
}

/**
 * Remove message by ID
 */
function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

/**
 * Update evidence panel with citations
 */
function updateEvidencePanel(citations, conflicts) {
    evidencePanel.classList.remove('hidden');

    if (!citations.length) {
        evidenceList.innerHTML = '<p class="placeholder">No citations found.</p>';
        return;
    }

    // Get conflict IDs for marking
    const conflictIds = new Set();
    conflicts.forEach(c => {
        conflictIds.add(c.source_a);
        conflictIds.add(c.source_b);
    });

    evidenceList.innerHTML = citations.map((citation, i) => {
        const hasConflict = citation.conflicts_with.length > 0;
        const icon = getModalityIcon(citation.modality);
        const location = formatLocation(citation.location, citation.modality);
        const confPercent = (citation.confidence * 100).toFixed(0);

        return `
            <div class="evidence-card ${hasConflict ? 'conflict' : ''}">
                <div class="evidence-card-header">
                    <div class="evidence-modality">
                        <span class="evidence-modality-icon">${icon}</span>
                        <span>${citation.source_file}</span>
                    </div>
                    <span class="evidence-confidence ${getConfidenceClass(citation.confidence)}">
                        ${confPercent}%
                    </span>
                </div>
                <div class="evidence-card-body">
                    <p class="evidence-snippet">${escapeHtml(citation.text_snippet || 'Visual content')}</p>
                    <p class="evidence-location">${location}</p>
                </div>
                <button class="view-source-btn" onclick="viewCitation(${i})">
                    ${hasConflict ? '⚠️ ' : ''}View Source
                </button>
            </div>
        `;
    }).join('');
}

/**
 * View a citation in the modal
 */
async function viewCitation(index) {
    const citation = currentCitations[index];
    if (!citation) return;

    modalTitle.textContent = `${getModalityIcon(citation.modality)} ${citation.source_file}`;
    modalBody.innerHTML = '<div class="spinner"></div>';
    viewerModal.classList.remove('hidden');

    try {
        // Fetch evidence details
        const response = await fetch(`${API_URL}/evidence/${citation.chunk_id}`);
        if (!response.ok) throw new Error('Failed to load evidence');

        const evidence = await response.json();
        
        // Render based on modality
        renderViewer(evidence, citation);

    } catch (error) {
        console.error('View error:', error);
        modalBody.innerHTML = `<p style="color: var(--error)">Failed to load evidence: ${error.message}</p>`;
    }
}

/**
 * Render the appropriate viewer based on modality
 */
function renderViewer(evidence, citation) {
    const { modality, content_url, location, text_content } = evidence;

    switch (modality) {
        case 'text':
        case 'ocr':
            renderTextViewer(text_content, location);
            break;

        case 'image':
        case 'video_frame':
            renderImageViewer(content_url, location);
            break;

        case 'audio_transcript':
            renderAudioPlayer(content_url, location, text_content);
            break;

        case 'pdf':
        default:
            if (content_url.endsWith('.pdf')) {
                renderPDFViewer(content_url, location);
            } else {
                renderTextViewer(text_content, location);
            }
    }
}

/**
 * Render PDF viewer using PDF.js
 */
async function renderPDFViewer(url, location) {
    const page = location?.page || 1;
    
    modalBody.innerHTML = `
        <div class="pdf-container" id="pdf-container">
            <p>Loading PDF page ${page}...</p>
        </div>
    `;

    try {
        const fullUrl = `${API_URL}${url}`;
        const pdf = await pdfjsLib.getDocument(fullUrl).promise;
        const pageObj = await pdf.getPage(page);
        
        const scale = 1.5;
        const viewport = pageObj.getViewport({ scale });
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        
        await pageObj.render({
            canvasContext: context,
            viewport: viewport
        }).promise;

        const container = document.getElementById('pdf-container');
        container.innerHTML = '';
        container.appendChild(canvas);

        // Draw highlight if bbox provided
        if (location?.bbox) {
            drawHighlight(canvas, location.bbox, viewport);
        }

    } catch (error) {
        console.error('PDF render error:', error);
        modalBody.innerHTML = `<p style="color: var(--error)">Failed to render PDF</p>`;
    }
}

/**
 * Render text viewer with highlighted lines
 */
function renderTextViewer(text, location) {
    let displayText = text || 'No text content available';
    
    // Highlight relevant portion
    if (location?.line_start && location?.line_end) {
        displayText = `Lines ${location.line_start}-${location.line_end}:\n\n${displayText}`;
    }

    modalBody.innerHTML = `
        <div class="text-viewer">
            <pre class="text-highlight">${escapeHtml(displayText)}</pre>
        </div>
    `;
}

/**
 * Render image viewer with bbox overlay
 */
function renderImageViewer(url, location) {
    const fullUrl = `${API_URL}${url}`;
    
    modalBody.innerHTML = `
        <div class="image-viewer" id="image-container">
            <img src="${fullUrl}" alt="Evidence" onload="addBboxOverlay(this, ${JSON.stringify(location?.bbox || null)})">
        </div>
    `;
}

/**
 * Add bounding box overlay to image
 */
function addBboxOverlay(img, bbox) {
    if (!bbox || bbox.length !== 4) return;

    const container = img.parentElement;
    const [x1, y1, x2, y2] = bbox;

    const overlay = document.createElement('div');
    overlay.className = 'bbox-overlay';
    overlay.style.left = (x1 * 100) + '%';
    overlay.style.top = (y1 * 100) + '%';
    overlay.style.width = ((x2 - x1) * 100) + '%';
    overlay.style.height = ((y2 - y1) * 100) + '%';

    container.style.position = 'relative';
    container.appendChild(overlay);
}

/**
 * Render audio/video player with timestamp
 */
function renderAudioPlayer(url, location, transcript) {
    const fullUrl = `${API_URL}${url}`;
    const isVideo = url.includes('.mp4') || url.includes('.mkv') || url.includes('.avi');
    const startTime = location?.timestamp_start || 0;

    const mediaElement = isVideo ? 
        `<video controls id="media-player"><source src="${fullUrl}" type="video/mp4"></video>` :
        `<audio controls id="media-player"><source src="${fullUrl}" type="audio/mpeg"></audio>`;

    modalBody.innerHTML = `
        <div class="media-player">
            ${mediaElement}
            <div class="transcript-display">
                <h4>Transcript (${formatTimestamp(startTime)})</h4>
                <p>${escapeHtml(transcript || 'No transcript available')}</p>
            </div>
        </div>
    `;

    // Jump to timestamp
    const player = document.getElementById('media-player');
    player.addEventListener('loadedmetadata', () => {
        player.currentTime = startTime;
    });
}

/**
 * Handle file upload
 */
async function handleFileUpload(e) {
    const files = e.target.files;
    if (!files.length) return;

    uploadProgress.classList.remove('hidden');

    for (const file of files) {
        progressText.textContent = `Uploading ${file.name}...`;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_URL}/ingest`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const result = await response.json();
            addMessage(`✅ Indexed "${file.name}": ${result.chunks_created} chunks (${result.modalities.join(', ')})`, 'assistant');

        } catch (error) {
            console.error('Upload error:', error);
            addMessage(`❌ Failed to upload "${file.name}": ${error.message}`, 'assistant');
        }
    }

    uploadProgress.classList.add('hidden');
    fileInput.value = '';
}

// Load Obsidian settings on page load
window.addEventListener('DOMContentLoaded', () => {
    loadObsidianSettings();
});

/**
 * Load Obsidian settings from sessionStorage
 */
function loadObsidianSettings() {
    const apiKey = sessionStorage.getItem('obsidian_api_key') || '';
    const apiUrl = sessionStorage.getItem('obsidian_api_url') || 'http://localhost:27123';
    
    document.getElementById('obsidian-api-key').value = apiKey;
    document.getElementById('obsidian-api-url').value = apiUrl;
}

/**
 * Save Obsidian settings to sessionStorage
 */
function saveObsidianSettings() {
    const apiKey = document.getElementById('obsidian-api-key').value.trim();
    const apiUrl = document.getElementById('obsidian-api-url').value.trim();
    
    sessionStorage.setItem('obsidian_api_key', apiKey);
    sessionStorage.setItem('obsidian_api_url', apiUrl);
    
    // Visual feedback
    const saveBtn = event.target;
    const originalText = saveBtn.textContent;
    saveBtn.textContent = '✓ Saved!';
    saveBtn.style.background = '#10b981';
    
    setTimeout(() => {
        saveBtn.textContent = originalText;
        saveBtn.style.background = '#000000';
    }, 1500);
}

/**
 * Toggle password visibility
 */
function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    input.type = input.type === 'password' ? 'text' : 'password';
}

/**
 * Export conversation as markdown download
 */
async function exportAsMarkdown() {
    if (!currentResponse) {
        addMessage('❌ No query to export. Please ask a question first.', 'assistant');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/export/markdown`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResponse)
        });
        
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `TRACE_Query_${Date.now()}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        addMessage('✅ Downloaded markdown file', 'assistant');
        
    } catch (error) {
        console.error('Export error:', error);
        addMessage(`❌ Export failed: ${error.message}`, 'assistant');
    }
}

/**
 * Save conversation to Obsidian (uses sessionStorage credentials)
 */
async function saveToObsidian() {
    if (!currentResponse) {
        addMessage('❌ No query to export. Please ask a question first.', 'assistant');
        return;
    }
    
    // Get credentials from sessionStorage
    const apiKey = sessionStorage.getItem('obsidian_api_key') || '';
    const apiUrl = sessionStorage.getItem('obsidian_api_url') || 'http://localhost:27123';
    
    if (!apiKey) {
        addMessage('❌ Obsidian API key not configured. Please set it in the Obsidian settings panel.', 'assistant');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/export/obsidian`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...currentResponse,
                obsidian_api_key: apiKey,
                obsidian_api_url: apiUrl
            })
        });
        
        if (!response.ok) throw new Error('Export failed');
        
        const result = await response.json();
        
        if (result.status === 'success') {
            addMessage(`✅ ${result.message}`, 'assistant');
        } else {
            addMessage(`❌ ${result.message}`, 'assistant');
        }
        
    } catch (error) {
        console.error('Obsidian export error:', error);
        addMessage(`❌ Failed to save to Obsidian: ${error.message}`, 'assistant');
    }
}

// Utility Functions

function getModalityIcon(modality) {
    const icons = {
        'pdf': '📄',
        'text': '📝',
        'docx': '📝',
        'markdown': '📝',
        'image': '🖼️',
        'ocr': '🔍',
        'audio_transcript': '🎵',
        'video_frame': '🎬',
    };
    return icons[modality] || '📎';
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

function formatLocation(location, modality) {
    if (!location) return '';
    
    const parts = [];
    if (location.page) parts.push(`Page ${location.page}`);
    if (location.timestamp_start !== undefined) {
        parts.push(formatTimestamp(location.timestamp_start));
    }
    if (location.line_start) {
        parts.push(`Lines ${location.line_start}-${location.line_end || location.line_start}`);
    }
    
    return parts.join(' | ') || modality;
}

function formatTimestamp(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function drawHighlight(canvas, bbox, viewport) {
    if (!bbox || bbox.length !== 4) return;
    
    const ctx = canvas.getContext('2d');
    const [x1, y1, x2, y2] = bbox;
    
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 3;
    ctx.strokeRect(
        x1 * canvas.width,
        y1 * canvas.height,
        (x2 - x1) * canvas.width,
        (y2 - y1) * canvas.height
    );
}

// Initialize PDF.js worker
if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

console.log('CHAKRAVYUH initialized');
