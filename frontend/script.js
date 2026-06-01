let currentMode = "rag";

// ── MODE SELECTION ─────────────────────────────────────
function setMode(mode, btn) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
}

// ── QUICK SEND ─────────────────────────────────────────
function sendQuick(text) {
    document.getElementById("user-input").value = text;
    sendMessage();
}

// ── CLEAR CHAT ─────────────────────────────────────────
function clearChat() {
    const messages = document.getElementById("messages");
    messages.innerHTML = `
        <div class="message bot-message">
            <div class="avatar bot-avatar">🤖</div>
            <div class="bubble bot-bubble">
                <p>Chat cleared! Ask me anything about placement records from 2023–2025.</p>
                <div class="bubble-time">Just now</div>
            </div>
        </div>`;
}

// ── TIME HELPER ────────────────────────────────────────
function getTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ── FORMAT BOT ANSWER ──────────────────────────────────
function formatAnswer(text) {
    // Detect numbered list like "1. Company\n2. Company"
    const lines = text.split("\n").filter(l => l.trim());

    const isNumberedList = lines.filter(l => /^\d+\.\s/.test(l.trim())).length > 3;

    if (isNumberedList) {
        const intro = lines.filter(l => !/^\d+\.\s/.test(l.trim())).join("<br>");
        const items = lines
            .filter(l => /^\d+\.\s/.test(l.trim()))
            .map(l => `<div class="company-item">${l.trim()}</div>`)
            .join("");
        return `<p>${intro}</p><div class="company-list">${items}</div>`;
    }

    // Normal text — replace newlines with <br>
    return "<p>" + lines.join("</p><p>") + "</p>";
}

// ── MAIN SEND ──────────────────────────────────────────
async function sendMessage() {
    const input   = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const messages = document.getElementById("messages");
    const typing  = document.getElementById("typing");

    const question = input.value.trim();
    if (!question) return;

    // Add user bubble
    messages.innerHTML += `
        <div class="message user-message">
            <div class="avatar user-avatar">U</div>
            <div class="bubble user-bubble">
                ${question}
                <div class="bubble-time">${getTime()}</div>
            </div>
        </div>`;

    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    typing.style.display = "flex";
    messages.scrollTop = messages.scrollHeight;

    try {
        const response = await fetch("http://127.0.0.1:8000/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, mode: currentMode })
        });

        const data = await response.json();
        const formatted = formatAnswer(data.answer || "No response received.");

        messages.innerHTML += `
            <div class="message bot-message">
                <div class="avatar bot-avatar">🤖</div>
                <div class="bubble bot-bubble">
                    ${formatted}
                    <div class="bubble-time">${getTime()} · ${currentMode === "rag" ? "RAG" : "Zero Shot"}</div>
                </div>
            </div>`;

    } catch (err) {
        messages.innerHTML += `
            <div class="message bot-message">
                <div class="avatar bot-avatar">🤖</div>
                <div class="bubble bot-bubble" style="border-color:#ef4444;">
                    <p>⚠️ Could not connect to backend. Make sure <code>app.py</code> is running.</p>
                    <div class="bubble-time">${getTime()}</div>
                </div>
            </div>`;
    }

    typing.style.display = "none";
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
}

// ── ENTER KEY ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("user-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

async function handleResumeUpload() {
    const fileInput = document.getElementById('resumeFile');
    const statusDiv = document.getElementById('resumeStatus');
    const messagesContainer = document.getElementById('messages');
    
    if (!fileInput.files || fileInput.files.length === 0) return;
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading text
    statusDiv.style.display = "block";
    statusDiv.innerText = "Analyzing resume keywords...";
    
    try {
        const response = await fetch("http://127.0.0.1:8000/api/upload-resume", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        statusDiv.style.display = "none";
        
        if (data.success && data.suggestions.length > 0) {
            // Build a clean message list displaying matching results
            let resultsHtml = `<p>📊 <strong>Resume Analysis Complete!</strong></p>`;
            resultsHtml += `<p>Based on your profile skills and past recruitment records (2023-2025), here are the top companies you are highly suitable for:</p><ol style="margin-top: 8px; padding-left: 16px;">`;
            
            data.suggestions.forEach(item => {
                resultsHtml += `<li style="margin-bottom: 4px;"><strong>${item.company}</strong></li>`;
            });
            resultsHtml += `</ol><p style="margin-top: 8px; font-size: 13px; color: #8e95b0;">Try asking PlaceBot about deadlines or interview requirements for these corporations!</p>`;
            
            // Push the result into the chat container
            messagesContainer.innerHTML += `
                <div class="message bot-message">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="bubble bot-bubble">
                        ${resultsHtml}
                        <div class="bubble-time">Just now · Profile Match</div>
                    </div>
                </div>`;
        } else {
            const msg = data.message || "We couldn't find explicit historical overlap matches for keywords in this specific document configuration.";
            messagesContainer.innerHTML += `
                <div class="message bot-message">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="bubble bot-bubble" style="border-color:#ef4444;">
                        <p>⚠️ ${msg}</p>
                    </div>
                </div>`;
        }
    } catch (err) {
        statusDiv.style.display = "none";
        alert("Could not connect to the backend server. Please make sure app.py is actively running.");
    }
    
    // Clear the input so you can re-upload if needed
    fileInput.value = "";
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}