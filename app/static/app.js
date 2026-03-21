// ── Helpers ──────────────────────────────────────────────────────────────

function appendMessage(text, type) {
  const chat = document.getElementById("chat-window");
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function appendHTML(html, type) {
  const chat = document.getElementById("chat-window");
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function showTyping() {
  const chat = document.getElementById("chat-window");
  const indicator = document.createElement("div");
  indicator.className = "typing-indicator";
  indicator.id = "typingIndicator";
  indicator.innerHTML = "<span></span><span></span><span></span>";
  chat.appendChild(indicator);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

// ── Upload ───────────────────────────────────────────────────────────────
async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const btn = document.getElementById("uploadBtn");
  const status = document.getElementById("uploadStatus");

  const file = fileInput.files[0];

  if (!file) {
    status.innerHTML = `<span class="text-warning">⚠️ Please select a PDF file first.</span>`;
    return;
  }

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    status.innerHTML = `<span class="text-danger">❌ Only PDF files are supported.</span>`;
    return;
  }

  btn.disabled = true;
  status.innerHTML = `<span class="text-info"><span class="spinner-border spinner-border-sm me-1"></span> Uploading...</span>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (response.status === 202 && result.job_id) {
      // Background processing started — poll for completion
      status.innerHTML = `<span class="text-info"><span class="spinner-border spinner-border-sm me-1"></span> Analyzing document...</span>`;
      pollUploadStatus(result.job_id, file.name, btn, status);
    } else if (result.error) {
      status.innerHTML = `<span class="text-danger">❌ ${result.error}</span>`;
      btn.disabled = false;
    } else {
      status.innerHTML = `<span class="text-danger">❌ Upload failed. Please try again.</span>`;
      btn.disabled = false;
    }
  } catch (err) {
    status.innerHTML = `<span class="text-danger">❌ Network error. Please try again.</span>`;
    btn.disabled = false;
  }
}

function pollUploadStatus(jobId, fileName, btn, status) {
  const interval = setInterval(async () => {
    try {
      const response = await fetch("/upload_status/" + jobId);
      const data = await response.json();

      if (data.status === "done") {
        clearInterval(interval);
        status.innerHTML = `<span class="text-success">${data.message}</span>`;
        appendHTML(
          `📄 <strong>${fileName}</strong> has been analyzed. What would you like to know?`,
          "ai-msg",
        );
        btn.disabled = false;
      } else if (data.status === "error") {
        clearInterval(interval);
        status.innerHTML = `<span class="text-danger">❌ ${data.message}</span>`;
        btn.disabled = false;
      }
      // if "processing" — keep polling
    } catch (err) {
      clearInterval(interval);
      status.innerHTML = `<span class="text-danger">❌ Status check failed. Please try again.</span>`;
      btn.disabled = false;
    }
  }, 3000); // poll every 3 seconds
}

// ── Chat ─────────────────────────────────────────────────────────────────

async function askQuestion() {
  const input = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const question = input.value.trim();
  if (!question) return;

  appendMessage(question, "user-msg");
  input.value = "";
  sendBtn.disabled = true;
  showTyping();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // FIX #6: Always sends valid JSON
      body: JSON.stringify({ question }),
    });

    removeTyping();
    const result = await response.json();

    if (result.answer) {
      appendMessage(result.answer, "ai-msg");
    } else {
      appendMessage("Something went wrong. Please try again.", "ai-msg error");
    }
  } catch (err) {
    removeTyping();
    appendMessage(
      "Network error. Please check your connection and try again.",
      "ai-msg error",
    );
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

// ── New Chat ──────────────────────────────────────────────────────────────
// FIX #13: This function was missing entirely — now wired to the button

async function newChat() {
  const btn = document.getElementById("newChatBtn");
  btn.disabled = true;

  try {
    const response = await fetch("/new_chat", { method: "POST" });
    const result = await response.json();

    // Clear the UI
    const chat = document.getElementById("chat-window");
    chat.innerHTML = `<div class="ai-msg msg">🆕 Session cleared. Upload a new document to begin.</div>`;

    // Reset sidebar
    document.getElementById("fileInput").value = "";
    document.getElementById("uploadStatus").innerHTML = "";
  } catch (err) {
    appendMessage(
      "Could not clear session. Please refresh the page.",
      "ai-msg error",
    );
  } finally {
    btn.disabled = false;
  }
}
