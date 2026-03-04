const STORAGE_KEY = "ec3_chat_threads_v2";
const THINKING_MODE_KEY = "ec3_thinking_mode";

const GRAPH_NODES = [
  { id: "user",         label: "User",          icon: "person", col: 0, row: 1 },
  { id: "database",     label: "Database",      icon: "book",   col: 1, row: 0 },
  { id: "orchestrator", label: "Orchestrator",  icon: "brain",  col: 1, row: 1 },
  { id: "tools",        label: "Tools",         icon: "wrench", col: 1, row: 2 },
  { id: "response",     label: "Response",      icon: "check",  col: 2, row: 1 },
];

const GRAPH_EDGES = [
  { id: "u_o",  from: "user",         to: "orchestrator" },
  { id: "o_d",  from: "orchestrator", to: "database"     },
  { id: "o_t",  from: "orchestrator", to: "tools"        },
  { id: "o_r",  from: "orchestrator", to: "response"     },
];

const NODE_ICONS = {
  person: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
  brain: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c-1.8-1.4-4.8-1.2-6.5.4-1.8 1.7-2 4.8-.5 6.8-.9 1.3-.9 3.2.1 4.5.8 1 2.1 1.6 3.4 1.6.7 2 2.6 3.4 4.5 3.4"/><path d="M12 3c1.8-1.4 4.8-1.2 6.5.4 1.8 1.7 2 4.8.5 6.8.9 1.3.9 3.2-.1 4.5-.8 1-2.1 1.6-3.4 1.6-.7 2-2.6 3.4-4.5 3.4"/><path d="M12 3v16"/><path d="M8.2 7.2c.8-.8 2-.9 2.8-.1"/><path d="M15.8 7.2c-.8-.8-2-.9-2.8-.1"/><path d="M7.6 11c1-.7 2.3-.7 3.2.1"/><path d="M16.4 11c-1-.7-2.3-.7-3.2.1"/><path d="M8.6 14.7c.9-.4 1.7-.3 2.4.3"/><path d="M15.4 14.7c-.9-.4-1.7-.3-2.4.3"/></svg>`,
  book: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>`,
  wrench: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>`,
  citation: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const form = $("#chat-form");
const input = $("#prompt-input");
const thinkingModeSelect = $("#thinking-mode");
const thinkingModeTrigger = $("#thinking-mode-trigger");
const thinkingModeMenu = $("#thinking-mode-menu");
const thinkingModeLabel = $("#thinking-mode-label");
const sendBtn = $("#send-btn");
const messagesEl = $("#messages");
const template = $("#message-template");
const welcome = $("#welcome");
const threadList = $("#thread-list");
const newChatBtn = $("#new-chat-btn");
const chatSearch = $("#chat-search");
const devToggle = $("#dev-mode-toggle");
const devPanel = $("#dev-panel");
const sidebarToggle = $("#sidebar-toggle");
const sidebar = $("#sidebar");
const signInBtn = $("#signin-btn");
const registerBtn = $("#register-btn");
const sidebarSigninBtn = $("#sidebar-signin-btn");
const sidebarSignupBtn = $("#sidebar-signup-btn");
const attachTrigger = $("#attach-trigger");
const attachMenu = $("#attach-menu");
const photoInput = $("#photo-input");
const fileInput = $("#file-input");
const attachmentsPreview = $("#attachments-preview");


const state = {
  threads: [],
  activeThreadId: null,
  guestThread: null,
  filter: "",
  devMode: false,
  thinkingMode: "thinking",
  attachments: [],
};

function uid() {
  return crypto?.randomUUID?.() || `id_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}
function now() { return new Date().toISOString(); }
function emptyThread(title = "New chat") { return { id: uid(), title, createdAt: now(), updatedAt: now(), messages: [] }; }

function fmtTime(iso) {
  return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function truncTitle(v) {
  const c = (v || "").trim().replace(/\s+/g, " ");
  return !c ? "New chat" : c.length > 50 ? c.slice(0, 50) + "..." : c;
}

function clamp(v, max = 42) {
  const c = String(v || "").replace(/\s+/g, " ").trim();
  return c.length > max ? c.slice(0, max - 1) + "..." : c;
}

function renderMd(text) {
  if (typeof marked !== "undefined") {
    try { return marked.parse(text || ""); } catch { /* fall through */ }
  }
  return `<pre>${escHtml(text || "")}</pre>`;
}

function escHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ---- Attachments ----
function fmtFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function isImageFile(file) {
  return file.type.startsWith("image/");
}

function closeAttachMenu() {
  if (!attachMenu) return;
  attachMenu.classList.add("hidden");
  attachTrigger?.setAttribute("aria-expanded", "false");
  attachTrigger?.closest(".attach-wrap")?.classList.remove("open");
}

async function addAttachments(files) {
  for (const file of files) {
    if (state.attachments.length >= 10) break;
    const att = {
      id: uid(),
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      isImage: isImageFile(file),
      dataUrl: null,
    };
    if (att.isImage) {
      try { att.dataUrl = await readFileAsDataUrl(file); } catch { /* skip preview */ }
    }
    state.attachments.push(att);
  }
  renderAttachmentsPreview();
}

function removeAttachment(id) {
  state.attachments = state.attachments.filter(a => a.id !== id);
  renderAttachmentsPreview();
}

function clearAttachments() {
  state.attachments = [];
  renderAttachmentsPreview();
}

function renderAttachmentsPreview() {
  if (!attachmentsPreview) return;
  attachmentsPreview.innerHTML = "";
  if (!state.attachments.length) {
    attachmentsPreview.classList.add("hidden");
    return;
  }
  attachmentsPreview.classList.remove("hidden");
  for (const att of state.attachments) {
    const chip = document.createElement("div");
    chip.className = att.isImage ? "attachment-chip photo-chip" : "attachment-chip";

    if (att.isImage && att.dataUrl) {
      const img = document.createElement("img");
      img.src = att.dataUrl;
      img.alt = att.name;
      chip.appendChild(img);
    } else {
      const icon = document.createElement("span");
      icon.className = "att-icon";
      icon.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;
      const name = document.createElement("span");
      name.className = "att-name";
      name.textContent = att.name;
      name.title = att.name;
      const size = document.createElement("span");
      size.className = "att-size";
      size.textContent = fmtFileSize(att.size);
      chip.appendChild(icon);
      chip.appendChild(name);
      chip.appendChild(size);
    }

    const removeBtn = document.createElement("button");
    removeBtn.className = "att-remove";
    removeBtn.type = "button";
    removeBtn.innerHTML = "&times;";
    removeBtn.title = "Remove";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeAttachment(att.id);
    });
    chip.appendChild(removeBtn);

    attachmentsPreview.appendChild(chip);
  }
}

function buildAttachmentHtml(attachments) {
  if (!attachments || !attachments.length) return "";
  let html = '<div class="msg-attachments">';
  for (const att of attachments) {
    if (att.isImage && att.dataUrl) {
      html += `<div class="msg-photo-thumb" data-src="${escHtml(att.dataUrl)}"><img src="${escHtml(att.dataUrl)}" alt="${escHtml(att.name)}" /></div>`;
    } else {
      html += `<div class="msg-file-tag"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><span class="file-tag-name" title="${escHtml(att.name)}">${escHtml(att.name)}</span><span class="file-tag-size">${fmtFileSize(att.size)}</span></div>`;
    }
  }
  html += "</div>";
  return html;
}

function attachLightboxListeners(container) {
  const thumbs = container.querySelectorAll(".msg-photo-thumb");
  for (const thumb of thumbs) {
    thumb.addEventListener("click", () => {
      const src = thumb.dataset.src || thumb.querySelector("img")?.src;
      if (!src) return;
      const overlay = document.createElement("div");
      overlay.className = "photo-lightbox";
      const img = document.createElement("img");
      img.src = src;
      overlay.appendChild(img);
      overlay.addEventListener("click", () => overlay.remove());
      document.body.appendChild(overlay);
    });
  }
}

function isValidThinkingMode(mode) {
  return mode === "standard" || mode === "thinking" || mode === "extended";
}

function thinkingModeLabelText(mode) {
  if (mode === "standard") return "Standard";
  if (mode === "extended") return "Extended Thinking";
  return "Thinking";
}

function closeThinkingModeMenu() {
  if (!thinkingModeMenu) return;
  thinkingModeMenu.classList.add("hidden");
  thinkingModeTrigger?.setAttribute("aria-expanded", "false");
  thinkingModeTrigger?.closest(".thinking-mode-wrap")?.classList.remove("open");
}

function syncThinkingModeUi() {
  if (thinkingModeSelect && thinkingModeSelect.value !== state.thinkingMode) {
    thinkingModeSelect.value = state.thinkingMode;
  }
  if (thinkingModeLabel) {
    thinkingModeLabel.textContent = thinkingModeLabelText(state.thinkingMode);
  }
  for (const option of $$(".thinking-mode-option")) {
    const selected = option.dataset.mode === state.thinkingMode;
    option.classList.toggle("selected", selected);
    option.setAttribute("aria-selected", selected ? "true" : "false");
  }
}

function setThinkingModeDisabled(disabled) {
  if (thinkingModeSelect) thinkingModeSelect.disabled = disabled;
  if (thinkingModeTrigger) thinkingModeTrigger.disabled = disabled;
  for (const option of $$(".thinking-mode-option")) {
    option.disabled = disabled;
  }
  if (disabled) closeThinkingModeMenu();
}

function loadThinkingModePreference() {
  const saved = localStorage.getItem(THINKING_MODE_KEY);
  if (isValidThinkingMode(saved)) {
    state.thinkingMode = saved;
  }
  syncThinkingModeUi();
}

function setThinkingMode(mode) {
  state.thinkingMode = isValidThinkingMode(mode) ? mode : "thinking";
  localStorage.setItem(THINKING_MODE_KEY, state.thinkingMode);
  syncThinkingModeUi();
}

// ---- State persistence ----
function canUseStoredThreads() {
  if (!auth.ready) return false;
  return !!auth.user;
}

function storageKey() {
  if (!auth.user) return STORAGE_KEY;
  const identity = auth.user.user_id || auth.user.email || "user";
  return `${STORAGE_KEY}:${identity}`;
}

function resetThreadState() {
  state.threads = [];
  state.activeThreadId = null;
}

function resetGuestThread() {
  state.guestThread = emptyThread("Temporary chat");
}

function save() {
  if (!canUseStoredThreads()) return;
  if (auth.threadsSync) return;
  localStorage.setItem(storageKey(), JSON.stringify({ threads: state.threads, activeThreadId: state.activeThreadId }));
}

async function load() {
  resetThreadState();
  if (!canUseStoredThreads()) return;
  if (auth.threadsSync) {
    try {
      const res = await fetchWithAuth("/api/threads");
      if (!res.ok) return;
      const data = await res.json();
      const threads = (data.threads || []).map((t) => ({
        id: t.id,
        title: t.title || "New chat",
        createdAt: t.createdAt,
        updatedAt: t.updatedAt,
        messages: [],
      }));
      state.threads = threads;
      if (threads.length && !state.activeThreadId) {
        state.activeThreadId = threads[0].id;
      }
      if (state.activeThreadId) {
        const full = await loadThreadFromApi(state.activeThreadId);
        if (full) {
          const idx = state.threads.findIndex((t) => t.id === full.id);
          if (idx >= 0) state.threads[idx] = full;
        }
      }
    } catch {
      state.threads = [];
    }
    return;
  }
  try {
    const key = storageKey();
    const current = localStorage.getItem(key);
    const fallback = key !== STORAGE_KEY ? localStorage.getItem(STORAGE_KEY) : null;
    const raw = current || fallback;
    if (!raw) return;
    const p = JSON.parse(raw);
    if (p?.threads) { state.threads = p.threads; state.activeThreadId = p.activeThreadId || null; }
    if (!current && fallback && key !== STORAGE_KEY) {
      localStorage.setItem(key, fallback);
    }
  } catch { state.threads = []; state.activeThreadId = null; }
}

async function addMessageToApi(threadId, role, content, responsePayload = null) {
  const res = await fetchWithAuth(`/api/threads/${threadId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content, response_payload: responsePayload }),
  });
  return res.ok;
}

async function truncateThreadApi(threadId, keepCount, updatedContent = null) {
  const res = await fetchWithAuth(`/api/threads/${threadId}/truncate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keep_count: keepCount, updated_content: updatedContent }),
  });
  return res.ok;
}

async function updateThreadTitleApi(threadId, title) {
  const res = await fetchWithAuth(`/api/threads/${threadId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return res.ok;
}

async function loadThreadFromApi(threadId) {
  const res = await fetchWithAuth(`/api/threads/${threadId}`);
  if (!res.ok) return null;
  const t = await res.json();
  return {
    id: t.id,
    title: t.title || "New chat",
    createdAt: t.createdAt,
    updatedAt: t.updatedAt,
    messages: (t.messages || []).map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content || "",
      responsePayload: m.responsePayload,
      createdAt: m.createdAt,
    })),
  };
}

async function createThread(title = "New chat") {
  if (!canUseStoredThreads()) {
    state.guestThread = emptyThread("Temporary chat");
    return state.guestThread;
  }
  if (auth.threadsSync) {
    try {
      const res = await fetchWithAuth("/api/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title || "New chat" }),
      });
      if (!res.ok) throw new Error("Failed to create thread");
      const t = await res.json();
      const thread = {
        id: t.id,
        title: t.title || "New chat",
        createdAt: t.createdAt,
        updatedAt: t.updatedAt,
        messages: [],
      };
      state.threads.unshift(thread);
      state.activeThreadId = thread.id;
      return thread;
    } catch {
      const t = emptyThread(title);
      state.threads.unshift(t);
      state.activeThreadId = t.id;
      save();
      return t;
    }
  }
  const t = emptyThread(title);
  state.threads.unshift(t);
  state.activeThreadId = t.id;
  save();
  return t;
}

function activeThread() { return state.threads.find((t) => t.id === state.activeThreadId) || null; }
function currentThread() { return canUseStoredThreads() ? activeThread() : state.guestThread; }
async function ensureThread() { return currentThread() || await createThread(canUseStoredThreads() ? "New chat" : "Temporary chat"); }

async function setActive(id) {
  if (!canUseStoredThreads()) return;
  if (!state.threads.some((t) => t.id === id)) return;
  state.activeThreadId = id;
  if (auth.threadsSync) {
    const full = await loadThreadFromApi(id);
    if (full) {
      const idx = state.threads.findIndex((t) => t.id === id);
      if (idx >= 0) state.threads[idx] = full;
    }
  } else {
    save();
  }
  renderThreadList();
  renderMessages();
}

function updateWelcome() {
  const a = currentThread();
  const hasMessages = !!(a && a.messages.length);
  welcome.classList.toggle("hidden", hasMessages);
  document.body.classList.toggle("chat-started", hasMessages);
  document.body.classList.toggle("pre-chat", !hasMessages);
}

// ---- Thread list ----
function renderThreadList() {
  const chatsSection = $("#sidebar-chats-section");
  const ctaSection = $("#sidebar-chats-cta");
  const authDisabledSection = $("#sidebar-auth-disabled");

  if (!auth.ready) {
    threadList.innerHTML = '<li class="empty-threads">Loading...</li>';
    if (chatsSection) chatsSection.classList.remove("hidden");
    if (ctaSection) ctaSection.classList.add("hidden");
    if (authDisabledSection) authDisabledSection.classList.add("hidden");
    return;
  }

  if (!auth.enabled) {
    if (chatsSection) chatsSection.classList.add("hidden");
    if (ctaSection) ctaSection.classList.add("hidden");
    if (authDisabledSection) authDisabledSection.classList.remove("hidden");
    threadList.innerHTML = "";
    return;
  }

  if (!canUseStoredThreads()) {
    if (chatsSection) chatsSection.classList.add("hidden");
    if (ctaSection) ctaSection.classList.remove("hidden");
    if (authDisabledSection) authDisabledSection.classList.add("hidden");
    threadList.innerHTML = "";
    return;
  }

  if (chatsSection) chatsSection.classList.remove("hidden");
  if (ctaSection) ctaSection.classList.add("hidden");
  if (authDisabledSection) authDisabledSection.classList.add("hidden");

  threadList.innerHTML = "";
  const f = state.filter.trim().toLowerCase();
  const vis = state.threads.filter(t => !f || t.title.toLowerCase().includes(f));
  if (!vis.length) {
    threadList.innerHTML = '<li class="empty-threads">No chats yet</li>';
    return;
  }
  for (const t of vis) {
    const li = document.createElement("li");
    li.className = "thread-item" + (t.id === state.activeThreadId ? " active" : "");
    li.innerHTML = `<span class="thread-title">${escHtml(t.title || "New chat")}</span><span class="thread-meta">${fmtTime(t.updatedAt || t.createdAt)}</span>`;
    li.addEventListener("click", () => setActive(t.id));
    threadList.appendChild(li);
  }
}

// ---- Flow graph ----
const GRID = {
  colW: 186,
  rowH: 112,
  padX: 28,
  padY: 44,
  nodeW: 156,
  nodeH: 62,
  popupGap: 10,
  popupMaxRows: 2,
  popupMaxItems: 4,
  popupRowH: 22,
  popupRowGap: 6,
  edgePadY: 12,
};

function popupLaneReserve() {
  return GRID.edgePadY
    + GRID.popupGap
    + (GRID.popupMaxRows * GRID.popupRowH)
    + ((GRID.popupMaxRows - 1) * GRID.popupRowGap);
}

function getGraphLayout() {
  const minCol = Math.min(...GRAPH_NODES.map(n => n.col));
  const maxCol = Math.max(...GRAPH_NODES.map(n => n.col));
  const minRow = Math.min(...GRAPH_NODES.map(n => n.row));
  const maxRow = Math.max(...GRAPH_NODES.map(n => n.row));
  const popupReserveY = popupLaneReserve();

  const originX = GRID.padX - minCol * GRID.colW;
  const originY = GRID.padY + popupReserveY - minRow * GRID.rowH;
  const spanCols = maxCol - minCol;
  const spanRows = maxRow - minRow;

  return {
    originX,
    originY,
    totalW: originX + spanCols * GRID.colW + GRID.nodeW + GRID.padX,
    totalH: originY + spanRows * GRID.rowH + GRID.nodeH + GRID.padY + popupReserveY,
  };
}

function getNodePos(node, layout = { originX: GRID.padX, originY: GRID.padY }) {
  return {
    x: layout.originX + node.col * GRID.colW,
    y: layout.originY + node.row * GRID.rowH,
    cx: layout.originX + node.col * GRID.colW + GRID.nodeW / 2,
    cy: layout.originY + node.row * GRID.rowH + GRID.nodeH / 2,
  };
}

function edgePath(fromNode, toNode, layout) {
  const f = getNodePos(fromNode, layout);
  const t = getNodePos(toNode, layout);

  if (fromNode.col === toNode.col) {
    const midX = f.cx;
    const startY = fromNode.row < toNode.row ? f.y + GRID.nodeH : f.y;
    const endY = fromNode.row < toNode.row ? t.y : t.y + GRID.nodeH;
    return `M ${midX} ${startY} L ${midX} ${endY}`;
  }

  const rightward = fromNode.col < toNode.col;
  const startX = rightward ? f.x + GRID.nodeW : f.x;
  const endX = rightward ? t.x : t.x + GRID.nodeW;
  const startY = f.cy;
  const endY = t.cy;
  const dx = endX - startX;
  return `M ${startX} ${startY} C ${startX + dx * 0.5} ${startY} ${endX - dx * 0.5} ${endY} ${endX} ${endY}`;
}

function initFlowGraph(msgNode, prompt) {
  const block = msgNode.querySelector(".thinking-block");
  const graph = msgNode.querySelector(".flow-graph");
  if (!block || !graph) return;
  block.classList.remove("hidden");
  setThinkingState(msgNode, true);
  graph.innerHTML = "";

  const canvas = document.createElement("div");
  canvas.className = "flow-canvas";

  const layout = getGraphLayout();
  const totalW = layout.totalW;
  const totalH = layout.totalH;
  canvas.style.width = totalW + "px";
  canvas.style.height = totalH + "px";

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("class", "flow-svg");
  svg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);

  const nodeMap = Object.fromEntries(GRAPH_NODES.map(n => [n.id, n]));
  for (const e of GRAPH_EDGES) {
    const fromN = nodeMap[e.from], toN = nodeMap[e.to];
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", edgePath(fromN, toN, layout));
    path.setAttribute("class", "flow-edge idle");
    path.dataset.edge = e.id;

    const markerId = `arrow-${e.id}`;
    const defs = svg.querySelector("defs") || svg.insertBefore(document.createElementNS(svgNS, "defs"), svg.firstChild);
    const marker = document.createElementNS(svgNS, "marker");
    marker.setAttribute("id", markerId);
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("orient", "auto-start-reverse");
    const arrow = document.createElementNS(svgNS, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    arrow.setAttribute("fill", "currentColor");
    arrow.setAttribute("class", "flow-arrow");
    marker.appendChild(arrow);
    defs.appendChild(marker);
    path.setAttribute("marker-end", `url(#${markerId})`);

    svg.appendChild(path);
  }
  canvas.appendChild(svg);

  const nodeEls = {};
  for (const n of GRAPH_NODES) {
    const pos = getNodePos(n, layout);
    const el = document.createElement("div");
    el.className = "flow-node idle";
    el.dataset.node = n.id;
    el.style.left = pos.x + "px";
    el.style.top = pos.y + "px";
    el.style.width = GRID.nodeW + "px";
    el.style.height = GRID.nodeH + "px";

    const icon = document.createElement("div");
    icon.className = "flow-node-icon";
    icon.innerHTML = NODE_ICONS[n.icon] || "";
    const title = document.createElement("div");
    title.className = "flow-node-title";
    title.textContent = n.label;
    el.appendChild(icon);
    el.appendChild(title);

    canvas.appendChild(el);
    nodeEls[n.id] = el;
  }

  const docPos = getNodePos(nodeMap.database, layout);
  const toolPos = getNodePos(nodeMap.tools, layout);

  const docPopups = document.createElement("div");
  docPopups.className = "flow-popups above";
  docPopups.style.left = docPos.cx + "px";
  docPopups.style.top = (docPos.y - GRID.popupGap) + "px";
  canvas.appendChild(docPopups);

  const toolPopups = document.createElement("div");
  toolPopups.className = "flow-popups below";
  toolPopups.style.left = toolPos.cx + "px";
  toolPopups.style.top = (toolPos.y + GRID.nodeH + GRID.popupGap) + "px";
  canvas.appendChild(toolPopups);

  graph.appendChild(canvas);

  const ns = {}, es = {};
  GRAPH_NODES.forEach(n => { ns[n.id] = "idle"; });
  GRAPH_EDGES.forEach(e => { es[e.id] = "idle"; });

  ns["user"] = "done";
  es["u_o"] = "active";

  msgNode.__flow = {
    ns,
    es,
    nodeEls,
    popupLanes: { docs: docPopups, tools: toolPopups },
    popupRefs: { docs: new Map(), tools: new Map() },
  };
  msgNode.__thinkStart = Date.now();
  msgNode.__stepCount = 0;
  applyFlow(msgNode);
}

function setNS(f, id, s) {
  const c = f.ns[id] || "idle";
  if (c === "error" && s !== "error") return;
  f.ns[id] = s;
}
function setES(f, id, s) {
  const c = f.es[id] || "idle";
  if (c === "error" && s !== "error") return;
  f.es[id] = s;
}

function triggerPopupPulse(chip) {
  if (!chip) return;
  chip.classList.remove("pulse");
  void chip.offsetWidth;
  chip.classList.add("pulse");
}

function addPopupChip(f, lane, key, label) {
  if (!lane || !key || !label) return;
  const laneEl = f.popupLanes?.[lane];
  const refs = f.popupRefs?.[lane];
  if (!laneEl || !refs) return;

  const existing = refs.get(key);
  if (existing) {
    triggerPopupPulse(existing);
    return;
  }

  const chip = document.createElement("div");
  chip.className = "flow-popup-chip";
  chip.textContent = clamp(label, 48);
  laneEl.appendChild(chip);
  refs.set(key, chip);
  requestAnimationFrame(() => chip.classList.add("show"));
  triggerPopupPulse(chip);

  while (laneEl.children.length > GRID.popupMaxItems) {
    const first = laneEl.firstElementChild;
    if (!first) break;
    laneEl.removeChild(first);
    for (const [k, el] of refs.entries()) {
      if (el === first) refs.delete(k);
    }
  }
}

function formatDocBadge(entry) {
  if (!entry || typeof entry !== "object") return "";
  const rawDoc = String(entry.doc_id || "").trim();
  const file = rawDoc ? `${rawDoc.replace(/\.json$/i, "")}.json` : "document.json";
  const clause = String(entry.clause_id || "").trim();
  if (!clause) return file;
  const prefix = /^\d/.test(clause) ? "Cl. " : "";
  return `${file} · ${prefix}${clause}`;
}

function pushDocBadges(f, entries) {
  if (!Array.isArray(entries)) return;
  for (const entry of entries) {
    const label = formatDocBadge(entry);
    if (!label) continue;
    const key = `${entry.doc_id || "unknown"}:${entry.clause_id || "?"}`;
    addPopupChip(f, "docs", key, label);
  }
}

function pushToolBadge(f, tool) {
  const raw = normTool(tool);
  if (!raw) return;
  addPopupChip(f, "tools", raw.toLowerCase(), raw);
}

function applyFlow(n) {
  const f = n.__flow;
  if (!f) return;
  for (const nd of GRAPH_NODES) {
    const el = f.nodeEls[nd.id], s = f.ns[nd.id] || "idle";
    el.classList.remove("idle", "active", "done", "error");
    el.classList.add(s);
  }
  n.querySelectorAll(".flow-edge").forEach(e => {
    const s = f.es[e.dataset.edge] || "idle";
    e.classList.remove("idle", "active", "done", "error");
    e.classList.add(s);
  });
}

function normTool(n) { return String(n || "").replace(/_ec3/g, "").replace(/_/g, " ").trim(); }

function processEvent(f, ev) {
  const s = ev.status || "active", node = ev.node, m = ev.meta || {};
  if (node === "intake") {
    if (s === "active") {
      setNS(f, "user", "done");
      setNS(f, "orchestrator", "active");
      setES(f, "u_o", "active");
      return;
    }
    if (s === "error") {
      setNS(f, "user", "done");
      setNS(f, "orchestrator", "error");
      setES(f, "u_o", "error");
      return;
    }
    setNS(f, "orchestrator", "done");
    setES(f, "u_o", "done");
  } else if (node === "plan") {
    setNS(f, "orchestrator", s === "done" ? "done" : "active");
  } else if (node === "inputs") {
    setNS(f, "orchestrator", s === "done" ? "done" : "active");
  } else if (node === "retrieval") {
    const skipped = m.skipped === true || /skipped/i.test(String(ev.detail || ""));
    setNS(f, "orchestrator", "done");
    if (skipped) {
      setNS(f, "database", "idle");
      setES(f, "o_d", "idle");
      return;
    }
    setNS(f, "database", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setES(f, "o_d", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    if (m.top?.length) pushDocBadges(f, m.top);
    if (m.top_clauses?.length) pushDocBadges(f, m.top_clauses);
  } else if (node === "tools") {
    setNS(f, "orchestrator", "done");
    if (ev.skipped) {
      // No tools used — keep TOOLS node idle (unhighlighted)
      setNS(f, "tools", "idle");
      setES(f, "o_t", "idle");
    } else {
      setNS(f, "tools", s === "error" ? "error" : (s === "done" ? "done" : "active"));
      setES(f, "o_t", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    }
    if (m.tool) pushToolBadge(f, m.tool);
  } else if (node === "compose") {
    setNS(f, "orchestrator", s === "done" ? "done" : (s === "error" ? "error" : "active"));
    if (s === "error") setNS(f, "response", "error");
    setES(f, "o_r", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    if (m.used_tools?.length) {
      setNS(f, "tools", "done");
      for (const t of m.used_tools) pushToolBadge(f, t);
    }
  } else if (node === "output") {
    setNS(f, "orchestrator", "done");
    setNS(f, "response", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setES(f, "o_r", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setNS(f, "user", "done");
  }
}

function appendLog(msgNode, text) {
  const log = msgNode.querySelector(".machine-log");
  if (!log) return;
  const li = document.createElement("li");
  li.className = "log-item";
  const ts = document.createElement("span");
  ts.className = "log-ts";
  ts.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const msg = document.createElement("span");
  msg.className = "log-msg";
  msg.textContent = clamp(text, 260);
  li.appendChild(ts);
  li.appendChild(msg);
  log.prepend(li);
  while (log.children.length > 12) log.removeChild(log.lastChild);
}

// ---- Witty rotating phrases while thinking ----
const _THINKING_PHRASES = [
  "Sharpening the pencil",
  "Warming up the calculator",
  "Flipping through the textbook",
  "Putting on the hard hat",
  "Rolling up the sleeves",
  "Brewing a strong coffee",
  "Dusting off the slide rule",
  "Squinting at the fine print",
  "Stretching before the heavy lifting",
  "Cracking the knuckles",
  "Gathering the evidence",
  "Connecting the dots",
  "Thinking really hard",
  "Almost there, probably",
  "Doing the engineering thing",
  "Consulting the sacred texts",
];

function _startPhraseRotation(msgNode) {
  if (msgNode.__phraseTimer) return;
  let idx = Math.floor(Math.random() * _THINKING_PHRASES.length);
  const label = msgNode.querySelector(".thinking-label");
  if (!label) return;
  label.textContent = _THINKING_PHRASES[idx];
  msgNode.__phraseTimer = setInterval(() => {
    idx = (idx + 1) % _THINKING_PHRASES.length;
    label.classList.add("phrase-fade");
    setTimeout(() => {
      label.textContent = _THINKING_PHRASES[idx];
      label.classList.remove("phrase-fade");
    }, 200);
  }, 2800);
}

function _stopPhraseRotation(msgNode) {
  if (msgNode.__phraseTimer) {
    clearInterval(msgNode.__phraseTimer);
    msgNode.__phraseTimer = null;
  }
}

function updateThinkingLabel(msgNode, text) {
  _stopPhraseRotation(msgNode);
  const label = msgNode.querySelector(".thinking-label");
  if (label) label.textContent = clamp(text, 120);
}

function setThinkingState(msgNode, active) {
  const block = msgNode.querySelector(".thinking-block");
  if (!block) return;
  block.classList.toggle("is-thinking", !!active);
  if (active) {
    _startPhraseRotation(msgNode);
  } else {
    _stopPhraseRotation(msgNode);
  }
}

function previewPairs(obj, max = 3) {
  if (!obj || typeof obj !== "object") return "";
  const pairs = Object.entries(obj)
    .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
    .slice(0, max)
    .map(([k, v]) => `${k}=${String(v)}`);
  return pairs.join(", ");
}

function previewClauses(entries, max = 3) {
  if (!Array.isArray(entries)) return "";
  return entries
    .map((e) => String(e?.clause_id || "").trim())
    .filter(Boolean)
    .slice(0, max)
    .join(", ");
}

function describeMachineStep(ev) {
  if (!ev || !ev.node) return "";
  const s = ev.status || "active";
  const m = ev.meta || {};

  if (ev.node === "intake") {
    if (s === "active") return "Read your request and started orchestrator intake.";
    if (s === "done") return "Parsed the request and moved to planning.";
    return "Stopped during intake due to an error.";
  }

  if (ev.node === "plan") {
    const thinking = String(m.thinking_mode || "");
    const mode = String(m.mode || "retrieval_only").replace(/_/g, " ");
    const tools = Array.isArray(m.tools) && m.tools.length ? m.tools.map(normTool).join(" -> ") : "no tools";
    const modeLabel = thinking ? `${thinking} mode` : "default mode";
    return `Planned ${mode} path (${modeLabel}) with tool chain: ${tools}.`;
  }

  if (ev.node === "inputs") {
    if (ev.skipped) return null;  // no tools → no inputs step needed
    if (s === "active") return "Resolving user-provided values and defaults.";
    if (s === "done") {
      const provided = Object.keys(m.user_inputs || {}).length;
      const defaulted = Object.keys(m.assumed_inputs || {}).length;
      const sample = previewPairs(m.user_inputs);
      return sample
        ? `Resolved inputs (${provided} provided, ${defaulted} defaulted). Key values: ${sample}.`
        : `Resolved inputs (${provided} provided, ${defaulted} defaulted).`;
    }
    return "Input resolution failed.";
  }

  if (ev.node === "retrieval") {
    const skipped = m.skipped === true || /skipped/i.test(String(ev.detail || ""));
    if (skipped) return "Skipped database retrieval for calculator-only path.";
    if (s === "active") {
      const iteration = m.iteration?.iteration || m.iteration?.pass || "";
      const clauses = previewClauses(m.top || m.top_clauses);
      if (iteration && clauses) return `Search pass ${iteration}: top EC3 clauses ${clauses}.`;
      if (iteration) return `Search pass ${iteration}: updating EC3 evidence ranking.`;
      return "Searching EC3 clauses and ranking relevance.";
    }
    if (s === "done") {
      const count = Number(m.retrieved_count || 0);
      const clauses = previewClauses(m.top_clauses || m.top);
      return clauses
        ? `Selected ${count} relevant clause(s). Top hits: ${clauses}.`
        : `Selected ${count} relevant clause(s) for evidence.`;
    }
    return "Retrieval step failed.";
  }

  if (ev.node === "tools") {
    if (ev.skipped) return null;  // no tools used — suppress log line
    const toolName = normTool(m.tool || "");
    if (s === "error") {
      return toolName ? `Tool ${toolName} failed; answer support may be limited.` : "Tool execution failed.";
    }
    if (toolName && m.status === "ok") return `Tool ${toolName} completed successfully.`;
    if (toolName && s === "active") return `Running ${toolName} with resolved inputs.`;
    if (s === "done") return "Tool execution finished.";
    return "Executing tool chain.";
  }

  if (ev.node === "compose") {
    const usedTools = Array.isArray(m.used_tools) ? m.used_tools.length : 0;
    const usedSources = Array.isArray(m.used_sources) ? m.used_sources.length : 0;
    if (s === "active") {
      return (usedTools || usedSources)
        ? "Composing response from tool outputs and retrieved clauses."
        : (ev.detail || "Generating response...");
    }
    if (s === "done") {
      return (usedTools || usedSources)
        ? `Draft complete with ${usedTools} tool(s) and ${usedSources} source citation(s).`
        : (ev.detail || "Response ready.");
    }
    return "Could not fully ground the draft in available evidence.";
  }

  if (ev.node === "output") {
    if (s === "active") return "Streaming response to chat.";
    if (s === "done") return "Response delivered.";
    return "Output stage failed.";
  }

  return ev.detail || `${ev.node}: ${s}`;
}

function updateFlow(msgNode, ev) {
  const f = msgNode.__flow;
  if (!f || !ev?.node) return;
  msgNode.__stepCount = (msgNode.__stepCount || 0) + 1;
  processEvent(f, ev);
  applyFlow(msgNode);
  const detail = describeMachineStep(ev);
  // Don't call updateThinkingLabel here — the phrase rotator handles it
  if (detail !== null) {
    appendLog(msgNode, detail || ev.detail || `${ev.node}: ${ev.status}`);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function finalizeThinking(msgNode, payload) {
  const f = msgNode.__flow;
  if (!f) return;
  setNS(f, "response", payload.supported ? "done" : "error");
  setNS(f, "user", "done");
  setES(f, "o_r", payload.supported ? "done" : "error");
  applyFlow(msgNode);
  setThinkingState(msgNode, false);

  const elapsed = ((Date.now() - (msgNode.__thinkStart || Date.now())) / 1000).toFixed(1);
  const steps = msgNode.__stepCount || 0;
  const meta = msgNode.querySelector(".thinking-meta");
  if (meta) meta.textContent = `${steps} steps · ${elapsed}s`;
  updateThinkingLabel(msgNode, "Reasoning complete. Expand to review steps.");
}

function setTrace(msgNode, payload) {
  const trace = msgNode.querySelector(".trace");
  const body = msgNode.querySelector(".trace-body");
  if (!trace || !body) return;
  const lines = [];
  if (payload.what_i_used?.length) lines.push(...payload.what_i_used.map(i => `• ${i}`));
  if (payload.tool_trace?.length) {
    lines.push("", "Tool chain:");
    for (const s of payload.tool_trace) lines.push(`  ${s.status === "ok" ? "✓" : "✗"} ${s.tool_name}: ${s.status}`);
  }
  if (payload.assumptions?.length) {
    lines.push("", "Assumptions:");
    for (const a of payload.assumptions) lines.push(`  → ${a}`);
  }
  if (!lines.length) return;
  body.textContent = lines.join("\n");
  trace.classList.remove("hidden");
}

// ---- Messages ----
function createMsg(role, content = "", opts = {}) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".role").textContent = role === "assistant" ? "Assistant" : "You";

  const contentEl = node.querySelector(".content");
  if (role === "assistant") {
    contentEl.innerHTML = renderMd(content);
    // Assistant messages: hide edit button, keep copy
    node.querySelector(".edit-btn")?.remove();
    if (opts.showThinking !== false) {
      initFlowGraph(node, opts.prompt || "");
      const startCollapsed = opts.startCollapsed !== false;
      if (startCollapsed) node.querySelector(".thinking-block")?.classList.add("collapsed");
    } else {
      node.querySelector(".thinking-block")?.classList.add("hidden");
    }
    if (opts.responsePayload) {
      setTrace(node, opts.responsePayload);
      if (node.querySelector(".thinking-block")) {
        node.querySelector(".thinking-block").classList.add("collapsed");
      }
    }
  } else {
    const attHtml = buildAttachmentHtml(opts.attachments);
    if (attHtml) {
      contentEl.innerHTML = attHtml;
      const textNode = document.createElement("span");
      textNode.textContent = content;
      contentEl.appendChild(textNode);
      attachLightboxListeners(contentEl);
    } else {
      contentEl.textContent = content;
    }
    node.querySelector(".thinking-block")?.remove();
    node.querySelector(".trace")?.remove();

    // Store original prompt + attachments for edit & resubmit
    node.__editContent = content;
    node.__editAttachments = opts.attachments || [];
  }

  const copyBtn = node.querySelector(".copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const text = role === "assistant" ? (contentEl.innerText || contentEl.textContent) : content;
      navigator.clipboard?.writeText(text);
      copyBtn.title = "Copied!";
      setTimeout(() => { copyBtn.title = "Copy to clipboard"; }, 1500);
    });
  }

  // Edit & resubmit for user messages
  const editBtn = node.querySelector(".edit-btn");
  if (editBtn && role === "user") {
    editBtn.addEventListener("click", () => editAndResubmit(node));
  }

  const toggle = node.querySelector(".thinking-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const block = node.querySelector(".thinking-block");
      if (block) block.classList.toggle("collapsed");
    });
  }

  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

// ---- Inline Edit & Resubmit ----
function editAndResubmit(userMsgNode) {
  // Don't allow editing while a request is in flight or already editing
  if (sendBtn.disabled) return;
  if (userMsgNode.classList.contains("editing")) return;

  const contentEl = userMsgNode.querySelector(".content");
  const originalHtml = contentEl.innerHTML;
  const prompt = userMsgNode.__editContent || "";
  const attachments = userMsgNode.__editAttachments || [];

  userMsgNode.classList.add("editing");

  // Build inline editor
  const textarea = document.createElement("textarea");
  textarea.className = "inline-edit-input";
  textarea.value = prompt;

  const btnRow = document.createElement("div");
  btnRow.className = "inline-edit-actions";

  const saveBtn = document.createElement("button");
  saveBtn.className = "inline-edit-save";
  saveBtn.textContent = "Save & Submit";
  saveBtn.type = "button";

  const cancelBtn = document.createElement("button");
  cancelBtn.className = "inline-edit-cancel";
  cancelBtn.textContent = "Cancel";
  cancelBtn.type = "button";

  btnRow.appendChild(saveBtn);
  btnRow.appendChild(cancelBtn);

  // Replace content with textarea + buttons
  contentEl.innerHTML = "";
  contentEl.appendChild(textarea);
  contentEl.appendChild(btnRow);

  // Auto-size textarea to fit content
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
  textarea.focus();
  textarea.selectionStart = textarea.selectionEnd = textarea.value.length;

  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
  });

  // Cancel → restore original content
  cancelBtn.addEventListener("click", () => {
    contentEl.innerHTML = originalHtml;
    userMsgNode.classList.remove("editing");
    if (attachments.length) attachLightboxListeners(contentEl);
  });

  // Save → resubmit from this point
  saveBtn.addEventListener("click", async () => {
    const newText = textarea.value.trim();
    if (!newText && !attachments.length) return;

    userMsgNode.classList.remove("editing");

    // Update the displayed content
    const attHtml = buildAttachmentHtml(attachments);
    if (attHtml) {
      contentEl.innerHTML = attHtml;
      const span = document.createElement("span");
      span.textContent = newText;
      contentEl.appendChild(span);
      attachLightboxListeners(contentEl);
    } else {
      contentEl.textContent = newText;
    }

    // Update stored edit content
    userMsgNode.__editContent = newText;

    // Remove all messages AFTER this one from the DOM
    const allMsgs = [...messagesEl.querySelectorAll(".message")];
    const idx = allMsgs.indexOf(userMsgNode);
    if (idx < 0) return;
    const toRemove = allMsgs.slice(idx + 1);
    for (const el of toRemove) el.remove();

    // Update thread: keep this message (with new content), drop everything after
    const thread = currentThread();
    if (thread && thread.messages.length > idx) {
      thread.messages[idx].content = newText;
      thread.messages[idx].attachments = attachments;
      thread.messages.length = idx + 1;
      thread.updatedAt = now();
    }

    // Build full prompt with attachment markers
    let fullPrompt = newText;
    if (attachments.length) {
      const fileDescs = attachments.map(a =>
        a.isImage ? `[Attached image: ${a.name}]` : `[Attached file: ${a.name} (${fmtFileSize(a.size)})]`
      ).join(" ");
      fullPrompt = fullPrompt ? `${fileDescs}\n\n${fullPrompt}` : fileDescs;
    }

    // Lock UI before any async work to prevent concurrent edits/submissions
    sendBtn.disabled = true;
    setThinkingModeDisabled(true);

    if (canUseStoredThreads()) {
      if (auth.threadsSync) {
        try {
          await truncateThreadApi(thread.id, idx + 1, newText);
        } catch (e) {
          console.warn("Thread truncation sync failed:", e);
        }
      } else {
        save();
      }
    }

    // Create assistant node and stream response
    const assistantNode = createMsg("assistant", "", { showThinking: true, prompt: fullPrompt });

    try {
      await streamChat(fullPrompt, assistantNode, thread, state.thinkingMode, attachments, { isEdit: true });
    } catch (err) {
      const errMsg = `Error: ${err.message}`;
      setThinkingState(assistantNode, false);
      updateThinkingLabel(assistantNode, "Reasoning failed.");
      assistantNode.querySelector(".content").innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
      appendLog(assistantNode, `Transport error: ${err.message}`);
      thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
      thread.updatedAt = now();
      if (canUseStoredThreads()) {
        if (auth.threadsSync) {
          await addMessageToApi(thread.id, "assistant", errMsg, null);
        } else {
          save();
        }
      }
    } finally {
      sendBtn.disabled = false;
      setThinkingModeDisabled(false);
    }
  });

  // Keyboard shortcuts: Ctrl/Cmd+Enter to submit, Escape to cancel
  textarea.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      saveBtn.click();
    }
    if (e.key === "Escape") {
      e.preventDefault();
      cancelBtn.click();
    }
  });
}

function renderMessages() {
  messagesEl.innerHTML = "";
  const t = currentThread();
  if (!t) { updateWelcome(); return; }
  for (const m of t.messages || []) {
    if (m.role === "assistant") {
      createMsg("assistant", m.content || "", { showThinking: false, responsePayload: m.responsePayload });
    } else {
      createMsg("user", m.content || "", { attachments: m.attachments });
    }
  }
  updateWelcome();
}

// ---- Streaming ----
async function streamChat(prompt, assistantNode, thread, thinkingMode = "thinking", attachments = [], { isEdit = false } = {}) {
  const contentEl = assistantNode.querySelector(".content");
  contentEl.innerHTML = "";
  let accumulated = "";
  let renderTimer = null;

  function scheduleRender() {
    if (renderTimer) return;
    renderTimer = setTimeout(() => {
      contentEl.innerHTML = renderMd(accumulated);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      renderTimer = null;
    }, 60);
  }

  const threadMsgs = thread.messages || [];
  const prevMsgs = threadMsgs.slice(0, -1);
  const history = prevMsgs.slice(-6).map(m => ({
    role: m.role,
    content: m.role === "assistant" ? (m.content || "").slice(0, 500) : (m.content || ""),
  }));

  // Build attachments payload for the API (with base64 data for images)
  const apiAttachments = attachments.map(a => ({
    name: a.name,
    type: a.type || "",
    size: a.size || 0,
    is_image: !!a.isImage,
    data_url: a.isImage ? (a.dataUrl || null) : null,
  }));

  const res = await fetchWithAuth("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: prompt,
      history,
      thinking_mode: thinkingMode,
      attachments: apiAttachments,
      is_edit: isEdit,
    }),
  });
  if (!res.ok || !res.body) throw new Error(`Request failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalized = false;
  let lastPayload = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try { event = JSON.parse(line); } catch { continue; }

      if (event.type === "machine") updateFlow(assistantNode, event);

      if (event.type === "delta") {
        accumulated += event.delta;
        scheduleRender();
      }

      if (event.type === "final") {
        if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
        const payload = event.response;
        lastPayload = payload;
        contentEl.innerHTML = renderMd(payload.answer);
        setTrace(assistantNode, payload);
        finalizeThinking(assistantNode, payload);
        appendLog(assistantNode, "Response complete.");

        if (state.devMode && payload.tool_trace?.length) {
          showDevActivity(payload);
        }

        if (!finalized) {
          thread.messages.push({ id: uid(), role: "assistant", content: payload.answer, responsePayload: payload, createdAt: now() });
          thread.updatedAt = now();
          if (canUseStoredThreads()) {
            if (auth.threadsSync) {
              await addMessageToApi(thread.id, "assistant", payload.answer, payload);
            } else {
              save();
            }
          }
          renderThreadList();
          finalized = true;
        }
      }

      if (event.type === "error") {
        if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
        const errMsg = `Error: ${event.detail || "Unknown error"}`;
        setThinkingState(assistantNode, false);
        updateThinkingLabel(assistantNode, "Reasoning failed.");
        contentEl.innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
        appendLog(assistantNode, errMsg);
        if (!finalized) {
          thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
          thread.updatedAt = now();
          if (canUseStoredThreads()) {
            if (auth.threadsSync) {
              await addMessageToApi(thread.id, "assistant", errMsg, null);
            } else {
              save();
            }
          }
          renderThreadList();
          finalized = true;
        }
      }
    }
  }
}

// ---- Developer mode ----
function showDevActivity(payload) {
  const out = $("#dev-activity");
  if (!out) return;
  out.classList.remove("hidden");

  const lines = [];
  if (payload.tool_trace?.length) {
    lines.push("Tools executed:");
    for (const t of payload.tool_trace) {
      const status = t.status === "ok" ? "✓" : "✗";
      lines.push(`  ${status} ${t.tool_name}`);
      if (t.inputs) {
        for (const [k, v] of Object.entries(t.inputs)) {
          lines.push(`    ${k}: ${JSON.stringify(v)}`);
        }
      }
    }
  }
  if (payload.sources?.length) {
    lines.push("\nSources used:");
    const seen = new Set();
    for (const s of payload.sources) {
      const key = `${s.clause_id}`;
      if (seen.has(key) || key === "0") continue;
      seen.add(key);
      const clauseId = String(s.clause_id || "").trim();
      const prefix = /^\d/.test(clauseId) ? "Cl. " : "";
      lines.push(`  ${prefix}${clauseId} — ${s.clause_title || ""}`);
    }
  }
  out.textContent = lines.join("\n");
}

function initDevMode() {
  devToggle.addEventListener("change", () => {
    state.devMode = devToggle.checked;
    devPanel.classList.toggle("hidden", !state.devMode);
    document.body.classList.toggle("dev-active", state.devMode);
  });
  $("#dev-panel-close")?.addEventListener("click", () => {
    devToggle.checked = false;
    state.devMode = false;
    devPanel.classList.add("hidden");
    document.body.classList.remove("dev-active");
  });
  $("#tool-writer-btn")?.addEventListener("click", async () => {
    const desc = $("#tool-writer-input")?.value?.trim();
    if (!desc) return;
    const out = $("#tool-writer-output");
    out.classList.remove("hidden");
    out.textContent = "Generating tool...";
    try {
      const res = await fetch("/api/tools/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: desc }),
      });
      const data = await res.json();
      out.textContent = data.code || data.error || JSON.stringify(data, null, 2);
    } catch (e) {
      out.textContent = `Error: ${e.message}`;
    }
  });
}

// ---- Auth ----
const auth = {
  token: null,
  user: null,
  refreshToken: null,
  expiresAt: null,
  enabled: true,
  threadsSync: false,
  ready: false,
  mode: "login",
};

function syncAuthControls() {
  const signedIn = !!auth.user;
  const showAuthArea = auth.ready && auth.enabled && !signedIn;

  const authArea = $("#sidebar-auth-area");
  const userArea = $("#sidebar-user-area");
  if (authArea) authArea.classList.toggle("hidden", !showAuthArea);
  if (userArea) userArea.classList.toggle("hidden", showAuthArea);

  if (chatSearch) {
    const canSearch = canUseStoredThreads();
    chatSearch.disabled = !canSearch;
    chatSearch.placeholder = !auth.ready
      ? "Loading chats..."
      : (!auth.enabled
        ? "History unavailable without auth"
        : (chatSearch.disabled ? "Sign in to search saved chats..." : "Search chats..."));
    if (chatSearch.disabled) {
      chatSearch.value = "";
      state.filter = "";
    }
  }
}

async function applyAuthState() {
  if (canUseStoredThreads()) {
    await load();
    await ensureThread();
  } else {
    resetThreadState();
    resetGuestThread();
  }
  updateUserPill();
  syncAuthControls();
  renderThreadList();
  renderMessages();
}

async function checkAuthStatus() {
  auth.ready = false;
  try {
    const res = await fetch("/api/auth/status");
    const data = await res.json();
    auth.enabled = data.enabled === true;
    auth.threadsSync = data.threads_sync === true;
  } catch { auth.enabled = false; auth.threadsSync = false; }

  if (!auth.enabled) {
    auth.ready = true;
    return;
  }

  const saved = sessionStorage.getItem("ec3_auth");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      auth.token = parsed.access_token || null;
      auth.user = { user_id: parsed.user_id, email: parsed.email };
      auth.refreshToken = parsed.refresh_token || null;
      auth.expiresAt = parsed.expires_at || null;
      await maybeRefreshSession();
    } catch {
      sessionStorage.removeItem("ec3_auth");
    }
  }
  auth.ready = true;
}

function persistAuth(data) {
  auth.token = data.access_token;
  auth.refreshToken = data.refresh_token ?? auth.refreshToken;
  auth.expiresAt = data.expires_at ?? auth.expiresAt;
  if (data.user_id || data.email) {
    auth.user = { ...auth.user, user_id: data.user_id || auth.user?.user_id, email: data.email || auth.user?.email };
  }
  const toStore = {
    access_token: auth.token,
    refresh_token: auth.refreshToken,
    expires_at: auth.expiresAt,
    user_id: auth.user?.user_id,
    email: auth.user?.email,
  };
  sessionStorage.setItem("ec3_auth", JSON.stringify(toStore));
}

function isTokenExpiringSoon() {
  if (!auth.expiresAt) return false;
  const secLeft = auth.expiresAt - Math.floor(Date.now() / 1000);
  return secLeft < 300;
}

async function refreshSession() {
  if (!auth.refreshToken) return false;
  try {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: auth.refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    persistAuth({
      access_token: data.access_token,
      refresh_token: data.refresh_token ?? auth.refreshToken,
      expires_at: data.expires_at,
      user_id: auth.user?.user_id,
      email: auth.user?.email,
    });
    return true;
  } catch {
    return false;
  }
}

async function maybeRefreshSession() {
  if (isTokenExpiringSoon() && auth.refreshToken) {
    await refreshSession();
  }
}

async function fetchWithAuth(url, opts = {}) {
  await maybeRefreshSession();
  opts.headers = { ...authHeaders(), ...opts.headers };
  let res = await fetch(url, opts);
  if (res.status === 401 && auth.refreshToken) {
    const ok = await refreshSession();
    if (ok) {
      opts.headers = { ...authHeaders(), ...opts.headers };
      res = await fetch(url, opts);
    }
  }
  return res;
}

function showAuthOverlay(mode = "login") {
  if (!auth.enabled) return;
  switchAuthTab(mode);
  $("#auth-error")?.classList.add("hidden");
  $("#auth-overlay")?.classList.remove("hidden");
  if (mode === "login") $("#auth-email-login")?.focus();
  else $("#auth-email-signup")?.focus();
}

function hideAuthOverlay() {
  $("#auth-overlay")?.classList.add("hidden");
}

function updateSidebarUser() {
  const avatarEl = $("#sidebar-user-avatar");
  const nameEl = $("#sidebar-user-name");
  if (!auth.user) return;

  const email = auth.user.email || "";
  const name = email.split("@")[0] || "User";
  const initials = name.slice(0, 2).toUpperCase();

  if (avatarEl) avatarEl.textContent = initials;
  if (nameEl) nameEl.textContent = name;
}

function updateUserPill() {
  updateSidebarUser();
}

function initAuth() {
  const overlay = $("#auth-overlay");
  const errorEl = $("#auth-error");
  const closeBtn = $("#auth-close");
  const tabLogin = $("#auth-tab-login");
  const tabSignup = $("#auth-tab-signup");
  const formLogin = $("#auth-form-login");
  const formSignup = $("#auth-form-signup");
  const forgotLink = $("#auth-forgot-password");

  // Sidebar: open auth modal
  signInBtn?.addEventListener("click", () => showAuthOverlay("login"));
  registerBtn?.addEventListener("click", () => showAuthOverlay("signup"));
  sidebarSigninBtn?.addEventListener("click", () => showAuthOverlay("login"));
  sidebarSignupBtn?.addEventListener("click", () => showAuthOverlay("signup"));

  // Sidebar user area: toggle menu & logout (event delegation)
  sidebar?.addEventListener("click", (e) => {
    if (e.target.closest("#sidebar-user-btn")) {
      if (e.target.closest("#sidebar-logout-btn")) return;
      const btn = $("#sidebar-user-btn");
      const menu = $("#sidebar-user-menu");
      const expanded = btn?.getAttribute("aria-expanded") === "true";
      btn?.setAttribute("aria-expanded", !expanded);
      menu?.classList.toggle("hidden", expanded);
    }
    if (e.target.closest("#sidebar-logout-btn")) {
      (async () => {
        try { await fetch("/api/auth/logout", { method: "POST" }); } catch {}
        sessionStorage.removeItem("ec3_auth");
        auth.token = null;
        auth.user = null;
        auth.refreshToken = null;
        auth.expiresAt = null;
        $("#sidebar-user-menu")?.classList.add("hidden");
        hideAuthOverlay();
        applyAuthState();
      })();
    }
  });

  // Auth modal: close
  closeBtn?.addEventListener("click", hideAuthOverlay);
  overlay?.addEventListener("click", (e) => {
    if (e.target === overlay) hideAuthOverlay();
  });

  // Auth tabs
  tabLogin?.addEventListener("click", () => switchAuthTab("login"));
  tabSignup?.addEventListener("click", () => switchAuthTab("signup"));

  // Forgot password
  forgotLink?.addEventListener("click", (e) => {
    e.preventDefault();
    handleForgotPassword();
  });

  // Login form
  formLogin?.addEventListener("submit", (e) => {
    e.preventDefault();
    handleAuthSubmit("login");
  });

  // Signup form
  formSignup?.addEventListener("submit", (e) => {
    e.preventDefault();
    handleAuthSubmit("signup");
  });
}

function switchAuthTab(mode) {
  auth.mode = mode;
  const isLogin = mode === "login";
  $("#auth-tab-login")?.classList.toggle("active", isLogin);
  $("#auth-tab-signup")?.classList.toggle("active", !isLogin);
  $("#auth-form-login")?.classList.toggle("hidden", !isLogin);
  $("#auth-form-signup")?.classList.toggle("hidden", isLogin);
  const errEl = $("#auth-error");
  errEl?.classList.add("hidden");
  errEl?.classList.remove("auth-success");
}

async function handleForgotPassword() {
  const emailInput = $("#auth-email-login");
  const email = emailInput?.value?.trim();
  if (!email) {
    $("#auth-error").textContent = "Enter your email address.";
    $("#auth-error").classList.remove("hidden");
    return;
  }
  const btn = $("#auth-submit-login");
  const btnText = btn?.querySelector(".auth-btn-text");
  const spinner = btn?.querySelector(".auth-btn-spinner");
  btnText && (btnText.textContent = "Sending...");
  spinner?.classList.remove("hidden");
  btn?.setAttribute("disabled", "true");
  const errEl = $("#auth-error");
  try {
    const res = await fetch("/api/auth/forgot-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    const data = await res.json().catch(() => ({}));
    errEl.textContent = data.message || "If an account exists, you will receive a reset link.";
    errEl.classList.remove("hidden");
    errEl.classList.add("auth-success");
  } catch (err) {
    errEl.textContent = err.message || "Failed to send reset email.";
    errEl.classList.remove("hidden");
    errEl.classList.remove("auth-success");
  } finally {
    btnText && (btnText.textContent = "Sign in");
    spinner?.classList.add("hidden");
    btn?.removeAttribute("disabled");
  }
}

async function handleAuthSubmit(mode) {
  const isLogin = mode === "login";
  const emailInput = isLogin ? $("#auth-email-login") : $("#auth-email-signup");
  const passInput = isLogin ? $("#auth-password-login") : $("#auth-password-signup");
  const passConfirm = $("#auth-password-confirm");
  const submitBtn = isLogin ? $("#auth-submit-login") : $("#auth-submit-signup");
  const btnText = submitBtn?.querySelector(".auth-btn-text");
  const spinner = submitBtn?.querySelector(".auth-btn-spinner");

  const email = emailInput?.value?.trim();
  const password = passInput?.value;
  if (!email || !password) return;

  if (!isLogin) {
    const confirmVal = passConfirm?.value;
    if (password !== confirmVal) {
      $("#auth-error").textContent = "Passwords do not match.";
      $("#auth-error").classList.remove("hidden");
      return;
    }
  }

  $("#auth-error")?.classList.add("hidden");
  submitBtn?.setAttribute("disabled", "true");
  btnText && (btnText.classList.add("hidden"));
  spinner?.classList.remove("hidden");

  const endpoint = isLogin ? "/api/auth/login" : "/api/auth/signup";
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Auth failed");
    if (!data.access_token) throw new Error("Account created. Confirm your email, then sign in.");

    auth.token = data.access_token;
    auth.user = { user_id: data.user_id, email: data.email };
    auth.refreshToken = data.refresh_token || null;
    auth.expiresAt = data.expires_at || null;
    persistAuth(data);
    applyAuthState();
    hideAuthOverlay();
  } catch (err) {
    $("#auth-error").textContent = err.message;
    $("#auth-error").classList.remove("hidden");
  } finally {
    submitBtn?.removeAttribute("disabled");
    btnText?.classList.remove("hidden");
    spinner?.classList.add("hidden");
  }
}

function authHeaders() {
  if (auth.token) return { Authorization: `Bearer ${auth.token}` };
  return {};
}

async function handleAuthRedirectFromHash() {
  const hash = window.location.hash;
  if (!hash) return;
  const params = new URLSearchParams(hash.slice(1));
  const accessToken = params.get("access_token");
  if (!accessToken) return;

  const refreshToken = params.get("refresh_token") || "";
  const expiresAt = params.get("expires_at");
  let user = { user_id: params.get("user_id") || "", email: params.get("email") || "" };

  auth.token = accessToken;
  auth.refreshToken = refreshToken;
  auth.expiresAt = expiresAt ? parseInt(expiresAt, 10) : null;
  auth.user = user;

  if (!user.user_id || !user.email) {
    try {
      const res = await fetch("/api/auth/me", { headers: { Authorization: `Bearer ${accessToken}` } });
      if (res.ok) {
        const data = await res.json();
        user = { user_id: data.user_id || "", email: data.email || "" };
        auth.user = user;
      }
    } catch {}
  }

  const toStore = {
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at: auth.expiresAt,
    user_id: user.user_id,
    email: user.email,
  };
  sessionStorage.setItem("ec3_auth", JSON.stringify(toStore));

  window.history.replaceState(null, "", window.location.pathname + window.location.search);
  applyAuthState();
}


// ---- Init ----
async function initialize() {
  resetGuestThread();
  renderThreadList();
  renderMessages();
  initDevMode();
  initAuth();
  syncAuthControls();
  loadThinkingModePreference();

  thinkingModeSelect?.addEventListener("change", (e) => {
    setThinkingMode(e.target.value);
  });
  thinkingModeTrigger?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (thinkingModeTrigger.disabled || !thinkingModeMenu) return;
    closeAttachMenu();
    const open = thinkingModeMenu.classList.contains("hidden");
    if (open) {
      thinkingModeMenu.classList.remove("hidden");
      thinkingModeTrigger.setAttribute("aria-expanded", "true");
      thinkingModeTrigger.closest(".thinking-mode-wrap")?.classList.add("open");
      return;
    }
    closeThinkingModeMenu();
  });
  thinkingModeMenu?.addEventListener("click", (e) => {
    e.stopPropagation();
  });
  for (const option of $$(".thinking-mode-option")) {
    option.addEventListener("click", () => {
      setThinkingMode(option.dataset.mode || "thinking");
      closeThinkingModeMenu();
    });
  }
  document.addEventListener("click", (e) => {
    const wrap = thinkingModeTrigger?.closest(".thinking-mode-wrap");
    if (!wrap) return;
    if (!(e.target instanceof Node)) return;
    if (!wrap.contains(e.target)) closeThinkingModeMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeThinkingModeMenu();
      closeAttachMenu();
    }
  });

  // ---- Attach button ----
  attachTrigger?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!attachMenu) return;
    const isOpen = !attachMenu.classList.contains("hidden");
    closeThinkingModeMenu();
    if (isOpen) {
      closeAttachMenu();
    } else {
      attachMenu.classList.remove("hidden");
      attachTrigger.setAttribute("aria-expanded", "true");
      attachTrigger.closest(".attach-wrap")?.classList.add("open");
    }
  });
  attachMenu?.addEventListener("click", (e) => {
    e.stopPropagation();
  });
  for (const opt of $$(".attach-option")) {
    opt.addEventListener("click", () => {
      const type = opt.dataset.type;
      closeAttachMenu();
      if (type === "photo") {
        photoInput?.click();
      } else if (type === "file") {
        fileInput?.click();
      }
    });
  }
  photoInput?.addEventListener("change", () => {
    if (photoInput.files?.length) addAttachments(photoInput.files);
    photoInput.value = "";
  });
  fileInput?.addEventListener("change", () => {
    if (fileInput.files?.length) addAttachments(fileInput.files);
    fileInput.value = "";
  });
  document.addEventListener("click", (e) => {
    const wrap = attachTrigger?.closest(".attach-wrap");
    if (wrap && e.target instanceof Node && !wrap.contains(e.target)) closeAttachMenu();
  });

  // Drag-and-drop on the composer
  const composerEl = $(".composer");
  if (composerEl) {
    let dragCounter = 0;
    composerEl.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dragCounter++;
      composerEl.style.borderColor = "var(--accent)";
      composerEl.style.boxShadow = "0 0 0 2px var(--accent-glow), 0 8px 24px rgba(0,0,0,0.25)";
    });
    composerEl.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        composerEl.style.borderColor = "";
        composerEl.style.boxShadow = "";
      }
    });
    composerEl.addEventListener("dragover", (e) => {
      e.preventDefault();
    });
    composerEl.addEventListener("drop", (e) => {
      e.preventDefault();
      dragCounter = 0;
      composerEl.style.borderColor = "";
      composerEl.style.boxShadow = "";
      if (e.dataTransfer?.files?.length) {
        addAttachments(e.dataTransfer.files);
      }
    });
  }

  await handleAuthRedirectFromHash();

  newChatBtn.addEventListener("click", async () => {
    await createThread();
    renderThreadList();
    renderMessages();
    input.focus();
  });

  chatSearch.addEventListener("input", e => {
    state.filter = e.target.value || "";
    renderThreadList();
  });

  sidebarToggle?.addEventListener("click", () => {
    sidebar.classList.toggle("sidebar-open");
  });

  for (const btn of $$(".example-btn")) {
    btn.addEventListener("click", () => {
      const p = btn.dataset.prompt;
      if (p) { input.value = p; form.requestSubmit(); }
    });
  }

  checkAuthStatus().finally(() => {
    applyAuthState();
  });

  setInterval(() => maybeRefreshSession(), 45 * 60 * 1000);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = input.value.trim();
    const hasAttachments = state.attachments.length > 0;
    if (!prompt && !hasAttachments) return;
    const thinkingMode = thinkingModeSelect?.value || state.thinkingMode;
    setThinkingMode(thinkingMode);

    // Capture attachments before clearing
    const currentAttachments = state.attachments.map(a => ({
      id: a.id, name: a.name, size: a.size, type: a.type, isImage: a.isImage, dataUrl: a.dataUrl,
    }));

    // Build a prompt that includes file context for the AI
    let fullPrompt = prompt;
    if (currentAttachments.length) {
      const fileDescs = currentAttachments.map(a =>
        a.isImage ? `[Attached image: ${a.name}]` : `[Attached file: ${a.name} (${fmtFileSize(a.size)})]`
      ).join(" ");
      fullPrompt = fullPrompt ? `${fileDescs}\n\n${fullPrompt}` : fileDescs;
    }

    const thread = await ensureThread();
    if (canUseStoredThreads() && (!thread.messages.length || thread.title === "New chat")) {
      thread.title = truncTitle(prompt || currentAttachments[0]?.name || "New chat");
      if (auth.threadsSync) {
        await updateThreadTitleApi(thread.id, thread.title);
        const idx = state.threads.findIndex((t) => t.id === thread.id);
        if (idx >= 0) state.threads[idx].title = thread.title;
      }
    } else if (!canUseStoredThreads() && (!thread.messages.length || thread.title === "Temporary chat")) {
      thread.title = truncTitle(prompt || currentAttachments[0]?.name || "Temporary chat");
    }

    thread.messages.push({ id: uid(), role: "user", content: prompt, attachments: currentAttachments, createdAt: now() });
    thread.updatedAt = now();
    if (canUseStoredThreads()) {
      if (auth.threadsSync) {
        await addMessageToApi(thread.id, "user", fullPrompt);
      } else {
        save();
      }
    }
    renderThreadList();

    input.value = "";
    clearAttachments();
    closeAttachMenu();
    sendBtn.disabled = true;
    setThinkingModeDisabled(true);
    createMsg("user", prompt, { attachments: currentAttachments });
    const assistantNode = createMsg("assistant", "", { showThinking: true, prompt: fullPrompt });
    updateWelcome();

    if (state.devMode) {
      const out = $("#dev-activity");
      if (out) { out.classList.remove("hidden"); out.textContent = "Processing query..."; }
    }

    try {
      await streamChat(fullPrompt, assistantNode, thread, state.thinkingMode, currentAttachments);
    } catch (err) {
      const errMsg = `Error: ${err.message}`;
      setThinkingState(assistantNode, false);
      updateThinkingLabel(assistantNode, "Reasoning failed.");
      assistantNode.querySelector(".content").innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
      appendLog(assistantNode, `Transport error: ${err.message}`);
      thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
      thread.updatedAt = now();
      if (canUseStoredThreads()) {
        if (auth.threadsSync) {
          await addMessageToApi(thread.id, "assistant", errMsg, null);
        } else {
          save();
        }
      }
      renderThreadList();
    } finally {
      sendBtn.disabled = false;
      setThinkingModeDisabled(false);
      input.focus();
    }
  });
}

initialize();
