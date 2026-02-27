const STORAGE_KEY = "ec3_chat_threads_v2";

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

const state = { threads: [], activeThreadId: null, filter: "", devMode: false };

function uid() {
  return crypto?.randomUUID?.() || `id_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}
function now() { return new Date().toISOString(); }

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

// ---- State persistence ----
function save() { localStorage.setItem(STORAGE_KEY, JSON.stringify({ threads: state.threads, activeThreadId: state.activeThreadId })); }
function load() {
  try {
    const p = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (p?.threads) { state.threads = p.threads; state.activeThreadId = p.activeThreadId || null; }
  } catch { state.threads = []; state.activeThreadId = null; }
}

function createThread(title = "New chat") {
  const t = { id: uid(), title, createdAt: now(), updatedAt: now(), messages: [] };
  state.threads.unshift(t);
  state.activeThreadId = t.id;
  save();
  return t;
}

function activeThread() { return state.threads.find(t => t.id === state.activeThreadId) || null; }
function ensureThread() { return activeThread() || createThread(); }

function setActive(id) {
  if (!state.threads.some(t => t.id === id)) return;
  state.activeThreadId = id;
  save();
  renderThreadList();
  renderMessages();
}

function updateWelcome() {
  const a = activeThread();
  welcome.classList.toggle("hidden", !!(a && a.messages.length));
}

// ---- Thread list ----
function renderThreadList() {
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
  return clause ? `${file} · Cl. ${clause}` : file;
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
    setNS(f, "orchestrator", "done");
    setNS(f, "database", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setES(f, "o_d", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    if (m.top?.length) pushDocBadges(f, m.top);
    if (m.top_clauses?.length) pushDocBadges(f, m.top_clauses);
  } else if (node === "tools") {
    setNS(f, "orchestrator", "done");
    setNS(f, "tools", s === "error" ? "error" : (s === "done" ? "done" : "active"));
    setES(f, "o_t", s === "error" ? "error" : (s === "done" ? "done" : "active"));
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

function updateThinkingLabel(msgNode, text) {
  const label = msgNode.querySelector(".thinking-label");
  if (label) label.textContent = clamp(text, 120);
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
    const mode = String(m.mode || "retrieval_only").replace(/_/g, " ");
    const tools = Array.isArray(m.tools) && m.tools.length ? m.tools.map(normTool).join(" -> ") : "no tools";
    return `Planned ${mode} path with tool chain: ${tools}.`;
  }

  if (ev.node === "inputs") {
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
    if (s === "active") return "Composing response from tool outputs and retrieved clauses.";
    if (s === "done") {
      const usedTools = Array.isArray(m.used_tools) ? m.used_tools.length : 0;
      const usedSources = Array.isArray(m.used_sources) ? m.used_sources.length : 0;
      return `Draft complete with ${usedTools} tool(s) and ${usedSources} source citation(s).`;
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
  updateThinkingLabel(msgNode, detail || ev.detail || `${ev.node}: ${ev.status || "active"}`);
  appendLog(msgNode, detail || ev.detail || `${ev.node}: ${ev.status}`);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function finalizeThinking(msgNode, payload) {
  const f = msgNode.__flow;
  if (!f) return;
  setNS(f, "response", payload.supported ? "done" : "error");
  setNS(f, "user", "done");
  setES(f, "o_r", payload.supported ? "done" : "error");
  applyFlow(msgNode);

  const elapsed = ((Date.now() - (msgNode.__thinkStart || Date.now())) / 1000).toFixed(1);
  const steps = msgNode.__stepCount || 0;
  const meta = msgNode.querySelector(".thinking-meta");
  if (meta) meta.textContent = `${steps} steps · ${elapsed}s`;
  updateThinkingLabel(msgNode, "Reasoning complete. Review orchestrator steps below.");
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
    if (opts.showThinking !== false) {
      initFlowGraph(node, opts.prompt || "");
      if (opts.startCollapsed) node.querySelector(".thinking-block")?.classList.add("collapsed");
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
    contentEl.textContent = content;
    node.querySelector(".thinking-block")?.remove();
    node.querySelector(".trace")?.remove();
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

function renderMessages() {
  messagesEl.innerHTML = "";
  const t = activeThread();
  if (!t) { updateWelcome(); return; }
  for (const m of t.messages || []) {
    if (m.role === "assistant") {
      createMsg("assistant", m.content || "", { showThinking: false, responsePayload: m.responsePayload });
    } else {
      createMsg("user", m.content || "");
    }
  }
  updateWelcome();
}

// ---- Streaming ----
async function streamChat(prompt, assistantNode, thread) {
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

  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message: prompt, history }),
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
          save();
          renderThreadList();
          finalized = true;
        }
      }

      if (event.type === "error") {
        if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
        const errMsg = `Error: ${event.detail || "Unknown error"}`;
        contentEl.innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
        appendLog(assistantNode, errMsg);
        if (!finalized) {
          thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
          thread.updatedAt = now();
          save();
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
      lines.push(`  Cl. ${s.clause_id} — ${s.clause_title || ""}`);
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
const auth = { token: null, user: null, enabled: false, mode: "login" };

async function checkAuthStatus() {
  try {
    const res = await fetch("/api/auth/status");
    const data = await res.json();
    auth.enabled = data.enabled === true;
  } catch { auth.enabled = false; }

  if (!auth.enabled) return true;

  const saved = sessionStorage.getItem("ec3_auth");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      auth.token = parsed.access_token;
      auth.user = parsed;
      updateUserPill();
      return true;
    } catch { /* stale */ }
  }
  return false;
}

function showAuthOverlay() {
  $("#auth-overlay")?.classList.remove("hidden");
}

function hideAuthOverlay() {
  $("#auth-overlay")?.classList.add("hidden");
}

function updateUserPill() {
  const pill = $("#user-pill");
  if (!pill) return;
  if (auth.user) {
    pill.innerHTML = `<span>${escHtml(auth.user.email)}</span><button id="logout-btn" type="button">Sign Out</button>`;
    pill.classList.remove("hidden");
    $("#logout-btn")?.addEventListener("click", async () => {
      try { await fetch("/api/auth/logout", { method: "POST" }); } catch {}
      sessionStorage.removeItem("ec3_auth");
      auth.token = null;
      auth.user = null;
      pill.classList.add("hidden");
      if (auth.enabled) showAuthOverlay();
    });
  } else {
    pill.classList.add("hidden");
  }
}

function initAuth() {
  const overlay = $("#auth-overlay");
  const form = $("#auth-form");
  const emailInput = $("#auth-email");
  const passInput = $("#auth-password");
  const submitBtn = $("#auth-submit");
  const switchBtn = $("#auth-switch-btn");
  const switchText = $("#auth-switch-text");
  const errorEl = $("#auth-error");

  switchBtn?.addEventListener("click", () => {
    auth.mode = auth.mode === "login" ? "signup" : "login";
    submitBtn.textContent = auth.mode === "login" ? "Sign In" : "Sign Up";
    switchText.textContent = auth.mode === "login" ? "Don't have an account?" : "Already have an account?";
    switchBtn.textContent = auth.mode === "login" ? "Sign Up" : "Sign In";
    errorEl.classList.add("hidden");
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = emailInput.value.trim();
    const password = passInput.value;
    if (!email || !password) return;

    submitBtn.disabled = true;
    errorEl.classList.add("hidden");

    const endpoint = auth.mode === "login" ? "/api/auth/login" : "/api/auth/signup";
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Auth failed");

      auth.token = data.access_token;
      auth.user = data;
      sessionStorage.setItem("ec3_auth", JSON.stringify(data));
      updateUserPill();
      hideAuthOverlay();
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.classList.remove("hidden");
    } finally {
      submitBtn.disabled = false;
    }
  });
}

function authHeaders() {
  if (auth.token) return { Authorization: `Bearer ${auth.token}` };
  return {};
}

// ---- Init ----
function initialize() {
  load();
  ensureThread();
  renderThreadList();
  renderMessages();
  initDevMode();
  initAuth();

  newChatBtn.addEventListener("click", () => {
    createThread();
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

  checkAuthStatus().then(ok => { if (!ok) showAuthOverlay(); });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = input.value.trim();
    if (!prompt) return;

    const thread = ensureThread();
    if (!thread.messages.length || thread.title === "New chat") thread.title = truncTitle(prompt);

    thread.messages.push({ id: uid(), role: "user", content: prompt, createdAt: now() });
    thread.updatedAt = now();
    save();
    renderThreadList();

    input.value = "";
    sendBtn.disabled = true;
    createMsg("user", prompt);
    const assistantNode = createMsg("assistant", "", { showThinking: true, prompt });
    updateWelcome();

    if (state.devMode) {
      const out = $("#dev-activity");
      if (out) { out.classList.remove("hidden"); out.textContent = "Processing query..."; }
    }

    try {
      await streamChat(prompt, assistantNode, thread);
    } catch (err) {
      const errMsg = `Error: ${err.message}`;
      assistantNode.querySelector(".content").innerHTML = `<div class="error-msg">${escHtml(errMsg)}</div>`;
      appendLog(assistantNode, `Transport error: ${err.message}`);
      thread.messages.push({ id: uid(), role: "assistant", content: errMsg, responsePayload: null, createdAt: now() });
      thread.updatedAt = now();
      save();
      renderThreadList();
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });
}

initialize();
