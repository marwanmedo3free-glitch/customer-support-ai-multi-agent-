const $ = (id) => document.getElementById(id);

/* ================= Tabs ================= */
const views = { chat: $("view-chat"), agent: $("view-agent") };
function showView(name) {
  for (const [key, el] of Object.entries(views)) el.hidden = key !== name;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  history.replaceState(null, "", name === "agent" ? "#agent" : "#");
  name === "agent" ? startAgentPolling() : clearInterval(agentTimer);
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

/* ================= Customer chat ================= */
const log = $("log"), input = $("input"), sendBtn = $("send"), statusEl = $("status");
let threadId = null, busy = false, pollTimer = null;
const DEBUG = location.search.includes("debug");   // open /?debug to see the path each message took

function add(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;               // textContent: never inject model output as HTML
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function setBusy(state, label) {
  busy = state;
  sendBtn.disabled = input.disabled = state;
  statusEl.textContent = label || (state ? "Thinking…" : "Online");
  if (!state) input.focus();
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Customer-Id": $("customer").value },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : "Request failed (" + res.status + ")");
  }
  return data;
}

async function send(text) {
  if (busy || !text.trim()) return;
  add("user", text);
  input.value = "";
  setBusy(true);
  const typing = add("bot typing", "Typing…");
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, thread_id: threadId }),
    });
    threadId = data.thread_id;
    typing.remove();
    add("bot", data.reply);
    if (DEBUG && data.trace) add("note", "path: " + data.trace.join(" → "));
    if (data.status === "pending_human_approval") {
      add("note", "A team member is reviewing your request. This page will update automatically.");
      setBusy(true, "Waiting for a team member…");
      startStatusPolling();
    } else {
      setBusy(false);
    }
  } catch (err) {
    typing.remove();
    add("note", "Something went wrong: " + err.message);
    setBusy(false);
  }
}

function startStatusPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const s = await api("/api/status/" + encodeURIComponent(threadId));
      if (!s.waiting_for_human) {
        clearInterval(pollTimer);
        add("bot", s.last_message || "Your request has been reviewed.");
        setBusy(false);
      }
    } catch (err) {
      clearInterval(pollTimer);
      add("note", "Lost connection while waiting: " + err.message);
      setBusy(false);
    }
  }, 3000);
}

function resetChat() {
  clearInterval(pollTimer);
  threadId = null;
  log.innerHTML = "";
  add("bot", "Hi! I can track orders, change a delivery address, handle returns and refunds, and answer questions about our policies and products.");
  setBusy(false);
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
$("chips").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") send(e.target.textContent); });
$("new").addEventListener("click", resetChat);
$("customer").addEventListener("change", resetChat);   // a thread belongs to one customer

/* ================= Agent console ================= */
const tokenEl = $("token"), listEl = $("list"), errEl = $("error"), countEl = $("count");
let agentTimer = null;
tokenEl.value = sessionStorage.getItem("agentToken") || "";

function showError(msg) { errEl.hidden = !msg; errEl.textContent = msg || ""; }

async function agentApi(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Agent-Token": tokenEl.value },
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 403) throw new Error("Invalid agent token");
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
  return data;
}

function row(dl, label, value) {
  const dt = document.createElement("dt"), dd = document.createElement("dd");
  dt.textContent = label; dd.textContent = value;
  dl.append(dt, dd);
}

function reviewCard(id, p) {
  const el = document.createElement("article");
  el.className = "card";
  const h = document.createElement("h2");
  h.textContent = "Refund for order #" + p.order_id + " · $" + Number(p.amount).toFixed(2);
  const dl = document.createElement("dl");
  row(dl, "Customer id", p.customer_id);
  row(dl, "Prior refunds", p.prior_refunds);
  row(dl, "Thread", id);
  const quote = document.createElement("blockquote");
  quote.textContent = p.customer_message;

  const feedback = document.createElement("input");
  feedback.placeholder = "Message to customer if rejecting (optional)";
  const ok = document.createElement("button"); ok.className = "good"; ok.textContent = "Approve refund";
  const no = document.createElement("button"); no.className = "bad"; no.textContent = "Reject";

  async function decide(approved) {
    ok.disabled = no.disabled = true;
    try {
      await agentApi("/api/approve", {
        method: "POST",
        body: JSON.stringify({ thread_id: id, approved, feedback: feedback.value || null }),
      });
      showError("");
      loadPending();
    } catch (err) { showError(err.message); ok.disabled = no.disabled = false; }
  }
  ok.onclick = () => decide(true);
  no.onclick = () => decide(false);

  const actions = document.createElement("div");
  actions.className = "row";
  actions.append(feedback, ok, no);
  el.append(h, dl, quote, actions);
  return el;
}

async function loadPending() {
  if (!tokenEl.value) return;
  try {
    const pending = await agentApi("/api/pending");
    showError("");
    const entries = Object.entries(pending);
    countEl.textContent = entries.length + " pending review" + (entries.length === 1 ? "" : "s");
    listEl.replaceChildren(...(entries.length
      ? entries.map(([id, p]) => reviewCard(id, p))
      : [Object.assign(document.createElement("p"), { className: "empty", textContent: "Nothing waiting for review." })]));
  } catch (err) { showError(err.message); clearInterval(agentTimer); }
}

function startAgentPolling() {
  clearInterval(agentTimer);
  if (!tokenEl.value) return;
  loadPending();
  agentTimer = setInterval(loadPending, 5000);
}

$("save").addEventListener("click", () => {
  sessionStorage.setItem("agentToken", tokenEl.value);
  startAgentPolling();
});

/* ================= Start ================= */
resetChat();
showView(location.hash === "#agent" ? "agent" : "chat");
