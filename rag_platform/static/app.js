const API = {
  health: "/health",
  rag: "/api/v1/rag/ask",
  sql: "/api/v1/sql/query",
  upload: "/api/v1/documents/upload",
  reindex: "/api/v1/documents/reindex",
};

const VIEW_META = {
  rag: {
    title: "智能问答",
    sub: "基于私有运维知识库的故障排查助手",
  },
  sql: {
    title: "数据查询",
    sub: "用自然语言查询巡检记录，自动生成只读 SQL",
  },
  docs: {
    title: "文档管理",
    sub: "上传运维文档并维护向量知识库",
  },
};

const ICON = {
  bot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><rect x="4" y="8.5" width="16" height="11.5" rx="3.5"></rect><line x1="12" y1="4.5" x2="12" y2="8.5"></line><circle cx="12" cy="3.4" r="1.4"></circle><circle cx="9" cy="13.5" r="1.3" fill="currentColor" stroke="none"></circle><circle cx="15" cy="13.5" r="1.3" fill="currentColor" stroke="none"></circle><line x1="9" y1="17" x2="15" y2="17"></line></svg>',
  user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="8" r="3.6"></circle><path d="M5 20c0-3.3 3.1-5.5 7-5.5s7 2.2 7 5.5"></path></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"></path></svg>',
};

const state = {
  health: null,
  activeView: "rag",
  busy: { rag: false, sql: false, docs: false },
};

const $ = (selector, root = document) => root.querySelector(selector);

/* ------------------------------------------------------------- 基础工具 */
function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function errorMessage(detail) {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item && (item.msg || item.message)).filter(Boolean).join("；");
  }
  if (typeof detail === "object") {
    if (detail.error) return String(detail.error);
    if (detail.detail) return errorMessage(detail.detail);
    if (detail.message) return String(detail.message);
  }
  return String(detail);
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const text = await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    if (!response.ok) {
      const detail =
        data && Object.prototype.hasOwnProperty.call(data, "detail") ? data.detail : data;
      const error = new Error(errorMessage(detail) || `请求失败（HTTP ${response.status}）`);
      error.status = response.status;
      error.detail = detail;
      throw error;
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

function setNote(element, message, isError = false) {
  element.textContent = message;
  element.className = `result-note show${isError ? " error" : ""}`;
}

/* ------------------------------------------------------------- 视图切换 */
function switchView(view) {
  if (!VIEW_META[view]) return;
  state.activeView = view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.id === `view-${view}`);
  });
  $("#page-title").textContent = VIEW_META[view].title;
  $("#page-sub").textContent = VIEW_META[view].sub;
  if (view === "rag") $("#rag-input").focus();
  if (view === "sql") $("#sql-input").focus();
}

/* ------------------------------------------------------------- 健康检查 */
function statusClass(status) {
  if (status === "up") return "up";
  if (status === "degraded") return "degraded";
  return "down";
}

function renderHealth(data) {
  state.health = data;
  const pills = [
    { label: "向量库", status: data.qdrant.status, extra: `${data.qdrant.chunks} 切片` },
    { label: "缓存", status: data.redis.status, extra: data.redis.backend },
    { label: "模型", status: data.llm.status, extra: data.llm.chat_model },
    { label: "数据库", status: data.sqlite.status, extra: `${data.sqlite.rows} 条` },
  ];
  $("#status-group").innerHTML = pills
    .map(
      (pill) =>
        `<div class="status-pill ${statusClass(pill.status)}" title="${escapeHtml(
          pill.label
        )}：${escapeHtml(pill.status)}">${escapeHtml(pill.label)} · ${escapeHtml(pill.extra)}</div>`
    )
    .join("");

  const sideDot = $("#side-dot");
  const sideText = $("#side-status");
  if (data.status === "ok") {
    sideDot.className = "dot is-ok";
    sideText.textContent = "全部依赖正常";
  } else {
    sideDot.className = "dot is-warn";
    sideText.textContent = "部分依赖降级";
  }

  $("#stat-collection").textContent = data.qdrant.collection;
  $("#stat-chunks").textContent = `${data.qdrant.chunks} 个切片`;
  $("#stat-chat-model").textContent = data.llm.chat_model;
  $("#stat-embed-model").textContent = data.llm.embed_model;
}

function renderHealthError() {
  $("#status-group").innerHTML = '<div class="status-pill down">服务未连接</div>';
  $("#side-dot").className = "dot is-err";
  $("#side-status").textContent = "服务未连接";
}

async function refreshHealth() {
  try {
    renderHealth(await api(API.health));
  } catch {
    renderHealthError();
  }
}

/* ------------------------------------------------------------------ 问答 */
function hideWelcome() {
  const welcome = $("#welcome");
  if (welcome) welcome.style.display = "none";
}

function appendMessage(role, contentNode) {
  hideWelcome();
  const row = document.createElement("div");
  row.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = role === "user" ? ICON.user : ICON.bot;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.appendChild(contentNode);

  row.append(avatar, bubble);
  $("#chat").appendChild(row);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  return row;
}

function textNode(text) {
  const div = document.createElement("div");
  div.style.whiteSpace = "pre-wrap";
  div.textContent = text;
  return div;
}

function appendTyping(label) {
  const wrap = document.createElement("div");
  wrap.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;
  const caption = document.createElement("div");
  caption.style.cssText = "margin-top:8px;color:#667085;font-size:12.5px;";
  caption.textContent = label;
  wrap.appendChild(caption);
  return appendMessage("bot", wrap);
}

function setRagBusy(busy) {
  $("#rag-send").disabled = busy;
  $("#rag-input").disabled = busy;
}

function renderRagAnswer(data) {
  const wrap = document.createElement("div");
  wrap.appendChild(textNode(data.answer || ""));

  if (Array.isArray(data.sources) && data.sources.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    data.sources.forEach((item) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.title = `${item.source} · ${item.title || "切片 " + item.chunk_id}`;
      const score = typeof item.score === "number" ? item.score.toFixed(3) : "—";
      chip.innerHTML = `${ICON.spark}<span>${escapeHtml(item.source)} #${escapeHtml(
        item.chunk_id
      )} · <b>${escapeHtml(score)}</b></span>`;
      sources.appendChild(chip);
    });
    wrap.appendChild(sources);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  const confidence = Math.round(Number(data.confidence || 0) * 100);
  meta.innerHTML = [
    data.found ? `置信度 ${confidence}%` : "知识库未命中",
    `耗时 ${escapeHtml(data.latency_ms ?? 0)} ms`,
    data.cached ? '<span class="cached">缓存命中</span>' : "实时生成",
  ].join("<span>·</span>");
  wrap.appendChild(meta);

  appendMessage("bot", wrap);
}

function renderRagError(error) {
  const wrap = document.createElement("div");
  wrap.style.whiteSpace = "pre-wrap";
  const title = document.createElement("strong");
  title.textContent =
    error.status === 503 ? "服务暂不可用" : error.status === 422 ? "请求未通过校验" : "请求失败";
  const message = document.createElement("div");
  message.style.marginTop = "6px";
  message.textContent = error.message || "未知错误";
  wrap.append(title, message);
  const row = appendMessage("bot", wrap);
  row.querySelector(".bubble").classList.add("error");
}

async function askRag(rawQuestion) {
  const question = String(rawQuestion || "").trim();
  if (!question || state.busy.rag) return;

  state.busy.rag = true;
  setRagBusy(true);
  appendMessage("user", textNode(question));
  const typing = appendTyping("正在检索知识库并生成答案…");

  try {
    const data = await api(API.rag, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    typing.remove();
    renderRagAnswer(data);
  } catch (error) {
    typing.remove();
    renderRagError(error);
  } finally {
    state.busy.rag = false;
    setRagBusy(false);
    $("#rag-input").focus();
  }
}

/* ------------------------------------------------------------- Text2SQL */
function setSqlBusy(busy) {
  $("#sql-send").disabled = busy;
  $("#sql-input").disabled = busy;
}

function renderSqlSuccess(data) {
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const columns =
    Array.isArray(data.columns) && data.columns.length
      ? data.columns
      : rows.length
        ? Object.keys(rows[0])
        : [];

  let tableHtml;
  if (!rows.length) {
    tableHtml = `<div class="empty-state"><p>查询成功，但没有匹配记录</p><span>可以换个时间范围或筛选条件再试。</span></div>`;
  } else {
    const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
    const body = rows
      .map((row) => {
        const cells = columns
          .map((column) => {
            let value = row[column];
            if (value === null || value === undefined) value = "—";
            else if (typeof value === "object") value = JSON.stringify(value);
            return `<td>${escapeHtml(value)}</td>`;
          })
          .join("");
        return `<tr>${cells}</tr>`;
      })
      .join("");
    tableHtml = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  const badges = [
    `<span class="badge">${escapeHtml(data.row_count ?? rows.length)} 行</span>`,
    `<span class="badge">${escapeHtml(data.latency_ms ?? 0)} ms</span>`,
    data.cached ? '<span class="badge ok">缓存命中</span>' : '<span class="badge">实时生成</span>',
    '<span class="badge ok">只读校验通过</span>',
  ];
  if (data.truncated) badges.push('<span class="badge warn">结果已截断</span>');

  const result = $("#sql-result");
  result.className = "sql-result";
  result.innerHTML = `
    <div class="result-card">
      <div class="result-top">
        <div class="explain">${ICON.spark}<span>${escapeHtml(data.explain || "未提供查询说明")}</span></div>
        <div class="badges">${badges.join("")}</div>
      </div>
      <pre class="sql-block">${escapeHtml(data.sql || "")}</pre>
      ${tableHtml}
    </div>`;
}

function renderSqlError(error) {
  const detail = error.detail && typeof error.detail === "object" ? error.detail : null;
  const sql = detail && detail.sql ? String(detail.sql) : "";
  const message = error.message || "未知错误";
  const result = $("#sql-result");
  result.className = "sql-result";
  result.innerHTML = `
    <div class="result-error">
      <strong>${error.status === 422 ? "SQL 未通过安全校验，请求已拦截" : "查询失败"}</strong>
      <span>${escapeHtml(message)}</span>
      ${sql ? `<code>${escapeHtml(sql)}</code>` : ""}
    </div>`;
}

async function runSql(rawQuestion) {
  const question = String(rawQuestion || "").trim();
  if (!question || state.busy.sql) return;

  state.busy.sql = true;
  setSqlBusy(true);
  const result = $("#sql-result");
  result.className = "sql-result";
  result.innerHTML = `<div class="result-card"><div class="explain">${ICON.spark}<span>正在生成 SQL 并执行…</span></div></div>`;

  try {
    renderSqlSuccess(await api(API.sql, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }));
  } catch (error) {
    renderSqlError(error);
  } finally {
    state.busy.sql = false;
    setSqlBusy(false);
    $("#sql-input").focus();
  }
}

/* ------------------------------------------------------------- 文档管理 */
function setDocsBusy(busy) {
  state.busy.docs = busy;
  $("#reindex-btn").disabled = busy;
}

async function uploadFile(file) {
  if (!file || state.busy.docs) return;
  const name = file.name || "";
  if (!/\.(md|txt)$/i.test(name)) {
    setNote($("#upload-result"), "只支持 .md / .txt 文件", true);
    return;
  }
  if (file.size > 2 * 1024 * 1024) {
    setNote($("#upload-result"), "文件超过 2 MB 限制", true);
    return;
  }

  setDocsBusy(true);
  setNote($("#upload-result"), `正在上传并向量化 ${name}…`);

  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api(API.upload, { method: "POST", body: form });
    setNote(
      $("#upload-result"),
      `已入库：${data.source || name} · ${data.chunks ?? 0} 个切片，现在就可以在问答中检索到。`
    );
    refreshHealth();
  } catch (error) {
    setNote($("#upload-result"), error.message || "上传失败", true);
  } finally {
    setDocsBusy(false);
    $("#file-input").value = "";
  }
}

async function reindex() {
  if (state.busy.docs) return;
  const confirmed = window.confirm("重建会重新向量化全部内置文档，期间问答可能短暂不可用，继续吗？");
  if (!confirmed) return;

  setDocsBusy(true);
  setNote($("#reindex-result"), "正在重建知识库…");
  try {
    const data = await api(API.reindex, { method: "POST" });
    const failed = Array.isArray(data.failed) ? data.failed.length : 0;
    setNote(
      $("#reindex-result"),
      `重建完成：${data.documents ?? 0} 篇文档 / ${data.chunks ?? 0} 个切片${failed ? `，${failed} 个失败` : ""}。`
    );
    refreshHealth();
  } catch (error) {
    setNote($("#reindex-result"), error.message || "重建失败", true);
  } finally {
    setDocsBusy(false);
  }
}

/* ------------------------------------------------------------------ 绑定 */
function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });

  const ragForm = $("#rag-form");
  const ragInput = $("#rag-input");

  ragForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = ragInput.value;
    ragInput.value = "";
    ragInput.style.height = "auto";
    askRag(question);
  });

  ragInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ragForm.requestSubmit();
    }
  });

  ragInput.addEventListener("input", () => {
    ragInput.style.height = "auto";
    ragInput.style.height = `${Math.min(ragInput.scrollHeight, 140)}px`;
  });

  document.querySelectorAll("#rag-samples .chip").forEach((chip) => {
    chip.addEventListener("click", () => askRag(chip.textContent));
  });

  const sqlForm = $("#sql-form");
  const sqlInput = $("#sql-input");

  sqlForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runSql(sqlInput.value);
  });

  document.querySelectorAll("#sql-samples .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      sqlInput.value = chip.textContent;
      runSql(chip.textContent);
    });
  });

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");

  fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  $("#reindex-btn").addEventListener("click", reindex);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshHealth();
  });
}

/* ------------------------------------------------------------------ 启动 */
function init() {
  bindEvents();
  refreshHealth();
  setInterval(refreshHealth, 15000);
}

init();
