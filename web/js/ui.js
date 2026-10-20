// Small DOM helpers shared by every page.

export function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function icon(name, cls = "") {
  return `<i data-lucide="${name}" class="${cls}" aria-hidden="true"></i>`;
}

export function refreshIcons() {
  const L = window.lucide;
  if (!L || !L.createIcons) return;   // CDN unavailable: icons are decorative, text labels remain
  try {
    L.createIcons(L.icons ? { icons: L.icons } : undefined);
  } catch (e) { /* ignore icon errors */ }
}

export function $(sel, root = document) { return root.querySelector(sel); }
export function $$(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }

export function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "Not evaluated";
  return Number(v).toFixed(digits);
}
export function pct(v, digits = 0) {
  if (v === null || v === undefined) return "-";
  return (v * 100).toFixed(digits) + "%";
}
export function num(v) {
  if (v === null || v === undefined) return "-";
  return Number(v).toLocaleString("en-IN");
}
export function marks(v) {
  return Number.isInteger(v) ? String(v) : Number(v).toFixed(2).replace(/\.?0+$/, "");
}
export function plural(n, word) { return `${num(n)} ${word}${n === 1 ? "" : "s"}`; }

export const VERDICT_COLOR = { "Correct": "green", "Partially Correct": "amber", "Incorrect": "red" };
export const LABEL_COLOR = {
  correct: "green", partially_correct_incomplete: "amber", contradictory: "red", irrelevant: "red", non_domain: "red",
};
export const LABEL_NAME = {
  correct: "Correct", contradictory: "Contradictory", partially_correct_incomplete: "Partially Correct",
  irrelevant: "Irrelevant", non_domain: "Non-Answer",
};

export function verdictBadge(verdict) {
  const c = VERDICT_COLOR[verdict] || "gray";
  const ic = c === "green" ? "circle-check" : c === "amber" ? "circle-alert" : "circle-x";
  return `<span class="badge ${c}">${icon(ic)}${esc(verdict)}</span>`;
}
export function labelBadge(label) {
  return `<span class="badge ${LABEL_COLOR[label] || "gray"}">${esc(LABEL_NAME[label] || label)}</span>`;
}

export function alertBox(kind, text, ic) {
  const icons = { info: "info", warn: "triangle-alert", error: "circle-x", success: "circle-check" };
  return `<div class="alert ${kind}" role="${kind === "error" ? "alert" : "status"}">${icon(ic || icons[kind])}<div>${text}</div></div>`;
}

export function emptyState(title, text, ic = "inbox") {
  return `<div class="empty">${icon(ic)}<h3>${esc(title)}</h3><p>${text}</p></div>`;
}

export function loading(text = "Loading…") {
  return `<div class="loading-block"><span class="spinner dark"></span>${esc(text)}</div>`;
}

let toastTimer;
export function toast(message) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.className = "toast";
    el.setAttribute("role", "status");
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

export function sparkline(values, color = "#16a34a") {
  if (!values || values.length < 2 || values.every((v) => v === 0)) return "";
  const w = 100, h = 30, max = Math.max(...values), min = Math.min(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - 3 - ((v - min) / span) * (h - 6)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline fill="none" stroke="${color}" stroke-width="1.6" points="${pts}"/></svg>`;
}

export function probRows(probabilities) {
  return probabilities.map((p) => {
    const c = LABEL_COLOR[p.label] === "green" ? "" : LABEL_COLOR[p.label] === "amber" ? "amber" : "red";
    return `<div class="prob-row"><span>${esc(p.name)}</span><div class="bar ${c}"><div style="width:${(p.p * 100).toFixed(1)}%"></div></div><span class="num">${pct(p.p, 1)}</span></div>`;
  }).join("");
}

export function pager(total, offset, limit) {
  const from = total ? offset + 1 : 0;
  const to = Math.min(total, offset + limit);
  return `<div class="pager"><span>${num(from)}-${num(to)} of ${num(total)}</span>
    <div class="btn-row"><button class="btn sm" data-page="prev" ${offset <= 0 ? "disabled" : ""}>Previous</button>
    <button class="btn sm" data-page="next" ${to >= total ? "disabled" : ""}>Next</button></div></div>`;
}

export function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

export function errorMessage(e) {
  return esc(e && e.message ? e.message : String(e));
}
