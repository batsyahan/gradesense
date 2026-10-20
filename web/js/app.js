// GradeSense app shell: loads server metadata, wires the sidebar, search and router.
import { api } from "./api.js";
import { destroyCharts } from "./charts.js";
import { state } from "./state.js";
import { $, $$, alertBox, debounce, esc, refreshIcons } from "./ui.js";
import * as about from "./pages/about.js";
import * as batch from "./pages/batch.js";
import * as config from "./pages/config.js";
import * as dashboard from "./pages/dashboard.js";
import * as dataset from "./pages/dataset.js";
import * as errors from "./pages/errors.js";
import * as grade from "./pages/grade.js";
import * as history from "./pages/history.js";
import * as insights from "./pages/insights.js";
import * as legal from "./pages/legal.js";
import * as playground from "./pages/playground.js";

const ROUTES = {
  dashboard: { title: "Dashboard", render: dashboard.render },
  grade: { title: "Grade One", render: grade.render },
  batch: { title: "Batch Grading", render: batch.render },
  insights: { title: "Model Insights", render: insights.render },
  dataset: { title: "Dataset Explorer", render: dataset.render },
  errors: { title: "Error Analysis", render: errors.render },
  history: { title: "Grading History", render: history.render },
  playground: { title: "Model Playground", render: playground.render },
  config: { title: "Model Configuration", render: config.render },
  about: { title: "About", render: about.render },
  privacy: { title: "Privacy Policy", render: legal.privacy },
  terms: { title: "Terms and Conditions", render: legal.terms },
};

const view = document.getElementById("view");
let cleanup = null;

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [name, query] = raw.split("?");
  return { name: ROUTES[name] ? name : "dashboard", params: Object.fromEntries(new URLSearchParams(query || "")) };
}

async function route() {
  const { name, params } = parseHash();
  if (typeof cleanup === "function") { try { cleanup(); } catch (e) { /* ignore */ } }
  cleanup = null;
  destroyCharts();
  $$(".nav a").forEach((a) => {
    const on = a.dataset.route === name;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  document.title = `${ROUTES[name].title} - GradeSense`;
  closeSidebar();
  window.scrollTo(0, 0);
  try {
    cleanup = await ROUTES[name].render(view, params);
  } catch (e) {
    console.error(e);
    view.innerHTML = alertBox("error", `This page failed to load: ${esc(e.message)}`);
  }
  refreshIcons();
  view.focus({ preventScroll: true });
}

// ------------------------------------------------------------------ sidebar status
function setStatus(model) {
  const dot = $("#side-model-dot");
  const label = $("#side-model");
  const map = {
    loaded: ["ok", "DeBERTa-v3 (Fine-tuned)"],
    loading: ["warn", "DeBERTa-v3 - loading…"],
    not_loaded: ["warn", "DeBERTa-v3 - starting…"],
    error: ["err", "Model not loaded"],
  };
  const [cls, text] = map[model.status] || ["", model.status];
  dot.className = `dot ${cls}`;
  label.textContent = text;
  label.title = model.error || "";
}

async function pollStatus() {
  try {
    const h = await api("/api/health");
    if (state.meta) state.meta.model.status = h.model;
    if (h.model === "loaded" || h.model === "error") {
      const meta = await api("/api/meta");
      state.meta = meta;
      setStatus(meta.model);
      $("#side-data-dot").className = `dot ${meta.dataset.status === "loaded" ? "ok" : meta.dataset.status === "error" ? "err" : "warn"}`;
      if (meta.dataset.status !== "loaded" && meta.dataset.status !== "error") setTimeout(pollStatus, 4000);
      return;
    }
    setStatus({ status: h.model });
  } catch (e) { /* server unreachable: keep polling */ }
  setTimeout(pollStatus, 3000);
}

// ------------------------------------------------------------------ search
function setupSearch() {
  const input = $("#search-input");
  const box = $("#search-results");
  let items = [];
  let active = -1;

  const close = () => { box.classList.remove("open"); input.setAttribute("aria-expanded", "false"); active = -1; };
  const go = (href) => { close(); input.blur(); location.hash = href; };

  const run = debounce(async () => {
    const q = input.value.trim();
    if (q.length < 2) { close(); return; }
    let d;
    try { d = await api("/api/search", { params: { q } }); } catch (e) { return; }
    items = [];
    let html = "";
    if (d.questions.length) {
      html += `<div class="search-group">Dataset questions</div>`;
      d.questions.forEach((r) => {
        items.push(`#/dataset?split=${r.split}&q=${encodeURIComponent(q)}`);
        html += `<a class="search-item" role="option" href="${items[items.length - 1]}">${esc(r.question.slice(0, 120))}<small>${esc(r.split_name)}</small></a>`;
      });
    }
    if (d.history.length) {
      html += `<div class="search-group">Your grading history</div>`;
      d.history.forEach((r) => {
        items.push(`#/history?q=${encodeURIComponent(q)}`);
        html += `<a class="search-item" role="option" href="${items[items.length - 1]}">${esc(r.answer.slice(0, 100))}<small>${esc(r.verdict)} · ${esc(r.question.slice(0, 70))}</small></a>`;
      });
    }
    if (!items.length) html = `<div class="search-empty">No matches for “${esc(q)}”.</div>`;
    box.innerHTML = html;
    box.classList.add("open");
    input.setAttribute("aria-expanded", "true");
    $$(".search-item", box).forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); go(a.getAttribute("href")); }));
  }, 250);

  input.addEventListener("input", run);
  input.addEventListener("keydown", (e) => {
    const links = $$(".search-item", box);
    if (e.key === "Escape") { close(); return; }
    if (!links.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      active = (active + (e.key === "ArrowDown" ? 1 : links.length - 1)) % links.length;
      links.forEach((l, i) => l.classList.toggle("active", i === active));
    } else if (e.key === "Enter") {
      e.preventDefault();
      go(links[Math.max(0, active)].getAttribute("href"));
    }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) close(); });
}

// ------------------------------------------------------------------ mobile navigation
function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#scrim").classList.remove("open");
  $("#menu-btn").setAttribute("aria-expanded", "false");
}
function setupMenu() {
  $("#menu-btn").addEventListener("click", () => {
    const open = !$("#sidebar").classList.contains("open");
    $("#sidebar").classList.toggle("open", open);
    $("#scrim").classList.toggle("open", open);
    $("#menu-btn").setAttribute("aria-expanded", String(open));
  });
  $("#scrim").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });
}

// ------------------------------------------------------------------ start
async function start() {
  setupMenu();
  refreshIcons();
  try {
    state.meta = await api("/api/meta");
  } catch (e) {
    view.innerHTML = alertBox("error", `Cannot connect to the GradeSense server: ${esc(e.message)}`);
    refreshIcons();
    setTimeout(start, 4000);
    return;
  }
  const m = state.meta;
  $("#side-version").textContent = `v${m.version}`;
  setStatus(m.model);
  if (m.profile.name) {
    $("#profile").hidden = false;
    $("#profile-name").textContent = m.profile.name;
    $("#profile-sub").textContent = m.profile.subtitle || "";
    $("#profile-initials").textContent = m.profile.name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("");
  }
  setupSearch();
  window.addEventListener("hashchange", route);
  route();
  if (m.model.status !== "loaded" || m.dataset.status !== "loaded") pollStatus();
}

// Wait for the CDN scripts (deferred) - module scripts run after them, but guard anyway
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
else start();
