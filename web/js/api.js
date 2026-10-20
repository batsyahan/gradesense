// GradeSense API client. Every request carries a random per-browser id so each visitor only sees their own history.

function newId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "c-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12);
}

export function clientId() {
  let id = null;
  try { id = localStorage.getItem("gs_client_id"); } catch (e) { /* storage blocked */ }
  if (!id) {
    id = newId();
    try { localStorage.setItem("gs_client_id", id); } catch (e) { /* keep for this page only */ }
  }
  return id;
}
const CLIENT = clientId();

export class ApiError extends Error {
  constructor(message, status, details) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function parse(res) {
  const type = res.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    let msg = (body && body.detail) || `Request failed (${res.status})`;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg || String(d)).join(" ");
    throw new ApiError(String(msg), res.status, body && body.errors);
  }
  return body;
}

export async function api(path, { method = "GET", body, params } = {}) {
  let url = path;
  if (params) {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") q.set(k, v); });
    const s = q.toString();
    if (s) url += (url.includes("?") ? "&" : "?") + s;
  }
  const headers = { "X-Client-Id": CLIENT };
  const opts = { method, headers };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw new ApiError("Cannot reach the GradeSense server. Check that it is running.", 0);
  }
  return parse(res);
}

// Download a file from an endpoint that needs the client header
export async function download(path, params, fallbackName) {
  const q = params ? "?" + new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)) : "";
  let res;
  try {
    res = await fetch(path + q, { headers: { "X-Client-Id": CLIENT } });
  } catch (e) {
    throw new ApiError("Cannot reach the GradeSense server.", 0);
  }
  if (!res.ok) await parse(res);
  const blob = await res.blob();
  const cd = res.headers.get("content-disposition") || "";
  const m = cd.match(/filename="?([^"]+)"?/);
  saveBlob(blob, m ? m[1] : fallbackName);
}

export function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
