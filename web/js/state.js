// Shared client state: server metadata, this browser's preferences, the last grading result.

const PREF_KEY = "gs_prefs";
export const LABEL_KEYS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"];

export const state = {
  meta: null,
  lastResult: null,   // { inputs, result } from the most recent grading in this tab
  draft: null,        // inputs handed from History / Playground to the grading form
};

function readPrefs() {
  try { return JSON.parse(localStorage.getItem(PREF_KEY) || "{}"); } catch (e) { return {}; }
}

export function prefs() {
  const p = readPrefs();
  const m = state.meta || {};
  return {
    defaultMaxMarks: p.defaultMaxMarks ?? 2,
    explain: p.explain ?? true,
    showConfidence: p.showConfidence ?? (m.show_confidence ?? true),
    marksMapping: p.marksMapping ?? null,   // null = use the server default
  };
}

export function savePrefs(update) {
  const p = { ...readPrefs(), ...update };
  try { localStorage.setItem(PREF_KEY, JSON.stringify(p)); } catch (e) { /* ignore */ }
  return prefs();
}

export function resetPrefs() {
  try { localStorage.removeItem(PREF_KEY); } catch (e) { /* ignore */ }
  return prefs();
}

export function effectiveMapping() {
  return prefs().marksMapping || (state.meta && state.meta.marks_mapping) || null;
}

try {
  const saved = sessionStorage.getItem("gs_last_result");
  if (saved) state.lastResult = JSON.parse(saved);
} catch (e) { /* ignore */ }

export function setLastResult(value) {
  state.lastResult = value;
  try { sessionStorage.setItem("gs_last_result", JSON.stringify(value)); } catch (e) { /* ignore */ }
}
