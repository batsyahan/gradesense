// Chart.js wrappers. Charts are tracked per canvas so re-rendering never leaks old charts.

const registry = new Map();
const PALETTE = ["#1e3a8a", "#2563eb", "#93c5fd", "#64748b", "#16a34a"];

function ready() { return typeof window.Chart !== "undefined"; }

export function destroyCharts(root) {
  for (const [canvas, chart] of registry) {
    if (!root || root.contains(canvas) || !document.body.contains(canvas)) {
      chart.destroy();
      registry.delete(canvas);
    }
  }
}

function make(canvas, config) {
  if (!canvas) return null;
  if (!ready()) {
    canvas.replaceWith(Object.assign(document.createElement("p"), {
      className: "help", textContent: "Charts could not load (no connection to the chart library CDN).",
    }));
    return null;
  }
  if (registry.has(canvas)) registry.get(canvas).destroy();
  const chart = new window.Chart(canvas.getContext("2d"), config);
  registry.set(canvas, chart);
  return chart;
}

const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart, args, opts) {
    if (!opts || !opts.enabled) return;
    const { ctx } = chart;
    ctx.save();
    ctx.font = "11px Inter, Segoe UI, sans-serif";
    ctx.fillStyle = "#334155";
    ctx.textAlign = "center";
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden) return;
      meta.data.forEach((bar, j) => {
        const v = ds.data[j];
        if (v === null || v === undefined) return;
        ctx.fillText(Number(v).toFixed(opts.digits ?? 2), bar.x, bar.y - 4);
      });
    });
    ctx.restore();
  },
};

const baseFont = { family: "Inter, Segoe UI, sans-serif", size: 12 };

export function groupedBar(canvas, { labels, series, yTitle, max = 1, showValues = true, digits = 2 }) {
  return make(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: series.map((s, i) => ({
        label: s.label, data: s.data, backgroundColor: s.color || PALETTE[i % PALETTE.length],
        borderRadius: 2, maxBarThickness: 34,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { top: 16 } },
      plugins: {
        legend: { position: "top", labels: { boxWidth: 12, font: baseFont } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.raw === null ? "not evaluated" : Number(c.raw).toFixed(3)}` } },
        valueLabels: { enabled: showValues, digits },
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: baseFont, maxRotation: 0, autoSkip: false } },
        y: { min: 0, max, title: { display: !!yTitle, text: yTitle, font: baseFont }, ticks: { font: baseFont } },
      },
    },
    plugins: [valueLabels],
  });
}

export function simpleBar(canvas, { labels, data, colors, yTitle, xTitle, integer = true }) {
  return make(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: colors || "#2563eb", borderRadius: 2, maxBarThickness: 44 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, title: { display: !!xTitle, text: xTitle, font: baseFont }, ticks: { font: baseFont } },
        y: { beginAtZero: true, title: { display: !!yTitle, text: yTitle, font: baseFont },
             ticks: { font: baseFont, precision: integer ? 0 : undefined } },
      },
    },
  });
}

export function lineChart(canvas, { labels, series, yTitle, xTitle }) {
  return make(canvas, {
    type: "line",
    data: {
      labels,
      datasets: series.map((s, i) => ({
        label: s.label, data: s.data, borderColor: s.color || PALETTE[i + 1], backgroundColor: s.color || PALETTE[i + 1],
        yAxisID: s.axis || "y", tension: 0.25, pointRadius: 3,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top", labels: { boxWidth: 12, font: baseFont } } },
      scales: {
        x: { title: { display: !!xTitle, text: xTitle, font: baseFont }, grid: { display: false } },
        y: { position: "left", title: { display: !!yTitle, text: yTitle, font: baseFont } },
        y2: { position: "right", min: 0, max: 1, grid: { drawOnChartArea: false },
              title: { display: true, text: "Validation macro-F1", font: baseFont } },
      },
    },
  });
}
