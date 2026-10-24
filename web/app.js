const PRIMARY_SVD_RANK = 16;
const PRIMARY_SVD_PREFIX = `results_svd_hybrid_r${PRIMARY_SVD_RANK}`;
const PRIMARY_FINALIZED_FOLDER = "results_svd_hybrid_r16_20260402_030346";
const SCENARIO_ORDER = ["clean", "attack", "defended", "defended_dp"];
const SCENARIO_META = {
  clean: { label: "Clean", badge: "clean", color: "#5ea8ff" },
  attack: { label: "Attack", badge: "attack", color: "#f59e0b" },
  defended: { label: "Defended", badge: "defended", color: "#34d399" },
  defended_dp: { label: "Defended + DP", badge: "defendedDp", color: "#fb7185" },
};
const visible = { clean: true, attack: true, defended: true, defended_dp: true };

const state = {
  selectedFolder: null,
  selectedFolderMtime: null,
  selectedFiles: [],
  lastTruth: null,
  lastCurves: null,
  modelFilesInFolder: [],
  preferredDefault: null,
  didAutoLoadDefaultModel: false,
  folders: [],
  modelStatus: null,
  reference: {
    primaryFolder: null,
    quantumBaselineFolder: null,
    svdFolders: [],
    classicalFolders: {},
    rankRows: [],
    classicalRows: [],
  },
  folderFilesCache: new Map(),
  folderCurvesCache: new Map(),
  fileTextCache: new Map(),
};

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function setStatus(text, kind = "neutral") {
  const pill = el("statusPill");
  if (!pill) return;
  pill.textContent = text;
  pill.classList.toggle("danger", kind === "error");
}

function showGlobalMessage(text, kind = "neutral") {
  const box = el("globalMessage");
  if (!box) return;
  box.textContent = text;
  box.classList.remove("error", "success");
  if (kind === "error") box.classList.add("error");
  if (kind === "success") box.classList.add("success");
  box.style.display = "block";
}

function clearGlobalMessage() {
  const box = el("globalMessage");
  if (!box) return;
  box.style.display = "none";
  box.textContent = "";
  box.classList.remove("error", "success");
}

function prettyLabel(lbl) {
  // Loader convention: 0 = Attack, 1 = Normal.
  if (lbl === "Class 0") return "Attack";
  if (lbl === "Class 1") return "Normal";
  return lbl || "—";
}

function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = Number(bytes);
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}

function fmtPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNum(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function fmtDelta(delta) {
  if (delta === null || delta === undefined || Number.isNaN(Number(delta))) {
    return { text: "No delta", cls: "deltaFlat" };
  }
  const value = Number(delta);
  if (Math.abs(value) < 1e-9) return { text: "No change", cls: "deltaFlat" };
  return {
    text: `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)} pts`,
    cls: value > 0 ? "deltaUp" : "deltaDown",
  };
}

function scenarioBadge(key, labelOverride) {
  const meta = SCENARIO_META[key] || { label: key, badge: "neutral" };
  return `<span class="scenarioBadge ${meta.badge}">${escapeHtml(labelOverride || meta.label)}</span>`;
}

function scenarioLabel(key) {
  return SCENARIO_META[key]?.label || key;
}

function isPrimarySvdFolder(name) {
  return (name || "").toLowerCase().startsWith(PRIMARY_SVD_PREFIX.toLowerCase());
}

function inferRunType(folderName, files) {
  const names = (files || []).map((file) => (typeof file === "string" ? file : file.name).toLowerCase());
  if ((folderName || "").toLowerCase().includes("svd") || names.some((name) => name.includes("_svd_"))) {
    return "SVD Hybrid QNN";
  }
  if ((folderName || "").toLowerCase().includes("classical")) {
    return "Classical Baseline";
  }
  return "Baseline HQNN";
}

function parseRankFromName(name) {
  const match = String(name || "").match(/_r(\d+)/i);
  return match ? Number(match[1]) : null;
}

function getScenarioKeysFromFiles(files) {
  const names = new Set((files || []).map((file) => (file.name || file).toLowerCase()));
  return SCENARIO_ORDER.filter((key) => names.has(`${key}.csv`));
}

function getVisibleKeys(curves) {
  return Object.keys(curves || {}).filter((key) => visible[key] !== false);
}

function average(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function latestFolderByPrefix(prefix) {
  const candidates = state.folders.filter((folder) => folder.name.startsWith(prefix));
  if (!candidates.length) return null;
  candidates.sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  return candidates[0].name;
}

function pickDefaultFolder(folders) {
  if (!folders?.length) return null;

  const finalized = folders.find((folder) => folder.name === PRIMARY_FINALIZED_FOLDER);
  if (finalized) return finalized;

  if (state.preferredDefault?.folder) {
    const exact = folders.find((folder) => folder.name === state.preferredDefault.folder);
    if (exact) return exact;
  }

  const svdCandidates = folders.filter((folder) => isPrimarySvdFolder(folder.name));
  if (svdCandidates.length) {
    svdCandidates.sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
    return svdCandidates[0];
  }

  return folders[0];
}

function chooseQuantumBaselineFolder() {
  const preferred = ["results_full_throttle_100pct", "results_full_throttle"];
  for (const name of preferred) {
    if (state.folders.some((folder) => folder.name === name)) return name;
  }

  const fallback = state.folders.find((folder) => {
    const lower = folder.name.toLowerCase();
    return lower.startsWith("results_") && !lower.includes("svd") && !lower.includes("classical");
  });
  return fallback?.name || null;
}

function updatePrimaryPathTexts() {
  const primaryFolder = state.reference.primaryFolder || state.preferredDefault?.folder;
  const source = state.preferredDefault?.source;
  const headline = el("primaryPathHeadline");
  const meta = el("primaryPathMeta");
  const note = el("primaryPathNote");
  const sidebarNote = el("sidebarPrimaryNote");

  if (headline) headline.textContent = primaryFolder ? "SVD-hybrid r16" : "Waiting for runs";
  if (meta) {
    meta.textContent = primaryFolder
      ? `${primaryFolder}${source ? ` • source: ${source}` : ""}`
      : "No primary run found.";
  }
  const noteText = primaryFolder
    ? "Preferred run from current artifacts."
    : "Primary run will appear here.";
  if (note) note.textContent = noteText;
  if (sidebarNote) sidebarNote.textContent = noteText;
}

function renderScenarioAvailability(keys) {
  const container = el("scenarioAvailability");
  const meta = el("scenarioAvailabilityMeta");
  if (!container) return;

  if (!keys?.length) {
    container.innerHTML = `<span class="scenarioBadge neutral">No scenario CSVs found</span>`;
    if (meta) meta.textContent = "No scenario CSVs in this run.";
    return;
  }

  container.innerHTML = keys.map((key) => scenarioBadge(key)).join("");
  if (meta) meta.textContent = `${keys.length} scenario CSVs found.`;
}

function updateRunSummary(folderName, files, curves) {
  const nameEl = el("activeRunName");
  const typeEl = el("activeRunType");
  const updatedEl = el("activeRunUpdated");
  const notesEl = el("activeRunNotes");

  if (nameEl) nameEl.textContent = folderName || "No run";
  if (typeEl) typeEl.textContent = inferRunType(folderName, files);

  if (updatedEl) {
    updatedEl.textContent = state.selectedFolderMtime
      ? `Updated: ${new Date(state.selectedFolderMtime * 1000).toLocaleString()}`
      : "Updated: —";
  }

  const modelCount = (files || []).filter((file) => file.name?.toLowerCase().startsWith("model_") && file.name?.toLowerCase().endsWith(".pt")).length;
  const curveCount = Object.keys(curves || {}).length;
  const rank = parseRankFromName(folderName);
  const summaryText = inferRunType(folderName, files) === "SVD Hybrid QNN"
    ? `SVD artifacts present${rank ? ` • rank ${rank}` : ""} • ${curveCount}/4 scenario curves • ${modelCount} checkpoint(s)`
    : `${curveCount}/4 scenario curves • ${modelCount} checkpoint(s) exposed in this run`;

  if (notesEl) notesEl.textContent = folderName ? summaryText : "Waiting for run data";
  renderScenarioAvailability(getScenarioKeysFromFiles(files));
}

function setLoadedModelHeadline(status) {
  const headline = el("loadedModelHeadline");
  const meta = el("loadedModelMeta");
  const mini = el("loadedMini");

  if (!status?.loaded) {
    if (headline) headline.textContent = "No model loaded";
    if (meta) meta.textContent = "Load from Inference";
    if (mini) mini.textContent = "—";
    return;
  }

  if (headline) headline.textContent = status.model_file || "Model loaded";
  if (meta) {
    const extras = [
      status.model_type || "Unknown type",
      status.svd_enabled ? `rank ${status.svd_rank ?? PRIMARY_SVD_RANK}` : null,
      status.folder || null,
    ].filter(Boolean);
    meta.textContent = extras.join(" • ");
  }
  if (mini) mini.textContent = status.model_file || "Loaded";
}

function showView(viewId) {
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  el(viewId)?.classList.add("active");

  document.querySelectorAll(".navItem").forEach((button) => button.classList.remove("active"));
  document.querySelector(`.navItem[data-view="${viewId}"]`)?.classList.add("active");

  const title = el("pageTitle");
  const hint = el("pageHint");
  if (viewId === "viewOverview") {
    if (title) title.textContent = "Overview";
    if (hint) hint.textContent = "Runs, metrics, charts";
  } else if (viewId === "viewAnalytics") {
    if (title) title.textContent = "Analytics";
    if (hint) hint.textContent = "Curves, ranks, compare";
    setTimeout(() => {
      if (state.lastCurves) drawCurves(state.lastCurves);
    }, 60);
  } else {
    if (title) title.textContent = "Inference";
    if (hint) hint.textContent = "Load, test, predict";
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const raw = await response.text();
  let data = {};
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch (_) {
    data = { detail: raw };
  }
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  return data;
}

async function fetchTextArtifact(folder, file) {
  const key = `${folder}::${file}`;
  if (state.fileTextCache.has(key)) return state.fileTextCache.get(key);
  const response = await fetch(`/api/results/download?folder=${encodeURIComponent(folder)}&file=${encodeURIComponent(file)}`);
  const text = await response.text();
  if (!response.ok) throw new Error(text || `Could not read ${file}`);
  state.fileTextCache.set(key, text);
  return text;
}

function parseCsv(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return [];
  const header = lines[0].split(",").map((cell) => cell.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",").map((cell) => cell.trim());
    const row = {};
    header.forEach((key, index) => {
      row[key] = cells[index];
    });
    return row;
  });
}

async function getFolderFiles(folderName) {
  if (state.folderFilesCache.has(folderName)) return state.folderFilesCache.get(folderName);
  const data = await fetchJson(`/api/results/files?folder=${encodeURIComponent(folderName)}`);
  state.folderFilesCache.set(folderName, data);
  return data;
}

async function getFolderCurves(folderName) {
  if (state.folderCurvesCache.has(folderName)) return state.folderCurvesCache.get(folderName);
  const data = await fetchJson(`/api/results/curves?folder=${encodeURIComponent(folderName)}`);
  state.folderCurvesCache.set(folderName, data);
  return data;
}

async function getSummaryJson(folderName) {
  try {
    return JSON.parse(await fetchTextArtifact(folderName, "summary.json"));
  } catch (_) {
    return null;
  }
}

async function hydrateReferenceData() {
  const primaryFolder = state.preferredDefault?.folder || pickDefaultFolder(state.folders)?.name || null;
  const quantumBaselineFolder = chooseQuantumBaselineFolder();
  const svdFolders = state.folders
    .filter((folder) => folder.name.toLowerCase().startsWith("results_svd_hybrid_r"))
    .sort((a, b) => (parseRankFromName(a.name) || 0) - (parseRankFromName(b.name) || 0))
    .map((folder) => folder.name);

  state.reference.primaryFolder = primaryFolder;
  state.reference.quantumBaselineFolder = quantumBaselineFolder;
  state.reference.svdFolders = svdFolders;
  state.reference.classicalFolders = {
    logreg: latestFolderByPrefix("results_classical_logreg"),
    mlp: latestFolderByPrefix("results_classical_mlp"),
    tree: latestFolderByPrefix("results_classical_tree"),
  };

  updatePrimaryPathTexts();

  const rankRows = [];
  for (const folderName of svdFolders) {
    try {
      const curvePayload = await getFolderCurves(folderName);
      const curves = curvePayload.curves || {};
      const available = SCENARIO_ORDER.filter((key) => curves[key]?.final !== undefined && curves[key]?.final !== null);
      const finals = {};
      available.forEach((key) => {
        finals[key] = curves[key].final;
      });
      rankRows.push({
        folder: folderName,
        rank: parseRankFromName(folderName),
        finals,
        average: average(Object.values(finals)),
        isPrimary: folderName === primaryFolder,
      });
    } catch (_) {
      // Skip folders with unreadable curves.
    }
  }
  state.reference.rankRows = rankRows;

  const classicalRows = [];
  for (const [label, folderName] of Object.entries(state.reference.classicalFolders)) {
    if (!folderName) continue;
    const summary = await getSummaryJson(folderName);
    if (!summary) continue;
    classicalRows.push({
      folder: folderName,
      key: label,
      model: summary.model || label,
      description: summary.description || "",
      metrics: summary.metrics || {},
      confusion: summary.confusion_matrix || [],
      labels: summary.confusion_matrix_labels || [],
      mapping: summary.label_mapping_from_loader || {},
      use20Percent: summary.use_20_percent,
    });
  }
  state.reference.classicalRows = classicalRows;
}

function renderFolderList() {
  const list = el("folderList");
  const count = el("foldersCount");
  if (!list) return;

  if (count) count.textContent = String(state.folders.length);
  if (!state.folders.length) {
    list.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No result folders found</div>
        <div class="emptySub">Expected folders like <code>results_*</code> or <code>sanity*</code>.</div>
      </div>
    `;
    return;
  }

  list.innerHTML = "";
  state.folders.forEach((folder) => {
    const isSelected = state.selectedFolder === folder.name;
    const isPrimary = state.reference.primaryFolder === folder.name;
    const rank = parseRankFromName(folder.name);
    const lower = folder.name.toLowerCase();
    const badges = [];
    if (isPrimary) badges.push(`<span class="chip strong">Primary path</span>`);
    if (lower.includes("classical")) badges.push(`<span class="chip">Classical</span>`);
    if (lower.includes("svd")) badges.push(`<span class="chip">SVD${rank ? ` r${rank}` : ""}</span>`);
    if (!lower.includes("svd") && !lower.includes("classical")) badges.push(`<span class="chip">Quantum baseline</span>`);

    const div = document.createElement("div");
    div.className = `item clickable${isSelected ? " active" : ""}`;
    div.innerHTML = `
      <div class="itemTitle">${escapeHtml(folder.name)}</div>
      <div class="small">${new Date((folder.mtime || 0) * 1000).toLocaleString()}</div>
      <div class="summaryBadges">${badges.join("")}</div>
    `;
    div.onclick = () => selectFolder(folder.name);
    list.appendChild(div);
  });
}

function fillModelDropdown(modelFiles) {
  const select = el("modelSelect");
  if (!select) return;

  select.innerHTML = "";
  if (!state.selectedFolder) {
    select.innerHTML = `<option value="">Select a results folder first</option>`;
    return;
  }
  if (!modelFiles.length) {
    select.innerHTML = `<option value="">No model_*.pt found in this folder</option>`;
    return;
  }

  const preferredOrder = [
    `model_clean_svd_r${PRIMARY_SVD_RANK}.pt`,
    `model_defended_svd_r${PRIMARY_SVD_RANK}.pt`,
    `model_defended_dp_svd_r${PRIMARY_SVD_RANK}.pt`,
    `model_attack_svd_r${PRIMARY_SVD_RANK}.pt`,
    "model_clean.pt",
    "model_defended.pt",
    "model_defended_dp.pt",
    "model_attack.pt",
  ];

  const ordered = [...modelFiles].sort((left, right) => {
    const li = preferredOrder.indexOf(left);
    const ri = preferredOrder.indexOf(right);
    if (li === -1 && ri === -1) return left.localeCompare(right);
    if (li === -1) return 1;
    if (ri === -1) return -1;
    return li - ri;
  });

  ordered.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });

  if (state.preferredDefault?.model_file && ordered.includes(state.preferredDefault.model_file)) {
    select.value = state.preferredDefault.model_file;
  }
}

function renderFileList(files) {
  const list = el("fileList");
  const count = el("filesCount");
  const text = el("selectedFolderText");
  if (!list) return;

  if (count) count.textContent = String(files.length);
  if (text) text.textContent = state.selectedFolder ? `Artifacts in ${state.selectedFolder}` : "Select a folder to inspect files and checkpoints";

  if (!files.length) {
    list.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No files found</div>
        <div class="emptySub">This folder does not expose downloadable artifacts yet.</div>
      </div>
    `;
    return;
  }

  list.innerHTML = "";
  files.forEach((file) => {
    const lower = file.name.toLowerCase();
    const isModel = lower.startsWith("model_") && lower.endsWith(".pt");
    const isCsv = lower.endsWith(".csv");
    const dl = `/api/results/download?folder=${encodeURIComponent(state.selectedFolder)}&file=${encodeURIComponent(file.name)}`;
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <div class="row" style="justify-content:space-between; align-items:flex-start;">
        <div>
          <div class="itemTitle">${escapeHtml(file.name)}</div>
          <div class="small">${fmtBytes(file.size_bytes)} • ${escapeHtml(file.ext || "file")}</div>
          <div class="summaryBadges" style="margin-top:8px;">
            ${isModel ? `<span class="chip strong">${escapeHtml(inferRunType(state.selectedFolder, [file]))}</span>` : ""}
            ${isCsv ? `<span class="chip">CSV artifact</span>` : ""}
          </div>
        </div>
        <div class="toolbarRow">
          ${isModel ? `<button class="btn smallBtn" data-load="${escapeHtml(file.name)}">Load</button>` : ""}
          <a class="btn smallBtn ghost" href="${dl}">Download</a>
        </div>
      </div>
    `;
    if (isModel) {
      div.querySelector(`[data-load="${CSS.escape(file.name)}"]`)?.addEventListener("click", async () => {
        await loadModel(state.selectedFolder, file.name);
        showView("viewInference");
      });
    }
    list.appendChild(div);
  });
}

function setFinalChips(curves) {
  const targets = {
    finalClean: "clean",
    finalAttack: "attack",
    finalDef: "defended",
    finalDP: "defended_dp",
  };

  Object.entries(targets).forEach(([id, key]) => {
    const node = el(id);
    if (!node) return;
    node.textContent = `${scenarioLabel(key)}: ${fmtPct(curves?.[key]?.final, 1)}`;
  });
}

function renderOverviewMetrics() {
  const container = el("overviewMetricCards");
  if (!container) return;

  const curves = state.lastCurves || {};
  const keys = SCENARIO_ORDER.filter((key) => curves[key]);
  if (!keys.length) {
    container.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No scenario curves for this run</div>
          <div class="emptySub">No scenario CSVs in this folder.</div>
      </div>
    `;
    renderOverviewMetricChart(curves);
    return;
  }

  container.innerHTML = keys.map((key) => {
    const curve = curves[key];
    const lastRound = curve.rounds?.[curve.rounds.length - 1];
    return `
      <div class="metricCard">
        <div class="metricCardHeader">
          <div class="metricCardTitle">${escapeHtml(scenarioLabel(key))}</div>
          ${scenarioBadge(key)}
        </div>
        <div class="metricCardValue">${fmtPct(curve.final, 1)}</div>
        <div class="metricCardMeta">Final accuracy at round ${lastRound ?? "—"}</div>
      </div>
    `;
  }).join("");
  renderOverviewMetricChart(curves);
}

function renderOverviewMetricChart(curves) {
  const container = el("overviewMetricChart");
  if (!container) return;
  const keys = SCENARIO_ORDER.filter((key) => curves[key]?.final !== undefined);
  if (!keys.length) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = `
    <div class="barList">
      ${keys.map((key) => `
        <div class="barRow">
          <div class="barLabel">${escapeHtml(scenarioLabel(key))}</div>
          <div class="barTrack">
            <div class="barFill" style="width:${Math.max(0, Math.min(100, (curves[key].final || 0) * 100))}%; background:${SCENARIO_META[key]?.color || "#fff"};"></div>
          </div>
          <div class="barValue">${fmtPct(curves[key].final, 1)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderScenarioComparison() {
  const body = el("scenarioCompareBody");
  const sub = el("scenarioCompareSub");
  if (!body) return;

  const primaryFolder = state.reference.primaryFolder;
  const baselineFolder = state.reference.quantumBaselineFolder;
  const primaryCurves = state.folderCurvesCache.get(primaryFolder)?.curves || null;
  const baselineCurves = state.folderCurvesCache.get(baselineFolder)?.curves || null;

  if (!primaryFolder || !baselineFolder || !primaryCurves || !baselineCurves) {
    body.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No quantum comparison</div>
        <div class="emptySub">Need primary and baseline runs.</div>
      </div>
    `;
    return;
  }

  if (sub) sub.textContent = `${primaryFolder} compared with ${baselineFolder}`;

  body.innerHTML = `
    <div class="comparisonGrid">
      ${SCENARIO_ORDER.map((key) => {
        const primary = primaryCurves[key]?.final;
        const baseline = baselineCurves[key]?.final;
        const delta = fmtDelta(primary !== undefined && baseline !== undefined ? primary - baseline : null);
        const primaryWidth = Math.max(0, Math.min(100, (primary || 0) * 100));
        const baselineWidth = Math.max(0, Math.min(100, (baseline || 0) * 100));
        return `
          <div class="comparisonCard">
            <div class="comparisonCardHeader">
              <div class="comparisonTitle">${escapeHtml(scenarioLabel(key))}</div>
              ${scenarioBadge(key)}
            </div>
            <div class="comparisonValue">${fmtPct(primary, 1)}</div>
            <div class="barCompare">
              <div class="barCompareHeader"><span>SVD r16</span><span>${fmtPct(primary, 1)}</span></div>
              <div class="dualBarTrack">
                <div class="dualBar primary" style="width:${primaryWidth}%;"></div>
                <div class="dualBarLabel"><span></span><span></span></div>
              </div>
              <div class="barCompareHeader"><span>Baseline</span><span>${fmtPct(baseline, 1)}</span></div>
              <div class="dualBarTrack">
                <div class="dualBar secondary" style="width:${baselineWidth}%;"></div>
                <div class="dualBarLabel"><span></span><span></span></div>
              </div>
            </div>
            <div class="comparisonMeta">
              <span class="${delta.cls}">${delta.text}</span>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderRankStory() {
  const body = el("rankStoryBody");
  if (!body) return;

  const rows = state.reference.rankRows || [];
  if (!rows.length) {
    body.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No SVD runs</div>
        <div class="emptySub">No readable SVD folders.</div>
      </div>
    `;
    return;
  }

  body.innerHTML = `
    <div class="tableWrap" style="margin-top:12px;">
      <table class="tbl">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Folder</th>
            <th>Clean</th>
            <th>Attack</th>
            <th>Defended</th>
            <th>Defended + DP</th>
            <th>Average</th>
            <th>Tag</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td><b>${row.rank ?? "—"}</b></td>
              <td><code>${escapeHtml(row.folder)}</code></td>
              <td>${fmtPct(row.finals.clean, 1)}</td>
              <td>${fmtPct(row.finals.attack, 1)}</td>
              <td>${fmtPct(row.finals.defended, 1)}</td>
              <td>${fmtPct(row.finals.defended_dp, 1)}</td>
              <td><b>${fmtPct(row.average, 1)}</b></td>
              <td>${row.isPrimary ? `<span class="chip strong">Primary path</span>` : `<span class="chip">Available</span>`}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderClassicalMatrix(row) {
  const matrix = row.confusion || [];
  const labels = row.labels || [];
  if (matrix.length !== 2 || labels.length !== 2) {
    return `<div class="small">No 2x2 confusion matrix.</div>`;
  }

  const firstLabel = row.mapping?.[String(labels[0])] || `Class ${labels[0]}`;
  const secondLabel = row.mapping?.[String(labels[1])] || `Class ${labels[1]}`;

  return `
    <div class="matrixGrid">
      <div class="matrixCell good">
        <div class="matrixLabel">True ${escapeHtml(firstLabel)} / Pred ${escapeHtml(firstLabel)}</div>
        <div class="matrixValue">${escapeHtml(matrix[0]?.[0] ?? "—")}</div>
      </div>
      <div class="matrixCell bad">
        <div class="matrixLabel">True ${escapeHtml(firstLabel)} / Pred ${escapeHtml(secondLabel)}</div>
        <div class="matrixValue">${escapeHtml(matrix[0]?.[1] ?? "—")}</div>
      </div>
      <div class="matrixCell bad">
        <div class="matrixLabel">True ${escapeHtml(secondLabel)} / Pred ${escapeHtml(firstLabel)}</div>
        <div class="matrixValue">${escapeHtml(matrix[1]?.[0] ?? "—")}</div>
      </div>
      <div class="matrixCell good">
        <div class="matrixLabel">True ${escapeHtml(secondLabel)} / Pred ${escapeHtml(secondLabel)}</div>
        <div class="matrixValue">${escapeHtml(matrix[1]?.[1] ?? "—")}</div>
      </div>
    </div>
  `;
}

function renderClassicalComparison() {
  const body = el("classicalCompareBody");
  if (!body) return;

  const rows = state.reference.classicalRows || [];
  if (!rows.length) {
    body.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No classical data</div>
        <div class="emptySub">No readable summary.json files.</div>
      </div>
    `;
    return;
  }

  body.innerHTML = rows.map((row) => `
    <div class="comparisonCard" style="margin-bottom:12px;">
      <div class="comparisonCardHeader">
        <div>
          <div class="comparisonTitle">${escapeHtml(String(row.model).toUpperCase())}</div>
          <div class="comparisonMeta">${escapeHtml(row.description || row.folder)}</div>
        </div>
        <span class="chip">${row.use20Percent ? "20% split" : "Full split"}</span>
      </div>
      <div class="scoreGrid" style="margin-top:12px;">
        <div class="scoreCard"><div class="detailLabel">Accuracy</div><div class="comparisonValue">${fmtPct(row.metrics.accuracy, 1)}</div></div>
        <div class="scoreCard"><div class="detailLabel">Precision</div><div class="comparisonValue">${fmtPct(row.metrics.precision, 1)}</div></div>
        <div class="scoreCard"><div class="detailLabel">Recall</div><div class="comparisonValue">${fmtPct(row.metrics.recall, 1)}</div></div>
        <div class="scoreCard"><div class="detailLabel">F1</div><div class="comparisonValue">${fmtPct(row.metrics.f1, 1)}</div></div>
      </div>
      ${renderClassicalMatrix(row)}
    </div>
  `).join("");
}

function renderRunSnapshot() {
  const body = el("runSnapshotBody");
  if (!body) return;

  if (!state.selectedFolder) {
    body.innerHTML = `
      <div class="emptyState">
        <div class="emptyTitle">No run selected</div>
        <div class="emptySub">Pick a run.</div>
      </div>
    `;
    return;
  }

  const files = state.selectedFiles || [];
  const curves = state.lastCurves || {};
  const modelFiles = files.filter((file) => file.name?.toLowerCase().startsWith("model_") && file.name?.toLowerCase().endsWith(".pt"));
  const csvFiles = files.filter((file) => file.name?.toLowerCase().endsWith(".csv"));
  const metaFiles = files.filter((file) => file.name?.toLowerCase().includes("meta"));

  body.innerHTML = `
    <div class="detailGrid">
      <div class="detailItem">
        <div class="detailLabel">Run type</div>
        <div class="detailValue">${escapeHtml(inferRunType(state.selectedFolder, files))}</div>
      </div>
      <div class="detailItem">
        <div class="detailLabel">Folder role</div>
        <div class="detailValue">${state.selectedFolder === state.reference.primaryFolder ? "Primary path" : "Compare run"}</div>
      </div>
      <div class="detailItem">
        <div class="detailLabel">Checkpoint files</div>
        <div class="detailValue">${modelFiles.length}</div>
      </div>
      <div class="detailItem">
        <div class="detailLabel">CSV artifacts</div>
        <div class="detailValue">${csvFiles.length}</div>
      </div>
      <div class="detailItem">
        <div class="detailLabel">Scenario curves</div>
        <div class="detailValue">${Object.keys(curves).length} of 4</div>
      </div>
      <div class="detailItem">
        <div class="detailLabel">Meta files</div>
        <div class="detailValue">${metaFiles.map((file) => escapeHtml(file.name)).join(", ") || "None"}</div>
      </div>
    </div>
  `;
}

function renderFindings() {
  const body = el("findingsBody");
  if (!body) return;

  const items = [];
  if (state.reference.primaryFolder) {
    items.push({
      title: "Primary",
      body: `${state.reference.primaryFolder}`,
    });
  }

  if (state.lastCurves && Object.keys(state.lastCurves).length) {
    const entries = SCENARIO_ORDER
      .filter((key) => state.lastCurves[key]?.final !== undefined)
      .map((key) => ({ key, value: state.lastCurves[key].final }));
    if (entries.length) {
      entries.sort((a, b) => a.value - b.value);
      const lowest = entries[0];
      const highest = entries[entries.length - 1];
      items.push({
        title: "Range",
        body: `${scenarioLabel(highest.key)} ${fmtPct(highest.value, 1)} • ${scenarioLabel(lowest.key)} ${fmtPct(lowest.value, 1)}`,
      });
    }
  }

  if (state.reference.rankRows.length) {
    const primary = state.reference.rankRows.find((row) => row.isPrimary);
    items.push({
      title: "Ranks",
      body: primary
        ? `Primary rank: ${primary.rank ?? "?"}`
        : "SVD rank rows available.",
    });
  }

  if (state.reference.classicalRows.length) {
    items.push({
      title: "Classical",
      body: `${state.reference.classicalRows.length} baseline summaries loaded.`,
    });
  } else {
    items.push({
      title: "Classical",
      body: "No readable baseline summaries.",
    });
  }

  body.innerHTML = `<div class="findingList">${items.map((item) => `
    <div class="findingItem">
      <div class="findingTitle">${escapeHtml(item.title)}</div>
      <div class="findingBody">${escapeHtml(item.body)}</div>
    </div>
  `).join("")}</div>`;
}

function renderCompareTable(payload) {
  const out = el("compareOut");
  const summary = el("compareSummary");
  if (!out) return;

  const rows = payload.rows || [];
  const goodRows = rows.filter((row) => row.status === "ok");
  const best = goodRows[0];

  if (summary) {
    summary.textContent = best
      ? `Best: ${best.model_file} • ${fmtPct(best.accuracy, 1)}`
      : "No successful compare rows.";
  }

  renderCompareBarChart(rows);

  out.style.display = "block";
  out.innerHTML = `
    <div class="itemTitle">Compare Results</div>
    <div class="small">Folder: ${escapeHtml(payload.folder)} • n=${payload.n} • target=${escapeHtml(payload.target)}</div>
    <div class="tableWrap" style="margin-top:12px;">
      <table class="tbl">
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th>Status</th>
            <th>Accuracy</th>
            <th>TN</th>
            <th>FP</th>
            <th>FN</th>
            <th>TP</th>
            <th>Threshold</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td><code>${escapeHtml(row.model_file)}</code></td>
              <td>${escapeHtml(row.model_type || "—")}${row.svd_rank !== null && row.svd_rank !== undefined ? ` • r${row.svd_rank}` : ""}</td>
              <td>${escapeHtml(row.status || "—")}</td>
              <td><b>${row.accuracy !== undefined ? fmtPct(row.accuracy, 1) : "—"}</b></td>
              <td>${row.tn ?? "—"}</td>
              <td>${row.fp ?? "—"}</td>
              <td>${row.fn ?? "—"}</td>
              <td>${row.tp ?? "—"}</td>
              <td>${row.threshold !== undefined ? fmtNum(row.threshold, 3) : "—"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderCompareBarChart(rows) {
  const container = el("compareBarChart");
  if (!container) return;
  const okRows = (rows || []).filter((row) => row.status === "ok" && row.accuracy !== undefined);
  if (!okRows.length) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = `
    <div class="barList">
      ${okRows.slice(0, 8).map((row) => `
        <div class="barRow">
          <div class="barLabel">${escapeHtml(row.model_file.replace("model_", "").replace(".pt", ""))}</div>
          <div class="barTrack">
            <div class="barFill" style="width:${Math.max(0, Math.min(100, row.accuracy * 100))}%; background:${row.model_type === "SVD Hybrid QNN" ? "#5f8dff" : "#34d399"};"></div>
          </div>
          <div class="barValue">${fmtPct(row.accuracy, 1)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function setModelUI(status) {
  state.modelStatus = status;
  const pill = el("modelPill");
  const inDimPill = el("inDimPill");
  const thresholdPill = el("thresholdPill");
  const loadedFolder = el("loadedFolderValue");
  const loadedModel = el("loadedModelValue");
  const loadedType = el("loadedTypeValue");
  const loadedMeta = el("loadedMetaValue");

  if (pill) {
    pill.textContent = status?.loaded ? "Loaded" : "Not Loaded";
    pill.classList.toggle("danger", !status?.loaded);
  }
  if (inDimPill) inDimPill.textContent = status?.in_dim ?? "?";
  if (thresholdPill) thresholdPill.textContent = status?.threshold ?? "?";
  if (loadedFolder) loadedFolder.textContent = status?.folder || "—";
  if (loadedModel) loadedModel.textContent = status?.model_file || "—";
  if (loadedType) {
    loadedType.textContent = status?.loaded
      ? `${status.model_type || "Unknown"}${status.svd_enabled ? ` • rank ${status.svd_rank ?? PRIMARY_SVD_RANK}` : ""}`
      : "—";
  }
  if (loadedMeta) loadedMeta.textContent = status?.meta_file || "—";
  setLoadedModelHeadline(status);
}

async function checkModelStatus() {
  try {
    const status = await fetchJson("/api/model/status");
    setModelUI(status);
    return status;
  } catch (err) {
    setStatus("Error", "error");
    showGlobalMessage(`Model status check failed: ${err.message}`, "error");
    return null;
  }
}

async function loadModel(folder, modelFile) {
  try {
    const status = await fetchJson(`/api/model/load?folder=${encodeURIComponent(folder)}&model_file=${encodeURIComponent(modelFile)}`, {
      method: "POST",
    });
    setModelUI(status);
    showGlobalMessage(`Loaded model: ${modelFile}`, "success");
  } catch (err) {
    showGlobalMessage(`Load failed: ${err.message}`, "error");
  }
}

function parseVector(text) {
  const parts = String(text || "").split(",").map((part) => part.trim()).filter(Boolean);
  const nums = parts.map((value) => Number(value));
  return nums.some((num) => Number.isNaN(num)) ? null : nums;
}

function renderPrediction(data) {
  const out = el("predictOut");
  if (!out) return;

  const prob = data.prob_attack !== undefined ? data.prob_attack : data.probability_class1;
  const predicted = prettyLabel(data.pred_label || `Class ${data.pred_class ?? "?"}`);

  out.style.display = "block";
  out.innerHTML = `
    <div class="itemTitle">Prediction</div>
    <div class="scoreGrid" style="margin-top:12px;">
      <div class="scoreCard"><div class="detailLabel">P(Attack)</div><div class="comparisonValue">${fmtPct(prob, 1)}</div></div>
      <div class="scoreCard"><div class="detailLabel">Threshold</div><div class="comparisonValue">${fmtNum(data.threshold ?? 0.5, 3)}</div></div>
      <div class="scoreCard"><div class="detailLabel">Predicted</div><div class="comparisonValue">${escapeHtml(predicted)}</div></div>
      <div class="scoreCard"><div class="detailLabel">Truth</div><div class="comparisonValue">${escapeHtml(state.lastTruth || "Unknown")}</div></div>
    </div>
  `;
}

async function predictNow() {
  const vector = parseVector(el("featureInput")?.value || "");
  if (!vector) {
    showGlobalMessage("Invalid input. Paste comma-separated numbers only.", "error");
    return;
  }

  try {
    const data = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: vector }),
    });
    renderPrediction(data);
    showGlobalMessage("Prediction complete.", "success");
  } catch (err) {
    showGlobalMessage(`Predict failed: ${err.message}`, "error");
  }
}

async function loadRandom(target) {
  try {
    const data = await fetchJson(`/api/test/random?target=${encodeURIComponent(target)}&use_20_percent=0`);
    state.lastTruth = prettyLabel(data.true_label_name);
    if (el("truthPill")) el("truthPill").textContent = `truth: ${state.lastTruth}`;
    if (el("featureInput")) el("featureInput").value = (data.features || []).join(",");
    showGlobalMessage(`Loaded ${target} sample.`, "success");
  } catch (err) {
    showGlobalMessage(`Random sample failed: ${err.message}`, "error");
  }
}

function renderBatchResult(data) {
  const out = el("batchOut");
  if (!out) return;
  const c = data.confusion || {};
  const precision = (c.tp + c.fp) > 0 ? c.tp / (c.tp + c.fp) : null;
  const recall = (c.tp + c.fn) > 0 ? c.tp / (c.tp + c.fn) : null;
  const specificity = (c.tn + c.fp) > 0 ? c.tn / (c.tn + c.fp) : null;

  out.style.display = "block";
  out.innerHTML = `
    <div class="itemTitle">Batch Test</div>
    <div class="small">n=${data.n} • target=${escapeHtml(data.target)} • threshold=${fmtNum(data.threshold, 3)}</div>
    <div class="scoreGrid" style="margin-top:12px;">
      <div class="scoreCard"><div class="detailLabel">Accuracy</div><div class="comparisonValue">${fmtPct(data.accuracy, 1)}</div></div>
      <div class="scoreCard"><div class="detailLabel">Precision</div><div class="comparisonValue">${fmtPct(precision, 1)}</div></div>
      <div class="scoreCard"><div class="detailLabel">Recall</div><div class="comparisonValue">${fmtPct(recall, 1)}</div></div>
      <div class="scoreCard"><div class="detailLabel">Specificity</div><div class="comparisonValue">${fmtPct(specificity, 1)}</div></div>
    </div>
    <div class="matrixGrid">
      <div class="matrixCell good">
        <div class="matrixLabel">TN</div>
        <div class="matrixValue">${c.tn ?? "—"}</div>
      </div>
      <div class="matrixCell bad">
        <div class="matrixLabel">FP</div>
        <div class="matrixValue">${c.fp ?? "—"}</div>
      </div>
      <div class="matrixCell bad">
        <div class="matrixLabel">FN</div>
        <div class="matrixValue">${c.fn ?? "—"}</div>
      </div>
      <div class="matrixCell good">
        <div class="matrixLabel">TP</div>
        <div class="matrixValue">${c.tp ?? "—"}</div>
      </div>
    </div>
  `;
}

async function batchTest(target) {
  try {
    const data = await fetchJson(`/api/eval/batch?n=50&target=${encodeURIComponent(target)}&use_20_percent=0`);
    renderBatchResult(data);
    showGlobalMessage(`Batch done: n=${data.n}`, "success");
  } catch (err) {
    showGlobalMessage(`Batch test failed: ${err.message}`, "error");
  }
}

async function compareModels(n) {
  if (!state.selectedFolder) {
    alert("Select a results folder first.");
    return;
  }

  const pill = el("comparePill");
  const out = el("compareOut");
  if (pill) pill.textContent = "Running";
  if (out) {
    out.style.display = "none";
    out.innerHTML = "";
  }
  if (el("compareSummary")) el("compareSummary").textContent = "Running compare...";

  try {
    const data = await fetchJson(`/api/eval/compare?folder=${encodeURIComponent(state.selectedFolder)}&n=${encodeURIComponent(n)}&target=any&use_20_percent=0`);
    if (pill) pill.textContent = `Done (n=${data.n})`;
    renderCompareTable(data);
    showGlobalMessage(`Compare done: ${data.rows?.length || 0} models`, "success");
  } catch (err) {
    if (pill) pill.textContent = "Failed";
    if (el("compareSummary")) el("compareSummary").textContent = "Compare failed.";
    renderCompareBarChart([]);
    showGlobalMessage(`Compare failed: ${err.message}`, "error");
  }
}

function drawCurves(curves, hoverRound = null) {
  const canvas = el("curveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const wrapWidth = canvas.parentElement?.clientWidth || 900;
  const width = canvas.width = Math.max(760, wrapWidth - 4);
  const height = canvas.height = 380;
  ctx.clearRect(0, 0, width, height);

  const keysAll = Object.keys(curves || {});
  const keys = getVisibleKeys(curves || {});
  if (!keysAll.length) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.font = "14px system-ui";
    ctx.fillText("No scenario curves.", 18, 32);
    return;
  }
  if (!keys.length) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.font = "14px system-ui";
    ctx.fillText("All lines hidden.", 18, 32);
    return;
  }

  let minY = 1;
  let maxY = 0;
  let maxRound = 0;
  keys.forEach((key) => {
    (curves[key]?.acc || []).forEach((value) => {
      minY = Math.min(minY, value);
      maxY = Math.max(maxY, value);
    });
    (curves[key]?.rounds || []).forEach((round) => {
      maxRound = Math.max(maxRound, round);
    });
  });
  minY = Math.max(0, minY - 0.03);
  maxY = Math.min(1, maxY + 0.03);

  const padL = 66;
  const padR = 26;
  const padT = 30;
  const padB = 58;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  function toX(round) {
    return padL + ((round - 1) / Math.max(1, maxRound - 1)) * plotW;
  }
  function toY(acc) {
    return padT + (1 - ((acc - minY) / Math.max(1e-9, maxY - minY))) * plotH;
  }

  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.font = "600 14px system-ui";
  ctx.fillText("Accuracy vs Round", padL, 18);

  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  ctx.font = "12px system-ui";
  for (let i = 0; i <= 5; i += 1) {
    const y = padT + (plotH * i / 5);
    const value = maxY - ((maxY - minY) * i / 5);
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.58)";
    ctx.fillText(value.toFixed(2), 14, y + 4);
  }

  const ticks = [1, Math.ceil(maxRound / 2), maxRound].filter((value, index, array) => value > 0 && array.indexOf(value) === index);
  ticks.forEach((tick) => {
    const x = toX(tick);
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.58)";
    ctx.fillText(String(tick), x - 6, height - 24);
  });

  ctx.fillStyle = "rgba(255,255,255,0.62)";
  ctx.fillText("Round", padL + plotW / 2 - 16, height - 8);

  keys.forEach((key) => {
    const color = SCENARIO_META[key]?.color || "#ffffff";
    const rounds = curves[key]?.rounds || [];
    const accs = curves[key]?.acc || [];
    if (!rounds.length) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(toX(rounds[0]), toY(accs[0]));
    for (let i = 1; i < rounds.length; i += 1) {
      ctx.lineTo(toX(rounds[i]), toY(accs[i]));
    }
    ctx.stroke();

    ctx.fillStyle = color;
    rounds.forEach((round, index) => {
      if (index % 2 === 0 || index === rounds.length - 1) {
        ctx.beginPath();
        ctx.arc(toX(round), toY(accs[index]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  });

  if (hoverRound !== null) {
    const x = toX(hoverRound);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
  }

  let legendX = padL;
  const legendY = height - 28;
  keys.forEach((key) => {
    const color = SCENARIO_META[key]?.color || "#ffffff";
    ctx.fillStyle = color;
    ctx.fillRect(legendX, legendY - 9, 10, 10);
    ctx.fillStyle = "rgba(255,255,255,0.74)";
    ctx.fillText(scenarioLabel(key), legendX + 16, legendY);
    legendX += 126;
  });
}

function attachChartHover(curves) {
  const canvas = el("curveCanvas");
  const tip = el("chartTip");
  if (!canvas || !tip) return;

  const keys = getVisibleKeys(curves || {});
  const rounds = curves?.[keys[0]]?.rounds || [];
  if (!keys.length || !rounds.length) {
    canvas.onmousemove = null;
    canvas.onmouseleave = null;
    return;
  }

  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const padL = 66;
    const padR = 26;
    const plotW = canvas.width - padL - padR;
    const relative = (x - padL) / Math.max(1, plotW);
    const approx = 1 + relative * ((rounds[rounds.length - 1] || 1) - 1);

    let nearest = rounds[0];
    let distance = Infinity;
    rounds.forEach((round) => {
      const nextDistance = Math.abs(round - approx);
      if (nextDistance < distance) {
        distance = nextDistance;
        nearest = round;
      }
    });

    tip.innerHTML = `
      <div class="tipTitle">Round ${nearest}</div>
      ${keys.map((key) => {
        const idx = (curves[key]?.rounds || []).indexOf(nearest);
        const value = idx >= 0 ? curves[key].acc[idx] : null;
        return `<div class="tipRow"><span class="tipKey">${escapeHtml(scenarioLabel(key))}</span><span class="tipVal">${fmtPct(value, 1)}</span></div>`;
      }).join("")}
    `;
    tip.style.display = "block";
    tip.style.left = `${Math.min(rect.width - 180, Math.max(12, x + 12))}px`;
    tip.style.top = `${Math.min(rect.height - 130, Math.max(12, y - 12))}px`;
    drawCurves(curves, nearest);
  };

  canvas.onmouseleave = () => {
    tip.style.display = "none";
    drawCurves(curves, null);
  };
}

async function selectFolder(folderName) {
  const folder = state.folders.find((item) => item.name === folderName);
  state.selectedFolder = folderName;
  state.selectedFolderMtime = folder?.mtime || null;
  if (el("selectedMini")) el("selectedMini").textContent = folderName || "—";

  try {
    const [filePayload, curvePayload] = await Promise.all([
      getFolderFiles(folderName),
      getFolderCurves(folderName),
    ]);

    state.selectedFiles = filePayload.files || [];
    state.lastCurves = curvePayload.curves || {};
    state.modelFilesInFolder = state.selectedFiles
      .map((file) => file.name)
      .filter((name) => name.toLowerCase().startsWith("model_") && name.toLowerCase().endsWith(".pt"));

    renderFolderList();
    fillModelDropdown(state.modelFilesInFolder);
    renderFileList(state.selectedFiles);
    updateRunSummary(folderName, state.selectedFiles, state.lastCurves);
    setFinalChips(state.lastCurves);
    drawCurves(state.lastCurves);
    attachChartHover(state.lastCurves);
    renderOverviewMetrics();
    renderScenarioComparison();
    renderRankStory();
    renderClassicalComparison();
    renderFindings();
    renderRunSnapshot();

    if (
      !state.didAutoLoadDefaultModel &&
      state.modelFilesInFolder.length &&
      folderName === state.reference.primaryFolder
    ) {
      const preferred = state.preferredDefault?.model_file;
      const modelName = (preferred && state.modelFilesInFolder.includes(preferred)) ? preferred : state.modelFilesInFolder[0];
      if (modelName) {
        state.didAutoLoadDefaultModel = true;
        await loadModel(folderName, modelName);
      }
    }
  } catch (err) {
    setStatus("Error", "error");
    showGlobalMessage(`Failed to load folder data: ${err.message}`, "error");
  }
}

async function fetchFolders() {
  setStatus("Refreshing...");
  clearGlobalMessage();
  state.folderFilesCache.clear();
  state.folderCurvesCache.clear();
  state.fileTextCache.clear();

  try {
    const data = await fetchJson("/api/results/folders");
    state.folders = data.folders || [];
    await hydrateReferenceData();
    renderFolderList();

    const activeFolder = state.selectedFolder && state.folders.some((folder) => folder.name === state.selectedFolder)
      ? state.selectedFolder
      : pickDefaultFolder(state.folders)?.name || null;

    if (activeFolder) {
      await selectFolder(activeFolder);
      if (state.preferredDefault?.ambiguous) {
        showGlobalMessage(
          `Multiple r16 runs found. Using ${activeFolder}.`,
          "success"
        );
      }
    } else {
      state.selectedFolder = null;
      state.selectedFiles = [];
      state.lastCurves = null;
      renderFolderList();
      renderFileList([]);
      renderOverviewMetrics();
      renderScenarioComparison();
      renderRankStory();
      renderClassicalComparison();
      renderFindings();
      renderRunSnapshot();
    }
    setStatus("Idle");
  } catch (err) {
    setStatus("Error", "error");
    showGlobalMessage(`Failed to load folders: ${err.message}`, "error");
  }
}

function wireToggles() {
  const buttons = [
    ["toggleClean", "clean"],
    ["toggleAttack", "attack"],
    ["toggleDefended", "defended"],
    ["toggleDP", "defended_dp"],
  ];
  buttons.forEach(([id, key]) => {
    const button = el(id);
    if (!button) return;
    button.onclick = () => {
      visible[key] = !visible[key];
      button.classList.toggle("off", !visible[key]);
      if (state.lastCurves) {
        drawCurves(state.lastCurves);
        attachChartHover(state.lastCurves);
      }
    };
  });
}

async function bootstrapDefault() {
  try {
    state.preferredDefault = await fetchJson("/api/results/default");
  } catch (_) {
    state.preferredDefault = null;
  }
  await fetchFolders();
  await checkModelStatus();
}

function clearPredictionInputs() {
  if (el("featureInput")) el("featureInput").value = "";
  if (el("truthPill")) el("truthPill").textContent = "truth: ?";
  state.lastTruth = null;
  if (el("predictOut")) {
    el("predictOut").style.display = "none";
    el("predictOut").innerHTML = "";
  }
}

function setup() {
  document.querySelectorAll(".navItem").forEach((button) => {
    button.onclick = () => showView(button.dataset.view);
  });

  el("refreshBtn").onclick = fetchFolders;
  el("checkModelBtn").onclick = checkModelStatus;
  el("loadSelectedModelBtn").onclick = async () => {
    if (!state.selectedFolder) {
      alert("Select a results folder first.");
      return;
    }
    const modelName = el("modelSelect")?.value;
    if (!modelName) {
      alert("No model selected.");
      return;
    }
    await loadModel(state.selectedFolder, modelName);
  };

  el("randAnyBtn").onclick = () => loadRandom("any");
  el("randNormalBtn").onclick = () => loadRandom("normal");
  el("randAttackBtn").onclick = () => loadRandom("attack");
  el("predictBtn").onclick = predictNow;
  el("clearBtn").onclick = clearPredictionInputs;
  el("batchAnyBtn").onclick = () => batchTest("any");
  el("batchNormalBtn").onclick = () => batchTest("normal");
  el("batchAttackBtn").onclick = () => batchTest("attack");
  el("compareBtn").onclick = () => compareModels(200);
  el("compareBtnSmall").onclick = () => compareModels(50);

  wireToggles();
  renderScenarioAvailability([]);
  bootstrapDefault();
  setInterval(fetchFolders, 15000);
}

setup();
