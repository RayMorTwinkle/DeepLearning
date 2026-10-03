const ENTITY_TYPES = [
  "CROP", "VAR", "TRT", "GST", "GENE", "QTL",
  "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"
];

const RELATION_TYPES = ["CON", "USE", "HAS", "AFF", "OCI", "LOI"];
const BIOES_PREFIXES = ["B", "I", "E", "S"];

const samples = [
  {
    name: "样本 0",
    text: "VrFRO8 reduces SOD contents by influencing the Fe2+/Fe3+ ratio under salt stress. This study mined a key gene associated with salt tolerance and constructed a VrFRO8-related PPI network for salt tolerance.",
    entities: [
      { start: 0, end: 6, text: "VrFRO8", label: "GENE" },
      { start: 190, end: 204, text: "salt tolerance", label: "TRT" },
      { start: 15, end: 27, text: "SOD contents", label: "TRT" },
      { start: 69, end: 80, text: "salt stress", label: "ABS" },
      { start: 126, end: 140, text: "salt tolerance", label: "TRT" }
    ],
    relations: [
      { head: "VrFRO8", head_start: 0, head_end: 6, head_type: "GENE", tail: "SOD contents", tail_start: 15, tail_end: 27, tail_type: "TRT", label: "AFF" },
      { head: "salt stress", head_start: 69, head_end: 80, head_type: "ABS", tail: "SOD contents", tail_start: 15, tail_end: 27, tail_type: "TRT", label: "AFF" },
      { head: "VrFRO8", head_start: 0, head_end: 6, head_type: "GENE", tail: "salt tolerance", tail_start: 126, tail_end: 140, tail_type: "TRT", label: "LOI" },
      { head: "VrFRO8", head_start: 0, head_end: 6, head_type: "GENE", tail: "salt tolerance", tail_start: 190, tail_end: 204, tail_type: "TRT", label: "LOI" }
    ]
  },
  {
    name: "样本 1",
    text: "It reduced reactive oxygen species accumulation and increased chlorophyll content in sorghum leaves. Silencing SbMYC2 by virus-induced gene silencing compromised drought tolerance in sorghum seedlings.",
    entities: [
      { start: 11, end: 47, text: "reactive oxygen species accumulation", label: "TRT" },
      { start: 62, end: 81, text: "chlorophyll content", label: "TRT" },
      { start: 111, end: 117, text: "SbMYC2", label: "GENE" },
      { start: 162, end: 179, text: "drought tolerance", label: "TRT" },
      { start: 121, end: 149, text: "virus-induced gene silencing", label: "BM" },
      { start: 85, end: 92, text: "sorghum", label: "CROP" },
      { start: 183, end: 190, text: "sorghum", label: "CROP" }
    ],
    relations: [
      { head: "SbMYC2", head_start: 111, head_end: 117, head_type: "GENE", tail: "drought tolerance", tail_start: 162, tail_end: 179, tail_type: "TRT", label: "AFF" },
      { head: "virus-induced gene silencing", head_start: 121, head_end: 149, head_type: "BM", tail: "SbMYC2", tail_start: 111, tail_end: 117, tail_type: "GENE", label: "USE" }
    ]
  }
];

const state = {
  sampleIndex: 0,
  selectedEntityIndex: 0,
  selectedPairIndex: 0,
  activeNerStep: 0,
  negRatio: 3
};

const dependencyItems = [
  ["JSON 样本", "每条数据有 text、entities、relations。text 是原文，entities 是实体位置，relations 是实体之间的关系。"],
  ["字符 span", "start/end 是字符下标，好像用尺子在原文上量出一个词从哪里开始、在哪里结束。"],
  ["BIOES 标签", "把一个实体拆成开头 B、中间 I、结尾 E；如果只有一个字符，就用 S；普通位置是 O。"],
  ["tokenizer", "分词器，把文字切成模型能读的小块，并给每块一个 offset_mapping。"],
  ["tensor", "模型不直接读文字，它读数字数组，例如 input_ids、attention_mask、labels。"],
  ["实体对", "RE 不看单个实体，而是问：这个 head 和这个 tail 之间有没有某种关系。"],
  ["负样本", "没有标注关系的实体对会被当成 NONE，但不能太多，否则模型只会学会说没有关系。"],
  ["精确匹配", "评估时位置和类型必须全对；多报是 FP，漏报是 FN，报对是 TP。"]
];

const codeMapItems = [
  ["build_ner_label_map", "用 12 类实体和 BIOES 前缀生成 49 个 NER 标签。"],
  ["split_data", "用固定 seed 随机切训练集和验证集，让实验可以复现。"],
  ["entities_to_bioes", "把原始实体 span 变成每个字符一个 BIOES 标签。"],
  ["tokenize_and_align_ner", "用 offset_mapping 把字符标签对齐到 token 标签。"],
  ["NERDataset", "把 input_ids、attention_mask、labels 打包给 DataLoader。"],
  ["tokens_to_entities_with_offsets", "把模型预测的 token 标签还原成提交需要的 start/end。"],
  ["insert_entity_markers", "给当前关系候选对插入 [E1]/[E2] 标记。"],
  ["prepare_re_samples", "枚举有方向实体对，保留正样本，按 neg_ratio 抽负样本。"],
  ["add_entity_marker_tokens", "把 [E1]、[/E1]、[E2]、[/E2] 加进 tokenizer。"],
  ["compute_ner_metrics", "实体的 start、end、label 完全一致才算 TP。"],
  ["compute_re_metrics", "头实体、尾实体、实体类型、关系类型全一致才算 TP。"],
  ["compute_total_score", "总分 = 0.4 * NER_Score + 0.6 * RE_Score。"]
];

const nerSteps = [
  "原始 entities",
  "字符 BIOES",
  "token 对齐",
  "Dataset tensor",
  "预测还原"
];

const labelMaps = buildLabelMaps();

function buildLabelMaps() {
  const nerLabel2Id = { O: 0 };
  let idx = 1;
  for (const type of ENTITY_TYPES) {
    for (const prefix of BIOES_PREFIXES) {
      nerLabel2Id[`${prefix}-${type}`] = idx;
      idx += 1;
    }
  }
  const relLabel2Id = { NONE: 0 };
  RELATION_TYPES.forEach((type, i) => {
    relLabel2Id[type] = i + 1;
  });
  return { nerLabel2Id, relLabel2Id };
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function currentSample() {
  return samples[state.sampleIndex];
}

function relationKeyFromEntities(head, tail) {
  return `${head.start}:${head.end}->${tail.start}:${tail.end}`;
}

function relationKeyFromRelation(rel) {
  return `${rel.head_start}:${rel.head_end}->${rel.tail_start}:${rel.tail_end}`;
}

function renderDependencies() {
  const grid = document.querySelector("#dependencyGrid");
  grid.innerHTML = dependencyItems.map(([title, text]) => `
    <article class="dependencyCard">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(text)}</p>
    </article>
  `).join("");
}

function renderLabelTags() {
  document.querySelector("#entityTypeCount").textContent = ENTITY_TYPES.length;
  document.querySelector("#nerLabelCount").textContent = Object.keys(labelMaps.nerLabel2Id).length;
  document.querySelector("#relLabelCount").textContent = Object.keys(labelMaps.relLabel2Id).length;
  document.querySelector("#entityTags").innerHTML = ENTITY_TYPES.map(type => `
    <span class="tag"><strong>${type}</strong><span>${BIOES_PREFIXES.map(prefix => `${prefix}-${type}`).join(" / ")}</span></span>
  `).join("");
  document.querySelector("#relationTags").innerHTML = Object.entries(labelMaps.relLabel2Id).map(([label, id]) => `
    <span class="tag"><strong>${label}</strong><span>id ${id}</span></span>
  `).join("");
}

function renderSampleTabs() {
  const tabs = document.querySelector("#sampleTabs");
  tabs.innerHTML = samples.map((sample, index) => `
    <button class="${index === state.sampleIndex ? "active" : ""}" data-sample="${index}" type="button">
      ${escapeHtml(sample.name)}
    </button>
  `).join("");
  tabs.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      state.sampleIndex = Number(button.dataset.sample);
      state.selectedEntityIndex = 0;
      state.selectedPairIndex = 0;
      renderAll();
    });
  });
}

function renderRawText() {
  const sample = currentSample();
  const entities = [...sample.entities].sort((a, b) => a.start - b.start || a.end - b.end);
  let cursor = 0;
  let html = "";
  for (const entity of entities) {
    html += escapeHtml(sample.text.slice(cursor, entity.start));
    html += `<span class="entityMark" data-type="${escapeHtml(entity.label)}" title="${escapeHtml(entity.label)} ${entity.start}-${entity.end}">${escapeHtml(sample.text.slice(entity.start, entity.end))}</span>`;
    cursor = entity.end;
  }
  html += escapeHtml(sample.text.slice(cursor));
  document.querySelector("#rawText").innerHTML = html;
  document.querySelector("#sampleStats").textContent = `${sample.entities.length} entities / ${sample.relations.length} relations`;

  const seen = new Map();
  for (const entity of sample.entities) {
    seen.set(entity.label, (seen.get(entity.label) || 0) + 1);
  }
  document.querySelector("#entityLegend").innerHTML = [...seen.entries()].map(([label, count]) => `
    <span class="legendItem"><strong>${label}</strong>${count}</span>
  `).join("");
}

function entitiesToBioes(text, entities) {
  const labels = Array.from({ length: text.length }, () => "O");
  for (const entity of entities) {
    const length = entity.end - entity.start;
    if (length <= 0) continue;
    if (length === 1) {
      labels[entity.start] = `S-${entity.label}`;
    } else {
      labels[entity.start] = `B-${entity.label}`;
      for (let i = entity.start + 1; i < entity.end - 1; i += 1) {
        labels[i] = `I-${entity.label}`;
      }
      labels[entity.end - 1] = `E-${entity.label}`;
    }
  }
  return labels;
}

function simpleTokenize(text) {
  const matches = [...text.matchAll(/[A-Za-z0-9]+(?:[-+/][A-Za-z0-9]+)*|[^\sA-Za-z0-9]/g)];
  return matches.map(match => ({
    token: match[0],
    start: match.index,
    end: match.index + match[0].length
  }));
}

function chooseTokenLabel(charLabels) {
  const entityLabels = charLabels.filter(label => label !== "O");
  if (!entityLabels.length) return "O";
  return entityLabels.find(label => label.startsWith("S-"))
    || entityLabels.find(label => label.startsWith("B-"))
    || entityLabels.find(label => label.startsWith("E-"))
    || entityLabels[0];
}

function tokenId(token) {
  let hash = 17;
  for (let i = 0; i < token.length; i += 1) {
    hash = (hash * 31 + token.charCodeAt(i)) % 119000;
  }
  return hash + 1000;
}

function makeTokenRows(sample) {
  const charLabels = entitiesToBioes(sample.text, sample.entities);
  const tokens = simpleTokenize(sample.text);
  const rows = [
    { token: "[CLS]", start: 0, end: 0, label: "-100", labelId: -100, chars: [] }
  ];
  for (const token of tokens) {
    const labels = charLabels.slice(token.start, token.end);
    const label = chooseTokenLabel(labels);
    rows.push({
      ...token,
      label,
      labelId: labelMaps.nerLabel2Id[label],
      chars: labels
    });
  }
  rows.push({ token: "[SEP]", start: 0, end: 0, label: "-100", labelId: -100, chars: [] });
  while (rows.length < 32) {
    rows.push({ token: "[PAD]", start: 0, end: 0, label: "-100", labelId: -100, chars: [] });
  }
  return rows;
}

function renderNerStepper() {
  const stepper = document.querySelector("#nerStepper");
  stepper.innerHTML = nerSteps.map((step, index) => `
    <button class="${index === state.activeNerStep ? "active" : ""}" data-step="${index}" type="button">
      ${index + 1}. ${escapeHtml(step)}
    </button>
  `).join("");
  stepper.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      state.activeNerStep = Number(button.dataset.step);
      renderNerStepper();
      drawOffsetCanvas(performance.now());
    });
  });
}

function renderEntityPicker() {
  const sample = currentSample();
  const picker = document.querySelector("#entityPicker");
  picker.innerHTML = sample.entities.map((entity, index) => `
    <button class="${index === state.selectedEntityIndex ? "active" : ""}" data-entity="${index}" type="button">
      ${escapeHtml(entity.text)} <span>${escapeHtml(entity.label)}</span>
    </button>
  `).join("");
  picker.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      state.selectedEntityIndex = Number(button.dataset.entity);
      renderNer();
    });
  });
}

function renderCharRail() {
  const sample = currentSample();
  const selected = sample.entities[state.selectedEntityIndex] || sample.entities[0];
  const labels = entitiesToBioes(sample.text, sample.entities);
  document.querySelector("#selectedEntityPill").textContent =
    `${selected.text} / ${selected.label} / ${selected.start}-${selected.end}`;
  document.querySelector("#charRail").innerHTML = [...sample.text].map((char, index) => {
    const visibleChar = char === " " ? "space" : char;
    const active = index >= selected.start && index < selected.end;
    const label = labels[index];
    return `
      <div class="charCell ${active ? "active" : ""}">
        <span class="char">${escapeHtml(visibleChar)}</span>
        <span class="idx">${index}</span>
        <span class="bioes">${escapeHtml(label)}</span>
      </div>
    `;
  }).join("");
}

function labelsSnippet(labels) {
  const unique = [...new Set(labels)];
  if (!unique.length) return "-";
  if (unique.length <= 3) return unique.join(" ");
  return `${unique.slice(0, 3).join(" ")} ...`;
}

function renderTokenTable() {
  const sample = currentSample();
  const rows = makeTokenRows(sample);
  const tbody = document.querySelector("#tokenTable tbody");
  tbody.innerHTML = rows.slice(0, 38).map(row => {
    const offset = row.start === 0 && row.end === 0 ? "(0, 0)" : `(${row.start}, ${row.end})`;
    const labelClass = row.label !== "O" && row.label !== "-100" ? "labelHot" : "";
    return `
      <tr>
        <td><code>${escapeHtml(row.token)}</code></td>
        <td><code>${offset}</code></td>
        <td>${escapeHtml(labelsSnippet(row.chars))}</td>
        <td class="${labelClass}">${escapeHtml(row.label)}</td>
        <td><code>${row.labelId}</code></td>
      </tr>
    `;
  }).join("");

  const ids = rows.map(row => row.token === "[PAD]" ? 0 : row.token === "[CLS]" ? 101 : row.token === "[SEP]" ? 102 : tokenId(row.token));
  const mask = rows.map(row => row.token === "[PAD]" ? 0 : 1);
  const labels = rows.map(row => row.labelId);
  document.querySelector("#inputIdsPreview").textContent = previewArray(ids);
  document.querySelector("#maskPreview").textContent = previewArray(mask);
  document.querySelector("#labelsPreview").textContent = previewArray(labels);
}

function previewArray(values) {
  const head = values.slice(0, 18).join(", ");
  return `[${head}${values.length > 18 ? ", ..." : ""}]`;
}

function tokensToEntities(rows, text) {
  const entities = [];
  let current = null;
  const append = () => {
    if (!current) return;
    while (current.start < current.end && /\s/.test(text[current.start])) current.start += 1;
    while (current.end > current.start && /\s/.test(text[current.end - 1])) current.end -= 1;
    if (current.start < current.end) {
      entities.push({
        start: current.start,
        end: current.end,
        text: text.slice(current.start, current.end),
        label: current.type
      });
    }
  };

  for (const row of rows) {
    if (row.label === "-100") continue;
    if (row.label === "O") {
      append();
      current = null;
      continue;
    }
    const [prefix, type] = row.label.split("-");
    if (prefix === "S") {
      append();
      current = null;
      entities.push({ start: row.start, end: row.end, text: text.slice(row.start, row.end), label: type });
    } else if (prefix === "B") {
      append();
      current = { start: row.start, end: row.end, type };
    } else if (prefix === "I" || prefix === "E") {
      if (!current || current.type !== type) {
        append();
        current = { start: row.start, end: row.end, type };
      } else {
        current.end = row.end;
      }
      if (prefix === "E") {
        append();
        current = null;
      }
    }
  }
  append();
  return entities;
}

function renderRoundTrip() {
  const sample = currentSample();
  const rows = makeTokenRows(sample);
  const entities = tokensToEntities(rows, sample.text);
  document.querySelector("#roundTrip").innerHTML = entities.slice(0, 8).map(entity => `
    <div class="roundTripItem">
      <strong>${escapeHtml(entity.text)}</strong>
      <span>${escapeHtml(entity.label)} / start ${entity.start} / end ${entity.end}</span>
    </div>
  `).join("");
}

function positivePairMap(sample) {
  const map = new Map();
  for (const rel of sample.relations) {
    map.set(relationKeyFromRelation(rel), rel.label);
  }
  return map;
}

function getPairs(sample) {
  const positives = positivePairMap(sample);
  const pairs = [];
  for (let i = 0; i < sample.entities.length; i += 1) {
    for (let j = 0; j < sample.entities.length; j += 1) {
      if (i === j) continue;
      const head = sample.entities[i];
      const tail = sample.entities[j];
      const key = relationKeyFromEntities(head, tail);
      const label = positives.get(key) || "NONE";
      pairs.push({ head, tail, key, label, positive: label !== "NONE" });
    }
  }
  return pairs;
}

function seededShuffle(items, seed) {
  const copy = [...items];
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function getReSelection() {
  const sample = currentSample();
  const pairs = getPairs(sample);
  const pos = pairs.filter(pair => pair.positive);
  const neg = pairs.filter(pair => !pair.positive);
  const sampledNeg = seededShuffle(neg, 42).slice(0, Math.min(neg.length, pos.length * state.negRatio));
  const sampledKeys = new Set(sampledNeg.map(pair => pair.key));
  return { pairs, pos, neg, sampledNeg, sampledKeys };
}

function renderRe() {
  const sample = currentSample();
  const { pairs, pos, neg, sampledNeg, sampledKeys } = getReSelection();
  const selected = pairs[state.selectedPairIndex] || pairs[0];

  document.querySelector("#pairCountPill").textContent = `${sample.entities.length} entities -> ${pairs.length} directed pairs`;
  document.querySelector("#negRatioText").textContent = String(state.negRatio);
  document.querySelector("#pairSummary").innerHTML = `
    <div><strong>${pairs.length}</strong><span>全部有方向实体对 n*(n-1)</span></div>
    <div><strong>${pos.length}</strong><span>正样本，来自 relations</span></div>
    <div><strong>${neg.length}</strong><span>候选负样本 NONE</span></div>
    <div><strong>${sampledNeg.length}</strong><span>实际抽取负样本</span></div>
  `;
  document.querySelector("#pairGrid").innerHTML = pairs.map((pair, index) => {
    const sampled = pair.positive || sampledKeys.has(pair.key);
    return `
      <button class="pairCard ${pair.positive ? "positive" : "negative"} ${index === state.selectedPairIndex ? "active" : ""}" data-pair="${index}" type="button">
        <strong>${escapeHtml(pair.head.text)} -> ${escapeHtml(pair.tail.text)}</strong>
        <span>${escapeHtml(pair.head.label)} to ${escapeHtml(pair.tail.label)}</span>
        <span>${pair.positive ? `label ${escapeHtml(pair.label)}` : sampled ? "NONE sampled" : "NONE not used"}</span>
      </button>
    `;
  }).join("");
  document.querySelectorAll(".pairCard").forEach(button => {
    button.addEventListener("click", () => {
      state.selectedPairIndex = Number(button.dataset.pair);
      renderRe();
    });
  });

  document.querySelector("#markedText").innerHTML = renderMarkedText(sample.text, selected.head, selected.tail);
  const relId = labelMaps.relLabel2Id[selected.label] ?? 0;
  document.querySelector("#currentRelLabel").textContent = `${selected.label} -> ${relId}`;
  document.querySelector("#reDatasetPreview").textContent = JSON.stringify(makeReDatasetPreview(selected, sample.text), null, 2);
  drawReCanvas(performance.now());
}

function renderMarkedText(text, head, tail) {
  const insertions = [
    [head.start, "[E1] "],
    [head.end, " [/E1]"],
    [tail.start, "[E2] "],
    [tail.end, " [/E2]"]
  ].sort((a, b) => b[0] - a[0]);
  let result = text;
  for (const [position, marker] of insertions) {
    result = result.slice(0, position) + marker + result.slice(position);
  }
  return escapeHtml(result).replace(/\[\/?E[12]\]/g, match => `<code>${match}</code>`);
}

function makeReDatasetPreview(pair, text) {
  const marked = stripHtml(renderMarkedText(text, pair.head, pair.tail));
  const tokens = simpleTokenize(marked).map(item => item.token);
  const ids = tokens.map(tokenId);
  const mask = tokens.map(() => 1);
  while (ids.length < 18) {
    ids.push(0);
    mask.push(0);
  }
  return {
    input_ids: previewArray(ids),
    attention_mask: previewArray(mask),
    label_id: labelMaps.relLabel2Id[pair.label] ?? 0,
    label: pair.label
  };
}

function stripHtml(html) {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent || "";
}

function renderMetricsBoards() {
  document.querySelector("#nerMatchBoard").innerHTML = `
    <div class="matchRow tp"><strong>TP</strong><p>预测出 <code>VrFRO8 / 0-6 / GENE</code>，标注里也有，算报对。</p></div>
    <div class="matchRow fp"><strong>FP</strong><p>预测出 <code>FRO8 / 2-6 / GENE</code>，位置错了，哪怕文字很像也算多报。</p></div>
    <div class="matchRow fn"><strong>FN</strong><p>标注有 <code>salt stress / 69-80 / ABS</code>，模型没报出来，算漏报。</p></div>
  `;
  document.querySelector("#reMatchBoard").innerHTML = `
    <div class="matchRow tp"><strong>TP</strong><p>head、tail、实体类型、关系类型都一致，例如 <code>VrFRO8 -> SOD contents / AFF</code>。</p></div>
    <div class="matchRow fp"><strong>FP</strong><p>模型多报了一条标注没有的关系，比如把无关系实体对报成 <code>LOI</code>。</p></div>
    <div class="matchRow fn"><strong>FN</strong><p>标注里有关系但模型没报出，RE 会漏掉一条关键得分项。</p></div>
  `;
}

function renderMetricSliders() {
  const tp = Number(document.querySelector("#tpSlider").value);
  const fp = Number(document.querySelector("#fpSlider").value);
  const fn = Number(document.querySelector("#fnSlider").value);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
  const score = 0.5 * f1 + 0.25 * precision + 0.25 * recall;
  const rows = [
    ["Precision", precision, `TP / (TP + FP) = ${tp} / ${tp + fp}`],
    ["Recall", recall, `TP / (TP + FN) = ${tp} / ${tp + fn}`],
    ["F1", f1, "P 和 R 的折中"],
    ["Score", score, "0.5F1 + 0.25P + 0.25R"]
  ];
  document.querySelector("#metricBars").innerHTML = rows.map(([name, value, tip]) => `
    <div class="metricBar">
      <strong>${name}</strong>
      <div class="metricTrack" title="${escapeHtml(tip)}"><div class="metricFill" style="width: ${(value * 100).toFixed(1)}%"></div></div>
      <span>${value.toFixed(4)}</span>
    </div>
  `).join("");
  const demoNer = score;
  const demoRe = Math.max(0, Math.min(1, score * 0.92 + 0.04));
  const total = 0.4 * demoNer + 0.6 * demoRe;
  document.querySelector("#totalFormula").textContent =
    `Total demo = 0.4 * NER(${demoNer.toFixed(4)}) + 0.6 * RE(${demoRe.toFixed(4)}) = ${total.toFixed(4)}`;
}

function renderCodeMap() {
  document.querySelector("#codeMap").innerHTML = codeMapItems.map(([fn, text]) => `
    <article class="codeMapItem">
      <code>${escapeHtml(fn)}()</code>
      <p>${escapeHtml(text)}</p>
    </article>
  `).join("");
}

function renderNer() {
  renderNerStepper();
  renderEntityPicker();
  renderCharRail();
  renderTokenTable();
  renderRoundTrip();
  drawOffsetCanvas(performance.now());
}

function renderAll() {
  renderSampleTabs();
  renderRawText();
  renderNer();
  renderRe();
}

function setupEvents() {
  document.querySelector("#negRatio").addEventListener("input", event => {
    state.negRatio = Number(event.target.value);
    renderRe();
  });
  ["#tpSlider", "#fpSlider", "#fnSlider"].forEach(selector => {
    document.querySelector(selector).addEventListener("input", renderMetricSliders);
  });
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawFlowCanvas(time) {
  const canvas = document.querySelector("#flowCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, w, h);

  const nodes = [
    ["JSON", "text/entities/relations", 58, 76, "#e3f0f7"],
    ["Label Map", "49 NER / 7 RE", 270, 76, "#dff3ea"],
    ["Split", "train / val", 482, 76, "#fff0d8"],
    ["NERDataset", "input_ids + labels", 270, 244, "#eee8fb"],
    ["REDataset", "entity pair samples", 482, 244, "#fae7e4"],
    ["Metrics", "P / R / F1 / Score", 694, 160, "#f6f1df"]
  ];

  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(25, 33, 38, 0.16)";
  drawConnector(ctx, 210, 132, 270, 132);
  drawConnector(ctx, 422, 132, 482, 132);
  drawConnector(ctx, 574, 154, 574, 244);
  drawConnector(ctx, 362, 154, 362, 244);
  drawConnector(ctx, 632, 300, 694, 216);
  drawConnector(ctx, 420, 300, 694, 216);

  for (const [title, sub, x, y, color] of nodes) {
    roundedRect(ctx, x, y, 152, 76, 8);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(25, 33, 38, 0.18)";
    ctx.stroke();
    ctx.fillStyle = "#192126";
    ctx.font = "700 18px system-ui";
    ctx.fillText(title, x + 16, y + 31);
    ctx.fillStyle = "#627079";
    ctx.font = "13px system-ui";
    ctx.fillText(sub, x + 16, y + 54);
  }

  const t = (time / 1000) % 6;
  const packets = [
    pathPoint([[210, 132], [270, 132], [422, 132], [482, 132]], (t % 2) / 2),
    pathPoint([[574, 154], [574, 244], [632, 300], [694, 216]], ((t + 0.7) % 2.6) / 2.6),
    pathPoint([[362, 154], [362, 244], [420, 300], [694, 216]], ((t + 1.4) % 3) / 3)
  ];
  for (const [x, y] of packets) {
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#257a5a";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.fillStyle = "#627079";
  ctx.font = "14px system-ui";
  ctx.fillText("NER 问：每个 token 是什么实体标签？", 70, 354);
  ctx.fillText("RE 问：这两个实体之间是什么关系？", 560, 354);
}

function drawConnector(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  const midX = (x1 + x2) / 2;
  ctx.bezierCurveTo(midX, y1, midX, y2, x2, y2);
  ctx.stroke();
}

function pathPoint(points, progress) {
  const segments = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    const length = Math.hypot(x2 - x1, y2 - y1);
    segments.push({ x1, y1, x2, y2, length });
    total += length;
  }
  let distance = progress * total;
  for (const segment of segments) {
    if (distance <= segment.length) {
      const p = distance / segment.length;
      return [
        segment.x1 + (segment.x2 - segment.x1) * p,
        segment.y1 + (segment.y2 - segment.y1) * p
      ];
    }
    distance -= segment.length;
  }
  const last = points[points.length - 1];
  return [last[0], last[1]];
}

function drawOffsetCanvas(time) {
  const canvas = document.querySelector("#offsetCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const sample = currentSample();
  const rows = makeTokenRows(sample).filter(row => row.token !== "[PAD]").slice(1, 16);
  const selected = sample.entities[state.selectedEntityIndex] || sample.entities[0];
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, w, h);

  const start = Math.max(0, selected.start - 18);
  const end = Math.min(sample.text.length, selected.end + 42);
  const textSlice = sample.text.slice(start, end);
  ctx.fillStyle = "#192126";
  ctx.font = "18px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillText(textSlice, 24, 58);

  const charWidth = Math.min(11, (w - 48) / Math.max(1, textSlice.length));
  for (let i = 0; i < textSlice.length; i += 1) {
    const x = 24 + i * charWidth;
    const globalIndex = start + i;
    if (globalIndex >= selected.start && globalIndex < selected.end) {
      ctx.fillStyle = "rgba(37, 122, 90, 0.18)";
      ctx.fillRect(x - 1, 68, charWidth + 1, 18);
    }
    ctx.fillStyle = "#9aa7ad";
    ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
    if (i % 5 === 0) ctx.fillText(String(globalIndex), x, 104);
  }

  const visible = rows.filter(row => row.end > start && row.start < end);
  const pulse = 0.5 + Math.sin(time / 420) * 0.5;
  visible.forEach((row, index) => {
    const x1 = 24 + Math.max(0, row.start - start) * charWidth;
    const x2 = 24 + Math.min(textSlice.length, row.end - start) * charWidth;
    const y = 155 + (index % 4) * 46;
    roundedRect(ctx, x1, y, Math.max(42, x2 - x1), 30, 6);
    ctx.fillStyle = row.label !== "O" ? "#dff3ea" : "#f5f7f7";
    ctx.fill();
    ctx.strokeStyle = row.label !== "O" ? "#257a5a" : "#d9e1e4";
    ctx.stroke();
    ctx.fillStyle = "#192126";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(row.token.slice(0, 16), x1 + 7, y + 19);
    ctx.beginPath();
    ctx.moveTo((x1 + x2) / 2, y);
    ctx.lineTo((x1 + x2) / 2, 88);
    ctx.strokeStyle = `rgba(36, 107, 155, ${0.25 + pulse * 0.45})`;
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  ctx.fillStyle = "#627079";
  ctx.font = "13px system-ui";
  ctx.fillText("offset_mapping 的直觉：每个 token 都拉一根线，指回原文的字符区间。", 24, h - 24);
}

function drawReCanvas(time) {
  const canvas = document.querySelector("#reCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { pos, neg, sampledNeg } = getReSelection();
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, w, h);

  const boxes = [
    ["relations 标注", `${pos.length} 正样本`, 54, 74, "#dff3ea"],
    ["无标注实体对", `${neg.length} 候选 NONE`, 54, 216, "#fae7e4"],
    ["抽样后", `${sampledNeg.length} 负样本`, 404, 216, "#fff0d8"],
    ["REDataset", `${pos.length + sampledNeg.length} 条训练样本`, 730, 145, "#e3f0f7"]
  ];
  boxes.forEach(([title, sub, x, y, color]) => {
    roundedRect(ctx, x, y, 210, 72, 8);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(25, 33, 38, 0.18)";
    ctx.stroke();
    ctx.fillStyle = "#192126";
    ctx.font = "700 18px system-ui";
    ctx.fillText(title, x + 16, y + 30);
    ctx.fillStyle = "#627079";
    ctx.font = "14px system-ui";
    ctx.fillText(sub, x + 16, y + 53);
  });

  ctx.strokeStyle = "rgba(25, 33, 38, 0.18)";
  ctx.lineWidth = 3;
  drawConnector(ctx, 264, 110, 730, 181);
  drawConnector(ctx, 264, 252, 404, 252);
  drawConnector(ctx, 614, 252, 730, 181);

  const dots = pos.length + sampledNeg.length;
  for (let i = 0; i < Math.min(28, dots); i += 1) {
    const p = ((time / 900 + i / 12) % 1);
    const start = i < pos.length ? [264, 110] : [614, 252];
    const end = [730, 181];
    const x = start[0] + (end[0] - start[0]) * p;
    const y = start[1] + (end[1] - start[1]) * p + Math.sin((p + i) * Math.PI * 2) * 12;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = i < pos.length ? "#257a5a" : "#b44242";
    ctx.fill();
  }

  ctx.fillStyle = "#627079";
  ctx.font = "14px system-ui";
  ctx.fillText("prepare_re_samples 的关键：正样本全要，负样本只抽一部分，避免 NONE 太多。", 54, 342);
}

function animate(time) {
  drawFlowCanvas(time);
  drawOffsetCanvas(time);
  drawReCanvas(time);
  requestAnimationFrame(animate);
}

document.addEventListener("DOMContentLoaded", () => {
  renderDependencies();
  renderLabelTags();
  renderCodeMap();
  renderMetricsBoards();
  setupEvents();
  renderAll();
  renderMetricSliders();
  requestAnimationFrame(animate);
});
