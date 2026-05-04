const { useEffect, useMemo, useRef, useState } = React;

const METRIC_META = {
  miou: { label: "mIoU", higherIsBetter: true },
  omiou: { label: "O-mIoU", higherIsBetter: true },
  sre: { label: "SRE", higherIsBetter: true },
  srr: { label: "SRR", higherIsBetter: true },
  clip_global: { label: "CLIPGlobal", higherIsBetter: true },
  clip_local: { label: "CLIPLocal", higherIsBetter: true },
  fid: { label: "FID", higherIsBetter: false }
};

const METRIC_COLUMNS = ["miou", "omiou", "sre", "srr", "clip_global", "clip_local", "fid"];
const PAGES = ["overview", "leaderboard", "contact"];

function getDefaultDirection(metric) {
  return METRIC_META[metric].higherIsBetter ? "desc" : "asc";
}

function formatScore(value) {
  if (value === "N/A" || value === null || value === undefined) return "N/A";
  const num = Number(value);
  return Number.isNaN(num) ? "N/A" : num.toFixed(2);
}

function rankRows(rows, sortKey, sortDirection) {
  return [...rows].sort((a, b) => {
    const aRaw = a[sortKey];
    const bRaw = b[sortKey];
    const aMissing = aRaw === "N/A" || aRaw === null || aRaw === undefined || Number.isNaN(Number(aRaw));
    const bMissing = bRaw === "N/A" || bRaw === null || bRaw === undefined || Number.isNaN(Number(bRaw));
    if (aMissing && bMissing) return a.method.localeCompare(b.method);
    if (aMissing) return 1;
    if (bMissing) return -1;
    const aVal = Number(aRaw);
    const bVal = Number(bRaw);
    if (aVal === bVal) return a.method.localeCompare(b.method);
    if (sortDirection === "asc") return aVal - bVal;
    return bVal - aVal;
  });
}

function parsePubYear(pubText) {
  if (!pubText) return null;
  const match = String(pubText).match(/(19|20)\d{2}/);
  return match ? Number(match[0]) : null;
}

function parseMethodTime(meta) {
  if (!meta) return null;
  if (typeof meta.time === "string") {
    const t = Date.parse(meta.time);
    if (!Number.isNaN(t)) return t;
  }
  if (meta.time !== undefined && meta.time !== null && meta.time !== "") {
    const t = Number(meta.time);
    if (!Number.isNaN(t) && t > 1000000000000) return t;
    if (!Number.isNaN(t)) {
      const year = Math.floor(t);
      const frac = t - year;
      const dayOfYear = Math.max(1, Math.round(frac * 365));
      const d = new Date(Date.UTC(year, 0, dayOfYear));
      return d.getTime();
    }
  }
  const y = parsePubYear(meta.pub);
  return y ? Date.UTC(y, 0, 1) : null;
}

function getPageFromHash() {
  const hash = (window.location.hash || "#overview").replace("#", "");
  return PAGES.includes(hash) ? hash : "overview";
}

function SortableHeader({ label, metricKey, sortConfig, onSort }) {
  const isActive = sortConfig.key === metricKey;
  const higher = METRIC_META[metricKey].higherIsBetter;
  const titleHint = higher
    ? "\u2191 ascending = lower scores first, \u2193 descending = higher scores first"
    : "\u2191 ascending = lower FID first (better), \u2193 descending = higher FID first";
  return (
    <th
      className={`sortable-header ${isActive ? "active-sort" : ""}`}
      onClick={() => onSort(metricKey)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSort(metricKey);
      }}
      title={titleHint}
    >
      <span className="th-label">{label}</span>
      <span className="th-sort-arrows" aria-hidden="true">
        <span className={`th-arrow ${isActive && sortConfig.direction === "asc" ? "is-on" : ""}`}>{"\u2191"}</span>
        <span className={`th-arrow ${isActive && sortConfig.direction === "desc" ? "is-on" : ""}`}>{"\u2193"}</span>
      </span>
    </th>
  );
}

function TopNav({ page, onNavigate }) {
  const items = [
    { key: "overview", label: "Overview" },
    { key: "leaderboard", label: "Leaderboard" },
    { key: "contact", label: "Contact Us" }
  ];
  return (
    <div className="top-nav-wrap">
      <div className="container is-max-widescreen">
        <div className="top-nav">
          <div className="top-nav-brand">OverLayBench</div>
          <div className="top-nav-links">
            {items.map((item) => (
              <button
                key={item.key}
                className={`top-nav-link ${page === item.key ? "is-active" : ""}`}
                onClick={() => onNavigate(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Hero() {
  return (
    <section className="hero page-hero">
      <div className="hero-body">
        <div className="container is-max-widescreen">
          <div className="columns is-vcentered">
            <div className="column is-7">
              <p className="tagline">NeurIPS 2025 D&B Track</p>
              <h1 className="title is-1 publication-title">OverLayBench</h1>
              <h2 className="subtitle is-4 hero-subtitle">A Benchmark for Layout-to-Image Generation with Dense Overlaps</h2>
              <p className="hero-desc">
                {/* OverLayBench studies difficult L2I layouts where objects overlap heavily and have similar semantics. */}
                {/* This project page provides detailed paper breakdowns, interactive leaderboard views, and resources. */}
                Existing layout-to-image models still struggle with dense overlaps, especially when instances are both spatially entangled and semantically similar. We introduce OverLayScore to measure this difficulty, build OverLayBench with balanced simple/regular/complex splits and high-quality annotations, and present CreatiLayout-AM as an overlap-aware baseline with amodal-mask supervision.
              </p>
              <div className="hero-ctas">
                <a href="https://arxiv.org/abs/2509.19282" className="button is-dark is-rounded">
                  <span className="icon"><i className="ai ai-arxiv" /></span><span>Paper</span>
                </a>
                <a href="https://github.com/mlpc-ucsd/OverLayBench" className="button is-dark is-rounded is-outlined">
                  <span className="icon"><i className="fab fa-github" /></span><span>Code</span>
                </a>
                <a href="https://huggingface.co/datasets/cywang143/OverLayBench_Eval" className="button is-link is-rounded">
                  <span className="icon"><i className="far fa-images" /></span><span>Data</span>
                </a>
              </div>
            </div>
            <div className="column is-5">
              <div className="hero-figure-wrap">
                <img src="./static/images/teaser.webp" alt="OverLayBench teaser" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function OverviewPage() {
  return (
    <>
      <section className="section section-compact">
        <div className="container is-max-widescreen">
          <div className="columns">
            <div className="column"><div className="metric-card"><p className="metric-value">4,052</p><p className="metric-label">Curated samples</p></div></div>
            <div className="column"><div className="metric-card"><p className="metric-value">2,052 / 1,000 / 1,000</p><p className="metric-label">Simple / Regular / Complex</p></div></div>
            <div className="column"><div className="metric-card"><p className="metric-value">7</p><p className="metric-label">Evaluation metrics</p></div></div>
            <div className="column"><div className="metric-card"><p className="metric-value">2</p><p className="metric-label">Tracks (training-based/free)</p></div></div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container is-max-widescreen">
          <div className="columns is-variable is-8 is-vcentered">
            <div className="column is-6">
              <h2 className="title is-4">Why Dense Overlap Breaks Layout-to-Image</h2>
              <div className="content paper-detail">
                <p>
                  State-of-the-art layout-to-image (L2I) models still fail when multiple bounding boxes overlap strongly, especially when the objects are
                  also semantically similar. The paper highlights two entangled challenges: <strong>large overlapping regions</strong> and <strong>minimal semantic
                  distinction</strong> across instances (e.g., two people in similar clothing, or closely interacting subjects).
                </p>
                <p>This leads to recurring artifacts including:</p>
                <ul>
                  <li><strong>Object fusion</strong> — merged geometry or textures where boxes overlap.</li>
                  <li><strong>BBox misalignment</strong> — objects only partially occupy the intended box.</li>
                  <li><strong>Distortion / corruption</strong> — unrealistic limbs, faces, or boundaries.</li>
                  <li><strong>Wrong counts or categories</strong> — missing or duplicated instances, swapped identities.</li>
                </ul>
                <p>
                  The teaser in the header illustrates the benchmark&rsquo;s difficulty axis: layouts become harder as OverLayScore increases from
                  <em> simple</em> to <em>regular</em> to <em>complex</em>.
                </p>
              </div>
            </div>
            <div className="column is-6">
              <figure className="figure-with-caption">
                <img src="./static/images/failure_case.webp" alt="Failure cases of existing layout-to-image methods" />
                <figcaption className="fig-caption">
                  Typical failures under dense overlap: object fusion, bbox misalignment, distortion, incorrect object number, and category confusion.
                </figcaption>
              </figure>
              <figure className="figure-with-caption">
                <img src="./static/images/motivation.webp" alt="Generation quality vs IoU and semantic similarity" />
                <figcaption className="fig-caption">
                  Empirical trend (paper Fig.&nbsp;3(a)): as IoU rises, generation quality drops; at fixed IoU, higher semantic similarity makes generation harder.
                </figcaption>
              </figure>
            </div>
          </div>
        </div>
      </section>

      <section className="section section-tinted">
        <div className="container is-max-widescreen">
          <div className="columns is-variable is-8 is-vcentered">
            <div className="column is-6">
              <h2 className="title is-4">OverLayScore</h2>
              <div className="content paper-detail">
                <p>
                  OverLayScore summarizes how difficult a layout is for L2I by summing, over every overlapping pair of instances, the product of
                  <strong> spatial overlap (IoU)</strong> and <strong>CLIP cosine similarity</strong> between their instance-level captions.
                  Higher values mean more entangled spatial and semantic overlap across the scene.
                </p>
                <div className="formula-box">
                  {"$$\\text{OverLayScore}=\\sum_{(i,j):\\text{IoU}(B_i,B_j)>0}\\text{IoU}(B_i,B_j)\\cdot\\cos(\\langle p_i,p_j \\rangle)$$"}
                </div>
                <p>
                  The paper validates the metric on COCO toy subsets: model mIoU decreases monotonically as difficulty bins (by OverLayScore) get harder.
                  It also shows that popular benchmarks skew toward low OverLayScore, while OverLayBench is deliberately balanced.
                </p>
              </div>
            </div>
            <div className="column is-6">
              <figure className="figure-with-caption distribution-figure">
                <img src="./static/images/distribution.webp" alt="Difficulty distribution across benchmarks" />
                <figcaption className="fig-caption">
                  Distribution comparison (paper Fig.&nbsp;3(c)): existing benchmarks are biased toward low difficulty, while OverLayBench keeps a much more
                  balanced split over simple, regular, and complex layouts.
                </figcaption>
              </figure>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container is-max-widescreen">
          <div className="paper-detail">
            <h2 className="title is-4">OverLayBench Dataset Curation</h2>
          </div>
          <div className="columns is-variable is-8">
            <div className="column is-6">
              <div className="content paper-detail">
                <p>
                  The benchmark is built to stress-test overlap. After extensive VLM-assisted construction, human annotators verify boxes, captions,
                  and relationship phrases so the evaluation set is free of obvious hallucinations.
                </p>
                <h3>Stage I — Reference images</h3>
                <p>
                  Image captions are taken from the COCO training set using <strong>Qwen2.5-VL-7B</strong>, then used to generate diverse candidates with
                  <strong> Flux.1-dev </strong> (28 sampling steps in the paper). Roughly <strong>86k</strong> generated image–caption pairs are collected as candidates.
                </p>
                <h3>Stage II — Grounding &amp; relationships</h3>
                <p>
                  A second captioning pass refines global captions for consistency with pixels. <strong>Qwen2.5-VL-32B</strong> detects instances, emits boxes and
                  instance captions, and extracts <strong>pairwise relationship phrases</strong> for overlapping pairs. A valid overlap pair requires <strong>IoU &gt; 5%</strong> and
                  an intersection area <strong>&gt; 1% of image area</strong>; scenes keep roughly <strong>1–10</strong> valid overlapping pairs.
                </p>
                <h3>Stage III — Human curation &amp; balancing</h3>
                <p>
                  Annotators filter unrealistic generations and balance difficulty. The released set has <strong>2,052</strong> simple, <strong>1,000</strong> regular, and{' '}
                  <strong>1,000</strong> complex layouts after OverLayScore stratification.
                </p>
              </div>
            </div>
            <div className="column is-6">
              <figure className="figure-with-caption">
                <img src="./static/images/overlaybench_pipeline.webp" alt="OverLayBench data curation pipeline" />
                <figcaption className="fig-caption">
                  Three-stage pipeline (paper Fig.&nbsp;4): Flux generation → Qwen grounding, captioning, and relationship extraction → manual auditing and
                  difficulty balancing before release.
                </figcaption>
              </figure>
            </div>
          </div>
        </div>
      </section>

      <section className="section section-tinted">
        <div className="container is-max-widescreen">
          <h2 className="title is-4">Evaluation Metrics &amp; Protocol</h2>
          <div className="columns is-variable is-8">
            <div className="column is-6">
              <div className="content paper-detail">
                <p>Besides standard L2I scores, the paper adds overlap-focused and relationship-centric measures.</p>
                <ul>
                  <li>
                    <strong>mIoU</strong> — Hungarian matching between predicted and GT boxes.
                  </li>
                  <li>
                    <strong>O-mIoU</strong> — mIoU restricted to the <em>intersection</em> of each overlapping GT pair and the corresponding prediction regions;
                    emphasizes fidelity where objects entangle.
                  </li>
                  <li>
                    <strong>SRE / SRR</strong> — instance-level and relationship-level success via <strong>Qwen2.5-VL-32B</strong> QA with curated prompts (see appendix in the paper).
                  </li>
                  <li>
                    <strong>CLIP</strong> — global and local alignment using <strong>ViT-B/32</strong>.
                  </li>
                  <li>
                    <strong>FID</strong> — perceptual distance to references.
                  </li>
                </ul>
                <p>
                  For each layout, the paper samples <strong>three</strong> images per method with fixed seeds <code>20251202</code>, <code>20251203</code>, and <code>20251204</code> for fair comparison.
                </p>
              </div>
            </div>
            <div className="column is-6">
              <figure className="figure-with-caption">
                <img src="./static/images/qual-res.webp" alt="Qualitative comparison of L2I models on OverLayBench" />
                <figcaption className="fig-caption">
                  Qualitative comparison (paper Fig.&nbsp;6): multiple strong baselines on OverLayBench. Dense overlaps expose fusion and relationship failures that
                  are easy to miss on simpler layout benchmarks.
                </figcaption>
              </figure>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container is-max-widescreen">
          <div className="columns is-variable is-8 is-vcentered">
            <div className="column is-6">
              <h2 className="title is-4">CreatiLayout-AM — Amodal Supervision</h2>
              <div className="content paper-detail">
                <p>
                  To probe whether explicit occlusion reasoning helps, the authors fine-tune CreatiLayout with <strong>amodal masks</strong> (full object extent even
                  under overlap). A synthetic training set is built by pasting occluders onto Flux renders; SAM&nbsp;v2 supplies masks; roughly
                  <strong> 67.8k </strong> training images are produced with Qwen-2.5-VL captions.
                </p>
                <p>
                  Training adds two auxiliary losses aligned with instance attention maps: a token-level alignment term and a pixel-level cross-entropy,
                  combined with the standard diffusion objective (coefficients λ, β in the paper). On OverLayBench, <strong>CreatiLayout-AM</strong> improves overlap-sensitive
                  metrics—especially <strong>O-mIoU</strong>—on simple and regular splits versus the base model, with competitive behavior on complex splits where train/test
                  shift is largest.
                </p>
                <p>
                  The paper also reports an <strong>EliGen-AM</strong> variant; attention maps for EliGen are approximated by averaging over local text tokens (appendix).
                </p>
              </div>
            </div>
            <div className="column is-6">
              <figure className="figure-with-caption">
                <img src="./static/images/ours_vs_base.webp" alt="CreatiLayout-AM compared to CreatiLayout baseline" />
                <figcaption className="fig-caption">
                  Qualitative gain (paper Fig.&nbsp;5): CreatiLayout-AM tends to separate overlapping instances more cleanly than the base CreatiLayout,
                  reducing fused or ambiguous regions in challenging layouts.
                </figcaption>
              </figure>
            </div>
          </div>
          <div className="top-gap">
            <h3 className="title is-5">CreatiLayout vs CreatiLayout-AM (Table)</h3>
            <div className="table-container leaderboard-table-wrap">
              <table className="table is-fullwidth is-striped is-hoverable compact-result-table">
                <thead>
                  <tr>
                    <th>Split</th>
                    <th>Method</th>
                    <th>mIoU</th>
                    <th>O-mIoU</th>
                    <th>SRE</th>
                    <th>SRR</th>
                    <th>CLIPG</th>
                    <th>CLIPL</th>
                    <th>FID</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td rowSpan="2"><strong>Simple</strong></td><td>CreatiLayout</td><td>58.78</td><td>32.52</td><td>72.34</td><td>84.45</td><td>37.29</td><td>27.49</td><td>27.51</td></tr>
                  <tr><td>CreatiLayout-AM</td><td>61.16</td><td><strong>37.69</strong></td><td>73.33</td><td>84.84</td><td>37.17</td><td>27.44</td><td>27.76</td></tr>
                  <tr><td rowSpan="2"><strong>Regular</strong></td><td>CreatiLayout</td><td>47.04</td><td>20.67</td><td>62.60</td><td>78.31</td><td>36.67</td><td>25.55</td><td>45.57</td></tr>
                  <tr><td>CreatiLayout-AM</td><td>47.38</td><td><strong>21.79</strong></td><td>63.13</td><td>78.71</td><td>36.49</td><td>25.46</td><td>46.34</td></tr>
                  <tr><td rowSpan="2"><strong>Complex</strong></td><td>CreatiLayout</td><td>44.24</td><td>18.05</td><td>52.10</td><td>79.98</td><td>36.55</td><td>24.76</td><td>53.29</td></tr>
                  <tr><td>CreatiLayout-AM</td><td>43.97</td><td><strong>18.07</strong></td><td>52.49</td><td>79.77</td><td>36.32</td><td>24.72</td><td>53.48</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
      <section className="section" id="BibTeX">
        <div className="container is-max-desktop content">
          <h2 className="title is-4">BibTeX</h2>
          <pre><code>{`@article{li2025overlaybench,
  title={OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps},
  author={Li, Bingnan and Wang, Chen-Yu and Xu, Haiyang and Zhang, Xiang and Armand, Ethan and Srivastava, Divyansh and Shan, Xiaojun and Chen, Zeyuan and Xie, Jianwen and Tu, Zhuowen},
  journal={arXiv preprint arXiv:2509.19282},
  year={2025}
}`}</code></pre>
        <pre><code>{`@inproceedings{NEURIPS2025_329ad516,
 author = {Li, Bingnan and Wang, Chen-Yu and Xu, Haiyang and Zhang, Xiang and Armand, Ethan and Srivastava, Divyansh and Xiaojun, Shan and Chen, Zeyuan and Xie, Jianwen and Tu, Zhuowen},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/329ad516cf7a6ac306f29882e9c77558-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {38},
 year = {2025}
}`}</code></pre>
        </div>
      </section>
    </>
  );
}

function LeaderboardPage({ data, methodMeta }) {
  const [track, setTrack] = useState("training_based");
  const [split, setSplit] = useState("simple");
  const [sortConfig, setSortConfig] = useState({ key: "omiou", direction: "desc" });
  const trendChartRef = useRef(null);

  useEffect(() => {
    setSortConfig({ key: "omiou", direction: "desc" });
  }, [track, split]);

  const rows = useMemo(() => data[track][split] || [], [data, track, split]);
  const ranked = useMemo(() => rankRows(rows, sortConfig.key, sortConfig.direction), [rows, sortConfig]);
  const trendPoints = useMemo(() => {
    return ranked
      .map((row) => {
        const year = parseMethodTime(methodMeta[row.method]);
        const metricValue = row[sortConfig.key];
        const missingMetric = metricValue === "N/A" || metricValue === null || metricValue === undefined || Number.isNaN(Number(metricValue));
        if (!year || missingMetric) return null;
        return { method: row.method, backbone: row.backbone, time: year, value: Number(metricValue) };
      })
      .filter(Boolean)
      .sort((a, b) => a.time - b.time || a.method.localeCompare(b.method));
  }, [ranked, sortConfig, methodMeta]);

  useEffect(() => {
    if (!trendChartRef.current || trendPoints.length < 2 || !window.echarts) return undefined;
    const chart = window.echarts.init(trendChartRef.current);
    const backbones = [...new Set(trendPoints.map((d) => d.backbone))];
    const colorMap = {
      "U-Net": "#0ea5e9",
      "DiT": "#8b5cf6",
      "AR": "#f59e0b"
    };
    const series = backbones.flatMap((bb) => {
      const raw = trendPoints
        .filter((d) => d.backbone === bb)
        .map((d) => ({ value: [d.time, d.value], method: d.method, backbone: d.backbone }));
      const grouped = {};
      raw.forEach((d) => {
        const t = d.value[0];
        if (!grouped[t]) grouped[t] = [];
        grouped[t].push(d.value[1]);
      });
      const trend = Object.keys(grouped)
        .map((t) => {
          const arr = grouped[t];
          const avg = arr.reduce((s, v) => s + v, 0) / arr.length;
          return [Number(t), avg];
        })
        .sort((a, b) => a[0] - b[0]);
      return [
        {
          name: bb,
          type: "scatter",
          symbolSize: 9,
          data: raw,
          itemStyle: { color: colorMap[bb] || "#2563eb" }
        },
        {
          name: bb,
          type: "line",
          data: trend,
          symbol: "none",
          smooth: 0.15,
          lineStyle: { width: 2.2, opacity: 0.85 },
          itemStyle: { color: colorMap[bb] || "#2563eb" },
          tooltip: { show: false },
          emphasis: { disabled: true }
        }
      ];
    });
    const times = trendPoints.map((d) => d.time);
    const values = trendPoints.map((d) => d.value);
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const pad = Math.max(0.5, (maxVal - minVal) * 0.12);
    chart.setOption({
      animationDuration: 450,
      color: backbones.map((bb) => colorMap[bb] || "#2563eb"),
      grid: { left: 58, right: 28, top: 58, bottom: 62 },
      legend: { top: 18, textStyle: { color: "#334155", fontSize: 12 } },
      toolbox: {
        right: 8,
        feature: {
          saveAsImage: { title: "Save" },
          restore: { title: "Reset" }
        }
      },
      tooltip: {
        trigger: "item",
        borderColor: "#e2e8f0",
        backgroundColor: "rgba(255,255,255,0.98)",
        textStyle: { color: "#0f172a" },
        formatter: (p) => {
          if (!p.data || !p.data.method) return "";
          const v = p.data.value;
          const dt = new Date(v[0]).toISOString().slice(0, 10);
          return `<strong>${p.data.method}</strong><br/>Backbone: ${p.data.backbone}<br/>Date: ${dt}<br/>${METRIC_META[sortConfig.key].label}: ${v[1].toFixed(2)}`;
        }
      },
      xAxis: {
        type: "time",
        name: "Publication date",
        nameLocation: "middle",
        nameGap: 34,
        min: minTime - 86400000 * 7,
        max: maxTime + 86400000 * 7,
        axisLabel: {
          color: "#64748b",
          formatter: (v) => {
            const d = new Date(v);
            return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;
          }
        },
        splitLine: { lineStyle: { color: "#eef2f7" } }
      },
      yAxis: {
        type: "value",
        name: METRIC_META[sortConfig.key].label,
        min: minVal - pad,
        max: maxVal + pad,
        axisLabel: { color: "#64748b" },
        splitLine: { lineStyle: { color: "#e2e8f0" } }
      },
      dataZoom: [
        { type: "inside", filterMode: "none" },
        { type: "slider", height: 18, bottom: 8 }
      ],
      series
    });
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
    };
  }, [trendPoints, sortConfig]);

  function handleSort(metricKey) {
    setSortConfig((prev) => {
      if (prev.key === metricKey) {
        return { key: metricKey, direction: prev.direction === "asc" ? "desc" : "asc" };
      }
      return { key: metricKey, direction: getDefaultDirection(metricKey) };
    });
  }

  function exportCurrentTable() {
    const headers = ["rank", "method", "backbone", "miou", "omiou", "sre", "srr", "clip_global", "clip_local", "fid", "pub"];
    const lines = [headers.join(",")];
    ranked.forEach((row, idx) => {
      const values = [
        idx + 1,
        row.method,
        row.backbone,
        formatScore(row.miou),
        formatScore(row.omiou),
        formatScore(row.sre),
        formatScore(row.srr),
        formatScore(row.clip_global),
        formatScore(row.clip_local),
        formatScore(row.fid),
        methodMeta[row.method]?.pub || "N/A"
      ];
      const escaped = values.map((v) => `"${String(v).replace(/"/g, '""')}"`);
      lines.push(escaped.join(","));
    });
    const csvContent = lines.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const filename = `overlaybench_${track}_${split}_${sortConfig.key}_${sortConfig.direction}.csv`;
    a.setAttribute("href", url);
    a.setAttribute("download", filename);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <section className="section" id="leaderboard">
      <div className="container is-max-widescreen">
        <div className="section-header">
          <h2 className="title is-3">Leaderboard</h2>
          <p className="section-note">Click a metric header to sort; each header shows <span className="nowrap">{"\u2191\u00a0\u2193"}</span> (active direction highlighted).</p>
        </div>

        <div className="leaderboard-toolbar">
          <div className="leaderboard-controls leaderboard-controls-two">
            <div className="field">
              <label className="label" htmlFor="trackSelect">Track</label>
              <div className="control">
                <div className="select is-fullwidth">
                  <select id="trackSelect" value={track} onChange={(e) => setTrack(e.target.value)}>
                    <option value="training_based">Training-based</option>
                    <option value="training_free">Training-free</option>
                  </select>
                </div>
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="splitSelect">Split</label>
              <div className="control">
                <div className="select is-fullwidth">
                  <select id="splitSelect" value={split} onChange={(e) => setSplit(e.target.value)}>
                    <option value="simple">Simple</option>
                    <option value="regular">Regular</option>
                    <option value="complex">Complex</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
          <div className="leaderboard-export-row">
            <button className="button is-light export-btn" onClick={exportCurrentTable}>
              <span className="icon"><i className="fas fa-download" /></span>
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        <div className="metric-trend-wrap">
          <h3 className="title is-5 metric-trend-title">
            {METRIC_META[sortConfig.key].label} vs Publication Date (current view)
          </h3>
          {trendPoints.length < 2 ? (
            <p className="section-note">Not enough methods with valid year/metric to render chart.</p>
          ) : (
            <div ref={trendChartRef} className="metric-trend-echart" />
          )}
        </div>

        <div className="table-container leaderboard-table-wrap">
          <table className="table is-fullwidth is-striped is-hoverable leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Method</th>
                <th>Backbone</th>
                <SortableHeader label={<>mIoU(%)</>} metricKey="miou" sortConfig={sortConfig} onSort={handleSort} />
                <SortableHeader label={<>O-mIoU(%)</>} metricKey="omiou" sortConfig={sortConfig} onSort={handleSort} />
                <SortableHeader label={<>SR<sub>E</sub>(%)</>} metricKey="sre" sortConfig={sortConfig} onSort={handleSort} />
                <SortableHeader label={<>SR<sub>R</sub>(%)</>} metricKey="srr" sortConfig={sortConfig} onSort={handleSort} />
                <SortableHeader label={<>CLIP<sub>Global</sub></>} metricKey="clip_global" sortConfig={sortConfig} onSort={handleSort} />
                <SortableHeader label={<>CLIP<sub>Local</sub></>} metricKey="clip_local" sortConfig={sortConfig} onSort={handleSort} />
                <SortableHeader label={<>FID</>} metricKey="fid" sortConfig={sortConfig} onSort={handleSort} />
              </tr>
            </thead>
            <tbody>
              {ranked.map((row, index) => (
                <tr key={`${row.method}-${split}`}>
                  <td>{index + 1}</td>
                  <td>
                    <span className={`method-name ${row.backbone === "AR" ? "is-ar-method" : ""}`}>
                      <strong>{row.method}</strong>
                    </span>
                    {methodMeta[row.method]?.isNew ? <span className="new-tag">NEW</span> : null}
                    <div className="method-submeta">
                      <span className="pub-chip">{methodMeta[row.method]?.pub || "N/A"}</span>
                      {methodMeta[row.method]?.paper ? (
                        <a className="mini-link-btn" href={methodMeta[row.method].paper} target="_blank" rel="noopener noreferrer">Paper</a>
                      ) : (
                        <span className="mini-link-btn is-disabled">Paper</span>
                      )}
                      {methodMeta[row.method]?.code ? (
                        <a className="mini-link-btn" href={methodMeta[row.method].code} target="_blank" rel="noopener noreferrer">Code</a>
                      ) : (
                        <span className="mini-link-btn is-disabled">Code</span>
                      )}
                    </div>
                  </td>
                  <td>
                    <span
                      className={`backbone-tag ${
                        row.backbone === "DiT" ? "is-dit" : row.backbone === "AR" ? "is-ar" : "is-unet"
                      }`}
                    >
                      {row.backbone}
                    </span>
                  </td>
                  {METRIC_COLUMNS.map((metricKey) => (
                    <td
                      key={metricKey}
                      className={`${sortConfig.key === metricKey ? "active-metric" : ""} ${index === 0 && sortConfig.key === metricKey ? "top-metric" : ""} ${formatScore(row[metricKey]) === "N/A" ? "na-cell" : ""}`}
                    >
                      {formatScore(row[metricKey])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function ContactPage() {
  return (
    <>
      <section className="section">
        <div className="container is-max-widescreen">
          <h2 className="title is-3">Contact Us</h2>
          <div className="columns">
            <div className="column">
              <div className="metric-card">
                <p className="metric-value">General Inquiries</p>
                <p className="metric-label">Questions about benchmark scope, evaluation setup, and paper details.</p>
                <p className="content">
                  Reach out to the authors via the emails listed in the paper or open an issue in the GitHub repository.
                </p>
              </div>
            </div>
            <div className="column">
              <div className="metric-card">
                <p className="metric-value">Data / Leaderboard Updates</p>
                <p className="metric-label">Submit updates or corrections.</p>
                <p className="content">
                  <a href="https://github.com/mlpc-ucsd/OverLayBench/issues">GitHub Issues</a><br />
                  <a href="https://github.com/mlpc-ucsd/OverLayBench/pulls">GitHub Pull Requests</a>
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [methodMeta, setMethodMeta] = useState(null);
  const [error, setError] = useState("");
  const [page, setPage] = useState(getPageFromHash());

  useEffect(() => {
    fetch("./static/data/leaderboard.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load leaderboard data: ${res.status}`);
        return res.json();
      })
      .then((json) => setData(json))
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    fetch("./static/data/method_meta.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load method metadata: ${res.status}`);
        return res.json();
      })
      .then((json) => setMethodMeta(json))
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    const onHashChange = () => setPage(getPageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  }, [page, data]);

  function navigate(nextPage) {
    window.location.hash = nextPage;
  }

  if (error) {
    return (
      <section className="section">
        <div className="container">
          <article className="message is-danger">
            <div className="message-body">Leaderboard load error: {error}</div>
          </article>
        </div>
      </section>
    );
  }

  if (!data || !methodMeta) {
    return (
      <section className="section">
        <div className="container has-text-centered">
          <p>Loading page data...</p>
        </div>
      </section>
    );
  }

  return (
    <>
      <TopNav page={page} onNavigate={navigate} />
      <Hero />

      {page === "overview" && <OverviewPage />}
      {page === "leaderboard" && <LeaderboardPage data={data} methodMeta={methodMeta} />}
      {page === "contact" && <ContactPage />}
    </>
  );
}

const root = ReactDOM.createRoot(document.getElementById("app"));
root.render(<App />);
