/* AEOLUS web viewer: interactive 3D cloud of FermiNet-sampled electron positions.
   Data is loaded from data/hydrogen_v0.json (see src/aeolus/export_web.py). */
(() => {
  'use strict';

  const DATA_URL = 'data/hydrogen_v0.json';
  const el = id => document.getElementById(id);
  const cloud = el('cloud');
  const radial = el('radial');
  const state = { snap: 'all', rmax: 6, size: 3.5, cut: false, spin: false };
  let DATA = null;

  // ---- theme tokens, read from CSS so charts follow light/dark ----
  function tok() {
    const s = getComputedStyle(document.documentElement);
    const g = n => s.getPropertyValue(n).trim();
    return { text: g('--text'), muted: g('--muted'), line: g('--line'), bars: g('--bars'),
             curve: g('--curve'), nucleus: g('--nucleus') };
  }

  // ---- data selection ----
  const rOf = p => Math.hypot(p[0], p[1], p[2]);
  const selected = () => (state.snap === 'all' ? DATA.snaps.flat() : DATA.snaps[+state.snap]);

  // ---- 3D cloud ----
  function drawCloud() {
    const t = tok();
    const all = selected();
    const shown = all.filter(p => rOf(p) <= state.rmax && !(state.cut && p[1] > 0));
    const r = shown.map(rOf);
    const samples = {
      type: 'scatter3d', mode: 'markers', name: 'electron samples',
      x: shown.map(p => p[0]), y: shown.map(p => p[1]), z: shown.map(p => p[2]),
      marker: { size: state.size, color: r, colorscale: 'Viridis', cmin: 0, cmax: 5, opacity: 0.85,
                colorbar: { title: { text: 'r (bohr)', font: { color: t.muted, size: 12 } },
                            tickfont: { color: t.muted, size: 11 }, thickness: 10, len: 0.55,
                            outlinewidth: 0 } },
      hovertemplate: 'distance %{marker.color:.2f} bohr<br>x %{x:.2f}, y %{y:.2f}, z %{z:.2f}<extra></extra>'
    };
    const nucleus = {
      type: 'scatter3d', mode: 'markers', name: 'nucleus', x: [0], y: [0], z: [0],
      marker: { size: 7, color: t.nucleus, symbol: 'diamond', line: { color: t.text, width: 1 } },
      hovertemplate: 'nucleus (proton)<extra></extra>'
    };
    const ax = title => ({ title: { text: title, font: { color: t.muted, size: 12 } }, range: [-5, 5],
                           color: t.muted, gridcolor: t.line, zeroline: false, showbackground: false,
                           tickfont: { color: t.muted, size: 10 } });
    const layout = {
      margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: 'rgba(0,0,0,0)', showlegend: false,
      uirevision: 'keep',
      scene: { xaxis: ax('x (bohr)'), yaxis: ax('y (bohr)'), zaxis: ax('z (bohr)'),
               aspectmode: 'cube', bgcolor: 'rgba(0,0,0,0)',
               camera: { eye: { x: 1.5, y: 1.2, z: 0.8 } } },
      font: { family: 'Instrument Sans, system-ui, sans-serif', color: t.text }
    };
    Plotly.react(cloud, [samples, nucleus], layout, { responsive: true, displaylogo: false });
    el('count').textContent = `${shown.length} of ${all.length}`;
  }

  // ---- radial histogram against the exact 1s distribution ----
  function drawRadial() {
    const t = tok();
    const r = selected().map(rOf);
    const nb = 30, lo = 0, hi = 6, w = (hi - lo) / nb;
    const counts = new Array(nb).fill(0);
    r.forEach(v => { const k = Math.floor((v - lo) / w); if (k >= 0 && k < nb) counts[k]++; });
    const centers = counts.map((_, k) => lo + (k + 0.5) * w);
    const dens = counts.map(c => c / (r.length * w));
    const xs = Array.from({ length: 241 }, (_, i) => i * 0.025);
    const bars = { type: 'bar', x: centers, y: dens, width: w * 0.95, name: 'FermiNet samples',
                   marker: { color: t.bars, opacity: 0.75 }, hovertemplate: 'r %{x:.2f}: %{y:.3f}<extra></extra>' };
    const exact = { type: 'scatter', mode: 'lines', x: xs, y: xs.map(x => 4 * x * x * Math.exp(-2 * x)),
                    name: 'exact 1s', line: { color: t.curve, width: 2.5 }, hoverinfo: 'skip' };
    const layout = {
      margin: { l: 40, r: 8, t: 6, b: 38 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { title: { text: 'r (bohr)', font: { color: t.muted, size: 12 } }, color: t.muted,
               gridcolor: t.line, zeroline: false, range: [0, 6] },
      yaxis: { title: { text: 'P(r)', font: { color: t.muted, size: 12 } }, color: t.muted,
               gridcolor: t.line, zeroline: false },
      legend: { orientation: 'h', y: 1.02, x: 0, font: { color: t.muted, size: 11 } },
      font: { family: 'Instrument Sans, system-ui, sans-serif', color: t.text }
    };
    Plotly.react(radial, [bars, exact], layout, { responsive: true, displayModeBar: false });
    el('meanr').textContent = (r.reduce((a, b) => a + b, 0) / r.length).toFixed(3);
  }

  const redraw = () => { drawCloud(); drawRadial(); };

  // ---- optional rotation, only when the viewer turns it on ----
  let ang = Math.atan2(1.2, 1.5), raf = null;
  function tick() {
    if (!state.spin) return;
    ang += 0.007;
    Plotly.relayout(cloud, { 'scene.camera.eye': { x: 1.8 * Math.cos(ang), y: 1.8 * Math.sin(ang), z: 0.8 } });
    raf = requestAnimationFrame(tick);
  }
  function setSpin(on) {
    state.spin = on; el('spin').checked = on;
    if (raf) cancelAnimationFrame(raf);
    if (on) raf = requestAnimationFrame(tick);
  }

  // ---- wiring ----
  function init() {
    const sel = el('snap');
    const per = DATA.snaps[0].length;
    sel.add(new Option(`All ${DATA.steps.length} checkpoints (${DATA.steps.length * per} walkers)`, 'all'));
    DATA.steps.forEach((s, i) =>
      sel.add(new Option(`Step ${s.toLocaleString('en-US')} (${DATA.snaps[i].length} walkers)`, String(i))));

    const first = DATA.steps[0].toLocaleString('en-US');
    const last = DATA.steps[DATA.steps.length - 1].toLocaleString('en-US');
    el('foot').innerHTML = `Positions are walker coordinates stored in ${DATA.steps.length} checkpoints ` +
      `(steps ${first} to ${last}), ${per} walkers each, in bohr. The red curve is the exact ` +
      `hydrogen 1s result, 4r<sup>2</sup>e<sup>-2r</sup>.`;

    sel.addEventListener('change', e => { state.snap = e.target.value; redraw(); });
    el('rmax').addEventListener('input', e => {
      state.rmax = +e.target.value; el('rmaxv').textContent = `${state.rmax.toFixed(1)} bohr`; drawCloud();
    });
    el('size').addEventListener('input', e => {
      state.size = +e.target.value; el('sizev').textContent = state.size.toFixed(1); drawCloud();
    });
    el('cut').addEventListener('change', e => { state.cut = e.target.checked; drawCloud(); });
    el('spin').addEventListener('change', e => setSpin(e.target.checked));
    cloud.addEventListener('pointerdown', () => { if (state.spin) setSpin(false); });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
    new MutationObserver(redraw).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme'] });

    el('rmaxv').textContent = `${state.rmax.toFixed(1)} bohr`;
    el('sizev').textContent = state.size.toFixed(1);
    redraw();
  }

  function showError(msg) {
    const box = el('error');
    box.textContent = msg;
    box.hidden = false;
  }

  fetch(DATA_URL)
    .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
    .then(json => { DATA = json; init(); })
    .catch(err => showError(
      `Could not load ${DATA_URL} (${err.message}). If you opened index.html directly from disk, ` +
      `serve the folder instead: run "python -m http.server" inside web/ and open http://localhost:8000.`));
})();
