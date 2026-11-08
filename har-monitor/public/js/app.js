/* HAR Monitor dashboard logic. */
(function () {
  const { COLORS, LABEL, esc } = window.Charts;

  // ------------------------------------------------------------------ config
  const CONFIG = {
    streamEveryMs: 2560,           // one 2.56 s window at a time, like the real sensor
    timelineLength: 40,            // windows kept in the live timeline
    uploadChunkRows: 250,          // rows per /api/predict/batch request (stays < 4.5 MB)
    // Example responses per activity. Replace with your line's real rules.
    responses: {
      LAYING: 'Possible fall or person down: alert supervisor, stop nearby machines.',
      SITTING: 'Operator seated at station: normal operation.',
      STANDING: 'Operator standing at station: normal operation.',
      WALKING: 'Operator moving: keep robot cell at reduced speed.',
      WALKING_UPSTAIRS: 'Operator on stairs: pause forklift lane below.',
      WALKING_DOWNSTAIRS: 'Operator on stairs: pause forklift lane below.'
    }
  };

  const $ = (s, el = document) => el.querySelector(s);
  const setK = (k, v) => document.querySelectorAll(`[data-k="${k}"]`).forEach((e) => (e.textContent = v));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
  const api = async (path, opts) => {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.status + ' ' + r.statusText;
      try { const j = await r.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  };

  // ------------------------------------------------------------- background
  // Soft glow that follows the cursor (styled by .amb-glow in style.css).
  const ambGlow = document.querySelector('.amb-glow');
  if (ambGlow) {
    window.addEventListener('pointermove', (e) => {
      ambGlow.style.left = e.clientX + 'px';
      ambGlow.style.top = e.clientY + 'px';
    }, { passive: true });
  }

  let CLASSES = Object.keys(COLORS);

  // ---------------------------------------------------------------- health
  api('/api/health').then((h) => {
    $('#api-dot').className = 'dot ok';
    $('#api-text').textContent = 'FastAPI /predict · online';
    CLASSES = h.classes;
  }).catch(() => {
    $('#api-dot').className = 'dot bad';
    $('#api-text').textContent = 'API offline';
  });

  // ---------------------------------------------------------------- metrics
  api('/api/metrics').then((m) => {
    CLASSES = m.classes;
    const d = m.dataset, k = m.keras, rf = m.random_forest;
    setK('train', d.train_windows.toLocaleString()); setK('train_s', `${d.train_subjects} subjects`);
    setK('test', d.test_windows.toLocaleString()); setK('test_s', `${d.test_subjects} unseen subjects`);
    setK('feat', d.features); setK('cls', m.classes.length);
    setK('acc', pct(k.test_accuracy)); setK('acc_s', rf ? `Random Forest ${pct(rf.test_accuracy)}` : 'held-out subjects');
    setK('nn_acc', pct(k.test_accuracy)); setK('nn_f1', `Macro F1 ${k.macro_f1.toFixed(3)}`);
    if (rf) { setK('rf_acc', pct(rf.test_accuracy)); setK('rf_f1', `Macro F1 ${rf.macro_f1.toFixed(3)}`); }
    setK('val', pct(k.val_accuracy)); setK('params', k.params.toLocaleString());

    if (k.history) {
      Charts.curves($('#curve'), k.history, k.best_epoch);
      $('#curve-sub').textContent = `Subject-wise validation · early stopping restored epoch ${k.best_epoch}`;
    } else {
      $('#curve').innerHTML = '<text x="380" y="130" fill="#9A9AA6" font-size="13" text-anchor="middle">Export with --history to show training curves</text>';
    }
    Charts.f1($('#f1'), m.classes, k.f1, rf && rf.f1);
    Charts.confusion($('#cm'), m.classes, k.confusion);
    $('#cm-sub').textContent = `Rows = true activity, columns = predicted (${d.test_windows.toLocaleString()} test windows)`;
    Charts.distribution($('#dist'), m.classes, d.train_counts);
    renderLegend();
  }).catch((e) => console.error('metrics', e));

  // ---------------------------------------------------------------- pipeline
  const STEPS = [
    ['Wearable sensors', 'Accelerometer + gyroscope, 50 Hz', '#5B7CF7'],
    ['Windowing', '128 samples · 2.56 s · 50% overlap', '#A78BFA'],
    ['Feature extraction', '561 time + frequency features', '#4FE3D6'],
    ['Keras Dense', '512 → 256 → 128, BatchNorm, Dropout', '#FF8A5B'],
    ['FastAPI /predict', 'Activity + confidence → line response', '#FF6FB5']
  ];
  const arrow = '<div class="arr" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></div>';
  $('#pipe').innerHTML = STEPS.map(([h, d, c], i) =>
    `${i ? arrow : ''}<div class="st"><div class="n" style="background:${c}">${i + 1}</div><div class="h">${h}</div><div class="d">${d}</div></div>`).join('');

  function renderLegend() {
    $('#tleg').innerHTML = CLASSES.map((c) => `<span><i style="background:${COLORS[c]}"></i>${LABEL[c]}</span>`).join('') +
      '<span><i style="background:#555"></i>faded = low confidence</span>';
  }
  renderLegend();

  // ------------------------------------------------------------------ live
  const timeline = [];
  let seen = 0, correct = 0, latSum = 0, timer = null, paused = false;

  function showWindow(w, latency) {
    seen += 1; correct += w.correct ? 1 : 0; latSum += latency;
    Charts.donut($('#donut'), w.activity, w.confidence);
    const probs = Object.entries(w.probabilities).sort((a, b) => b[1] - a[1]);
    $('#dl').innerHTML = probs.map(([c, p], i) =>
      `<div class="${i === 0 ? 'top1' : ''}"><i style="background:${COLORS[c]}"></i><span>${LABEL[c]}</span><b>${pct(p, 0)}</b></div>`).join('');
    $('#truth').innerHTML = `True activity: <b>${esc(LABEL[w.true_activity])}</b> (subject ${w.subject}) · ` +
      (w.correct ? '<span class="ok">✓ correct</span>' : '<span class="no">✗ misclassified</span>');
    const r = $('#response');
    if (w.low_confidence) {
      r.className = 'flag';
      r.textContent = `Low confidence (${pct(w.confidence, 0)}): window sent for review, no machine action taken.`;
    } else {
      r.className = 'flag calm';
      r.textContent = 'Example response: ' + (CONFIG.responses[w.activity] || '');
    }
    timeline.push(w);
    if (timeline.length > CONFIG.timelineLength) timeline.shift();
    $('#tl').innerHTML = timeline.map((t) =>
      `<div class="${t.low_confidence ? 'lc' : ''}" style="background:${COLORS[t.activity]}" title="${LABEL[t.activity]} · ${pct(t.confidence, 0)}"></div>`).join('');
    $('#tl-n').textContent = timeline.length;
    $('#tl-acc').textContent = `${correct}/${seen} correct (${pct(correct / seen, 0)})`;
    $('#lat').textContent = (latSum / seen).toFixed(2) + ' ms';
  }

  async function tick() {
    try {
      const r = await api('/api/stream?n=1');
      showWindow(r.windows[0], r.latency_ms);
    } catch (e) {
      $('#response').className = 'flag';
      $('#response').textContent = 'Could not reach the API: ' + e.message;
    }
  }
  function start() { tick(); timer = setInterval(tick, CONFIG.streamEveryMs); }
  function stop() { clearInterval(timer); timer = null; }

  $('#live-toggle').addEventListener('click', (e) => {
    paused = !paused;
    e.currentTarget.textContent = paused ? 'Resume' : 'Pause';
    e.currentTarget.setAttribute('aria-pressed', String(paused));
    paused ? stop() : start();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop(); else if (!paused && !timer) start();
  });
  start();

  // ------------------------------------------------------------ sidebar
  const links = [...document.querySelectorAll('.side .ic[data-sec]')];
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) links.forEach((l) => l.classList.toggle('on', l.dataset.sec === en.target.id));
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  ['overview', 'model', 'live', 'predict'].forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el); });

  // ------------------------------------------------------------- upload
  function splitCSVLine(line) {
    const out = []; let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (q) {
        if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
        else if (ch === '"') q = false;
        else cur += ch;
      } else if (ch === '"') q = true;
      else if (ch === ',') { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }

  function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
    if (!lines.length) throw new Error('The file is empty.');
    const first = splitCSVLine(lines[0]);
    const hasHeader = first.some((v) => v.trim() !== '' && isNaN(Number(v)));
    let featIdx, labelIdx = -1;
    if (hasHeader) {
      const names = first.map((s) => s.trim());
      labelIdx = names.findIndex((n) => n.toLowerCase() === 'activity');
      featIdx = names.map((n, i) => i).filter((i) => i !== labelIdx && names[i].toLowerCase() !== 'subject');
    } else {
      featIdx = first.map((_, i) => i).slice(0, 561);
    }
    if (featIdx.length !== 561) throw new Error(`Expected 561 feature columns, found ${featIdx.length}.`);
    const rows = [], labels = [];
    for (let r = hasHeader ? 1 : 0; r < lines.length; r++) {
      const cells = lines[r].split(',');
      const row = featIdx.map((i) => Math.round(Number(cells[i]) * 1e6) / 1e6);
      if (row.some((v) => !Number.isFinite(v))) throw new Error(`Row ${r + 1} has a missing or non-numeric value.`);
      rows.push(row);
      if (labelIdx >= 0) labels.push((cells[labelIdx] || '').replace(/"/g, '').trim());
    }
    return { rows, labels: labelIdx >= 0 ? labels : null };
  }

  function toCSV(preds, conf, labels) {
    const head = labels ? 'row,predicted,confidence,true_activity,correct' : 'row,predicted,confidence';
    const body = preds.map((p, i) => labels
      ? `${i + 1},${p},${conf[i]},${labels[i]},${p === labels[i]}`
      : `${i + 1},${p},${conf[i]}`);
    return [head, ...body].join('\n');
  }

  async function handleFile(file) {
    const res = $('#res');
    if (!file) return;
    res.innerHTML = `<div><b>${esc(file.name)}</b> <span class="sub">· ${(file.size / 1e6).toFixed(2)} MB</span></div><div class="prog"><i id="pbar"></i></div><div class="sub" id="pstat">Reading file…</div>`;
    try {
      const { rows, labels } = parseCSV(await file.text());
      const preds = [], conf = [];
      let ms = 0;
      for (let i = 0; i < rows.length; i += CONFIG.uploadChunkRows) {
        const chunk = rows.slice(i, i + CONFIG.uploadChunkRows);
        const r = await api('/api/predict/batch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows: chunk })
        });
        preds.push(...r.predictions); conf.push(...r.confidence); ms += r.latency_ms;
        $('#pbar').style.width = pct(Math.min(1, (i + chunk.length) / rows.length));
        $('#pstat').textContent = `Classified ${preds.length.toLocaleString()} of ${rows.length.toLocaleString()} rows…`;
      }
      const counts = {}; CLASSES.forEach((c) => (counts[c] = 0));
      preds.forEach((p) => (counts[p] = (counts[p] || 0) + 1));
      const acc = labels ? preds.filter((p, i) => p === labels[i]).length / preds.length : null;
      const low = conf.filter((c) => c < 0.7).length;
      const mx = Math.max(...Object.values(counts));
      const csv = toCSV(preds, conf, labels);
      const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      res.innerHTML = `
        <div><b>${esc(file.name)}</b> <span class="sub">· done</span></div>
        <div class="stats">
          <div><span class="sub">Rows</span><b>${preds.length.toLocaleString()}</b></div>
          <div><span class="sub">${labels ? 'Accuracy' : 'Avg confidence'}</span><b>${labels ? pct(acc) : pct(conf.reduce((a, b) => a + b, 0) / conf.length)}</b></div>
          <div><span class="sub">Low confidence</span><b>${low.toLocaleString()}</b></div>
        </div>
        <div class="bars">${CLASSES.map((c) => `<div><span>${LABEL[c]}</span><span class="tr2"><i style="width:${(counts[c] / mx) * 100}%;background:${COLORS[c]}"></i></span><span>${counts[c]}</span></div>`).join('')}</div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
          <a class="btn primary" style="display:inline-flex;align-items:center;text-decoration:none" href="${url}" download="${esc(file.name.replace(/\.csv$/i, ''))}_predictions.csv">Download predictions CSV</a>
          <span class="sub">Server inference time: ${ms.toFixed(1)} ms total</span>
        </div>`;
    } catch (e) {
      res.innerHTML = `<div class="err">Could not process the file: ${esc(e.message)}</div>`;
    }
  }

  const drop = $('#drop'), input = $('#csv');
  input.addEventListener('change', () => handleFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
})();
