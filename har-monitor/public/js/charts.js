/* Small SVG chart helpers for the dashboard (no external libraries). */
(function () {
  const AX = '#9A9AA6', GRID = '#2C2C33', TRACK = '#24242A';

  const COLORS = {
    LAYING: '#5B7CF7', SITTING: '#A78BFA', STANDING: '#FFC24D',
    WALKING: '#FF6FB5', WALKING_DOWNSTAIRS: '#FF8A5B', WALKING_UPSTAIRS: '#4FE3D6'
  };
  const LABEL = {
    LAYING: 'Laying', SITTING: 'Sitting', STANDING: 'Standing', WALKING: 'Walking',
    WALKING_DOWNSTAIRS: 'Walking downstairs', WALKING_UPSTAIRS: 'Walking upstairs'
  };
  const SHORT = {
    LAYING: 'Laying', SITTING: 'Sitting', STANDING: 'Standing', WALKING: 'Walking',
    WALKING_DOWNSTAIRS: 'Downstairs', WALKING_UPSTAIRS: 'Upstairs'
  };
  const TINY = { LAYING: 'Lay', SITTING: 'Sit', STANDING: 'Stand', WALKING: 'Walk', WALKING_DOWNSTAIRS: 'Down', WALKING_UPSTAIRS: 'Up' };

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // Catmull-Rom → cubic Bézier for smooth lines
  function smooth(pts) {
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
      const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d;
  }

  function curves(el, hist, bestEpoch) {
    const tr = hist.accuracy, va = hist.val_accuracy, n = tr.length;
    const L = 46, R = 750, T = 18, B = 226;
    const lo = Math.max(0, Math.floor(Math.min(...tr, ...va) * 20) / 20 - 0.0), hi = 1.0;
    const X = (i) => L + i * (R - L) / (n - 1), Y = (v) => B - (v - lo) / (hi - lo) * (B - T);
    let g = `<defs>
      <linearGradient id="ga" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFB23E" stop-opacity=".28"/><stop offset="1" stop-color="#FFB23E" stop-opacity="0"/></linearGradient>
      <linearGradient id="gb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FF6FB5" stop-opacity=".32"/><stop offset="1" stop-color="#FF6FB5" stop-opacity="0"/></linearGradient></defs>`;
    for (let v = Math.ceil(lo * 20) / 20; v <= hi + 1e-9; v += 0.05) {
      g += `<line x1="${L}" x2="${R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${GRID}" stroke-dasharray="3 4"/>
            <text x="${L - 10}" y="${Y(v) + 4}" fill="${AX}" font-size="11" text-anchor="end">${Math.round(v * 100)}%</text>`;
    }
    const stepX = Math.max(1, Math.round(n / 8));
    for (let i = 0; i < n; i += stepX) g += `<text x="${X(i)}" y="${B + 20}" fill="${AX}" font-size="11" text-anchor="middle">${i + 1}</text>`;
    g += `<text x="${R}" y="${B + 34}" fill="${AX}" font-size="10.5" text-anchor="end">epoch</text>`;
    const draw = (arr, c, id) => {
      const d = smooth(arr.map((v, i) => [X(i), Y(Math.max(v, lo))]));
      return `<path d="${d} L${R},${B} L${L},${B} Z" fill="url(#${id})"/><path d="${d}" fill="none" stroke="${c}" stroke-width="2.4"/>`;
    };
    if (bestEpoch) {
      const be = bestEpoch - 1;
      g += `<line x1="${X(be)}" x2="${X(be)}" y1="${T}" y2="${B}" stroke="#EDEDF2" stroke-opacity=".3" stroke-dasharray="4 4"/>`;
    }
    g += draw(tr, '#FFB23E', 'ga') + draw(va, '#FF6FB5', 'gb');
    if (bestEpoch) {
      const be = bestEpoch - 1, vb = va[be], bx = Math.min(Math.max(X(be), L + 70), R - 70);
      g += `<circle cx="${X(be)}" cy="${Y(vb)}" r="5" fill="#111" stroke="#FF6FB5" stroke-width="2.5"/>
            <rect x="${bx - 68}" y="${Y(vb) - 44}" width="136" height="28" rx="6" fill="#0F0F12" stroke="#2A2A30"/>
            <text x="${bx}" y="${Y(vb) - 25}" fill="#fff" font-size="12" text-anchor="middle">Best · epoch ${bestEpoch} · ${(vb * 100).toFixed(1)}%</text>`;
    }
    el.innerHTML = g;
  }

  function f1(el, classes, nn, rf) {
    const L = 40, R = 590, T = 16, B = 222;
    const lo = Math.min(0.8, Math.floor(Math.min(...nn, ...(rf || [1])) * 20) / 20), hi = 1.0;
    const Y = (v) => B - (v - lo) / (hi - lo) * (B - T);
    let g = '';
    for (let v = lo; v <= hi + 1e-9; v += 0.05) g += `<line x1="${L}" x2="${R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${GRID}" stroke-dasharray="3 4"/><text x="${L - 8}" y="${Y(v) + 4}" fill="${AX}" font-size="11" text-anchor="end">${v.toFixed(2)}</text>`;
    const w = (R - L) / classes.length;
    classes.forEach((c, i) => {
      const cx = L + w * i + w / 2, bw = 17;
      const bars = [[nn[i], '#FF8A5B', rf ? -bw - 2 : -bw / 2, 'Keras']];
      if (rf) bars.push([rf[i], '#5B7CF7', 2, 'Random Forest']);
      bars.forEach(([v, col, o, who]) => {
        g += `<rect x="${cx + o}" y="${T}" width="${bw}" height="${B - T}" rx="3" fill="${TRACK}"/>
              <rect x="${cx + o}" y="${Y(v)}" width="${bw}" height="${B - Y(v)}" rx="3" fill="${col}"><title>${who} ${SHORT[c]}: ${v.toFixed(3)}</title></rect>`;
      });
      g += `<text x="${cx}" y="${B + 20}" fill="${AX}" font-size="11.5" text-anchor="middle">${SHORT[c]}</text>`;
    });
    el.innerHTML = g;
  }

  function confusion(el, classes, cm) {
    const L = 96, T = 12, cs = 60, ch = 40;
    const mix = (a, b, t) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`;
    const base = [27, 27, 31], good = [79, 227, 214], bad = [255, 111, 181];
    let g = '', worst = { v: 0 };
    cm.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((v, j) => {
        const f = v / tot;
        const fill = i === j ? mix(base, good, 0.35 + 0.65 * f) : (v ? mix(base, bad, Math.min(1, 0.25 + f * 3)) : '#202025');
        if (i !== j && v > worst.v) worst = { v, i, j };
        g += `<rect x="${L + j * cs + 1}" y="${T + i * ch + 1}" width="${cs - 2}" height="${ch - 2}" rx="4" fill="${fill}"><title>True ${SHORT[classes[i]]}, predicted ${SHORT[classes[j]]}: ${v}</title></rect>
              <text x="${L + j * cs + cs / 2}" y="${T + i * ch + ch / 2 + 4}" fill="${i === j ? '#0F0F12' : (v ? '#fff' : '#5A5A66')}" font-size="12" font-weight="${i === j ? 600 : 400}" text-anchor="middle">${v}</text>`;
      });
      g += `<text x="${L - 10}" y="${T + i * ch + ch / 2 + 4}" fill="${AX}" font-size="11.5" text-anchor="end">${SHORT[classes[i]]}</text>`;
    });
    classes.forEach((c, j) => g += `<text x="${L + j * cs + cs / 2}" y="${T + classes.length * ch + 18}" fill="${AX}" font-size="10.5" text-anchor="middle">${TINY[c]}</text>`);
    if (worst.v) g += `<text x="${L + 3 * cs}" y="${T + classes.length * ch + 38}" fill="#FF6FB5" font-size="11.5" text-anchor="middle">${worst.v} ${SHORT[classes[worst.i]]} windows predicted as ${SHORT[classes[worst.j]]}</text>`;
    el.innerHTML = g;
  }

  function distribution(el, classes, counts) {
    const order = classes.map((c, i) => i).sort((a, b) => counts[b] - counts[a]);
    const mx = Math.ceil(Math.max(...counts) / 500) * 500;
    const L = 118, R = 466, T = 10, rh = 40;
    let g = '';
    for (let v = 0; v <= mx; v += 500) {
      const x = L + v / mx * (R - L);
      g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${T + classes.length * rh}" stroke="${GRID}" stroke-dasharray="3 4"/><text x="${x}" y="${T + classes.length * rh + 18}" fill="${AX}" font-size="11" text-anchor="middle">${v}</text>`;
    }
    order.forEach((k, i) => {
      const c = classes[k], y = T + i * rh + 12, w = counts[k] / mx * (R - L);
      g += `<text x="${L - 10}" y="${y + 11}" fill="#CFCFD8" font-size="12" text-anchor="end">${SHORT[c]}</text>
            <rect x="${L}" y="${y}" width="${R - L}" height="14" rx="3" fill="${TRACK}"/>
            <rect x="${L}" y="${y}" width="${w}" height="14" rx="3" fill="${COLORS[c]}"/>
            <text x="${L + w - 6}" y="${y + 11}" fill="#0F0F12" font-size="11" font-weight="600" text-anchor="end">${counts[k].toLocaleString()}</text>`;
    });
    el.innerHTML = g;
  }

  function donut(el, top, conf) {
    const cx = 105, cy = 105, r = 80, sw = 22, C = 2 * Math.PI * r;
    const len = Math.max(C * conf, 2);
    el.innerHTML = `<defs><linearGradient id="dg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5EF0B0"/><stop offset="1" stop-color="#4F7CF0"/></linearGradient></defs>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${TRACK}" stroke-width="${sw}"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="url(#dg)" stroke-width="${sw}" stroke-linecap="round"
        stroke-dasharray="${len} ${C}" transform="rotate(-90 ${cx} ${cy})" style="transition:stroke-dasharray .5s ease"/>
      <text x="${cx}" y="${cy - 2}" fill="#fff" font-size="34" font-weight="300" text-anchor="middle">${Math.round(conf * 100)}%</text>
      <text x="${cx}" y="${cy + 22}" fill="${AX}" font-size="12.5" text-anchor="middle">${esc(LABEL[top] || top)}</text>`;
  }

  window.Charts = { curves, f1, confusion, distribution, donut, COLORS, LABEL, SHORT, esc };
})();
