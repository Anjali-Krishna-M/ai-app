async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

function renderResult(data) {
  const card = document.getElementById('resultCard');
  const result = document.getElementById('result');
  if (!card || !result) return;

  const verdictClass = data.verdict === 'Fake' ? 'danger' : 'safe';
  result.innerHTML = `
    <p><span class="tag ${verdictClass}">${data.verdict}</span> Risk Score: <strong>${data.risk_score}%</strong></p>
    <p>Category: <strong>${data.category}</strong></p>
    <p>Trigger words: ${(data.trigger_words || []).map(w => `<span class="tag danger">${w}</span>`).join(' ') || '<small>No explicit red flags found.</small>'}</p>
    <p>Safety Tips:</p>
    <ul>${(data.tips || []).map(t => `<li>${t}</li>`).join('')}</ul>
  `;
  card.hidden = false;
}

const scanBtn = document.getElementById('scanBtn');
if (scanBtn) {
  scanBtn.addEventListener('click', async () => {
    const text = document.getElementById('adText').value;
    const data = await postJSON('/api/scan', { text });
    renderResult(data);
  });
}

const ocrBtn = document.getElementById('ocrBtn');
if (ocrBtn) {
  ocrBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('adImage');
    if (!fileInput.files.length) return;

    const fd = new FormData();
    fd.append('image', fileInput.files[0]);

    const res = await fetch('/api/ocr-scan', { method: 'POST', body: fd });
    const data = await res.json();
    renderResult(data);
  });
}

async function loadDashboard() {
  const totalEl = document.getElementById('totalScans');
  if (!totalEl) return;

  const res = await fetch('/api/stats');
  const data = await res.json();

  document.getElementById('totalScans').textContent = data.total_scans;
  document.getElementById('fakeScans').textContent = data.fake_scans;
  document.getElementById('genuineScans').textContent = data.genuine_scans;

  const history = document.getElementById('history');
  history.innerHTML = (data.history || []).map((h) => `
    <article class="card">
      <p><strong>${h.verdict}</strong> · ${h.category} · ${(h.fake_probability * 100).toFixed(1)}%</p>
      <small>${new Date(h.created_at).toLocaleString()}</small>
      <p>${h.ad_text}</p>
    </article>
  `).join('') || '<small>No scans yet.</small>';

  const ctx = document.getElementById('scanChart');
  if (ctx) {
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Fake', 'Genuine'],
        datasets: [{ data: [data.fake_scans, data.genuine_scans], backgroundColor: ['#ff6b81', '#22c55e'] }],
      },
    });
  }
}

loadDashboard();
