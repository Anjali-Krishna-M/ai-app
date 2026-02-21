let probabilityChart;

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  return res.json();
}

function renderPrediction(data) {
  const card = document.getElementById('predictionCard');
  const featureCard = document.getElementById('featureCard');
  if (!card) return;

  const resultBox = document.getElementById('predictionResult');
  const colorClass = data.prediction === 'Phishing' ? 'danger' : 'safe';

  resultBox.innerHTML = `
    <p><span class="tag ${colorClass}">${data.prediction}</span></p>
    <p>Confidence: <strong>${data.confidence}%</strong></p>
    <p>Typosquatting Alert: <strong>${data.typosquat_hint ? 'Possible domain mimic detected' : 'No mimic pattern detected'}</strong></p>
  `;

  const featureList = document.getElementById('featureList');
  featureList.innerHTML = (data.top_features || []).map(
    (f) => `<li><strong>${f.feature}</strong>: ${f.impact.toFixed(3)}</li>`
  ).join('');

  card.hidden = false;
  featureCard.hidden = false;

  const ctx = document.getElementById('probabilityChart');
  if (ctx) {
    if (probabilityChart) probabilityChart.destroy();
    probabilityChart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: ['Legitimate', 'Phishing'],
        datasets: [{
          data: [data.probabilities.legitimate, data.probabilities.phishing],
          backgroundColor: ['#2ecc71', '#e74c3c'],
        }],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
        },
      },
    });
  }
}

const predictBtn = document.getElementById('predictBtn');
if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    const input = document.getElementById('urlInput');
    const url = input.value.trim();
    if (!url) return;

    const data = await fetchJSON('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });

    if (data.error) {
      alert(data.error);
      return;
    }
    renderPrediction(data);
  });
}

async function loadModelChart() {
  const canvas = document.getElementById('modelComparisonChart');
  if (!canvas) return;

  const data = await fetchJSON('/api/model-metrics');
  const models = data.metrics.map((m) => m.model);
  const f1Scores = data.metrics.map((m) => m.f1);
  const accScores = data.metrics.map((m) => m.accuracy);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        { label: 'Accuracy', data: accScores, backgroundColor: '#2ecc71' },
        { label: 'F1 Score', data: f1Scores, backgroundColor: '#3498db' },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `Best Model: ${data.best_model}`,
        },
      },
      scales: {
        y: { beginAtZero: true, max: 1 },
      },
    },
  });
}

loadModelChart();
