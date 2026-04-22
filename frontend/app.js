/**
 * Environmental Monitoring Dashboard — Frontend Logic
 *
 * Loads data from the Python API server and renders the dashboard:
 *   - City analysis search bar with live progress
 *   - Stats cards
 *   - Compliance report table (filterable)
 *   - Map gallery
 *   - Training metrics charts (Chart.js)
 */

const API = '';  // Same origin — served by server.py

// ─── State ──────────────────────────────────────────────
let reportData = [];
let currentFilter = 'all';
let pollingInterval = null;

// ─── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Smooth scrolling nav
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            renderTable();
        });
    });

    // Enter key on search
    document.getElementById('city-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') startAnalysis();
    });

    // Check if there's already a running job
    await checkExistingJob();

    // Load everything
    await Promise.all([
        loadConfig(),
        loadReport(),
        loadMaps(),
        loadMetrics(),
    ]);
});


// ═══════════════════════════════════════════════════════════
//  CITY ANALYSIS — Search + Progress
// ═══════════════════════════════════════════════════════════

async function startAnalysis() {
    const input = document.getElementById('city-input');
    const btn = document.getElementById('analyze-btn');
    const city = input.value.trim();

    if (!city) {
        input.focus();
        input.style.boxShadow = '0 0 0 2px rgba(239,68,68,0.5) inset';
        setTimeout(() => input.style.boxShadow = '', 1500);
        return;
    }

    // Read date inputs
    const t1Start = document.getElementById('t1-start').value;
    const t1End   = document.getElementById('t1-end').value;
    const t2Start = document.getElementById('t2-start').value;
    const t2End   = document.getElementById('t2-end').value;

    // Disable button
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Starting...';

    try {
        const res = await fetch(`${API}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                city,
                t1_start: t1Start,
                t1_end: t1End,
                t2_start: t2Start,
                t2_end: t2End,
            }),
        });
        const data = await res.json();

        if (data.status === 'already_running') {
            btn.querySelector('.btn-text').textContent = 'Already running';
            setTimeout(() => {
                btn.disabled = false;
                btn.querySelector('.btn-text').textContent = 'Analyze';
            }, 2000);
            return;
        }

        // Show progress panel and start polling
        showProgressPanel(city);
        startPolling();

    } catch (err) {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Analyze';
        alert('Failed to start analysis. Is the server running?');
    }
}

async function checkExistingJob() {
    try {
        const res = await fetch(`${API}/api/status`);
        const data = await res.json();
        if (data.status === 'running') {
            showProgressPanel(data.city);
            startPolling();
        }
    } catch {}
}

function showProgressPanel(city) {
    const panel = document.getElementById('progress-panel');
    const title = document.getElementById('progress-title');
    const log = document.getElementById('progress-log');

    panel.style.display = 'block';
    panel.className = 'progress-panel';
    title.textContent = `Analyzing ${city}...`;
    log.innerHTML = '<div class="log-line">Starting analysis...</div>';

    // Disable analyze button
    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Running...';
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(pollStatus, 2000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function pollStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        const data = await res.json();

        const panel = document.getElementById('progress-panel');
        const title = document.getElementById('progress-title');
        const log = document.getElementById('progress-log');
        const elapsed = document.getElementById('progress-elapsed');

        // Update elapsed time
        if (data.elapsed_seconds) {
            const mins = Math.floor(data.elapsed_seconds / 60);
            const secs = Math.floor(data.elapsed_seconds % 60);
            elapsed.textContent = `${mins}m ${secs}s`;
        }

        // Update log
        if (data.progress && data.progress.length > 0) {
            log.innerHTML = data.progress
                .map(line => `<div class="log-line">${escapeHtml(line)}</div>`)
                .join('');
            // Auto-scroll to bottom
            log.scrollTop = log.scrollHeight;
        }

        // Handle completion
        if (data.status === 'done') {
            stopPolling();
            panel.classList.add('progress-done');
            title.textContent = `✅ Analysis complete for ${data.city}`;

            const btn = document.getElementById('analyze-btn');
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'Analyze';

            // Reload dashboard data
            await Promise.all([loadReport(), loadMaps()]);

        } else if (data.status === 'error') {
            stopPolling();
            panel.classList.add('progress-error');
            title.textContent = `❌ Analysis failed for ${data.city}`;

            const btn = document.getElementById('analyze-btn');
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'Analyze';

        } else if (data.status === 'idle') {
            stopPolling();
            panel.style.display = 'none';

            const btn = document.getElementById('analyze-btn');
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'Analyze';
        }

    } catch (err) {
        console.error('Poll error:', err);
    }
}


// ═══════════════════════════════════════════════════════════
//  DATA LOADING
// ═══════════════════════════════════════════════════════════

// ─── Config ─────────────────────────────────────────────
async function loadConfig() {
    try {
        const res = await fetch(`${API}/api/config`);
        const cfg = await res.json();
        document.getElementById('tag-roi').textContent = cfg.roi || 'Unknown';
        document.getElementById('tag-period').textContent =
            `${cfg.t1_start || '?'} → ${cfg.t2_end || '?'}`;
    } catch {
        document.getElementById('tag-roi').textContent = 'Unavailable';
    }
}

// ─── Compliance Report ──────────────────────────────────
async function loadReport() {
    try {
        const res = await fetch(`${API}/api/report`);
        const raw = await res.json();

        // Handle both formats: {summary, regions} and flat array
        if (raw && raw.regions) {
            reportData = raw.regions;
            if (raw.summary) updateStatsFromSummary(raw.summary);
            else updateStats();
        } else if (Array.isArray(raw)) {
            reportData = raw;
            updateStats();
        } else {
            reportData = [];
            updateStats();
        }

        renderTable();
    } catch {
        document.getElementById('report-body').innerHTML =
            '<tr><td colspan="5" class="empty-state"><div class="icon">📭</div><p>No compliance report found. Run an analysis above.</p></td></tr>';
    }
}

function updateStatsFromSummary(summary) {
    document.getElementById('stat-total').textContent = summary.total_regions || 0;
    document.getElementById('stat-compliant').textContent = summary.compliant || 0;
    document.getElementById('stat-violations').textContent = summary.violations || 0;
    document.getElementById('stat-avg-veg').textContent =
        ((summary.avg_vegetation_percent || 0) * 100).toFixed(1) + '%';
    document.getElementById('stat-change-area').textContent =
        formatArea(summary.total_change_area_m2 || 0);
    document.getElementById('tag-regions').textContent = summary.total_regions || 0;
    document.getElementById('report-badge').textContent = `${summary.total_regions || 0} regions`;

    if (summary.roi) {
        document.getElementById('tag-roi').textContent = summary.roi;
    }
    if (summary.t1_period && summary.t2_period) {
        document.getElementById('tag-period').textContent =
            `${summary.t1_period} → ${summary.t2_period}`;
    }
}

function updateStats() {
    const total = reportData.length;
    const violations = reportData.filter(r => r.violation).length;
    const compliant = total - violations;
    const avgVeg = total > 0
        ? (reportData.reduce((s, r) => s + r.vegetation_percent, 0) / total * 100)
        : 0;
    const totalChange = reportData.reduce((s, r) => s + r.change_area_m2, 0);

    document.getElementById('stat-total').textContent = total;
    document.getElementById('stat-compliant').textContent = compliant;
    document.getElementById('stat-violations').textContent = violations;
    document.getElementById('stat-avg-veg').textContent = avgVeg.toFixed(1) + '%';
    document.getElementById('stat-change-area').textContent = formatArea(totalChange);
    document.getElementById('tag-regions').textContent = total;
    document.getElementById('report-badge').textContent = `${total} regions`;
}

function renderTable() {
    const tbody = document.getElementById('report-body');
    let filtered = reportData;

    if (currentFilter === 'violation') {
        filtered = reportData.filter(r => r.violation);
    } else if (currentFilter === 'compliant') {
        filtered = reportData.filter(r => !r.violation);
    }

    document.getElementById('table-count').textContent =
        `Showing ${filtered.length} of ${reportData.length}`;

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><div class="icon">✅</div><p>No matching regions.</p></td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(r => `
        <tr>
            <td><strong>#${r.region_id}</strong></td>
            <td>${r.size_px ? r.size_px.join(' × ') : '—'}</td>
            <td>${(r.vegetation_percent * 100).toFixed(1)}%</td>
            <td>${formatArea(r.change_area_m2)}</td>
            <td>
                <span class="status-badge ${r.violation ? 'violation' : 'compliant'}">
                    ${r.violation ? '🔴 Non-Compliant' : '🟢 Compliant'}
                </span>
            </td>
        </tr>
    `).join('');
}

// ─── Map Gallery ────────────────────────────────────────
async function loadMaps() {
    const gallery = document.getElementById('map-gallery');
    try {
        const res = await fetch(`${API}/api/maps`);
        const maps = await res.json();

        if (!maps.length) {
            gallery.innerHTML = '<div class="empty-state"><div class="icon">🗺️</div><p>No maps generated yet. Run an analysis above.</p></div>';
            return;
        }

        gallery.innerHTML = maps.map(m => `
            <div class="map-card fade-in">
                <img src="${API}/outputs/maps/${m}" alt="${m}" loading="lazy">
                <div class="map-info">
                    <span class="map-label">${m.replace(/_/g, ' ').replace('.png', '')}</span>
                </div>
            </div>
        `).join('');
    } catch {
        gallery.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><p>Could not load maps.</p></div>';
    }
}

// ─── Training Metrics ───────────────────────────────────
async function loadMetrics() {
    const grid = document.getElementById('metrics-grid');
    try {
        const res = await fetch(`${API}/api/metrics`);
        const metrics = await res.json();

        if (!metrics || Object.keys(metrics).length === 0) {
            grid.innerHTML = '<div class="empty-state"><div class="icon">📊</div><p>No training metrics found. Train models first.</p></div>';
            return;
        }

        grid.innerHTML = '';

        for (const [modelName, history] of Object.entries(metrics)) {
            const prettyName = modelName.replace(/_/g, ' ').replace('history', '').trim();

            if (history.loss) {
                const card = createMetricsCard(`${capitalize(prettyName)} — Loss`);
                grid.appendChild(card.el);
                createChart(card.canvas, history, 'loss', 'val_loss', 'Loss', '#ff6b6b', '#ffd93d');
            }

            if (history.accuracy) {
                const card = createMetricsCard(`${capitalize(prettyName)} — Accuracy`);
                grid.appendChild(card.el);
                createChart(card.canvas, history, 'accuracy', 'val_accuracy', 'Accuracy', '#6bcb77', '#4d96ff');
            }

            if (history.iou_metric) {
                const card = createMetricsCard(`${capitalize(prettyName)} — IoU`);
                grid.appendChild(card.el);
                createChart(card.canvas, history, 'iou_metric', 'val_iou_metric', 'IoU', '#ff9a3c', '#a855f7');
            }

            if (history.precision_metric) {
                const card = createMetricsCard(`${capitalize(prettyName)} — Precision & Recall`);
                grid.appendChild(card.el);
                createDualChart(card.canvas, history);
            }
        }

        // Also load curve images
        for (const name of ['vegetation_curves.png', 'change_detection_curves.png']) {
            try {
                const imgRes = await fetch(`${API}/outputs/metrics/${name}`, { method: 'HEAD' });
                if (imgRes.ok) {
                    const card = document.createElement('div');
                    card.className = 'metrics-card fade-in';
                    card.innerHTML = `<h3>${name.replace(/_/g, ' ').replace('.png', '')}</h3>
                        <img src="${API}/outputs/metrics/${name}" alt="${name}">`;
                    grid.appendChild(card);
                }
            } catch {}
        }
    } catch {
        grid.innerHTML = '<div class="empty-state"><div class="icon">📊</div><p>Could not load metrics.</p></div>';
    }
}


// ═══════════════════════════════════════════════════════════
//  CHART HELPERS
// ═══════════════════════════════════════════════════════════

function createMetricsCard(title) {
    const el = document.createElement('div');
    el.className = 'metrics-card fade-in';
    const h3 = document.createElement('h3');
    h3.textContent = title;
    const canvas = document.createElement('canvas');
    el.appendChild(h3);
    el.appendChild(canvas);
    return { el, canvas };
}

function createChart(canvas, history, trainKey, valKey, label, trainColor, valColor) {
    const epochs = Array.from({ length: history[trainKey].length }, (_, i) => i + 1);
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: `Train ${label}`,
                    data: history[trainKey],
                    borderColor: trainColor,
                    backgroundColor: trainColor + '20',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                },
                ...(history[valKey] ? [{
                    label: `Val ${label}`,
                    data: history[valKey],
                    borderColor: valColor,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0,
                }] : []),
            ],
        },
        options: chartOptions(label),
    });
}

function createDualChart(canvas, history) {
    const epochs = Array.from({ length: (history.precision_metric || []).length }, (_, i) => i + 1);
    const datasets = [];
    if (history.precision_metric)
        datasets.push({ label: 'Train Precision', data: history.precision_metric,
            borderColor: '#38bdf8', tension: 0.4, borderWidth: 2, pointRadius: 0 });
    if (history.val_precision_metric)
        datasets.push({ label: 'Val Precision', data: history.val_precision_metric,
            borderColor: '#38bdf8', borderDash: [5,5], tension: 0.4, borderWidth: 2, pointRadius: 0 });
    if (history.recall_metric)
        datasets.push({ label: 'Train Recall', data: history.recall_metric,
            borderColor: '#fb7185', tension: 0.4, borderWidth: 2, pointRadius: 0 });
    if (history.val_recall_metric)
        datasets.push({ label: 'Val Recall', data: history.val_recall_metric,
            borderColor: '#fb7185', borderDash: [5,5], tension: 0.4, borderWidth: 2, pointRadius: 0 });

    new Chart(canvas, {
        type: 'line',
        data: { labels: epochs, datasets },
        options: chartOptions('Score'),
    });
}

function chartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#94a3b8', font: { size: 11 } },
            },
        },
        scales: {
            x: {
                title: { display: true, text: 'Epoch', color: '#64748b' },
                ticks: { color: '#64748b' },
                grid: { color: 'rgba(255,255,255,0.04)' },
            },
            y: {
                title: { display: true, text: yLabel, color: '#64748b' },
                ticks: { color: '#64748b' },
                grid: { color: 'rgba(255,255,255,0.04)' },
            },
        },
    };
}


// ═══════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════

function formatArea(m2) {
    if (m2 >= 1e6) return (m2 / 1e6).toFixed(2) + ' km²';
    if (m2 >= 1e4) return (m2 / 1e4).toFixed(1) + ' ha';
    return m2.toFixed(0) + ' m²';
}

function capitalize(str) {
    return str.replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
