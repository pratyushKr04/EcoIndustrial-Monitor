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
        loadStaticMaps(),
        loadMetrics(),
    ]);

    // Initialize interactive map after DOM is ready
    initInteractiveMap();
    // Plot regions if report data was already loaded
    if (reportData.length > 0) plotRegionsOnMap(reportData);
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

// Pipeline steps detected from log output
const PIPELINE_STEPS = [
    { key: 'init',      label: 'Initializing',              pattern: /Starting analysis|Running:/ },
    { key: 'roi',       label: 'Resolving Region',          pattern: /ROI|roi|geocod/i },
    { key: 'osm',       label: 'Fetching OSM Data',         pattern: /OSM|industrial polygon|osm/i },
    { key: 'gee',       label: 'Google Earth Engine',       pattern: /GEE|Earth Engine|Authenticat/i },
    { key: 'satellite', label: 'Downloading Satellite Data', pattern: /Fetching t[12]|sentinel|image/i },
    { key: 'models',    label: 'Loading AI Models',         pattern: /Loading.*model|load_model|\.h5/i },
    { key: 'process',   label: 'Processing Regions',        pattern: /Processing region|region \d+/i },
    { key: 'report',    label: 'Generating Report',         pattern: /Compliance report|SUMMARY|saved/i },
];

let currentStepIndex = 0;

function showProgressPanel(city) {
    const panel = document.getElementById('progress-panel');
    const title = document.getElementById('progress-title');
    const log = document.getElementById('progress-log');

    panel.style.display = 'block';
    panel.className = 'progress-panel';
    title.textContent = `Analyzing ${city}`;
    currentStepIndex = 0;

    // Build step tracker HTML
    const stepsHtml = PIPELINE_STEPS.map((step, i) => `
        <div class="step-item ${i === 0 ? 'active' : ''}" id="step-${step.key}">
            <div class="step-indicator">
                <div class="step-dot"></div>
            </div>
            <span class="step-label">${step.label}</span>
        </div>
    `).join('');

    log.innerHTML = `
        <div class="step-tracker">${stepsHtml}</div>
        <div class="log-output" id="log-output">
            <div class="log-line">Initializing pipeline...</div>
        </div>
    `;

    // Disable analyze button
    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Running...';
}

function updateStepTracker(progressLines) {
    // Detect which step we're on based on log content
    const fullText = progressLines.join('\n');

    for (let i = PIPELINE_STEPS.length - 1; i >= 0; i--) {
        if (PIPELINE_STEPS[i].pattern.test(fullText)) {
            if (i > currentStepIndex) {
                currentStepIndex = i;
            }
            break;
        }
    }

    // Update step indicators
    PIPELINE_STEPS.forEach((step, i) => {
        const el = document.getElementById(`step-${step.key}`);
        if (!el) return;
        el.className = 'step-item';
        if (i < currentStepIndex) el.classList.add('completed');
        else if (i === currentStepIndex) el.classList.add('active');
    });
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(pollStatus, 1500);  // Poll every 1.5s for snappier updates
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
        const elapsed = document.getElementById('progress-elapsed');

        // Update elapsed time
        if (data.elapsed_seconds) {
            const mins = Math.floor(data.elapsed_seconds / 60);
            const secs = Math.floor(data.elapsed_seconds % 60);
            elapsed.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }

        // Update log output and step tracker
        if (data.progress && data.progress.length > 0) {
            updateStepTracker(data.progress);

            const logOutput = document.getElementById('log-output');
            if (logOutput) {
                logOutput.innerHTML = data.progress
                    .map(line => {
                        // Color-code different types of log lines
                        let cls = 'log-line';
                        if (/error|fail|❌/i.test(line)) cls += ' log-error';
                        else if (/✅|complete|saved|done/i.test(line)) cls += ' log-success';
                        else if (/Processing region/i.test(line)) cls += ' log-highlight';
                        else if (/Running:|Starting/i.test(line)) cls += ' log-dim';
                        return `<div class="${cls}">${escapeHtml(line)}</div>`;
                    })
                    .join('');
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        }

        // Handle completion
        if (data.status === 'done') {
            stopPolling();
            panel.classList.add('progress-done');
            title.textContent = `Analysis Complete — ${data.city}`;

            // Mark all steps as completed
            PIPELINE_STEPS.forEach(step => {
                const el = document.getElementById(`step-${step.key}`);
                if (el) el.className = 'step-item completed';
            });

            const btn = document.getElementById('analyze-btn');
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'Run Analysis';

            // Reload dashboard data
            await Promise.all([loadReport(), loadStaticMaps()]);
            plotRegionsOnMap(reportData);

        } else if (data.status === 'error') {
            stopPolling();
            panel.classList.add('progress-error');
            title.textContent = `Analysis Failed — ${data.city}`;

            const btn = document.getElementById('analyze-btn');
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'Run Analysis';

        } else if (data.status === 'idle') {
            stopPolling();
            panel.style.display = 'none';

            const btn = document.getElementById('analyze-btn');
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'Run Analysis';
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

        // Plot on interactive map if available
        if (leafletMap) plotRegionsOnMap(reportData);
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

    const countEl = document.getElementById('table-count');
    if (countEl) countEl.textContent = `Showing ${filtered.length} of ${reportData.length}`;

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg><p>No matching regions.</p></td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(r => `
        <tr>
            <td class="mono"><strong>#${r.region_id}</strong></td>
            <td class="mono">${r.size_px ? r.size_px.join(' × ') : '—'}</td>
            <td class="mono">${(r.vegetation_percent * 100).toFixed(1)}%</td>
            <td class="mono">${formatArea(r.change_area_m2)}</td>
            <td>
                <span class="status-indicator ${r.violation ? 'violation' : 'compliant'}">
                    <span class="status-dot"></span>
                    ${r.violation ? 'Non-Compliant' : 'Compliant'}
                </span>
            </td>
        </tr>
    `).join('');
}

// ─── Interactive Leaflet Map ────────────────────────────
let leafletMap = null;
let mapMarkers = [];

function initInteractiveMap() {
    const container = document.getElementById('interactive-map');
    if (!container || leafletMap) return;

    // Initialize map centered at (0, 0); we'll fitBounds after data loads
    leafletMap = L.map('interactive-map', {
        center: [20, 0],
        zoom: 3,
        zoomControl: true,
        attributionControl: true,
    });

    // Dark-themed tile layer (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
    }).addTo(leafletMap);
}

function plotRegionsOnMap(regions) {
    if (!leafletMap || !regions || regions.length === 0) return;

    // Clear existing markers/layers
    mapMarkers.forEach(m => leafletMap.removeLayer(m));
    mapMarkers = [];

    const boundsGroup = L.featureGroup();

    regions.forEach(r => {
        // Skip regions without coordinates
        if (r.lat === undefined || r.lon === undefined) return;

        const isViolation = r.violation;
        const fillColor = isViolation ? '#ef4444' : '#22c55e';
        const borderColor = '#3b82f6';

        // Draw polygon boundary if available
        if (r.polygon && r.polygon.length > 2) {
            const poly = L.polygon(r.polygon, {
                color: borderColor,
                weight: 2,
                opacity: 0.7,
                fillColor: fillColor,
                fillOpacity: 0.12,
            }).addTo(leafletMap);
            mapMarkers.push(poly);
            boundsGroup.addLayer(poly);
        }

        // Circle marker at centroid
        const marker = L.circleMarker([r.lat, r.lon], {
            radius: 8,
            fillColor: fillColor,
            color: '#fff',
            weight: 1.5,
            opacity: 1,
            fillOpacity: 0.85,
        }).addTo(leafletMap);

        // Build popup HTML
        const popupHtml = `
            <div class="popup-header">Region #${r.region_id}</div>
            <div class="popup-row">
                <span class="popup-label">Vegetation</span>
                <span class="popup-value">${(r.vegetation_percent * 100).toFixed(1)}%</span>
            </div>
            <div class="popup-row">
                <span class="popup-label">Change Area</span>
                <span class="popup-value">${formatArea(r.change_area_m2)}</span>
            </div>
            <div class="popup-row">
                <span class="popup-label">Dimensions</span>
                <span class="popup-value">${r.size_px ? r.size_px.join(' × ') + ' px' : '—'}</span>
            </div>
            <div class="popup-row">
                <span class="popup-label">Coordinates</span>
                <span class="popup-value">${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</span>
            </div>
            <div class="popup-status ${isViolation ? 'violation' : 'compliant'}">
                ${isViolation ? 'NON-COMPLIANT' : 'COMPLIANT'}
            </div>
        `;

        marker.bindPopup(popupHtml, { maxWidth: 280, minWidth: 220 });
        mapMarkers.push(marker);
        boundsGroup.addLayer(marker);
    });

    // Auto-zoom to fit all markers/polygons
    if (boundsGroup.getLayers().length > 0) {
        leafletMap.fitBounds(boundsGroup.getBounds().pad(0.15));
    }
}

// ─── Static Map Gallery ─────────────────────────────────
async function loadStaticMaps() {
    const gallery = document.getElementById('map-gallery');
    try {
        const res = await fetch(`${API}/api/maps`);
        const maps = await res.json();

        if (!maps.length) {
            gallery.innerHTML = '<div class="empty-state"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg><p>No maps generated yet. Run an analysis above.</p></div>';
            return;
        }

        gallery.innerHTML = maps.map(m => `
            <div class="map-card">
                <img src="${API}/outputs/maps/${m}" alt="${m}" loading="lazy">
                <div class="map-info">${m.replace(/_/g, ' ').replace('.png', '')}</div>
            </div>
        `).join('');
    } catch {
        gallery.innerHTML = '<div class="empty-state"><p>Could not load map images.</p></div>';
    }
}

// ─── Training Metrics ───────────────────────────────────
async function loadMetrics() {
    const grid = document.getElementById('metrics-grid');
    try {
        const res = await fetch(`${API}/api/metrics`);
        const metrics = await res.json();

        if (!metrics || Object.keys(metrics).length === 0) {
            grid.innerHTML = '<div class="empty-state"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg><p>No training metrics found. Train models first.</p></div>';
            return;
        }

        grid.innerHTML = '';

        for (const [modelName, history] of Object.entries(metrics)) {
            const prettyName = modelName.replace(/_/g, ' ').replace('history', '').trim();

            if (history.loss) {
                const card = createMetricsCard(`${capitalize(prettyName)} — Loss`);
                grid.appendChild(card.el);
                createChart(card.canvas, history, 'loss', 'val_loss', 'Loss', '#ef4444', '#f59e0b');
            }

            if (history.accuracy) {
                const card = createMetricsCard(`${capitalize(prettyName)} — Accuracy`);
                grid.appendChild(card.el);
                createChart(card.canvas, history, 'accuracy', 'val_accuracy', 'Accuracy', '#22c55e', '#3b82f6');
            }

            if (history.iou_metric) {
                const card = createMetricsCard(`${capitalize(prettyName)} — IoU`);
                grid.appendChild(card.el);
                createChart(card.canvas, history, 'iou_metric', 'val_iou_metric', 'IoU', '#8b5cf6', '#14b8a6');
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
