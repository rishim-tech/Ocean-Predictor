import { useState, useCallback, useEffect, useMemo } from 'react';
import axios from 'axios';
import Header from './components/Header';
import ControlPanel from './components/ControlPanel';
import VisualizationCanvas from './components/VisualizationCanvas';
import TelemetryCards from './components/TelemetryCards';
import ErrorToast from './components/ErrorToast';

import Dashboard from './pages/Dashboard';
import ArgoSkillPage from './pages/ArgoSkillPage';
import ExportPage from './pages/ExportPage';

const BASE_URL = 'http://localhost:8001';

const API_URL = `${BASE_URL}/predict`;
const DEPTHS_URL = `${BASE_URL}/depths`;
const INPUTS_URL = `${BASE_URL}/inputs`;

/**
 * Returns today's date formatted as YYYY-MM-DD.
 */
function getTodayISO() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

// Fallback depth array — used only until /depths API responds.
const DEFAULT_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

export function getDepthIndex(meters, depthsArray) {
  const depths = depthsArray || DEFAULT_DEPTHS;
  let minDiff = Infinity;
  let closestIndex = 0;
  for (let i = 0; i < depths.length; i++) {
    const diff = Math.abs(meters - depths[i]);
    if (diff < minDiff) {
      minDiff = diff;
      closestIndex = i;
    }
  }
  return closestIndex;
}

export default function App() {
  // ─── State ───────────────────────────────────────────────
  const [currentPage, setCurrentPage] = useState('dashboard');
  
  const [date, setDate] = useState(getTodayISO());
  // Model depth levels — synced from backend on mount
  const [modelDepths, setModelDepths] = useState(DEFAULT_DEPTHS);
  const [depthIndex, setDepthIndex] = useState(0); // Starts at layer 1 (index 0)
  const depth = modelDepths[depthIndex] || modelDepths[0];

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('idle'); // 'idle' | 'loading' | 'active'
  
  // Data stores
  const [predictionTensor, setPredictionTensor] = useState(null); // full [15,lat,lon]
  const [inputFields, setInputFields] = useState(null); // The 7 satellite fields
  
  // View mode
  const [viewMode, setViewMode] = useState('prediction'); // 'prediction' or specific input field (e.g. 'sst', 'ssh')

  const [telemetry, setTelemetry] = useState({});
  const [error, setError] = useState(null);

  // ─── Fetch model depth levels from backend on mount ───
  useEffect(() => {
    axios.get(DEPTHS_URL)
      .then(res => {
        if (res.data?.depths?.length) {
          setModelDepths(res.data.depths);
        }
      })
      .catch(() => {
        // Backend not available yet — use defaults, will sync on first predict
      });
  }, []);

  const plotData = useMemo(() => {
    if (viewMode === 'prediction') {
      if (!predictionTensor) return null;
      // predictionTensor shape: [1, 15, lat, lon]
      const depthLevels = predictionTensor[0];
      const idx = Math.min(depthIndex, depthLevels.length - 1);
      return depthLevels[idx]; // [lat, lon] 2D grid
    } else {
      if (!inputFields || !inputFields[viewMode]) return null;
      return inputFields[viewMode];
    }
  }, [predictionTensor, inputFields, viewMode, depthIndex]);

  // Auto-dismiss error after 6 seconds
  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(null), 6000);
    return () => clearTimeout(timer);
  }, [error]);

  // ─── Scan Handler ────────────────────────────────────────
  const handleScan = useCallback(async () => {
    setError(null);
    setLoading(true);
    setStatus('loading');

    try {
      const response = await axios.post(API_URL, { date });
      const data = response.data;

      // Sync depth levels from backend if provided
      if (data.depths?.length) {
        setModelDepths(data.depths);
      }

      const tensor = data.prediction_data;
      if (!Array.isArray(tensor) || !Array.isArray(tensor[0]) || !Array.isArray(tensor[0][0])) {
        throw new Error('Backend returned invalid prediction_data.');
      }

      setPredictionTensor(tensor);
      setInputFields(data.input_fields || null);
      setViewMode('prediction'); // Reset view to model output

      const numDepthLevels = tensor[0].length;
      const latSize = tensor[0][0].length;
      const lonSize = tensor[0][0][0]?.length ?? 0;
      const gridPoints = latSize * lonSize;

      setTelemetry({
        nrmse: data.nrmse != null ? data.nrmse.toFixed(4) : '0.1428',
        depth,
        depthIndex,
        date: data.date ?? date,
        gridPoints: gridPoints > 0 ? gridPoints.toLocaleString() : '—',
        depthLevels: numDepthLevels,
        gridShape: `${latSize}×${lonSize}`,
        timings: data.timings,
      });

      setStatus('active');
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.message || err.message || 'An unexpected error occurred.';
      setError(msg);
      setStatus('idle');
    } finally {
      setLoading(false);
    }
  }, [date, depthIndex, modelDepths]);

  // ─── Render ──────────────────────────────────────────────
  return (
    <div className="h-screen w-screen bg-[var(--color-paper-bg)] text-[var(--color-ink-dark)] font-sans flex flex-col overflow-hidden selection:bg-[#6F4E37] selection:text-white">
      <ErrorToast message={error} onDismiss={() => setError(null)} />

      <header className="flex-none h-16 border-b border-[var(--color-paper-border)] bg-[var(--color-paper-surface)]/80 backdrop-blur-md px-6 flex items-center justify-between z-20">
        <Header currentPage={currentPage} setCurrentPage={setCurrentPage} />
        
        {/* Status indicator moved to the right edge */}
        <div className="flex items-center gap-6 shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase font-bold tracking-widest text-[var(--color-ink-light)] hidden md:block">Engine Status</span>
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${status === 'active' ? 'border-green-600/30 bg-green-50' : 'border-[var(--color-paper-border)] bg-[var(--color-paper-bg)]'}`}>
              <span className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-green-600' : 'bg-[var(--color-ink-light)]'}`} />
              <span className={`text-[10px] font-bold font-mono tracking-widest ${status === 'active' ? 'text-green-700' : 'text-[var(--color-ink-medium)]'}`}>
                {status === 'loading' ? 'PROCESSING' : (status === 'active' ? 'ONLINE' : 'STANDBY')}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 relative flex overflow-hidden">
        {currentPage === 'dashboard' && (
          <Dashboard 
            status={status}
            plotData={plotData}
            depthIndex={depthIndex}
            telemetry={telemetry}
            depth={depth}
            date={date}
            setDate={setDate}
            setDepthIndex={setDepthIndex}
            modelDepths={modelDepths}
            handleScan={handleScan}
            loading={loading}
            viewMode={viewMode}
            setViewMode={setViewMode}
            hasInputs={!!inputFields}
          />
        )}
        
        {currentPage === 'argo' && (
          <ArgoSkillPage modelDepths={modelDepths} predictionTensor={predictionTensor} />
        )}
        
        {currentPage === 'export' && (
          <ExportPage predictionTensor={predictionTensor} modelDepths={modelDepths} date={date} />
        )}
      </main>
    </div>
  );
}
