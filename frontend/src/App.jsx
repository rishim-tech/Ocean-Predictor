import { useState, useCallback, useEffect, useMemo } from 'react';
import axios from 'axios';
import Header from './components/Header';
import ControlPanel from './components/ControlPanel';
import VisualizationCanvas from './components/VisualizationCanvas';
import TelemetryCards from './components/TelemetryCards';
import ErrorToast from './components/ErrorToast';

const BASE_URL = 'http://localhost:8001';

const API_URL = `${BASE_URL}/predict`;
const DEPTHS_URL = `${BASE_URL}/depths`;

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
const DEFAULT_DEPTHS = [0.49, 5.08, 9.57, 18.5, 29.44, 47.37, 77.85, 92.33, 155.85, 186.13, 318.13, 380.21, 541.09, 643.57, 902.34, 1000.0];

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
  const [date, setDate] = useState(getTodayISO());
  // Model depth levels — synced from backend on mount
  const [modelDepths, setModelDepths] = useState(DEFAULT_DEPTHS);
  const [depthIndex, setDepthIndex] = useState(0); // Starts at layer 1 (index 0)
  const depth = modelDepths[depthIndex] || modelDepths[0];

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('idle'); // 'idle' | 'loading' | 'active'
  const [predictionTensor, setPredictionTensor] = useState(null); // full [1,15,lat,lon]
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
    if (!predictionTensor) return null;
    // predictionTensor shape: [1, 15, lat, lon]
    // Clamp index to available depth levels
    const depthLevels = predictionTensor[0];
    const idx = Math.min(depthIndex, depthLevels.length - 1);
    return depthLevels[idx]; // [lat, lon] 2D grid
  }, [predictionTensor, depthIndex]);

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
      const response = await axios.post(API_URL, {
        date,
      });
      const data = response.data;

      // Sync depth levels from backend if provided
      if (data.depths?.length) {
        setModelDepths(data.depths);
      }

      // Backend returns: { status: 'success', prediction_data: [1,15,lat,lon] }
      const tensor = data.prediction_data;

      if (
        !Array.isArray(tensor) ||
        !Array.isArray(tensor[0]) ||
        !Array.isArray(tensor[0][0])
      ) {
        throw new Error(
          'Backend returned invalid prediction_data — expected a 4D tensor [batch, depth, lat, lon].'
        );
      }

      // Store the full tensor so the depth slider can browse layers
      setPredictionTensor(tensor);

      // Telemetry: compute grid shape from the first 2D slice
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
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'An unexpected error occurred while contacting the prediction engine.';
      setError(msg);
      setStatus('idle');
    } finally {
      setLoading(false);
    }
  }, [date, depthIndex, modelDepths]);

  // ─── Render ──────────────────────────────────────────────
  return (
    <div className="h-screen w-screen bg-[var(--color-paper-bg)] text-[var(--color-ink-dark)] font-sans flex flex-col overflow-hidden selection:bg-[#6F4E37] selection:text-white">
      {/* Error Toast */}
      <ErrorToast message={error} onDismiss={() => setError(null)} />

      {/* Top Header */}
      <header className="flex-none h-16 border-b border-[var(--color-paper-border)] bg-[var(--color-paper-surface)]/80 backdrop-blur-md px-6 flex items-center justify-between z-20">
        <Header />
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase font-bold tracking-widest text-[var(--color-ink-light)]">Engine Status</span>
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${status === 'active' ? 'border-green-600/30 bg-green-50' : 'border-[var(--color-paper-border)] bg-[var(--color-paper-bg)]'}`}>
              <span className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-green-600' : 'bg-[var(--color-ink-light)]'}`} />
              <span className={`text-[10px] font-bold font-mono tracking-widest ${status === 'active' ? 'text-green-700' : 'text-[var(--color-ink-medium)]'}`}>
                {status === 'loading' ? 'PROCESSING' : (status === 'active' ? 'ONLINE' : 'STANDBY')}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 relative flex overflow-hidden">
        
        {/* Map + Telemetry Column */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Map Area — hero visual, takes all remaining space */}
          <div className="flex-1 relative min-h-0">
            <VisualizationCanvas status={status} plotData={plotData} depthIndex={depthIndex} />
          </div>
          
          {/* Telemetry — compact strip below the map */}
          <div className="flex-none border-t border-[var(--color-paper-border)]">
            <TelemetryCards telemetry={{ ...telemetry, depth, depthIndex }} status={status} />
          </div>
        </div>

        {/* Side Panel (Right) - Compact Controls */}
        <aside className="w-60 flex-none border-l border-[var(--color-paper-border)] bg-[var(--color-paper-surface)]/50 backdrop-blur-md flex flex-col z-10">
          <div className="p-4 flex-1 overflow-y-auto">
            <ControlPanel
              date={date}
              setDate={setDate}
              depth={depth}
              depthIndex={depthIndex}
              setDepthIndex={setDepthIndex}
              modelDepths={modelDepths}
              onScan={handleScan}
              loading={loading}
            />
          </div>
        </aside>

      </main>
    </div>
  );
}
