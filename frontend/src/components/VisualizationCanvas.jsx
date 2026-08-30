import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Globe, SlidersHorizontal } from 'lucide-react';
import Plot from 'react-plotly.js';
import LoadingScreen from './LoadingScreen';
import { computeColorScale } from '../colorScale.js';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-50 rounded-xl border border-red-200 p-6 text-center">
          <h2 className="text-red-800 font-bold mb-2 font-display text-lg">Visualization Render Error</h2>
          <p className="text-red-600 text-sm max-w-md font-mono">{this.state.error?.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Geographic bounds for the expanded prediction grid ──────────
const LON_MIN = 45.0;
const LON_MAX = 100.0;
const LAT_MIN = -10.0;
const LAT_MAX = 30.0;

function IdleState() {
  return (
    <motion.div
      className="absolute inset-0 flex flex-col items-center justify-center wireframe-grid rounded-xl"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Globe icon */}
      <motion.div
        className="animate-pulse-soft mb-6"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <div className="relative p-6 rounded-full bg-[var(--color-paper-surface)] border border-[var(--color-paper-border)] shadow-sm">
          <Globe className="w-12 h-12 text-[var(--color-ink-light)] relative z-10" strokeWidth={1.5} />
        </div>
      </motion.div>

      <h3 className="text-[var(--color-ink-dark)] text-xl font-bold font-display mb-1">
        Awaiting Scan Initialization
      </h3>
      <p className="text-[var(--color-ink-light)] text-sm font-medium">
        Configure parameters and launch AI prediction
      </p>

      {/* Corner decorations */}
      {['top-6 left-6', 'top-6 right-6', 'bottom-6 left-6', 'bottom-6 right-6'].map(
        (pos, i) => (
          <div
            key={i}
            className={`absolute ${pos} w-4 h-4 border-[var(--color-ink-light)]/40 ${
              i === 0
                ? 'border-t-2 border-l-2'
                : i === 1
                ? 'border-t-2 border-r-2'
                : i === 2
                ? 'border-b-2 border-l-2'
                : 'border-b-2 border-r-2'
            }`}
          />
        )
      )}
    </motion.div>
  );
}


function ActiveState({ plotData, depthIndex, autoContrast }) {
  // 1. Grid preparation & data-driven color scale
  const { scaledZ, lonArray, latArray, finalZmin, finalZmax } = useMemo(() => {
    if (!plotData || !plotData.length || !plotData[0].length) return {};

    const numLat = plotData.length;
    const numLon = plotData[0].length;

    const latArray = Array.from({ length: numLat }, (_, i) => LAT_MIN + (i / (numLat - 1)) * (LAT_MAX - LAT_MIN));
    const lonArray = Array.from({ length: numLon }, (_, i) => LON_MIN + (i / (numLon - 1)) * (LON_MAX - LON_MIN));

    const scaledZ = plotData;

    // Compute scale using the tested, robust function
    const scale = computeColorScale(plotData, autoContrast
      ? { pLow: 0.02, pHigh: 0.98 }   // Auto-Contrast: tight 2nd-98th pctl
      : { pLow: 0.005, pHigh: 0.995 }  // Global: wider 0.5th-99.5th pctl
    );

    return {
      scaledZ,
      lonArray,
      latArray,
      finalZmin: scale.vmin,
      finalZmax: scale.vmax,
    };
  }, [plotData, depthIndex, autoContrast]);

  if (!scaledZ) return null;
  
  const CMOCEAN_THERMAL = [
    [0.0, '#042333'],
    [0.1, '#1b3472'],
    [0.2, '#3f3183'],
    [0.3, '#662479'],
    [0.4, '#8a1761'],
    [0.5, '#ab1a45'],
    [0.6, '#c6312a'],
    [0.7, '#da561a'],
    [0.8, '#e8841a'],
    [0.9, '#f1b72a'],
    [1.0, '#eff542']
  ];

  return (
    <motion.div
      className="absolute inset-0 rounded-xl overflow-hidden shadow-sm border border-[var(--color-paper-border)] bg-[var(--color-paper-surface)]"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <Plot
        data={[
          {
            // Dummy trace to force Plotly to render the geo layout map
            type: 'scattergeo',
            lat: [null],
            lon: [null],
            showlegend: false,
            hoverinfo: 'none',
          },
          {
            // 2. High-Fidelity Rendering with Dynamic Contrast Stretching
            type: 'heatmap',
            z: scaledZ,
            x: lonArray,
            y: latArray,
            zsmooth: 'best',
            colorscale: CMOCEAN_THERMAL,
            zmin: finalZmin,
            zmax: finalZmax,
            opacity: 1,
            colorbar: {
              title: {
                text: 'Potential Temp (°C)',
                font: { color: '#4A3428', size: 12, family: 'Inter', weight: 'bold' },
                side: 'right',
              },
              tickfont: { color: '#8B8379', size: 11, family: 'Inter', weight: '500' },
              thickness: 16,
              len: 0.8,
              outlinewidth: 1,
              outlinecolor: '#DDD5C8',
              bgcolor: 'rgba(251,249,244,0.8)',
              borderwidth: 0,
              tickcolor: '#8B8379',
              xpad: 15,
              dtick: undefined, // Let Plotly auto-compute tick spacing
              showticksuffix: 'last',
              ticksuffix: '°C'
            },
            hovertemplate:
              'Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Temp: %{z:.2f}°C<extra></extra>',
          },
        ]}
        layout={{
          datarevision: depthIndex + (autoContrast ? '-auto' : '-global'),
          // 3. Edge-to-Edge Visuals (Layout Settings)
          autosize: true,
          margin: { l: 0, r: 0, t: 0, b: 0 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'Inter', color: '#4A3428' },
          
          // Cartesian axes for the heatmap overlay (perfectly aligned with geo)
          xaxis: {
            range: [LON_MIN, LON_MAX],
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            visible: false,
          },
          yaxis: {
            range: [LAT_MIN, LAT_MAX],
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            visible: false,
            scaleanchor: 'x', // Force Cartesian grid to maintain equirectangular aspect ratio
          },
          
          // 4. Geographical Framing & Theme (Geo Settings)
          geo: {
            projection: { type: 'equirectangular' },
            lonaxis: { range: [LON_MIN, LON_MAX] },
            lataxis: { range: [LAT_MIN, LAT_MAX] },
            showcoastlines: true,
            coastlinecolor: 'rgba(0,229,255,0.4)',
            coastlinewidth: 1,
            showland: true,
            landcolor: '#E8E2D5', 
            showocean: true,
            oceancolor: 'transparent',
            bgcolor: 'transparent',
            showframe: false,
            resolution: 50,
          },
        }}
        config={{
          displayModeBar: false,
          responsive: true,
          scrollZoom: false,
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />
    </motion.div>
  );
}

export default function VisualizationCanvas({ status, plotData, depthIndex }) {
  const [autoContrast, setAutoContrast] = useState(true);

  return (
    <motion.div
      className="relative w-full h-full min-h-[500px]"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <ErrorBoundary>
        <AnimatePresence mode="wait">
          {status === 'idle' && <IdleState key="idle" />}
          {status === 'loading' && <LoadingScreen key="loading" />}
          {status === 'active' && <ActiveState key="active" plotData={plotData} depthIndex={depthIndex} autoContrast={autoContrast} />}
        </AnimatePresence>

        {/* Auto-Contrast Toggle Overlay (Temporarily Removed) */}
      </ErrorBoundary>
    </motion.div>
  );
}
