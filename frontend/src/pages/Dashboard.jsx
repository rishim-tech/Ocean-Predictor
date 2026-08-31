import React from 'react';
import VisualizationCanvas from '../components/VisualizationCanvas';
import TelemetryCards from '../components/TelemetryCards';
import ControlPanel from '../components/ControlPanel';

export default function Dashboard({
  status,
  plotData,
  depthIndex,
  telemetry,
  depth,
  date,
  setDate,
  setDepthIndex,
  modelDepths,
  handleScan,
  loading,
  viewMode,
  setViewMode,
  hasInputs
}) {
  return (
    <div className="flex-1 relative flex overflow-hidden w-full">
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
            viewMode={viewMode}
            setViewMode={setViewMode}
            hasInputs={hasInputs}
          />
        </div>
      </aside>
    </div>
  );
}
