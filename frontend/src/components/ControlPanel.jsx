import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Calendar, Anchor, Activity, Loader2, Crosshair } from 'lucide-react';
import { getDepthIndex } from '../App';

export default function ControlPanel({ date, setDate, depth, depthIndex, setDepthIndex, modelDepths, onScan, loading, viewMode, setViewMode, hasInputs }) {

  return (
    <div className="flex flex-col gap-5 h-full">
      {/* Title */}
      <div className="flex items-center gap-2 border-b border-[var(--color-paper-border)] pb-3">
        <Crosshair className="w-3.5 h-3.5 text-[var(--color-ink-medium)]" />
        <h2 className="text-[10px] font-bold text-[var(--color-ink-dark)] tracking-widest uppercase font-sans">
          Mission Parameters
        </h2>
      </div>

      {/* Date */}
      <div className="flex flex-col gap-1.5">
        <label className="flex items-center gap-2 text-[9px] font-bold text-[var(--color-ink-light)] uppercase tracking-widest">
          <Calendar className="w-3 h-3" />
          Observation Date
        </label>
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="w-full px-3 py-2 rounded bg-[var(--color-paper-bg)] border border-[var(--color-paper-border)] text-[var(--color-ink-dark)] text-sm font-medium font-sans focus:outline-none focus:border-[var(--color-coffee-main)] focus:ring-1 focus:ring-[var(--color-coffee-main)]/20 shadow-inner"
        />
      </div>

      {/* Depth */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-2 text-[9px] font-bold text-[var(--color-ink-light)] uppercase tracking-widest">
            <Anchor className="w-3 h-3" />
            Target Depth
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-lg font-bold tabular-nums font-display text-[var(--color-ink-dark)]">
              {Number(depth).toFixed(1).replace(/\.0$/, '')}
            </span>
            <span className="text-[10px] font-normal text-[var(--color-ink-light)]">m</span>
            <span className="text-[9px] font-bold text-[var(--color-coffee-main)] font-mono ml-1">
              L{depthIndex}
            </span>
          </div>
        </div>

        <div className="relative px-1 pb-2 mt-1">
          <input
            type="range"
            min={0}
            max={modelDepths.length ? modelDepths.length - 1 : 14}
            step="1"
            value={depthIndex}
            onChange={(e) => setDepthIndex(parseInt(e.target.value))}
            className="w-full"
            disabled={viewMode !== 'prediction'}
          />
          {/* Ticks */}
          <div className="flex justify-between mt-2 px-0.5">
             <span className="text-[9px] text-[var(--color-ink-light)] font-mono">{Math.round(modelDepths[0] || 0)}m</span>
             <span className="text-[9px] text-[var(--color-ink-light)] font-mono">{Math.round(modelDepths[modelDepths.length - 1] || 1000)}m</span>
          </div>
        </div>
      </div>

      {/* Variable / View Mode */}
      <div className="flex flex-col gap-1.5 mt-2">
        <label className="flex items-center gap-2 text-[9px] font-bold text-[var(--color-ink-light)] uppercase tracking-widest">
          <Activity className="w-3 h-3" />
          Active Layer Variable
        </label>
        <select
          value={viewMode}
          onChange={(e) => setViewMode(e.target.value)}
          className="w-full px-3 py-2 rounded bg-[var(--color-paper-bg)] border border-[var(--color-paper-border)] text-[var(--color-ink-dark)] text-sm font-medium font-sans focus:outline-none focus:border-[var(--color-coffee-main)] shadow-inner"
        >
          <option value="prediction">Model Output (Temperature)</option>
          {hasInputs && (
            <optgroup label="Satellite Inputs">
              <option value="sst">Sea Surface Temperature (SST)</option>
              <option value="sss">Sea Surface Salinity (SSS)</option>
              <option value="ssh">Sea Surface Height (SSH)</option>
              <option value="u_cur">Zonal Current (U)</option>
              <option value="v_cur">Meridional Current (V)</option>
              <option value="u_wind">Zonal Wind (U)</option>
              <option value="v_wind">Meridional Wind (V)</option>
            </optgroup>
          )}
        </select>
        {!hasInputs && viewMode === 'prediction' && (
          <p className="text-[9px] text-[var(--color-ink-light)] mt-1">Run scan to view inputs</p>
        )}
      </div>

      {/* Scan Button */}
      <div className="mt-auto pt-4 border-t border-[var(--color-paper-border)]">
        <motion.button
          onClick={onScan}
          disabled={loading}
          className="w-full py-3 rounded-lg font-bold text-xs tracking-widest uppercase font-sans bg-[var(--color-coffee-main)] text-white border border-[var(--color-coffee-hover)] shadow-md disabled:opacity-70 disabled:cursor-not-allowed transition-colors hover:bg-[var(--color-coffee-hover)] relative overflow-hidden"
          whileHover={!loading ? { scale: 1.02 } : {}}
          whileTap={!loading ? { scale: 0.98 } : {}}
        >
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.span
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center gap-2"
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> PROCESSING
              </motion.span>
            ) : (
              <motion.span
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center gap-2"
              >
                <Activity className="w-3.5 h-3.5" /> INITIALIZE SCAN
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      </div>
    </div>
  );
}
