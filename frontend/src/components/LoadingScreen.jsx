import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const STATUS_MESSAGES = [
  'Initiating secure handshake with Copernicus Marine API...',
  'Querying Spatio-Temporal NetCDF Telemetry...',
  'Extracting Surface Variables: SST, SSS, SSH...',
  'Parsing geospatial matrices & applying normalizations...',
  'Executing Deep CNN forward pass...',
  'Extrapolating Subsurface 3D Volumetric Thermal Profiles...',
  'Decoding Neural Output to geographic coordinates...',
  'Rendering Geospatial Heatmap...',
];

const TOTAL_DURATION = 40; // seconds
const MESSAGE_INTERVAL = 5; // seconds per message

export default function LoadingScreen() {
  const [messageIndex, setMessageIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const startTimeRef = useRef(Date.now());

  // Progress bar: fills over TOTAL_DURATION seconds (indeterminate/estimated)
  useEffect(() => {
    const interval = setInterval(() => {
      const dt = (Date.now() - startTimeRef.current) / 1000;
      setProgress(Math.min((dt / TOTAL_DURATION) * 100, 95)); // caps at 95% until complete
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Cycle through status messages
  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) =>
        prev < STATUS_MESSAGES.length - 1 ? prev + 1 : prev
      );
    }, MESSAGE_INTERVAL * 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-paper-surface)]/80 backdrop-blur-md rounded-2xl overflow-hidden z-20"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* ── Subtle Background Grid ── */}
      <div className="absolute inset-0 wireframe-grid opacity-50" />

      {/* ── Container ── */}
      <div className="relative z-10 flex flex-col items-center w-full max-w-lg px-8">
        
        {/* ── Marine Radar (Quiet, Sophisticated) ── */}
        <div className="relative w-40 h-40 mb-10 flex items-center justify-center opacity-80">
          {/* Base radar circle */}
          <div className="absolute inset-0 rounded-full border border-[var(--color-paper-border)] bg-[var(--color-paper-bg)]/50 shadow-inner" />
          
          {/* Concentric rings */}
          {[1, 2, 3].map((ring) => (
            <div
              key={ring}
              className="absolute rounded-full border border-[var(--color-ink-light)]/20"
              style={{ inset: `${ring * 12}px` }}
            />
          ))}

          {/* Crosshairs */}
          <div className="absolute w-full h-px bg-[var(--color-ink-light)]/20" />
          <div className="absolute h-full w-px bg-[var(--color-ink-light)]/20" />

          {/* Radar Sweep */}
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{
              background: 'conic-gradient(from 0deg, transparent 0deg, var(--color-coffee-main) 30deg, transparent 60deg)',
              opacity: 0.15
            }}
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
          />

          {/* Center blip */}
          <motion.div 
            className="absolute w-2 h-2 rounded-full bg-[var(--color-coffee-main)]"
            animate={{ scale: [1, 1.5, 1], opacity: [0.8, 1, 0.8] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
        </div>

        {/* ── Status Messages (Editorial style) ── */}
        <div className="w-full bg-[var(--color-paper-bg)] border border-[var(--color-paper-border)] p-6 rounded-xl shadow-sm mb-6">
          <div className="flex items-center gap-2 mb-4 border-b border-[var(--color-paper-border)] pb-2">
            <span className="w-2 h-2 rounded-full bg-[var(--color-coffee-main)] animate-pulse" />
            <span className="text-[10px] font-bold text-[var(--color-ink-medium)] uppercase tracking-widest">
              Processing Pipeline
            </span>
          </div>

          <div className="min-h-[60px] flex flex-col justify-center">
            <AnimatePresence mode="wait">
              <motion.div
                key={messageIndex}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.3 }}
                className="flex items-start gap-3"
              >
                <span className="text-[var(--color-ink-light)] font-mono text-xs mt-0.5">›</span>
                <p className="font-mono text-xs leading-relaxed text-[var(--color-ink-dark)] font-medium">
                  {STATUS_MESSAGES[messageIndex]}
                </p>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* ── Indeterminate Loading Bar ── */}
        <div className="w-full">
          <div className="h-1.5 w-full bg-[var(--color-paper-border)] rounded-full overflow-hidden relative">
            {/* Base estimated fill */}
            <motion.div 
              className="absolute top-0 left-0 h-full bg-[var(--color-ink-light)]/30 rounded-full"
              style={{ width: `${progress}%` }}
              transition={{ duration: 0.1 }}
            />
            {/* Shimmer sweep */}
            <motion.div
              className="absolute top-0 h-full w-1/3 bg-[var(--color-coffee-main)] rounded-full"
              animate={{ left: ['-100%', '200%'] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            />
          </div>
          <div className="flex justify-between mt-2 px-1">
            <span className="text-[9px] font-mono font-bold text-[var(--color-ink-light)] tracking-widest uppercase">
              Model Inference Active
            </span>
            <span className="text-[9px] font-mono text-[var(--color-ink-light)]">
              EST. WAIT ~45s
            </span>
          </div>
        </div>
        
      </div>
    </motion.div>
  );
}
