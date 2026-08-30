import { motion } from 'framer-motion';

function TelemetryItem({ label, value, subtext }) {
  return (
    <div className="bg-[var(--color-paper-surface)] py-2 px-4 flex-1 min-w-[100px] flex flex-col justify-center border-r border-[var(--color-paper-border)] last:border-r-0">
      <span className="text-[8px] font-bold uppercase tracking-widest text-[var(--color-ink-light)] mb-0.5">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className="text-sm font-bold font-display text-[var(--color-ink-dark)]">{value}</span>
        {subtext && <span className="text-[9px] font-mono text-[var(--color-ink-light)]">{subtext}</span>}
      </div>
    </div>
  );
}

export default function TelemetryCards({ telemetry, status }) {
  if (status !== 'active' && !telemetry.date) return null;

  return (
    <motion.div 
      className="bg-[var(--color-paper-surface)] border-[var(--color-paper-border)]"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex flex-wrap overflow-hidden">
        <TelemetryItem label="Depth" value={telemetry.depth != null ? telemetry.depth.toFixed(1) : '—'} subtext={telemetry.depth != null ? 'm' : null} />
        <TelemetryItem label="Layer" value={telemetry.depthIndex != null ? `L${telemetry.depthIndex}` : '—'} />
        <TelemetryItem label="NRMSE" value={telemetry.nrmse ?? '—'} />
        <TelemetryItem label="Date" value={telemetry.date ?? '—'} subtext="UTC" />
        <TelemetryItem label="Grid" value={telemetry.gridShape ?? '—'} subtext={telemetry.gridPoints ? `${telemetry.gridPoints} pts` : null} />
      </div>
    </motion.div>
  );
}
