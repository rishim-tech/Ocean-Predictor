import { motion } from 'framer-motion';
import { Compass } from 'lucide-react';

export default function Header() {
  return (
    <header className="flex items-center gap-4">
      <motion.div
        className="flex items-center justify-center w-10 h-10 rounded bg-[var(--color-coffee-main)] shadow-sm"
        animate={{ rotate: [0, 2, -2, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
      >
        <Compass className="w-6 h-6 text-white" />
      </motion.div>
      <div className="flex flex-col justify-center">
        <h1 className="text-xl font-bold tracking-tight text-[var(--color-ink-dark)] font-display leading-none">
          Ocean Predictor
        </h1>
        <span className="text-[10px] text-[var(--color-ink-light)] font-bold tracking-[0.15em] uppercase leading-none mt-1.5">
          Subsurface Thermal Analysis
        </span>
      </div>
    </header>
  );
}
