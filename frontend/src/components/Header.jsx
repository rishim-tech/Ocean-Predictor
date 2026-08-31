import { motion } from 'framer-motion';
import { Compass, BarChart2, Download, LayoutDashboard } from 'lucide-react';

export default function Header({ currentPage, setCurrentPage }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'argo', label: 'ARGO Validation', icon: BarChart2 },
    { id: 'export', label: 'Data Export', icon: Download },
  ];

  return (
    <header className="flex items-center gap-12 w-full">
      <div className="flex items-center gap-4">
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
      </div>

      <nav className="flex items-center gap-2">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setCurrentPage(item.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded font-semibold text-sm transition-colors ${
              currentPage === item.id
                ? 'bg-[var(--color-coffee-main)] text-white shadow-sm'
                : 'text-[var(--color-ink-medium)] hover:bg-[var(--color-paper-border)] hover:text-[var(--color-ink-dark)]'
            }`}
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
