import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, X } from 'lucide-react';

export default function ErrorToast({ message, onDismiss }) {
  return (
    <AnimatePresence>
      {message && (
        <motion.div
          className="fixed top-6 left-1/2 z-[100] w-full max-w-lg px-4"
          initial={{ opacity: 0, y: -40, x: '-50%' }}
          animate={{ opacity: 1, y: 0, x: '-50%' }}
          exit={{ opacity: 0, y: -40, x: '-50%' }}
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        >
          <div className="glass-panel-strong flex items-start gap-3 px-5 py-4
                          border border-red-500/20 shadow-[0_0_30px_rgba(239,68,68,0.15)]">
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-red-500/10 shrink-0 mt-0.5">
              <AlertTriangle className="w-5 h-5 text-red-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-red-300 mb-0.5">
                Prediction Failed
              </p>
              <p className="text-xs text-slate-400 leading-relaxed">
                {message}
              </p>
            </div>
            <button
              onClick={onDismiss}
              className="p-1 rounded-lg hover:bg-slate-700/50 transition-colors text-slate-500 hover:text-slate-300 shrink-0 cursor-pointer"
              aria-label="Dismiss error"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
