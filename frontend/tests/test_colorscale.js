import { computeColorScale } from '../src/colorScale.js';

/**
 * Minimal test runner — no external dependencies needed.
 * Run with: node frontend/tests/test_colorscale.js
 */

let passed = 0;
let failed = 0;
const results = [];

function assert(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

function test(name, fn) {
  try {
    fn();
    passed++;
    results.push({ name, status: 'PASS' });
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    results.push({ name, status: 'FAIL', error: e.message });
    console.log(`  FAIL  ${name}: ${e.message}`);
  }
}

function validateResult(r, label) {
  assert(Number.isFinite(r.vmin), `${label}: vmin must be finite, got ${r.vmin}`);
  assert(Number.isFinite(r.vmax), `${label}: vmax must be finite, got ${r.vmax}`);
  assert(r.vmin < r.vmax, `${label}: vmin (${r.vmin}) must be < vmax (${r.vmax})`);
}

// Helper: generate a grid filled with a value
function filledGrid(rows, cols, value) {
  return Array.from({ length: rows }, () => Array(cols).fill(value));
}

// Helper: generate random grid within [lo, hi]
function randomGrid(rows, cols, lo, hi) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => lo + Math.random() * (hi - lo))
  );
}

console.log('=== computeColorScale Test Suite ===\n');

// ──────────────────────────────────────────────────────
// 1. Normal surface field (~28°C mean, ~1.5°C std)
// ──────────────────────────────────────────────────────
test('1. Normal surface field', () => {
  const grid = randomGrid(101, 241, 25, 32);
  const r = computeColorScale(grid);
  validateResult(r, 'surface');
  assert(!r.isEmpty, 'Should not be empty');
  assert(r.vmin >= 24, `vmin ${r.vmin} should be >= 24`);
  assert(r.vmax <= 33, `vmax ${r.vmax} should be <= 33`);
});

// ──────────────────────────────────────────────────────
// 2. Warm surface field (~32°C peak tropical)
// ──────────────────────────────────────────────────────
test('2. Warm surface field', () => {
  const grid = randomGrid(101, 241, 29, 33);
  const r = computeColorScale(grid);
  validateResult(r, 'warm');
  assert(r.vmin >= 28, `vmin ${r.vmin} reasonable for warm`);
  assert(r.vmax <= 34, `vmax ${r.vmax} reasonable for warm`);
});

// ──────────────────────────────────────────────────────
// 3. Intermediate-depth field (~15°C mean at 150m)
// ──────────────────────────────────────────────────────
test('3. Intermediate-depth field', () => {
  const grid = randomGrid(101, 241, 12, 20);
  const r = computeColorScale(grid);
  validateResult(r, 'intermediate');
  assert(r.vmin >= 11, `vmin ${r.vmin}`);
  assert(r.vmax <= 21, `vmax ${r.vmax}`);
});

// ──────────────────────────────────────────────────────
// 4. Deep field (~7.5°C mean at 900m)
// ──────────────────────────────────────────────────────
test('4. Deep field', () => {
  const grid = randomGrid(101, 241, 5, 10);
  const r = computeColorScale(grid);
  validateResult(r, 'deep');
  assert(r.vmin >= 4, `vmin ${r.vmin}`);
  assert(r.vmax <= 11, `vmax ${r.vmax}`);
});

// ──────────────────────────────────────────────────────
// 5. Legitimate negative-temperature field
// ──────────────────────────────────────────────────────
test('5. Negative temperature field', () => {
  const grid = randomGrid(50, 50, -1.8, 2.0);
  const r = computeColorScale(grid);
  validateResult(r, 'negative');
  assert(r.vmin < 0, `vmin ${r.vmin} should be negative`);
});

// ──────────────────────────────────────────────────────
// 6. Narrow distribution (std < 0.1°C)
// ──────────────────────────────────────────────────────
test('6. Narrow distribution', () => {
  const grid = randomGrid(50, 50, 15.0, 15.1);
  const r = computeColorScale(grid);
  validateResult(r, 'narrow');
  assert(r.vmax - r.vmin >= 0.5, `span ${r.vmax - r.vmin} must be >= minSpan 0.5`);
});

// ──────────────────────────────────────────────────────
// 7. Nearly constant field (all ~ 15.0°C)
// ──────────────────────────────────────────────────────
test('7. Nearly constant field', () => {
  const grid = filledGrid(50, 50, 15.0);
  const r = computeColorScale(grid);
  validateResult(r, 'constant');
  assert(r.vmax - r.vmin >= 0.5, `span ${r.vmax - r.vmin} must be >= 0.5`);
});

// ──────────────────────────────────────────────────────
// 8. Field with outliers (99% in [10,20], a few at 40)
// ──────────────────────────────────────────────────────
test('8. Field with outliers', () => {
  const grid = randomGrid(100, 100, 10, 20);
  // Add outliers
  grid[0][0] = 40; grid[0][1] = 45; grid[99][99] = -5;
  const r = computeColorScale(grid);
  validateResult(r, 'outliers');
  // Percentiles should suppress outliers
  assert(r.vmax < 40, `vmax ${r.vmax} should be below outlier 40`);
  assert(r.vmin > -5, `vmin ${r.vmin} should be above outlier -5`);
});

// ──────────────────────────────────────────────────────
// 9. Field with land mask (50% null values)
// ──────────────────────────────────────────────────────
test('9. Field with land mask', () => {
  const grid = randomGrid(100, 100, 10, 25);
  // Set 50% to null (land)
  for (let i = 0; i < 50; i++) {
    for (let j = 0; j < 100; j++) {
      grid[i][j] = null;
    }
  }
  const r = computeColorScale(grid);
  validateResult(r, 'land-mask');
  assert(!r.isEmpty, 'Should not be empty with 50% ocean');
  assert(r.stats.count === 5000, `count should be 5000, got ${r.stats.count}`);
});

// ──────────────────────────────────────────────────────
// 10. Partially missing field (90% null)
// ──────────────────────────────────────────────────────
test('10. Partially missing field', () => {
  const grid = Array.from({ length: 100 }, () => Array(100).fill(null));
  // Only 10% valid
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 100; j++) {
      grid[i][j] = 12 + Math.random() * 3;
    }
  }
  const r = computeColorScale(grid);
  validateResult(r, 'mostly-missing');
  assert(!r.isEmpty, 'Not empty');
  assert(r.stats.count === 1000, `count ${r.stats.count}`);
});

// ──────────────────────────────────────────────────────
// 11. All-NaN field
// ──────────────────────────────────────────────────────
test('11. All-NaN field', () => {
  const grid = filledGrid(50, 50, NaN);
  const r = computeColorScale(grid);
  assert(r.isEmpty === true, 'Should be empty');
  validateResult(r, 'all-NaN');  // vmin=0, vmax=1 fallback
});

// ──────────────────────────────────────────────────────
// 12. All-invalid field (null + undefined)
// ──────────────────────────────────────────────────────
test('12. All-invalid field', () => {
  const grid = Array.from({ length: 20 }, (_, i) =>
    Array.from({ length: 20 }, (_, j) => (i + j) % 2 === 0 ? null : undefined)
  );
  const r = computeColorScale(grid);
  assert(r.isEmpty === true, 'Should be empty');
  validateResult(r, 'all-invalid');
});

// ──────────────────────────────────────────────────────
// 13. Single valid pixel
// ──────────────────────────────────────────────────────
test('13. Single valid pixel', () => {
  const grid = filledGrid(50, 50, null);
  grid[25][25] = 18.0;
  const r = computeColorScale(grid);
  validateResult(r, 'single-pixel');
  assert(!r.isEmpty, 'Not empty');
  assert(r.stats.count === 1, `count should be 1, got ${r.stats.count}`);
  assert(r.vmax - r.vmin >= 0.5, `span must be >= 0.5`);
});

// ──────────────────────────────────────────────────────
// 14. Very large valid spread (0°C to 35°C)
// ──────────────────────────────────────────────────────
test('14. Very large spread', () => {
  const grid = randomGrid(100, 100, 0, 35);
  const r = computeColorScale(grid);
  validateResult(r, 'large-spread');
  assert(r.vmax - r.vmin > 5, `span should be large, got ${r.vmax - r.vmin}`);
});

// ──────────────────────────────────────────────────────
// 15. Non-finite values (Inf, -Inf mixed with valid)
// ──────────────────────────────────────────────────────
test('15. Non-finite values', () => {
  const grid = randomGrid(50, 50, 10, 20);
  grid[0][0] = Infinity;
  grid[0][1] = -Infinity;
  grid[1][0] = NaN;
  const r = computeColorScale(grid);
  validateResult(r, 'non-finite');
  assert(r.vmin >= 9, `vmin ${r.vmin} should ignore Inf`);
  assert(r.vmax <= 21, `vmax ${r.vmax} should ignore -Inf`);
});

// ──────────────────────────────────────────────────────
// 16. Values near physical freezing (~-1.8°C)
// ──────────────────────────────────────────────────────
test('16. Near-freezing conditions', () => {
  const grid = randomGrid(50, 50, -1.8, 1.0);
  const r = computeColorScale(grid);
  validateResult(r, 'near-freezing');
  assert(r.vmin < 0, `vmin ${r.vmin} should be negative for near-freezing`);
});

// ──────────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────────
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
if (failed > 0) {
  console.log('\nFailed tests:');
  results.filter(r => r.status === 'FAIL').forEach(r => {
    console.log(`  - ${r.name}: ${r.error}`);
  });
  process.exit(1);
} else {
  console.log('All tests passed!');
  process.exit(0);
}
