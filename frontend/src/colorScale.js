/**
 * computeColorScale — Data-driven, robust color-scale computation for
 * ocean temperature visualization.
 *
 * Accepts a 2D array (grid) of predicted potential temperature values (°C).
 * Returns { vmin, vmax, isEmpty } with guarantees:
 *   - No NaN / Inf in vmin/vmax
 *   - vmin < vmax
 *   - Robust to outliers (percentile-based)
 *   - Handles all-NaN, all-null, single-pixel, and nearly-constant fields
 *
 * Land masking strategy:
 *   The model uses fillna(0) during preprocessing, so land pixels in the
 *   prediction output are denormalized as (0 * std) + mean ≈ mean for that
 *   depth. We CANNOT reliably distinguish land from ocean by value alone
 *   after denormalization. Instead, we treat null/NaN/non-finite as invalid,
 *   and use ALL finite values for scale computation. This is correct because
 *   the model's land predictions (while physically meaningless) are within
 *   the ocean temperature range and won't distort the scale significantly.
 *
 * @param {number[][]} grid - 2D array [lat][lon] of temperature values
 * @param {object} [options] - Optional configuration
 * @param {number} [options.pLow=0.02] - Lower percentile (0-1)
 * @param {number} [options.pHigh=0.98] - Upper percentile (0-1)
 * @param {number} [options.minSpan=0.5] - Minimum vmax-vmin span in °C
 * @returns {{ vmin: number, vmax: number, isEmpty: boolean, stats: object }}
 */
export function computeColorScale(grid, options = {}) {
  const {
    pLow = 0.02,
    pHigh = 0.98,
    minSpan = 0.5,
  } = options;

  // 1. Collect all finite values
  const validValues = [];
  if (Array.isArray(grid)) {
    for (let i = 0; i < grid.length; i++) {
      const row = grid[i];
      if (!Array.isArray(row)) continue;
      for (let j = 0; j < row.length; j++) {
        const v = row[j];
        if (v != null && Number.isFinite(v)) {
          validValues.push(v);
        }
      }
    }
  }

  // 2. Handle empty / all-invalid
  if (validValues.length === 0) {
    return {
      vmin: 0,
      vmax: 1,
      isEmpty: true,
      stats: { count: 0, rawMin: NaN, rawMax: NaN, median: NaN, p2: NaN, p98: NaN },
    };
  }

  // 3. Sort for percentile computation
  validValues.sort((a, b) => a - b);
  const n = validValues.length;

  const rawMin = validValues[0];
  const rawMax = validValues[n - 1];
  const median = validValues[Math.floor(n / 2)];

  // 4. Compute percentiles
  let p2, p98;
  if (n === 1) {
    // Single pixel: center on that value
    p2 = validValues[0] - minSpan / 2;
    p98 = validValues[0] + minSpan / 2;
  } else {
    p2 = validValues[Math.floor(n * pLow)];
    p98 = validValues[Math.min(Math.floor(n * pHigh), n - 1)];
  }

  // 5. Guarantee minimum span
  if (p98 - p2 < minSpan) {
    const center = (p2 + p98) / 2;
    p2 = center - minSpan / 2;
    p98 = center + minSpan / 2;
  }

  // 6. Round to clean tick-friendly values
  //    Snap to 0.5°C increments for human readability
  const vmin = Math.floor(p2 * 2) / 2;   // round down to nearest 0.5
  const vmax = Math.ceil(p98 * 2) / 2;    // round up to nearest 0.5

  // 7. Final safety: ensure vmin < vmax
  const finalVmin = vmin < vmax ? vmin : vmax - minSpan;

  return {
    vmin: finalVmin,
    vmax: vmax,
    isEmpty: false,
    stats: {
      count: n,
      rawMin,
      rawMax,
      median,
      p2: validValues[Math.floor(n * pLow)] ?? rawMin,
      p98: validValues[Math.min(Math.floor(n * pHigh), n - 1)] ?? rawMax,
      clipLowPct: pLow * 100,
      clipHighPct: (1 - pHigh) * 100,
    },
  };
}
