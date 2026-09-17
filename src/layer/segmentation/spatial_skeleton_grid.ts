/**
 * @license
 * Copyright 2016 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Choosing a grid level for a spatially-indexed skeleton source, and describing
 * the resulting spacing distribution to the render-scale histogram.
 *
 * Pure arithmetic over the level table -- no layer state, no DOM -- which is
 * why it sits beside `index.ts` rather than inside SegmentationUserLayer.
 */

import "#src/layer/segmentation/style.css";
import "#src/layer/segmentation/spatial_skeleton.css";

import {
  numRenderScaleHistogramBins,
  renderScaleHistogramBinSize,
  renderScaleHistogramOrigin,
} from "#src/render_scale_statistics.js";

import {
  getSpatialSkeletonGridSpacing,
  type SpatialSkeletonGridLevel,
} from "#src/skeleton/spatial_chunk_sizing.js";

export function findClosestSpatialSkeletonGridLevelBySpacing(
  levels: SpatialSkeletonGridLevel[],
  spacing: number,
): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < levels.length; ++i) {
    const gridSpacing = getSpatialSkeletonGridSpacing(levels[i].size);
    const distance = Math.abs(gridSpacing - spacing);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export function getSpatialSkeletonGridHistogramConfig(
  levels: SpatialSkeletonGridLevel[],
) {
  if (levels.length === 0) {
    return {
      origin: renderScaleHistogramOrigin,
      binSize: renderScaleHistogramBinSize,
    };
  }
  const logSpacings: number[] = [];
  let minLogSpacing = Number.POSITIVE_INFINITY;
  let maxLogSpacing = Number.NEGATIVE_INFINITY;
  for (const level of levels) {
    // Degenerate-input guard (avoids log2(0) === -Infinity below), not a
    // physical minimum -- 1e-6 (1 micron) silently floored every level of a
    // sub-micron skeleton/streamline pyramid to the same spacing, which
    // reads here as zero spread ("a single level ... no meaningful
    // spread") and permanently pins the histogram/marker at 1 micron,
    // however many genuinely distinct (sub-micron) levels actually exist.
    const spacing = Math.max(getSpatialSkeletonGridSpacing(level.size), 1e-30);
    const logSpacing = Math.log2(spacing);
    logSpacings.push(logSpacing);
    minLogSpacing = Math.min(minLogSpacing, logSpacing);
    maxLogSpacing = Math.max(maxLogSpacing, logSpacing);
  }
  if (!Number.isFinite(minLogSpacing) || !Number.isFinite(maxLogSpacing)) {
    return {
      origin: renderScaleHistogramOrigin,
      binSize: renderScaleHistogramBinSize,
    };
  }
  logSpacings.sort((a, b) => a - b);
  let minDelta = Number.POSITIVE_INFINITY;
  for (let i = 1; i < logSpacings.length; ++i) {
    const delta = logSpacings[i] - logSpacings[i - 1];
    if (delta > 0) minDelta = Math.min(minDelta, delta);
  }
  const span = maxLogSpacing - minLogSpacing;
  // Choose a bin size that spreads the levels across (most of) the widget
  // width.  Reserve a few bins of padding on each side so the extreme
  // levels aren't flush against the edges.  A single level (span 0) has no
  // meaningful spread — fall back to the default bin size.
  const coverageBinSize =
    span > 0
      ? span / Math.max(numRenderScaleHistogramBins - 4, 1)
      : renderScaleHistogramBinSize;
  // Never use a bin so large that two adjacent levels (minDelta apart in
  // log space) collapse into the same bin — that would merge distinct
  // scales into one bar.  When the coverage bin size already keeps them
  // distinct (the common case: few, well-separated pyramid levels), the
  // coverage value wins and the bars fan out across the full axis instead
  // of bunching into a narrow cluster in the middle.
  const maxBinSizeForDistinctBars = Number.isFinite(minDelta)
    ? minDelta * 0.9
    : Number.POSITIVE_INFINITY;
  let binSize = Math.max(
    0.05,
    Math.min(coverageBinSize, maxBinSizeForDistinctBars),
  );
  if (!Number.isFinite(binSize) || binSize <= 0) {
    binSize = renderScaleHistogramBinSize;
  }

  const range = numRenderScaleHistogramBins * binSize;
  const desiredPadding = binSize * 2;
  const minOrigin = maxLogSpacing + desiredPadding - range;
  const maxOrigin = minLogSpacing - desiredPadding;
  const centeredOrigin = (minLogSpacing + maxLogSpacing - range) / 2;
  const clampedOrigin = Math.min(
    Math.max(centeredOrigin, minOrigin),
    maxOrigin,
  );
  const roundedBinSize = Math.max(binSize, 1e-3);
  const roundedOrigin =
    Math.round(clampedOrigin / roundedBinSize) * roundedBinSize;
  return { origin: roundedOrigin, binSize: roundedBinSize };
}
