/**
 * @license
 * Copyright 2026 Google Inc.
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

import { describe, expect, it } from "vitest";

import { getSpatialSkeletonGridHistogramConfig } from "#src/layer/segmentation/spatial_skeleton_grid.js";
import { renderScaleHistogramBinSize } from "#src/render_scale_statistics.js";
import type { SpatialSkeletonGridLevel } from "#src/skeleton/spatial_chunk_sizing.js";

function level(spacingMeters: number): SpatialSkeletonGridLevel {
  return {
    size: { x: spacingMeters, y: spacingMeters, z: spacingMeters },
    lod: 0,
  };
}

describe("getSpatialSkeletonGridHistogramConfig", () => {
  it("spreads a sub-micron resolution pyramid across distinct bins instead of collapsing to one", () => {
    // 40 / 80 / 160 nm, expressed in meters -- all well below the 1 micron
    // (1e-6) floor this function used to apply. Before the fix, every level
    // floored to the identical 1e-6 spacing, `span` was 0, and the config
    // fell back to the "single level, no spread" default -- permanently
    // pinning the render-scale histogram/marker at 1 micron regardless of
    // which of the three genuinely distinct levels was active.
    const levels = [level(4e-8), level(8e-8), level(1.6e-7)];
    const { binSize } = getSpatialSkeletonGridHistogramConfig(levels);
    // A real (non-fallback) bin size is derived from the actual spread
    // between levels, which is only possible when the floor did not
    // collapse them all to one value first.
    expect(binSize).not.toBe(renderScaleHistogramBinSize);
    expect(binSize).toBeGreaterThan(0);
    expect(Number.isFinite(binSize)).toBe(true);
  });

  it("still falls back safely for a degenerate (zero-size) level", () => {
    // The floor's actual job: guard against log2(0) === -Infinity, not
    // impose a physical minimum. A single degenerate level must not throw
    // or produce a non-finite config.
    const { origin, binSize } = getSpatialSkeletonGridHistogramConfig([
      level(0),
    ]);
    expect(Number.isFinite(origin)).toBe(true);
    expect(Number.isFinite(binSize)).toBe(true);
    expect(binSize).toBeGreaterThan(0);
  });

  it("falls back to the default for a single real level (no spread to show)", () => {
    const { binSize } = getSpatialSkeletonGridHistogramConfig([level(4e-8)]);
    expect(binSize).toBe(renderScaleHistogramBinSize);
  });
});
