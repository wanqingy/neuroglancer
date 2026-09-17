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

import { describe, expect, it, vi } from "vitest";

import { RenderScaleWidget } from "#src/widget/render_scale_widget.js";

function makeWatchable<T>(initial: T) {
  let v = initial;
  return {
    get value() {
      return v;
    },
    set value(x: T) {
      v = x;
    },
    changed: { dispatch: () => {} },
    reset: () => {
      v = initial;
    },
  };
}

// `adjustViaWheel` only touches `logScaleOrigin`/`logScaleBinSize`/`target`/
// `hoverTarget`/`onManualTarget` -- none of which need the DOM elements the
// real constructor creates -- so a bare prototype instance with just those
// fields set exercises the exact method under test, following the same
// pattern used elsewhere in this codebase for private-method unit tests
// (e.g. `spatial_frontend.spec.ts`).
function makeWidget(
  logScaleOrigin: number,
  logScaleBinSize: number,
  targetValue: number,
) {
  return Object.assign(Object.create(RenderScaleWidget.prototype), {
    logScaleOrigin,
    logScaleBinSize,
    target: makeWatchable(targetValue),
    hoverTarget: makeWatchable<[number, number] | undefined>(undefined),
    onManualTarget: undefined,
  }) as RenderScaleWidget;
}

function wheelUp() {
  return { deltaY: 1, preventDefault: vi.fn() } as unknown as WheelEvent;
}

describe("RenderScaleWidget.adjustViaWheel axis clamp", () => {
  it("reaches the full default axis (origin=-4, binSize=0.5, 40 bins -> 20 octaves)", () => {
    // Before the fix, the upper clamp was `2 ** (round(origin + range) - 1)`
    // -- a fixed one-octave haircut baked in for this exact default axis, so
    // this case was never wrong in practice. It still must not regress:
    // starting one octave below the true top and scrolling up must reach the
    // true top (2**16), not fall short of it or move backwards.
    const widget = makeWidget(-4, 0.5, 2 ** 15);
    widget.adjustViaWheel(wheelUp());
    expect(widget.target.value).toBeCloseTo(2 ** 16, 6);
  });

  it("reaches a data-derived narrow axis's true top (the sub-micron pyramid case)", () => {
    // Mirrors `getSpatialSkeletonGridHistogramConfig`'s output for a real
    // 40/80/160nm (in meters: 4e-8/8e-8/1.6e-7) skeleton pyramid: origin
    // chosen so the axis's top sits just above the coarsest level, and a
    // binSize far smaller than the module default (0.5). Before the fix,
    // `2 ** (Math.round(origin + 40*binSize) - 1)` chopped a whole extra
    // octave off an axis that only spans ~2.2 octaves total, and the
    // coarsest level (160nm) was unreachable by wheel no matter how far the
    // user scrolled.
    const origin = -24.666666666666668;
    const binSize = 1 / 18; // ~0.0556, matching the production formula's order of magnitude
    const spacing160nm = 1.6e-7;
    const numBins = 40;
    const trueTop = 2 ** (origin + numBins * binSize);
    expect(trueTop).toBeGreaterThan(spacing160nm); // sanity: axis really covers it

    // Start at the 80nm level and scroll up (coarsen) twice: once should not
    // be enough (only a fraction of the remaining axis --binSize is small),
    // so scroll enough times to reach the ceiling and confirm it lands
    // exactly at the true top, not one octave short of it.
    const widget = makeWidget(origin, binSize, 8e-8);
    for (let i = 0; i < 40; ++i) {
      widget.adjustViaWheel(wheelUp());
    }
    expect(widget.target.value).toBeCloseTo(trueTop, 10);
    expect(widget.target.value).toBeGreaterThanOrEqual(spacing160nm);
  });

  it("still clamps at the bottom of the axis when scrolling down", () => {
    const widget = makeWidget(-4, 0.5, 2 ** -3);
    const wheelDown = {
      deltaY: -1,
      preventDefault: vi.fn(),
    } as unknown as WheelEvent;
    for (let i = 0; i < 10; ++i) {
      widget.adjustViaWheel(wheelDown);
    }
    expect(widget.target.value).toBeCloseTo(2 ** -4, 10);
  });
});
