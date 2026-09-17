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

import { SpatialSkeletonDetailFocus } from "#src/skeleton/spatial_chunk_sizing.js";
import type * as SliceViewBase from "#src/sliceview/base.js";
import { Uint64Set } from "#src/uint64_set.js";
import { DataType } from "#src/util/data_type.js";
import { mat4 } from "#src/util/geom.js";

// `forEachVisibleChunkSlot`'s per-cell arbitration branch calls this to walk
// the anchor level's visible cells; its real implementation needs a fully
// realized projection matrix and chunk layout. The arbitration-gating tests
// below only need to observe whether the branch is ENTERED (a spy on an
// internal method call already proves that), not what it enumerates, so the
// callback itself is left a no-op here -- everything else from the module
// stays real.
vi.mock("#src/sliceview/base.js", async (importOriginal) => {
  const actual = await importOriginal<typeof SliceViewBase>();
  return { ...actual, forEachVisibleVolumetricChunk: vi.fn() };
});

if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const {
  SpatiallyIndexedSkeletonLayer,
  getSpatialSkeletonCellKeyPrefix,
  resolveSpatiallyIndexedSkeletonSegmentPick,
  computeDiagonalModelToGlobalMetersScale,
  maybeUpdateAutoSpatialSkeletonGridResolutionTarget,
} = await import("#src/skeleton/spatial_frontend.js");

describe("resolveSpatiallyIndexedSkeletonSegmentPick", () => {
  it("returns the node segment id (bigint) for direct node picks (1-component)", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new Uint32Array([11, 13, 17]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 1, "node"),
    ).toBe(13n);
  });

  it("returns the first valid endpoint segment id for direct edge picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new Uint32Array([0, 19, 23]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 0, "edge"),
    ).toBe(19n);
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 1, "edge"),
    ).toBe(19n);
  });

  it("reconstructs a FULL uint64 (>2^32) id from a 2-component [lo,hi] column", () => {
    // Flywire-scale id 720575940612786691 = lo 0x0DE2_2603, hi 0x0A00_0002.
    const id = 720575940612786691n;
    const lo = Number(id & 0xffffffffn) >>> 0;
    const hi = Number((id >> 32n) & 0xffffffffn) >>> 0;
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    // interleaved [lo, hi] per vertex; vertex 1 carries the id.
    const segmentIds = new Uint32Array([0, 0, lo, hi, 0, 0]);
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        1,
        "node",
        2,
      ),
    ).toBe(id);
    // edge (0,1): first endpoint (vertex 0) is empty → falls back to vertex 1.
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        0,
        "edge",
        2,
      ),
    ).toBe(id);
  });

  it("returns undefined for out-of-range direct picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1]),
      numVertices: 2,
    };
    const segmentIds = new Uint32Array([5, 7]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 4, "node"),
    ).toBeUndefined();
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 2, "edge"),
    ).toBeUndefined();
  });
});

describe("SpatiallyIndexedSkeletonLayer browse node picks", () => {
  it("resolves browse node picks with node id and source state", () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    const segmentIds = new Uint32Array([11, 17]);
    const vertexBytes = new Uint8Array(
      positions.byteLength + segmentIds.byteLength,
    );
    vertexBytes.set(new Uint8Array(positions.buffer), 0);
    vertexBytes.set(new Uint8Array(segmentIds.buffer), positions.byteLength);
    const chunk = {
      vertexAttributes: vertexBytes,
      vertexAttributeOffsets: new Uint32Array([0, positions.byteLength]),
      numVertices: 2,
      indices: new Uint32Array([0, 1]),
      nodeIds: new Int32Array([101, 202]),
      nodeSourceStates: [
        { revisionToken: "2026-03-29T11:50:00Z" },
        { revisionToken: "2026-03-29T11:51:00Z" },
      ],
    };
    const layer = Object.create(SpatiallyIndexedSkeletonLayer.prototype);
    // The pick path locates the "segment" column by attribute index; provide
    // the [position, segment(uint32)] layout matching the packed bytes above.
    (layer as any).vertexAttributes = [
      { name: "position", dataType: DataType.FLOAT32, numComponents: 3 },
      { name: "segment", dataType: DataType.UINT32, numComponents: 1 },
    ];

    expect((layer as any).resolveNodePickFromChunk(chunk, 1)).toEqual({
      nodeId: 202,
      segmentId: 17n,
      position: new Float32Array([4, 5, 6]),
      sourceState: { revisionToken: "2026-03-29T11:51:00Z" },
    });
  });
});

describe("SpatiallyIndexedSkeletonLayer targeted source invalidation", () => {
  it("computes absolute half-open cell prefixes without lower-bound offsets", () => {
    expect(
      getSpatialSkeletonCellKeyPrefix(
        new Float32Array([100, 200, 300]),
        new Float32Array([100, 100, 100]),
      ),
    ).toBe("1,2,3|");
    expect(
      getSpatialSkeletonCellKeyPrefix(
        new Float32Array([99.999, 199.999, 299.999]),
        new Float32Array([100, 100, 100]),
      ),
    ).toBe("0,1,2|");
  });

  it("dedupes cell prefixes per unique source entry", () => {
    const invalidateCacheKeyPrefixes = vi.fn();
    const source = {
      spec: {
        chunkDataSize: new Float32Array([100, 100, 100]),
        lowerChunkBound: new Float32Array([10, 20, 30]),
      },
      invalidateCacheKeyPrefixes,
    };
    const source2d = {
      spec: {
        chunkDataSize: new Float32Array([50, 50, 50]),
      },
      invalidateCacheKeyPrefixes: vi.fn(),
    };
    const redrawNeeded = { dispatch: vi.fn() };
    const layer = {
      sources: [{ chunkSource: source }, { chunkSource: source }],
      sources2d: [{ chunkSource: source2d }],
      redrawNeeded,
    };

    const invalidated =
      SpatiallyIndexedSkeletonLayer.prototype.invalidateSourceCellsForPositions.call(
        layer,
        [
          new Float32Array([100, 200, 300]),
          new Float32Array([199.999, 200, 300]),
          new Float32Array([100, 200, 300]),
        ],
      );

    expect(invalidated).toBe(true);
    expect(invalidateCacheKeyPrefixes).toHaveBeenCalledTimes(1);
    expect([...invalidateCacheKeyPrefixes.mock.calls[0][0]]).toEqual([
      "1,2,3|",
    ]);
    expect(source2d.invalidateCacheKeyPrefixes).toHaveBeenCalledTimes(1);
    expect([...source2d.invalidateCacheKeyPrefixes.mock.calls[0][0]]).toEqual([
      "2,4,6|",
      "3,4,6|",
    ]);
    expect(redrawNeeded.dispatch).toHaveBeenCalledTimes(1);
  });
});

describe("SpatiallyIndexedSkeletonLayer browse exclusions", () => {
  it("includes suppressed browse segments even when no overlay segment is loaded", () => {
    const layer = Object.assign(
      Object.create(SpatiallyIndexedSkeletonLayer.prototype),
      {
        suppressedBrowseSegmentIds: new Set<number>(),
        browseExcludedSegments: new Uint64Set(),
        browseExcludedSegmentsKey: undefined,
        redrawNeeded: { dispatch: vi.fn() },
        getLoadedOverlaySegmentIds: () => [],
      },
    );

    expect(layer.suppressBrowseSegment(29)).toBe(true);
    expect(layer.redrawNeeded.dispatch).toHaveBeenCalledTimes(1);

    const excludedSegments = (layer as any).getBrowsePassExcludedSegments();
    expect(excludedSegments).toBeInstanceOf(Uint64Set);
    expect([...excludedSegments]).toEqual([29n]);
  });
});

function makeIdentityMappedTransform(
  modelToRenderLayerTransform: Float32Array,
) {
  return {
    rank: 3,
    unpaddedRank: 3,
    localToRenderLayerDimensions: [0, 1, 2],
    globalToRenderLayerDimensions: [0, 1, 2],
    channelToRenderLayerDimensions: [],
    channelToModelDimensions: [],
    channelSpaceShape: new Uint32Array(0),
    modelToRenderLayerTransform,
    modelDimensionNames: ["x", "y", "z"],
    layerDimensionNames: ["x", "y", "z"],
  };
}

function diagonalMat4(
  diag: readonly [number, number, number],
  offDiag?: { row: number; col: number; value: number },
): Float32Array {
  const m = new Float32Array(16);
  m[0] = diag[0];
  m[5] = diag[1];
  m[10] = diag[2];
  m[15] = 1;
  if (offDiag !== undefined) {
    m[offDiag.row + 4 * offDiag.col] = offDiag.value;
  }
  return m;
}

describe("computeDiagonalModelToGlobalMetersScale", () => {
  it("composes a diagonal model->renderLayer scale with per-axis global scales", () => {
    const transform = makeIdentityMappedTransform(diagonalMat4([2, 3, 4]));
    const result = computeDiagonalModelToGlobalMetersScale(
      transform as any,
      new Float64Array([10, 20, 30]),
    );
    expect(result).toBeDefined();
    expect(Array.from(result!)).toEqual([20, 60, 120]);
  });

  it("reflects a live 1000x output rescale (the mm-to-µm bug scenario)", () => {
    // The render-layer transform scales model coordinates by 1e-3
    // relative to the store's own declared unit -- e.g. the user
    // corrected the source's output dimensions from mm to µm.
    const transform = makeIdentityMappedTransform(
      diagonalMat4([1e-3, 1e-3, 1e-3]),
    );
    const result = computeDiagonalModelToGlobalMetersScale(
      transform as any,
      new Float64Array([1, 1, 1]),
    );
    expect(result).toBeDefined();
    for (const v of result!) {
      expect(v).toBeCloseTo(1e-3, 9);
    }
  });

  it("returns undefined when a model dimension is unmapped", () => {
    const transform = makeIdentityMappedTransform(diagonalMat4([2, 3, 4]));
    (transform as any).localToRenderLayerDimensions = [0, -1, 2];
    const result = computeDiagonalModelToGlobalMetersScale(
      transform as any,
      new Float64Array([1, 1, 1]),
    );
    expect(result).toBeUndefined();
  });

  it("returns undefined for a non-diagonal (rotated/sheared) transform", () => {
    const transform = makeIdentityMappedTransform(
      diagonalMat4([2, 3, 4], { row: 0, col: 1, value: 5 }),
    );
    const result = computeDiagonalModelToGlobalMetersScale(
      transform as any,
      new Float64Array([1, 1, 1]),
    );
    expect(result).toBeUndefined();
  });

  it("returns undefined when the corresponding global scale is invalid", () => {
    const transform = makeIdentityMappedTransform(diagonalMat4([2, 3, 4]));
    const result = computeDiagonalModelToGlobalMetersScale(
      transform as any,
      new Float64Array([10, 0, 30]),
    );
    expect(result).toBeUndefined();
  });
});

describe("maybeUpdateAutoSpatialSkeletonGridResolutionTarget bias stability", () => {
  function makeWatchable(initial: number) {
    let v = initial;
    return {
      get value() {
        return v;
      },
      set value(x: number) {
        v = x;
      },
      changed: { dispatch: () => {} },
    };
  }

  it("does not corrupt the persisted bias across repeated sub-threshold updates", () => {
    const target = makeWatchable(0);
    const bias = makeWatchable(1);
    const displayState = {
      autoSpatialSkeletonGridLevel3d: { value: true },
      spatialSkeletonGridResolutionTarget3d: target,
      spatialSkeletonGridResolutionBias3d: bias,
    } as any;

    // Identity view-projection: w=1 regardless of world position; only
    // the varying `width` below perturbs the computed target by a tiny
    // (sub-0.1%) amount frame to frame, matching the real skip-write path.
    const viewProjectionMat = mat4.create();
    const localPosition = new Float32Array(0);

    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      {
        viewProjectionMat,
        width: 1000,
        height: 1,
        globalPosition: new Float32Array(3),
      },
      localPosition,
      "3d",
    );
    // First call always writes (no prior `lastAuto`).
    expect(target.value).toBeCloseTo(0.2, 10);
    expect(bias.value).toBe(1);

    // Two more frames with a change small enough to land in the
    // "skip write" branch (< 0.1%), matching ordinary smooth camera
    // motion.  Before the fix, `lastAuto` advanced on the skipped
    // write anyway, drifted away from the un-changed `target.value`,
    // and on the very next frame got misread as a manual widget drag —
    // corrupting `bias`.
    for (let i = 0; i < 5; ++i) {
      maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
        displayState,
        {
          viewProjectionMat,
          width: 1000.1,
          height: 1,
          globalPosition: new Float32Array(3),
        },
        localPosition,
        "3d",
      );
    }

    expect(bias.value).toBe(1);
    expect(target.value).toBeCloseTo(0.2, 10);
  });

  it("does not read a reset's hard-default target as a deliberate recalibration", () => {
    // The only path that flips `autoSpatialSkeletonGridLevel3d` false->true is
    // `reset()` on the widget (a double-click), which sets it immediately
    // before `target.reset()` writes the class's hard default (1 metre).
    // Before the fix, the huge gap between that default and the last real
    // auto value was read as "the user just deliberately recalibrated",
    // baking an enormous multiplier into the persisted `bias` and pinning
    // the level at the coarsest available level forever.
    const target = makeWatchable(0);
    const bias = makeWatchable(1);
    const displayState = {
      autoSpatialSkeletonGridLevel3d: { value: true },
      spatialSkeletonGridResolutionTarget3d: target,
      spatialSkeletonGridResolutionBias3d: bias,
    } as any;
    const viewProjectionMat = mat4.create();
    const localPosition = new Float32Array(0);
    const projectionParameters = {
      viewProjectionMat,
      width: 1000,
      height: 1,
      globalPosition: new Float32Array(3),
    };

    // A real auto session: establishes `lastAuto` at the pixel-driven target.
    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      projectionParameters,
      localPosition,
      "3d",
    );
    expect(target.value).toBeCloseTo(0.2, 10);

    // A manual click: disables auto and jumps the target away (simulating
    // the widget's own "set" handler + `onManualTarget`).
    displayState.autoSpatialSkeletonGridLevel3d.value = false;
    target.value = 5;
    // While auto is off the function is still called every frame (the
    // caller does not gate on the flag) but must not touch `target`/`bias`.
    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      projectionParameters,
      localPosition,
      "3d",
    );
    expect(target.value).toBe(5);
    expect(bias.value).toBe(1);

    // A double-click reset: `onResetTarget` re-enables auto, then
    // `target.reset()` writes the class's hard default (1).
    displayState.autoSpatialSkeletonGridLevel3d.value = true;
    target.value = 1;

    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      projectionParameters,
      localPosition,
      "3d",
    );

    // Must snap straight back to the camera-derived value, not stay pinned
    // near the hard default, and must not corrupt the persisted bias.
    expect(target.value).toBeCloseTo(0.2, 10);
    expect(bias.value).toBe(1);
  });
});

describe("maybeUpdateAutoSpatialSkeletonGridResolutionTarget memory-target combination", () => {
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
    };
  }

  // Identity view-projection + width 1000 => pixelSize 0.001 => the
  // pixel-derived target is always 0.001 * 200 = 0.2 (see the bias-stability
  // suite above, which pins this same calibration).
  const viewProjectionMat = mat4.create();
  const localPosition = new Float32Array(0);
  const projectionParameters = {
    viewProjectionMat,
    width: 1000,
    height: 1,
    globalPosition: new Float32Array(3),
  };
  const PIXEL_DRIVEN_TARGET = 0.2;

  // Coarsest-first, matching `spatialSkeletonGridLevels` / per-cell-cost
  // convention: level 0 spacing 0.5 (coarse), level 1 spacing 0.1 (fine).
  const levels = [
    { size: { x: 0.5, y: 0.5, z: 0.5 }, lod: 0 },
    { size: { x: 0.1, y: 0.1, z: 0.1 }, lod: 1 },
  ];

  it("lets the camera drive the level when the whole pyramid easily fits in budget (a resolution pyramid)", () => {
    const target = makeWatchable(0);
    const displayState = {
      autoSpatialSkeletonGridLevel3d: { value: true },
      spatialSkeletonGridResolutionTarget3d: target,
      // Both levels cost far less than the budget: unconstrained.
      spatialSkeletonPerCellCostBytes: makeWatchable([100, 100]),
      spatialSkeletonGpuBudgetBytes: makeWatchable(1e9),
      spatialSkeletonGridLevels: makeWatchable(levels),
    } as any;

    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      projectionParameters,
      localPosition,
      "3d",
      /* visibleCellCount= */ 1,
    );

    // Before this fix, the memory target (the finest affordable spacing,
    // 0.1) unconditionally REPLACED the pixel-derived one, so the level
    // never changed no matter how far the camera zoomed. It must now be
    // the pixel-derived value driving the target -- larger than the
    // memory floor here, so the max() correctly picks it.
    expect(target.value).toBeCloseTo(PIXEL_DRIVEN_TARGET, 10);
  });

  it("still floors the target at the finest AFFORDABLE level when the budget is tight (a whole-brain sparsity pyramid)", () => {
    const target = makeWatchable(0);
    const displayState = {
      autoSpatialSkeletonGridLevel3d: { value: true },
      spatialSkeletonGridResolutionTarget3d: target,
      // Both levels cost far more than the budget: nothing fits, so
      // `targetSpacingForCellBudget` falls back to the coarsest level's
      // own spacing (0.5) -- the exact case commit 4c688c1c protects.
      spatialSkeletonPerCellCostBytes: makeWatchable([1e12, 1e12]),
      spatialSkeletonGpuBudgetBytes: makeWatchable(1e9),
      spatialSkeletonGridLevels: makeWatchable(levels),
    } as any;

    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      projectionParameters,
      localPosition,
      "3d",
      /* visibleCellCount= */ 1,
    );

    // The memory floor (0.5) is coarser than the pixel-derived target
    // (0.2), so it must dominate -- identical to the pre-fix behaviour
    // for a pyramid that cannot fit, regardless of the max() change.
    expect(target.value).toBeCloseTo(0.5, 10);
  });

  it("keeps the pure pixel-derived target when the layer publishes no per-cell costs", () => {
    const target = makeWatchable(0);
    const displayState = {
      autoSpatialSkeletonGridLevel3d: { value: true },
      spatialSkeletonGridResolutionTarget3d: target,
      // No spatialSkeletonPerCellCostBytes / GpuBudgetBytes / GridLevels at
      // all -- every non-tract spatially-indexed skeleton layer today.
    } as any;

    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      projectionParameters,
      localPosition,
      "3d",
      /* visibleCellCount= */ 1,
    );

    expect(target.value).toBeCloseTo(PIXEL_DRIVEN_TARGET, 10);
  });
});

describe("forEachVisibleChunkSlot LOCAL-focus arbitration gating", () => {
  // Two candidate levels with distinct chunkSource identities, standing in
  // for a 3-level resolution pyramid's finest and coarsest levels.
  function makeLevels() {
    const chunkSourceFine = { name: "fine" };
    const chunkSourceCoarse = { name: "coarse" };
    const selectedSources = [
      { chunkSource: chunkSourceFine },
      { chunkSource: chunkSourceCoarse },
    ];
    const transformedSources = [
      [
        { source: chunkSourceFine, chunkLayout: {} },
        { source: chunkSourceCoarse, chunkLayout: {} },
      ],
    ];
    return {
      chunkSourceFine,
      chunkSourceCoarse,
      selectedSources,
      transformedSources,
    };
  }

  function makeFakeLayer(partitioned: boolean, selectedSources: unknown[]) {
    return {
      detailFocus: { value: SpatialSkeletonDetailFocus.LOCAL },
      localPosition: { value: new Float32Array(3) },
      objectPartitionAvailable: vi.fn(() => partitioned),
      getSources: vi.fn(() => []),
      selectSourcesForViewAndGrid: vi.fn(() => selectedSources),
      selectSourcesForViewAndGridWithFallback: vi.fn(() => selectedSources),
      getChunkSpacing: vi.fn(() => 1),
      getMetersPerUnit: vi.fn(() => 1),
      getLevelSpacingsMeters: vi.fn(() => [4e-8, 1.6e-7]),
      getArbitrationTargetSpacingMeters3d: vi.fn(() => 8e-8),
      getReferencePixelSize: vi.fn(() => 1),
      getChunkCenterWorld: vi.fn(),
      getChunkGridPositionForWorldPoint: vi.fn(() => false),
      quantizeSpacingForArbitration: vi.fn((s: number) => s),
    };
  }

  it("skips per-cell arbitration when the levels do not partition objects", () => {
    const { selectedSources, transformedSources } = makeLevels();
    const layer = makeFakeLayer(/* partitioned= */ false, selectedSources);

    (
      SpatiallyIndexedSkeletonLayer.prototype as any
    ).forEachVisibleChunkSlot.call(
      layer,
      "3d",
      /* gridLevel= */ 0,
      transformedSources,
      {} as any,
      () => {},
    );

    expect(layer.objectPartitionAvailable).toHaveBeenCalledWith("3d");
    // The gate must short-circuit before any arbitration-only internals run
    // -- this is the exact condition the fix added.
    expect(layer.getArbitrationTargetSpacingMeters3d).not.toHaveBeenCalled();
    expect(layer.getLevelSpacingsMeters).not.toHaveBeenCalled();
  });

  it("still runs per-cell arbitration when the levels do partition objects", () => {
    const { selectedSources, transformedSources } = makeLevels();
    const layer = makeFakeLayer(/* partitioned= */ true, selectedSources);

    (
      SpatiallyIndexedSkeletonLayer.prototype as any
    ).forEachVisibleChunkSlot.call(
      layer,
      "3d",
      /* gridLevel= */ 0,
      transformedSources,
      {} as any,
      () => {},
    );

    expect(layer.objectPartitionAvailable).toHaveBeenCalledWith("3d");
    expect(layer.getArbitrationTargetSpacingMeters3d).toHaveBeenCalled();
    expect(layer.getLevelSpacingsMeters).toHaveBeenCalled();
  });
});
