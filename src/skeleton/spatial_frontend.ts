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

/**
 * Render half of the spatially-indexed skeleton source -- the chunk-pyramid
 * route, as opposed to upstream's one-blob-per-segment-id route in
 * `frontend.ts`.
 *
 * What lives here is what only a pyramid has: choosing a grid level from the
 * camera, picking and inspecting individual nodes, and the perspective and
 * cross-section render layers that draw a resident chunk set.
 *
 * It draws through `frontend.ts`'s RenderHelper rather than duplicating shader
 * construction, so both routes render identically and there is one place to
 * change when the shader changes. The runtime dependency runs one way -- this
 * module imports from `frontend.ts`, which imports only a type back.
 */

import { ChunkState, LayerChunkProgressInfo } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { ChunkRenderLayerFrontend } from "#src/chunk_manager/frontend.js";
import type { RoiGroupConfig } from "#src/datasource/zarr-vectors/roi.js";
import { hashCombine } from "#src/gpu_hash/hash_function.js";
import type { HashSetUint64 } from "#src/gpu_hash/hash_table.js";
import { GPUHashTable } from "#src/gpu_hash/shader.js";
import type {
  LayerView,
  MouseSelectionState,
  PickState,
  VisibleLayerInfo,
} from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type {
  PerspectiveViewReadyRenderContext,
  PerspectiveViewRenderContext,
} from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type {
  ChunkTransformParameters,
  RenderLayerTransform,
} from "#src/render_coordinate_transform.js";
import { getChunkTransformParameters } from "#src/render_coordinate_transform.js";
import type { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import type {
  RenderLayer,
  ThreeDimensionalRenderLayerAttachmentState,
} from "#src/renderlayer.js";
import { update3dRenderLayerAttachment } from "#src/renderlayer.js";
import { getVisibleSegments } from "#src/segmentation_display_state/base.js";
import { registerRedrawWhenSegmentationDisplayState3DChanged } from "#src/segmentation_display_state/frontend.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type {
  SpatiallyIndexedSkeletonNode,
  SpatialSkeletonSourceState,
} from "#src/skeleton/api.js";
import type {
  GeometryPrimitive,
  PackedSkeletonGeometry,
  SkeletonChunkBase,
  SkeletonGPUGeometry,
  SkeletonLayerDisplayState,
  SkeletonShaderContext,
  SkeletonShaderParameters,
  SpatiallyIndexedSkeletonPickData,
  VertexAttributeRenderInfo,
  ViewSpecificSkeletonRenderingOptions,
  VisibleChunk} from "#src/skeleton/frontend.js";
import {
  ChunkWireframeHelper,
  DEBUG_SPATIAL_SKELETON_CHUNKS,
  DEFAULT_FRAGMENT_MAIN,
  OVERLAY_SELECTED_FLOAT_ONE,
  OVERLAY_SELECTED_FLOAT_ZERO,
  RenderHelper,
  SkeletonRenderMode,
  freeSkeletonChunkGPUMemory,
  getSkeletonNodeDiameter,
  segmentAttribute,
  selectedNodeAttribute,
  setMouseStatePositionFromSpatialSkeletonNode,
  tempChunkKeyToColorMap,
  tempMat4,
  uploadSkeletonChunkToGPU,
  vertexPositionAttribute,
} from "#src/skeleton/frontend.js";
import type { PackedAttributeRange } from "#src/skeleton/packed_attributes.js";


import {
  buildSpatiallyIndexedSkeletonOverlayGeometry,
  type SpatiallyIndexedSkeletonOverlayGeometry,
} from "#src/skeleton/segment_overlay.js";
import {
  DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS,
  mergeSpatiallyIndexedSkeletonOverlaySegmentIds,
  retainSpatiallyIndexedSkeletonOverlaySegment,
} from "#src/skeleton/segment_overlay.js";
import {
  getSpatiallyIndexedSkeletonGridIndex,
  getSpatiallyIndexedSkeletonPartitionsObjects,
  getSpatiallyIndexedSkeletonSourceView,
  selectSpatiallyIndexedSkeletonEntriesForView,
  selectSpatiallyIndexedSkeletonEntriesForViewWithFallback,
  type SpatiallyIndexedSkeletonView,
} from "#src/skeleton/source_selection.js";
import type { VertexAttrStats } from "#src/skeleton/spatial_base.js";

import {
  forEachSpatialSkeletonVolumeCell,
  SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_VERTEX_ATTR_STATS_RPC_ID,
} from "#src/skeleton/spatial_base.js";
import {
  SpatialSkeletonDetailFocus,
  targetSpacingForCellBudget,
} from "#src/skeleton/spatial_chunk_sizing.js";
import {
  forEachVisibleVolumetricChunk,
  type SliceViewChunkSpecification,
  type SliceViewSourceOptions,
  type TransformedSource,
} from "#src/sliceview/base.js";
import type { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import type { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
  SliceViewChunk,
  SliceViewChunkSource,
  MultiscaleSliceViewChunkSource,
} from "#src/sliceview/frontend.js";
import type { SliceViewPanel } from "#src/sliceview/panel.js";
import type {
  SliceViewPanelRenderContext,
  SliceViewPanelReadyRenderContext,
} from "#src/sliceview/renderlayer.js";
import { SliceViewPanelRenderLayer } from "#src/sliceview/renderlayer.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import {
  makeCachedLazyDerivedWatchableValue,
  WatchableValue,
  registerNested,
} from "#src/trackable_value.js";

import { Uint64Set } from "#src/uint64_set.js";

import { hsvToRgb } from "#src/util/colorspace.js";
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import type { ValueOrError } from "#src/util/error.js";
import { makeValueOrError, valueOrThrow } from "#src/util/error.js";
import { kOneVec4, mat4, vec3 } from "#src/util/geom.js";

import { getObjectId } from "#src/util/object_id.js";
import { NullarySignal } from "#src/util/signal.js";

import { drawBoxEdges } from "#src/webgl/bounding_box.js";
import { GLBuffer } from "#src/webgl/buffer.js";

import type { GL } from "#src/webgl/context.js";

import type { ShaderProgram } from "#src/webgl/shader.js";

import {
  getFallbackBuilderState,
  parseShaderUiControls,
  setControlsInShader,
} from "#src/webgl/shader_ui_controls.js";
import {
  computeTextureFormat,
  setOneDimensionalTextureData,
  TextureFormat,
  updateOneDimensionalTextureElement,
} from "#src/webgl/texture_access.js";
import type { RPC } from "#src/worker_rpc.js";

export class SpatiallyIndexedSkeletonChunk
  extends SliceViewChunk
  implements SkeletonChunkBase
{
  declare source: SpatiallyIndexedSkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer!: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[] = [];
  nodeIds: Int32Array;
  nodeSourceStates: Array<SpatialSkeletonSourceState | undefined> = [];

  constructor(
    source: SpatiallyIndexedSkeletonSource,
    chunkData: PackedSkeletonGeometry,
  ) {
    super(source, chunkData);
    this.vertexAttributes = chunkData.vertexAttributes;
    const indices = (this.indices = chunkData.indices);
    this.numVertices = chunkData.numVertices;
    this.numIndices = indices.length;
    this.vertexAttributeOffsets = chunkData.vertexAttributeOffsets;
    this.nodeIds = chunkData.nodeIds ?? new Int32Array(0);
    const nodeSourceStates = chunkData.nodeSourceStates;
    this.nodeSourceStates = Array.isArray(nodeSourceStates)
      ? nodeSourceStates
      : [];
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    uploadSkeletonChunkToGPU(gl, this);
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    freeSkeletonChunkGPUMemory(gl, this);
  }
}

export interface SpatiallyIndexedSkeletonChunkSpecification
  extends SliceViewChunkSpecification {
  chunkLayout: ChunkLayout;
}

type SpatiallyIndexedSkeletonChunkListener = (
  key: string,
  chunk: SpatiallyIndexedSkeletonChunk,
) => void;

const spatiallyIndexedSkeletonTextureAttributeSpecs = Object.freeze([
  { name: "position", dataType: DataType.FLOAT32, numComponents: 3 },
  { name: "segment", dataType: DataType.UINT32, numComponents: 1 },
]);

export class SpatiallyIndexedSkeletonSource extends SliceViewChunkSource<
  SpatiallyIndexedSkeletonChunkSpecification,
  SpatiallyIndexedSkeletonChunk
> {
  vertexAttributes: VertexAttributeRenderInfo[];
  private attributeTextureFormats_?: TextureFormat[];
  private chunkListeners = new Set<SpatiallyIndexedSkeletonChunkListener>();

  constructor(chunkManager: ChunkManager, options: any) {
    super(chunkManager, options);
    this.vertexAttributes = [vertexPositionAttribute, segmentAttribute];
  }

  get attributeTextureFormats() {
    let attributeTextureFormats = this.attributeTextureFormats_;
    if (attributeTextureFormats === undefined) {
      attributeTextureFormats = this.attributeTextureFormats_ =
        spatiallyIndexedSkeletonTextureAttributeSpecs.map(
          ({ dataType, numComponents }) =>
            computeTextureFormat(new TextureFormat(), dataType, numComponents),
        );
    }
    return attributeTextureFormats;
  }

  static encodeSpec(spec: SpatiallyIndexedSkeletonChunkSpecification) {
    const base = SliceViewChunkSource.encodeSpec(spec);
    return { ...base, chunkLayout: spec.chunkLayout.toObject() };
  }

  addChunkListener(listener: SpatiallyIndexedSkeletonChunkListener) {
    this.chunkListeners.add(listener);
    return () => this.chunkListeners.delete(listener);
  }

  addChunk(key: string, chunk: SpatiallyIndexedSkeletonChunk) {
    super.addChunk(key, chunk);
    for (const listener of this.chunkListeners) {
      listener(key, chunk);
    }
  }

  getChunk(chunkData: PackedSkeletonGeometry) {
    return new SpatiallyIndexedSkeletonChunk(this, chunkData);
  }
}

// Options are provided by the SliceView framework for scale selection,
// but spatial skeleton sources expose all grid levels unconditionally.
// TODO (SKM): validate if this is an ok deviation from the SliceView
export const SPATIAL_SKELETON_SOURCE_OPTIONS: SliceViewSourceOptions = {
  displayRank: 0,
  multiscaleToViewTransform: new Float32Array(0),
  modelChannelDimensionIndices: [],
};

export function getSpatialSkeletonCellKeyPrefix(
  position: ArrayLike<number>,
  chunkDataSize: ArrayLike<number>,
) {
  const cell = new Array<number>(3);
  for (let i = 0; i < 3; ++i) {
    const coordinate = Number(position[i]);
    const chunkSize = Number(chunkDataSize[i]);
    if (
      !Number.isFinite(coordinate) ||
      !Number.isFinite(chunkSize) ||
      chunkSize <= 0
    ) {
      return undefined;
    }
    cell[i] = Math.floor(coordinate / chunkSize);
  }
  // A chunk key is the grid position plus the terminator, and nothing else
  // (`getChunk` in `skeleton/backend.ts`), so the whole key IS the prefix. The
  // terminator is what keeps the match exact -- see its docstring.
  return `${cell[0]},${cell[1]},${cell[2]}${SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR}`;
}

export abstract class MultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSliceViewChunkSource<SpatiallyIndexedSkeletonSource> {
  /**
   * When `true`, the segmentation layer enables
   * `autoSpatialSkeletonGridLevel{3d,2d}` on attach: the render layer
   * will overwrite `spatialSkeletonGridResolutionTarget*` every frame
   * from the camera projection.  Default `false` preserves the
   * existing manual-slider UX for sources that haven't opted in
   * (CATMAID).  Subclasses (e.g. zarr-vectors) override to `true`.
   */
  get prefersAutoSpatialSkeletonGridLevel(): boolean {
    return false;
  }

  /**
   * Fragment shader this source's geometry is best drawn with, or `undefined`
   * to keep the built-in `emitDefault()`.
   *
   * The layer applies it as the *default* rather than as a value, so a user's
   * explicit `skeletonRendering.shader` still wins and the state still
   * round-trips — see the consumer in `layer/segmentation/index.ts`. Existing
   * sources (CATMAID) do not override this and are unaffected.
   */
  get defaultFragmentMain(): string | undefined {
    return undefined;
  }

  getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    const sources = this.getSources(SPATIAL_SKELETON_SOURCE_OPTIONS);
    const flattened: SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] =
      [];
    for (const scale of sources) {
      if (scale.length > 0) {
        flattened.push(scale[0]);
      }
    }
    return flattened;
  }

  getSliceViewPanelSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    return this.getPerspectiveSources();
  }

  /**
   * @param liveScale Optional current effective meters-per-model-axis
   *     scale (see `computeDiagonalModelToGlobalMetersScale`), composing
   *     the datasource's own declared native-unit scale with any output
   *     `CoordinateSpaceTransform` the user has applied.  `undefined`
   *     when the caller couldn't reduce the live transform to a
   *     diagonal per-axis scale (rotation/shear) or it isn't available
   *     yet — implementations should fall back to a static scale in
   *     that case.  Subclasses that don't need transform-awareness
   *     (e.g. CATMAID, which never opts into
   *     `prefersAutoSpatialSkeletonGridLevel`) may ignore this.
   */
  getSpatialSkeletonGridSizes(
    liveScale?: Float64Array,
  ): { x: number; y: number; z: number }[] | undefined {
    liveScale;
    return undefined;
  }
}

type SpatiallyIndexedSkeletonSourceEntry =
  SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>;

// TODO (SKM): is all of this really optional?
interface SpatiallyIndexedSkeletonLayerOptions {
  gridLevel?: WatchableValueInterface<number>;
  lod?: WatchableValueInterface<number>;
  gridLevel2d?: WatchableValueInterface<number>;
  lod2d?: WatchableValueInterface<number>;
  /**
   * Which detail-focus mode the layer draws in; see
   * {@link SpatialSkeletonDetailFocus}. Decides whether each visible cell picks
   * its own level or every cell is pinned to the selected one.
   */
  detailFocus?: WatchableValueInterface<number>;
  /**
   * Share of the drawn level's new objects to decode, in [0, 1]; `1` disables
   * per-object admission. See `object_admission.ts`.
   */
  admissionFraction?: WatchableValueInterface<number>;
  sources2d?: SpatiallyIndexedSkeletonSourceEntry[];
  selectedNodeId?: WatchableValueInterface<number | undefined>;
  pendingNodePositionVersion?: WatchableValueInterface<number>;
  getPendingNodePosition?: (nodeId: number) => ArrayLike<number> | undefined;
  getCachedNode?: (nodeId: number) => SpatiallyIndexedSkeletonNode | undefined;
  inspectionState?: SpatiallyIndexedSkeletonInspectionState;
  maxRetainedOverlaySegments?: number;
}

interface SpatiallyIndexedSkeletonInspectionState {
  readonly nodeDataVersion: WatchableValueInterface<number>;
  readonly pendingNodePositionVersion: WatchableValueInterface<number>;
  getCachedSegmentNodes(
    segmentId: number,
  ): readonly SpatiallyIndexedSkeletonNode[] | undefined;
  getFullSegmentNodes(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    segmentId: number,
  ): Promise<readonly SpatiallyIndexedSkeletonNode[]>;
  evictInactiveSegmentNodes(activeSegmentIds: Iterable<number>): void;
}

class SkeletonOverlayChunk implements SkeletonGPUGeometry {
  readonly vertexAttributeTextures: (WebGLTexture | null)[];
  readonly indexBuffer: GLBuffer;
  readonly numIndices: number;
  readonly numVertices: number;
  readonly pickNodeIds: Int32Array;
  readonly pickNodePositions: Float32Array;
  readonly pickSegmentIds: Uint32Array;
  readonly pickEdgeSegmentIds: Uint32Array;
  private readonly nodeIdToVertexIndex: Map<number, number>;
  private readonly selectedFormat: TextureFormat;

  constructor(
    gl: GL,
    geometry: SpatiallyIndexedSkeletonOverlayGeometry,
    formats: TextureFormat[],
  ) {
    const attributeBuffers = [
      new Uint8Array(
        geometry.positions.buffer,
        geometry.positions.byteOffset,
        geometry.positions.byteLength,
      ),
      new Uint8Array(
        geometry.segmentIds.buffer,
        geometry.segmentIds.byteOffset,
        geometry.segmentIds.byteLength,
      ),
      new Uint8Array(
        geometry.selected.buffer,
        geometry.selected.byteOffset,
        geometry.selected.byteLength,
      ),
    ];
    const overlayTextures: (WebGLTexture | null)[] =
      (this.vertexAttributeTextures = []);
    for (let i = 0; i < attributeBuffers.length; i++) {
      const texture = gl.createTexture();
      gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, texture);
      setOneDimensionalTextureData(gl, formats[i], attributeBuffers[i]);
      overlayTextures[i] = texture;
    }
    gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
    this.indexBuffer = GLBuffer.fromData(
      gl,
      geometry.indices,
      WebGL2RenderingContext.ARRAY_BUFFER,
      WebGL2RenderingContext.STATIC_DRAW,
    );
    this.numIndices = geometry.indices.length;
    this.numVertices = geometry.numVertices;
    this.pickNodeIds = geometry.nodeIds;
    // positions and nodePositions were identical — reuse positions for picking.
    this.pickNodePositions = geometry.positions;
    this.pickSegmentIds = geometry.pickSegmentIds;
    this.pickEdgeSegmentIds = geometry.pickEdgeSegmentIds;
    const nodeIdToVertexIndex = new Map<number, number>();
    const { nodeIds } = geometry;
    for (let i = 0; i < nodeIds.length; i++) {
      const nodeId = nodeIds[i];
      if (nodeId > 0) nodeIdToVertexIndex.set(nodeId, i);
    }
    this.nodeIdToVertexIndex = nodeIdToVertexIndex;
    this.selectedFormat = formats[2];
  }

  // Updates the selected-node highlight in-place without a full GPU rebuild.
  // Clears oldNodeId's texel and sets newNodeId's texel.
  updateSelectedNode(
    gl: GL,
    oldNodeId: number | undefined,
    newNodeId: number | undefined,
  ) {
    if (oldNodeId === newNodeId) return;
    const texture = this.vertexAttributeTextures[2];
    if (texture === null) return;
    if (oldNodeId !== undefined) {
      const idx = this.nodeIdToVertexIndex.get(oldNodeId);
      if (idx !== undefined) {
        updateOneDimensionalTextureElement(
          gl,
          texture,
          this.selectedFormat,
          this.numVertices,
          idx,
          OVERLAY_SELECTED_FLOAT_ZERO,
        );
      }
    }
    if (newNodeId !== undefined) {
      const idx = this.nodeIdToVertexIndex.get(newNodeId);
      if (idx !== undefined) {
        updateOneDimensionalTextureElement(
          gl,
          texture,
          this.selectedFormat,
          this.numVertices,
          idx,
          OVERLAY_SELECTED_FLOAT_ONE,
        );
      }
    }
  }

  dispose(gl: GL) {
    for (const texture of this.vertexAttributeTextures) {
      if (texture) gl.deleteTexture(texture);
    }
    this.indexBuffer.dispose();
  }
}

/**
 * Attempts to compute, for each of a multiscale skeleton source's first 3
 * "model" (native/chunk-grid) dimensions, the CURRENT effective scale to
 * real-world meters — composing the datasource's own `modelToRenderLayer
 * Transform` (which reflects any output `CoordinateSpaceTransform` the user
 * has applied, e.g. correcting a source's declared voxel size from mm to
 * µm) with the corresponding global dimension's declared meters-per-unit
 * (`globalScales`, i.e. `coordinateSpace.scales`).
 *
 * Diagonal-only: returns `undefined` (caller should fall back to a static,
 * transform-oblivious scale) when a model dimension is unmapped, when a
 * render-layer dimension mixes contributions from more than one model
 * dimension by more than a small tolerance (rotation/shear — this function
 * does not attempt to reduce a non-axis-aligned transform to a per-axis
 * spacing), or when the corresponding global scale isn't a finite positive
 * number.
 *
 * This exists because `getSpatialSkeletonGridSizes()` previously reported
 * level sizes using only the datasource's own frozen, load-time native-unit
 * scale, which silently went stale relative to the camera-driven resolution
 * target (itself always computed from the live transform) whenever the
 * user edited the layer's output coordinate transform — producing a
 * systematic mismatch proportional to the rescale factor.
 */
export function computeDiagonalModelToGlobalMetersScale(
  transform: RenderLayerTransform,
  globalScales: Float64Array,
): Float64Array | undefined {
  const {
    rank: layerRank,
    localToRenderLayerDimensions,
    globalToRenderLayerDimensions,
    modelToRenderLayerTransform,
  } = transform;
  const result = new Float64Array(3);
  for (let modelDim = 0; modelDim < 3; ++modelDim) {
    const layerDim = localToRenderLayerDimensions[modelDim];
    if (layerDim === -1) return undefined;
    const diagValue =
      modelToRenderLayerTransform[layerDim + (layerRank + 1) * modelDim];
    // Reject if any OTHER model dimension also contributes non-negligibly
    // to this render-layer dimension (rotation/shear) — diagonal-only.
    for (let otherModelDim = 0; otherModelDim < layerRank; ++otherModelDim) {
      if (otherModelDim === modelDim) continue;
      const v =
        modelToRenderLayerTransform[layerDim + (layerRank + 1) * otherModelDim];
      if (Math.abs(v) > Math.abs(diagValue) * 1e-3 + 1e-12) {
        return undefined;
      }
    }
    let globalDim = -1;
    for (let g = 0; g < globalToRenderLayerDimensions.length; ++g) {
      if (globalToRenderLayerDimensions[g] === layerDim) {
        globalDim = g;
        break;
      }
    }
    if (globalDim === -1 || globalDim >= globalScales.length) {
      return undefined;
    }
    const metersPerGlobalUnit = globalScales[globalDim];
    if (!Number.isFinite(metersPerGlobalUnit) || metersPerGlobalUnit <= 0) {
      return undefined;
    }
    result[modelDim] = Math.abs(diagValue) * metersPerGlobalUnit;
  }
  return result;
}

/**
 * Physical units (meters) per screen pixel at `worldPoint`, derived from a
 * model-view-projection matrix and viewport dimensions with axis-aware
 * conversion from world units to meters.
 *
 * This is the same MVP-based formulation used by multiscale mesh LOD, but
 * adjusted for anisotropic coordinate systems (e.g. 4,4,40): each axis scale
 * is converted to pixels-per-meter before taking the max scale factor.
 *
 * Returns `+Infinity` for a behind-camera or invalid `w` so callers fall back
 * to the largest available level.
 */
function computePhysicalUnitsPerScreenPixel(
  modelViewProjection: mat4,
  viewportWidth: number,
  viewportHeight: number,
  worldPoint: Float32Array,
  displayDimensionScales?: Float64Array,
): number {
  const m = modelViewProjection;
  // Column-major mat4 indices.
  const m00 = m[0],
    m10 = m[1];
  const m01 = m[4],
    m11 = m[5];
  const m02 = m[8],
    m12 = m[9];
  const m30 = m[3],
    m31 = m[7],
    m32 = m[11],
    m33 = m[15];
  const w =
    m30 * worldPoint[0] + m31 * worldPoint[1] + m32 * worldPoint[2] + m33;
  if (!Number.isFinite(w) || w <= 0) return Number.POSITIVE_INFINITY;

  const sx =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 0 &&
    Number.isFinite(displayDimensionScales[0]) &&
    displayDimensionScales[0] > 0
      ? displayDimensionScales[0]
      : 1;
  const sy =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 1 &&
    Number.isFinite(displayDimensionScales[1]) &&
    displayDimensionScales[1] > 0
      ? displayDimensionScales[1]
      : sx;
  const sz =
    displayDimensionScales !== undefined &&
    displayDimensionScales.length > 2 &&
    Number.isFinite(displayDimensionScales[2]) &&
    displayDimensionScales[2] > 0
      ? displayDimensionScales[2]
      : sy;

  const xScale = Math.sqrt(
    ((m00 / sx) * viewportWidth) ** 2 + ((m10 / sx) * viewportHeight) ** 2,
  );
  const yScale = Math.sqrt(
    ((m01 / sy) * viewportWidth) ** 2 + ((m11 / sy) * viewportHeight) ** 2,
  );
  const zScale = Math.sqrt(
    ((m02 / sz) * viewportWidth) ** 2 + ((m12 / sz) * viewportHeight) ** 2,
  );
  const scaleFactor = Math.max(xScale, yScale, zScale);
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return w / scaleFactor;
}

/**
 * Multiplier from "world units per screen pixel" to the resolution
 * target the level picker matches against `levels[k].size`.  The picker
 * finds the level whose spacing is closest to the target, so target
 * needs to live in the same magnitude as a level spacing (typical
 * neuroscience streamline chunk = 50–500 µm/mm, while raw world-units-
 * per-pixel is a couple of orders of magnitude smaller for a normal
 * zoom).  200 means "a chunk should be ~200 screen pixels at the
 * matching level" — finest level when chunks are bigger than that on
 * screen, coarser otherwise.  Mirrors the mesh layer's `detailCutoff`
 * tunable in [src/mesh/multiscale.ts:202](src/mesh/multiscale.ts#L202).
 */
const AUTO_SPATIAL_SKELETON_GRID_DETAIL_CUTOFF = 200;
// Limit auto-target movement per update to avoid jumping across multiple
// pyramid levels in a single zoom step (especially in anisotropic spaces).
const AUTO_SPATIAL_SKELETON_GRID_MAX_OCTAVE_STEP = 0.5;

/**
 * If `displayState.autoSpatialSkeletonGridLevel{view}` is enabled,
 * compute the camera-derived "world units per screen pixel" at a
 * representative world position, multiply by the detail-cutoff
 * constant, and write the result to
 * `spatialSkeletonGridResolutionTarget{view}`.  Because
 * `findClosestSpatialSkeletonGridLevelBySpacing` re-runs on every
 * change of that target, the picker (and the histogram widget) snap
 * to the appropriate discrete level for the current zoom.
 *
 * No-op when:
 * - The flag is absent or false (default; CATMAID/legacy behaviour).
 * - The target watchable isn't wired up.
 * - The computed target isn't finite.
 *
 * Reference point: prefer `projectionParameters.globalPosition` (the
 * camera's look-at point in global space).  Fall back to
 * `localPosition` (per-layer position, usually empty for segmentation
 * layers without layer-local dimensions) and finally to the origin.
 * In orthographic projection `w` is independent of position, so the
 * choice only matters for perspective views; either way the helper
 * returns a meaningful value for any of these fallbacks.
 */
/**
 * The grid spacing this layer's memory budget sustains across `visibleCellCount`
 * cells, or `undefined` when the layer publishes no per-cell costs.
 */
function memoryAwareSpacingTarget(
  displayState: SpatiallyIndexedSkeletonLayerDisplayState,
  visibleCellCount: number,
): number | undefined {
  const perCellCost = displayState.spatialSkeletonPerCellCostBytes?.value;
  const budgetBytes = displayState.spatialSkeletonGpuBudgetBytes?.value;
  const levels = displayState.spatialSkeletonGridLevels?.value;
  if (
    perCellCost === undefined ||
    perCellCost.length === 0 ||
    budgetBytes === undefined ||
    levels === undefined
  ) {
    return undefined;
  }
  return targetSpacingForCellBudget(
    levels.map(({ size }) => Math.min(size.x, size.y, size.z)),
    perCellCost,
    visibleCellCount,
    budgetBytes,
  );
}

export function maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
  displayState: SpatiallyIndexedSkeletonLayerDisplayState,
  projectionParameters: {
    viewProjectionMat: mat4;
    width: number;
    height: number;
    globalPosition?: Float32Array;
    displayDimensionRenderInfo?: { displayDimensionScales: Float64Array };
  },
  localPosition: Float32Array,
  view: "2d" | "3d",
  /**
   * Cells the view covers, for the memory-aware target. Omit to keep the
   * pixel-derived target (every non-tract spatially-indexed skeleton layer).
   */
  visibleCellCount?: number,
): void {
  const autoFlag =
    view === "3d"
      ? displayState.autoSpatialSkeletonGridLevel3d
      : displayState.autoSpatialSkeletonGridLevel2d;
  if (autoFlag === undefined || autoFlag.value !== true) return;
  const target =
    view === "3d"
      ? displayState.spatialSkeletonGridResolutionTarget3d
      : displayState.spatialSkeletonGridResolutionTarget2d;
  if (target === undefined) return;
  // Pick the first non-empty reference position.  GlobalPosition is
  // populated for all 3D-view renders even when the segmentation
  // layer has no layer-local dimensions (the common case, where
  // `localPosition` is a zero-length array).
  let reference: Float32Array | undefined;
  if (
    projectionParameters.globalPosition !== undefined &&
    projectionParameters.globalPosition.length >= 3
  ) {
    reference = projectionParameters.globalPosition;
  } else if (localPosition.length >= 3) {
    reference = localPosition;
  } else {
    reference = new Float32Array(3); // origin fallback
  }
  const pixelSize = computePhysicalUnitsPerScreenPixel(
    projectionParameters.viewProjectionMat,
    projectionParameters.width,
    projectionParameters.height,
    reference,
    projectionParameters.displayDimensionRenderInfo?.displayDimensionScales,
  );
  if (!Number.isFinite(pixelSize) || pixelSize <= 0) return;
  // Camera-only target (no user bias yet).
  //
  // Combine the pixel- and memory-derived targets rather than letting one
  // replace the other. The pixel-derived figure is the right DETAIL question
  // for a pyramid whose levels differ in resolution — it is what lets the
  // camera drive the level at all. The memory-derived figure is the right
  // ADMISSIBILITY question for a pyramid whose levels differ in how many
  // whole objects they hold: `pixelSize * CUTOFF` alone saturates there
  // (already at or below the finest level's spacing at a whole-volume view,
  // and only falls further as you zoom in, so the level it selects never
  // changes over the entire useful range — the bug `targetSpacingForCellBudget`
  // was written to fix).
  //
  // Larger target -> a coarser level is picked (see
  // `findClosestSpatialSkeletonGridLevelBySpacing`), and `memoryTarget` is
  // "the finest spacing that still fits `visibleCellCount` cells in budget",
  // or the coarsest available spacing when nothing fits. So the correct
  // combinator is a MAX, applied whenever a memory target exists: it never
  // asks for a level FINER than the budget allows (nothing changes for a
  // sparsity pyramid near or over budget, e.g. the whole-brain tractogram
  // `targetSpacingForCellBudget` exists for — `memoryTarget` there is already
  // large and dominates), while for a pyramid that fits easily at every zoom
  // (a resolution pyramid on a small object, e.g. `object_sparsity: 1.0`
  // throughout) `memoryTarget` is small — at most the finest level's own
  // spacing — so the pixel-derived figure dominates instead and the camera
  // is free to drive the level. No pyramid-kind classification needed: the
  // bound is mathematically safe either way, not merely typical-case safe.
  const memoryTarget =
    visibleCellCount === undefined
      ? undefined
      : memoryAwareSpacingTarget(displayState, visibleCellCount);
  const pixelDriven = pixelSize * AUTO_SPATIAL_SKELETON_GRID_DETAIL_CUTOFF;
  const autoUnbiased =
    memoryTarget === undefined
      ? pixelDriven
      : Math.max(pixelDriven, memoryTarget);

  // The multiplicative detail `bias` (set when the user clicks/drags the
  // widget) lives in the *persistent* displayState so the calibration
  // survives a page refresh; the `lastAuto` value we wrote last frame is
  // transient and stays in a WeakMap keyed by the target.  A manual
  // interaction changes `target.value` away from `lastAuto`; we interpret
  // that as "make this zoom map to the level I picked" and fold it into
  // `bias`, so zoom keeps driving the level afterwards (the click rebiases
  // the offset, it does NOT freeze the target).
  const biasWatchable =
    view === "3d"
      ? displayState.spatialSkeletonGridResolutionBias3d
      : displayState.spatialSkeletonGridResolutionBias2d;
  let st = autoSpatialSkeletonBias.get(target);
  if (st === undefined) {
    st = { lastAuto: undefined };
    autoSpatialSkeletonBias.set(target, st);
  }
  let bias =
    biasWatchable !== undefined &&
    Number.isFinite(biasWatchable.value) &&
    biasWatchable.value > 0
      ? biasWatchable.value
      : 1;
  const cur = target.value as number;
  if (
    st.lastAuto !== undefined &&
    Math.abs(cur - st.lastAuto) > Math.max(st.lastAuto, 1e-30) * 1e-6 &&
    autoUnbiased > 0 &&
    Number.isFinite(cur) &&
    cur > 0
  ) {
    bias = cur / autoUnbiased;
    if (biasWatchable !== undefined && biasWatchable.value !== bias) {
      biasWatchable.value = bias;
    }
  }
  const next = autoUnbiased * bias;
  let stabilizedNext = next;
  if (
    st.lastAuto !== undefined &&
    Number.isFinite(st.lastAuto) &&
    st.lastAuto > 0 &&
    Number.isFinite(stabilizedNext) &&
    stabilizedNext > 0
  ) {
    const maxFactor = 2 ** AUTO_SPATIAL_SKELETON_GRID_MAX_OCTAVE_STEP;
    const upper = st.lastAuto * maxFactor;
    const lower = st.lastAuto / maxFactor;
    if (stabilizedNext > upper) stabilizedNext = upper;
    if (stabilizedNext < lower) stabilizedNext = lower;
  }
  // Only write when the value changes by more than 0.1% — the setter
  // dispatches `changed` unconditionally (level pick → re-attach chain).
  // Leave `lastAuto` untouched on this branch: it must only track the
  // last value actually WRITTEN to `target.value`.  Advancing it here
  // (as this used to do) lets it drift away from `cur` — which stays at
  // its last-written value while we skip the write — by up to just
  // under 0.1% per skipped frame.  That's comfortably past the 1e-6
  // "did the user manually move the widget" threshold above, so nearly
  // every skipped frame got misread as manual interaction on the very
  // next frame, corrupting the persisted `bias` continuously.
  if (
    st.lastAuto !== undefined &&
    Math.abs(cur - stabilizedNext) < Math.max(cur, stabilizedNext) * 1e-3
  ) {
    return;
  }
  target.value = stabilizedNext;
  st.lastAuto = stabilizedNext;
}

// Transient per-resolution-target auto-LOD state: the last auto-written
// target value, used to detect manual widget interaction frame-to-frame.
// The persistent detail `bias` lives in displayState (see
// `spatialSkeletonGridResolutionBias{2d,3d}`), not here.  Keyed weakly by
// the target.
const autoSpatialSkeletonBias = new WeakMap<
  WatchableValueInterface<number>,
  { lastAuto: number | undefined }
>();

export interface SpatiallyIndexedSkeletonLayerDisplayState
  extends SkeletonLayerDisplayState {
  spatialSkeletonGridLevel2d?: WatchableValueInterface<number>;
  spatialSkeletonGridLevel3d?: WatchableValueInterface<number>;
  /**
   * Bytes ONE cell of each pyramid level costs on the GPU, coarsest-first.
   * Drives the memory-aware resolution target; absent for sources that publish
   * no per-level costs, which keeps the pixel-derived target.
   */
  spatialSkeletonPerCellCostBytes?: WatchableValueInterface<number[]>;
  /**
   * The GPU byte budget the layer is sizing against. A watchable rather than a
   * number because render layers hold a spread copy of the display state, in
   * which a plain field would freeze at its value on activation.
   */
  spatialSkeletonGpuBudgetBytes?: WatchableValueInterface<number>;
  skeletonLod?: WatchableValueInterface<number>;
  spatialSkeletonLod2d?: WatchableValueInterface<number>;
  spatialSkeletonGridLevels?: WatchableValueInterface<
    Array<{ size: { x: number; y: number; z: number }; lod: number }>
  >;
  spatialSkeletonGridRenderScaleHistogram2d?: RenderScaleHistogram;
  spatialSkeletonGridRenderScaleHistogram3d?: RenderScaleHistogram;
  /**
   * Optional writable target that the picker is matched against;
   * paired with the `auto*` flags below.  When auto is enabled the
   * render layer overwrites this every frame from the camera
   * projection.
   */
  spatialSkeletonGridResolutionTarget2d?: WatchableValueInterface<number> & {
    value: number;
  };
  spatialSkeletonGridResolutionTarget3d?: WatchableValueInterface<number> & {
    value: number;
  };
  /**
   * Persistent multiplicative detail bias applied on top of the
   * camera-derived auto target (1 = pure camera).  Clicking/dragging the
   * render-scale widget folds the manual offset into this value so the
   * calibration survives a page refresh (the camera-derived target itself
   * is recomputed every frame and is not meaningful to persist).  Paired
   * with the `auto*` flags below.
   */
  spatialSkeletonGridResolutionBias2d?: WatchableValueInterface<number> & {
    value: number;
  };
  spatialSkeletonGridResolutionBias3d?: WatchableValueInterface<number> & {
    value: number;
  };
  /**
   * When `true` the render layer auto-derives `spatialSkeletonGridResolutionTarget*`
   * from the current camera projection at the layer's `localPosition`
   * (world units per screen pixel).  Default off; opt in to get
   * camera-driven LOD switching for spatially-indexed skeletons.
   */
  autoSpatialSkeletonGridLevel2d?: WatchableValueInterface<boolean>;
  autoSpatialSkeletonGridLevel3d?: WatchableValueInterface<boolean>;
}

/**
 * Resolve a picked node/edge offset to its owning segment as a full uint64
 * `bigint`.  `segmentIds` is the interleaved per-vertex segment column with
 * `segmentComponents` uint32 per vertex (2 = `[lo, hi]` full uint64; 1 = a
 * uint32 id with implicit high half 0).  Returns `undefined` for an absent /
 * zero id.
 */
export function resolveSpatiallyIndexedSkeletonSegmentPick(
  chunk: { indices: Uint32Array; numVertices: number },
  segmentIds: Uint32Array,
  pickedOffset: number,
  kind: "node" | "edge" | "face",
  segmentComponents = 1,
): bigint | undefined {
  const readId = (vertex: number): bigint | undefined => {
    const base = vertex * segmentComponents;
    if (vertex < 0 || base + segmentComponents > segmentIds.length) {
      return undefined;
    }
    const lo = BigInt(segmentIds[base] >>> 0);
    const hi = segmentComponents >= 2 ? BigInt(segmentIds[base + 1] >>> 0) : 0n;
    const id = lo | (hi << 32n);
    return id > 0n ? id : undefined;
  };
  if (pickedOffset < 0) return undefined;
  if (kind === "node") {
    if (pickedOffset >= chunk.numVertices) return undefined;
    return readId(pickedOffset);
  }
  // The picked primitive spans this many indices: an edge two, a triangle
  // three. Any of its corners can answer for the whole primitive -- they belong
  // to one object -- so take the first that has an id, which is also what makes
  // a bridge edge to a ghost vertex resolvable.
  const verticesPerPrimitive = kind === "face" ? 3 : 2;
  const indexOffset = pickedOffset * verticesPerPrimitive;
  if (indexOffset + verticesPerPrimitive > chunk.indices.length) {
    return undefined;
  }
  for (let i = 0; i < verticesPerPrimitive; ++i) {
    const id = readId(chunk.indices[indexOffset + i]);
    if (id !== undefined) return id;
  }
  return undefined;
}

export class SpatiallyIndexedSkeletonLayer
  extends RefCounted
  implements SkeletonShaderContext
{
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex: number | undefined;
  selectedNodeAttributeIndex: number | undefined;
  /** See {@link PackedAttributeRange}; set by sources that pack. */
  packedAttributeRange: PackedAttributeRange | undefined;
  /** What primitive this source's geometry is drawn as; see the constructor. */
  readonly geometryPrimitive: GeometryPrimitive;
  readonly browsePassLayerView: SkeletonShaderContext;
  readonly skeletonShaderParameters: WatchableValue<SkeletonShaderParameters>;
  readonly browsePassSkeletonShaderParameters: WatchableValueInterface<SkeletonShaderParameters>;
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );
  backend: ChunkRenderLayerFrontend;
  localPosition: WatchableValueInterface<Float32Array>;
  readonly chunkTransform: WatchableValueInterface<
    ValueOrError<ChunkTransformParameters>
  >;
  rpc: RPC | undefined;

  private overlayAttributeTextureFormats_?: TextureFormat[];
  private get overlayAttributeTextureFormats(): TextureFormat[] {
    return (this.overlayAttributeTextureFormats_ ??= this.vertexAttributes.map(
      ({ dataType, numComponents }) =>
        computeTextureFormat(new TextureFormat(), dataType, numComponents),
    ));
  }
  gridLevel: WatchableValueInterface<number>;
  lod: WatchableValueInterface<number>;
  gridLevel2d: WatchableValueInterface<number>;
  lod2d: WatchableValueInterface<number>;
  detailFocus: WatchableValueInterface<number>;
  admissionFraction: WatchableValueInterface<number>;
  private selectedNodeId:
    | WatchableValueInterface<number | undefined>
    | undefined;
  private pendingNodePositionVersion:
    | WatchableValueInterface<number>
    | undefined;
  private getPendingNodePositionOverride:
    | ((nodeId: number) => ArrayLike<number> | undefined)
    | undefined;
  private getCachedNodeInfo:
    | ((nodeId: number) => SpatiallyIndexedSkeletonNode | undefined)
    | undefined;
  private inspectionState: SpatiallyIndexedSkeletonInspectionState | undefined;
  private overlayChunk: SkeletonOverlayChunk | undefined;
  private overlayChunkKey: string | undefined;
  private overlayGeometryKey: string | undefined;
  private cachedSelectedNodeId: number | undefined;
  private overlayRebuildFrame = -1;
  private pendingOverlaySegmentLoads = new Set<number>();
  // Segments whose `getFullSegmentNodes` call rejected (e.g. the active
  // data source doesn't implement full-skeleton inspection at all, as is
  // the case for every zarr-vectors layer today). Without this, a
  // permanently-failing fetch is retried on every redraw forever: each
  // retry's `.finally()` disposes the overlay chunk and dispatches
  // another redraw, which triggers another retry — an infinite loop that
  // starves the render loop the moment any segment becomes visible.
  // Evicted alongside the success cache in `resolveSourceBackedOverlayChunk`
  // so a segment that becomes active again after going inactive gets one
  // more attempt, in case the earlier failure was transient.
  private failedOverlaySegmentLoads = new Set<number>();
  private browseExcludedSegments = new Uint64Set();
  private gpuBrowseExcludedSegmentsHashTable: GPUHashTable<HashSetUint64>;
  private browseExcludedSegmentsKey: string | undefined;
  private suppressedBrowseSegmentIds = new Set<number>();
  private retainedOverlaySegmentIds: number[] = [];
  private maxRetainedOverlaySegments: number;

  private disposeOverlayChunk() {
    this.overlayChunk?.dispose(this.gl);
    this.overlayChunk = undefined;
    this.overlayChunkKey = undefined;
    this.overlayGeometryKey = undefined;
    this.cachedSelectedNodeId = undefined;
  }

  private requestOverlaySegmentLoad(segmentId: number) {
    if (
      this.inspectionState === undefined ||
      this.pendingOverlaySegmentLoads.has(segmentId) ||
      this.failedOverlaySegmentLoads.has(segmentId)
    ) {
      return;
    }
    this.pendingOverlaySegmentLoads.add(segmentId);
    let failed = false;
    void this.inspectionState
      .getFullSegmentNodes(this, segmentId)
      .catch(() => {
        failed = true;
      })
      .finally(() => {
        this.pendingOverlaySegmentLoads.delete(segmentId);
        if (failed) {
          this.failedOverlaySegmentLoads.add(segmentId);
        }
        this.disposeOverlayChunk();
        this.redrawNeeded.dispatch();
      });
  }

  private getOverlayGeometryKey(segmentIds: readonly number[]) {
    return [
      segmentIds.join(","),
      `pending:${this.pendingNodePositionVersion?.value ?? ""}`,
      `data:${this.inspectionState?.nodeDataVersion.value ?? ""}`,
    ].join("|");
  }

  private getActiveEditableSegmentIds() {
    const segments = getVisibleSegments(
      this.displayState.segmentationGroupState.value,
    );
    const segmentIds: number[] = [];
    for (const segmentId of segments.keys()) {
      const normalizedSegmentId = Number(segmentId);
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      segmentIds.push(normalizedSegmentId);
    }
    segmentIds.sort((a, b) => a - b);
    return segmentIds;
  }

  getRetainedOverlaySegmentIds() {
    return this.retainedOverlaySegmentIds;
  }

  retainOverlaySegment(segmentId: number) {
    const nextRetainedOverlaySegmentIds =
      retainSpatiallyIndexedSkeletonOverlaySegment(
        this.retainedOverlaySegmentIds,
        segmentId,
        { maxRetained: this.maxRetainedOverlaySegments },
      );
    if (
      nextRetainedOverlaySegmentIds.length ===
        this.retainedOverlaySegmentIds.length &&
      nextRetainedOverlaySegmentIds.every(
        (candidateSegmentId, index) =>
          candidateSegmentId === this.retainedOverlaySegmentIds[index],
      )
    ) {
      return false;
    }
    this.retainedOverlaySegmentIds = nextRetainedOverlaySegmentIds;
    this.redrawNeeded.dispatch();
    return true;
  }

  suppressBrowseSegment(segmentId: number) {
    const normalizedSegmentId = Math.round(Number(segmentId));
    if (
      !Number.isSafeInteger(normalizedSegmentId) ||
      normalizedSegmentId <= 0 ||
      this.suppressedBrowseSegmentIds.has(normalizedSegmentId)
    ) {
      return false;
    }
    this.suppressedBrowseSegmentIds.add(normalizedSegmentId);
    this.redrawNeeded.dispatch();
    return true;
  }

  private getOverlayRenderSegmentIds() {
    return mergeSpatiallyIndexedSkeletonOverlaySegmentIds(
      this.getActiveEditableSegmentIds(),
      this.retainedOverlaySegmentIds,
    );
  }

  private getLoadedOverlaySegmentIds(
    segmentIds: readonly number[] = this.getOverlayRenderSegmentIds(),
  ) {
    if (this.inspectionState === undefined) {
      return [];
    }
    return segmentIds.filter(
      (segmentId) =>
        this.inspectionState?.getCachedSegmentNodes(segmentId) !== undefined,
    );
  }

  private getNormalizedBrowsePassExcludedSegmentIds() {
    const segmentIds = new Set<number>();
    for (const segmentId of this.getLoadedOverlaySegmentIds()) {
      const normalizedSegmentId = Math.round(Number(segmentId));
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      segmentIds.add(normalizedSegmentId);
    }
    for (const segmentId of this.suppressedBrowseSegmentIds) {
      const normalizedSegmentId = Math.round(Number(segmentId));
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      segmentIds.add(normalizedSegmentId);
    }
    return [...segmentIds].sort((a, b) => a - b);
  }

  private getBrowsePassExcludedSegments() {
    const segmentIds = this.getNormalizedBrowsePassExcludedSegmentIds();
    if (segmentIds.length === 0) {
      if (this.browseExcludedSegments.size !== 0) {
        this.browseExcludedSegments.clear();
      }
      this.browseExcludedSegmentsKey = undefined;
      return undefined;
    }
    const excludedSegmentsKey = segmentIds.join(",");
    if (this.browseExcludedSegmentsKey !== excludedSegmentsKey) {
      this.browseExcludedSegments.clear();
      this.browseExcludedSegments.add(
        segmentIds
          .filter(
            (segmentId) => Number.isSafeInteger(segmentId) && segmentId > 0,
          )
          .map((segmentId) => BigInt(segmentId)),
      );
      this.browseExcludedSegmentsKey = excludedSegmentsKey;
    }
    return this.browseExcludedSegments;
  }

  private resolveSourceBackedOverlayChunk(): SkeletonOverlayChunk | undefined {
    const frameNumber =
      this.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
    // Cache result for the entire frame — both slice and perspective draw calls
    // share the same chunk, and "no overlay" is also cached to avoid per-frame
    // allocation when the inspection overlay is inactive.
    if (this.overlayRebuildFrame === frameNumber) {
      return this.overlayChunk;
    }
    this.overlayRebuildFrame = frameNumber;
    if (this.inspectionState === undefined) {
      this.disposeOverlayChunk();
      return undefined;
    }
    const overlaySegmentIds = this.getOverlayRenderSegmentIds();
    if (overlaySegmentIds.length === 0) {
      this.disposeOverlayChunk();
      return undefined;
    }
    this.inspectionState.evictInactiveSegmentNodes(overlaySegmentIds);
    if (this.failedOverlaySegmentLoads.size > 0) {
      const activeSegmentIds = new Set(overlaySegmentIds);
      for (const segmentId of this.failedOverlaySegmentLoads) {
        if (!activeSegmentIds.has(segmentId)) {
          this.failedOverlaySegmentLoads.delete(segmentId);
        }
      }
    }

    // Pass 1: cheap scan to determine which segments are loaded and check cache.
    const loadedSegmentIds: number[] = [];
    for (const segmentId of overlaySegmentIds) {
      if (this.inspectionState.getCachedSegmentNodes(segmentId) !== undefined) {
        loadedSegmentIds.push(segmentId);
      } else {
        this.requestOverlaySegmentLoad(segmentId);
      }
    }
    if (loadedSegmentIds.length === 0) {
      this.disposeOverlayChunk();
      return undefined;
    }

    const overlayGeometryKey = this.getOverlayGeometryKey(loadedSegmentIds);
    const selectedNodeId = this.selectedNodeId?.value;
    const overlayChunkKey = `${overlayGeometryKey}|selected:${selectedNodeId ?? ""}`;

    if (this.overlayChunk !== undefined) {
      if (this.overlayGeometryKey === overlayGeometryKey) {
        // Geometry unchanged — update only the selected-node highlight in-place
        // rather than reallocating all GPU textures.
        if (this.overlayChunkKey !== overlayChunkKey) {
          this.overlayChunk.updateSelectedNode(
            this.gl,
            this.cachedSelectedNodeId,
            selectedNodeId,
          );
          this.overlayChunkKey = overlayChunkKey;
          this.cachedSelectedNodeId = selectedNodeId;
        }
        return this.overlayChunk;
      }
    }

    // Pass 2: geometry cache miss — collect node sets and rebuild.
    const segmentNodeSets: (readonly SpatiallyIndexedSkeletonNode[])[] = [];
    for (const segmentId of loadedSegmentIds) {
      const segmentNodes =
        this.inspectionState.getCachedSegmentNodes(segmentId);
      if (segmentNodes !== undefined) {
        segmentNodeSets.push(segmentNodes);
      }
    }
    this.disposeOverlayChunk();
    const geometry = buildSpatiallyIndexedSkeletonOverlayGeometry(
      segmentNodeSets,
      {
        selectedNodeId,
        getPendingNodePosition: this.getPendingNodePositionOverride,
      },
    );
    this.overlayChunk = new SkeletonOverlayChunk(
      this.gl,
      geometry,
      this.overlayAttributeTextureFormats,
    );
    this.overlayChunkKey = overlayChunkKey;
    this.overlayGeometryKey = overlayGeometryKey;
    this.cachedSelectedNodeId = selectedNodeId;
    return this.overlayChunk;
  }

  sources: SpatiallyIndexedSkeletonSourceEntry[];
  sources2d: SpatiallyIndexedSkeletonSourceEntry[];
  source: SpatiallyIndexedSkeletonSource;

  constructor(
    public chunkManager: ChunkManager,
    sources:
      | SpatiallyIndexedSkeletonSourceEntry[]
      | SpatiallyIndexedSkeletonSource,
    public displayState: SpatiallyIndexedSkeletonLayerDisplayState & {
      localPosition: WatchableValueInterface<Float32Array>;
    },
    options: SpatiallyIndexedSkeletonLayerOptions = {},
  ) {
    super();
    this.registerDisposer(() => {
      this.disposeOverlayChunk();
    });
    let sources3d: SpatiallyIndexedSkeletonSourceEntry[];
    let sources2d = options.sources2d ?? [];
    if (Array.isArray(sources)) {
      sources3d = sources;
    } else {
      sources3d = [
        {
          chunkSource: sources,
          chunkToMultiscaleTransform: mat4.create(),
        },
      ];
    }
    if (sources3d.length === 0 && sources2d.length > 0) {
      sources3d = sources2d;
    }
    if (sources2d.length === 0) {
      sources2d = sources3d;
    }
    if (sources3d.length === 0) {
      throw new Error(
        "SpatiallyIndexedSkeletonLayer requires at least one source.",
      );
    }
    this.sources = sources3d;
    this.sources2d = sources2d;
    this.source = sources3d[0].chunkSource;
    this.localPosition = displayState.localPosition;
    this.chunkTransform = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (modelTransform) =>
          makeValueOrError(() =>
            getChunkTransformParameters(valueOrThrow(modelTransform)),
          ),
        this.displayState.transform,
      ),
    );
    this.gridLevel =
      options.gridLevel ??
      displayState.spatialSkeletonGridLevel3d ??
      new WatchableValue(0);
    this.lod = options.lod ?? displayState.skeletonLod ?? new WatchableValue(0);
    this.gridLevel2d =
      options.gridLevel2d ??
      displayState.spatialSkeletonGridLevel2d ??
      this.gridLevel;
    this.lod2d = options.lod2d ?? displayState.spatialSkeletonLod2d ?? this.lod;
    // A source that names no mode gets LOCAL, which is the behaviour every
    // spatially-indexed skeleton layer had before the mode existed.
    this.detailFocus =
      options.detailFocus ??
      new WatchableValue<number>(SpatialSkeletonDetailFocus.LOCAL);
    this.admissionFraction =
      options.admissionFraction ?? new WatchableValue<number>(1);
    this.selectedNodeId = options.selectedNodeId;
    this.pendingNodePositionVersion = options.pendingNodePositionVersion;
    this.getPendingNodePositionOverride = options.getPendingNodePosition;
    this.getCachedNodeInfo = options.getCachedNode;
    this.inspectionState = options.inspectionState;
    this.maxRetainedOverlaySegments = Math.max(
      1,
      Math.round(
        options.maxRetainedOverlaySegments ??
          DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS,
      ),
    );
    registerRedrawWhenSegmentationDisplayState3DChanged(displayState, this);
    this.displayState.shaderError.value = undefined;
    const { skeletonRenderingOptions: renderingOptions } = displayState;
    this.registerDisposer(
      renderingOptions.shader.changed.add(() => {
        this.displayState.shaderError.value = undefined;
        this.redrawNeeded.dispatch();
      }),
    );

    this.vertexAttributes = [
      ...this.source.vertexAttributes,
      selectedNodeAttribute,
    ];
    // Indices are into the source's list, and this only appends, so they carry
    // over unchanged.
    this.packedAttributeRange = (
      this.source as { packedAttributeRange?: PackedAttributeRange }
    ).packedAttributeRange;
    // What the geometry IS, as opposed to how the user prefers to see it. A
    // zarr-vectors point cloud has no edges and a zarr-vectors mesh has no line
    // segments, so neither can honour the lines/points preference. Structural
    // read, like `isRoiHighDetailSource` below -- other datasources sharing this
    // class declare nothing and keep the `"lines"` default.
    this.geometryPrimitive =
      (this.source as { geometryPrimitive?: GeometryPrimitive })
        .geometryPrimitive ?? "lines";
    // Constant for the layer's lifetime: the passing set is attached at layer
    // creation for zarr-vectors tract layers and never for others.
    const hasRoiFilter = this.displayState.roiPassingSegments !== undefined;
    const hasRoiSegmentColors =
      this.displayState.roiSegmentColors !== undefined;
    // The pass-2 layer runs this same shader over the same display state, so the
    // hide tier must be compiled ONLY into pass 1 -- otherwise pass 2 would hide
    // exactly the tracts it exists to draw, and nothing would render at all.
    // The source is what distinguishes them.
    const isRoiHighDetailLayer =
      (this.source as { isRoiHighDetailSource?: boolean })
        .isRoiHighDetailSource === true;
    const hasRoiHighDetailHide =
      hasRoiFilter &&
      !isRoiHighDetailLayer &&
      this.displayState.roiHighDetailSegments !== undefined;
    // Background per-object value tier: pass 1 only (see hasRoiHighDetailHide;
    // the background tracts it governs are the coarse bulk pass 2 never draws).
    const hasRoiObjectValues =
      this.displayState.roiObjectValues !== undefined && !isRoiHighDetailLayer;
    this.skeletonShaderParameters =
      new WatchableValue<SkeletonShaderParameters>({
        dynamicSegmentAppearance: true,
        hasRoiFilter,
        hasRoiHighDetailHide,
        hasRoiSegmentColors,
        hasRoiObjectValues,
        hasSegmentStatedColors: false,
        hasSegmentDefaultColor: false,
        hoverHighlight: false,
        spatialChunkCulling: false,
      });
    const updateSkeletonShaderParameters = () => {
      const colorGroupState =
        this.displayState.segmentationColorGroupState.value;
      this.skeletonShaderParameters.value = {
        dynamicSegmentAppearance: true,
        hasRoiFilter,
        hasRoiHighDetailHide,
        hasRoiSegmentColors,
        hasRoiObjectValues,
        hasSegmentStatedColors: colorGroupState.segmentStatedColors.size !== 0,
        hasSegmentDefaultColor:
          colorGroupState.segmentDefaultColor.value !== undefined ||
          DEBUG_SPATIAL_SKELETON_CHUNKS,
        hoverHighlight: this.displayState.hoverHighlight.value,
        spatialChunkCulling: false,
      };
    };
    this.registerDisposer(
      registerNested((context, colorGroupState) => {
        context.registerDisposer(
          colorGroupState.segmentStatedColors.changed.add(
            updateSkeletonShaderParameters,
          ),
        );
        context.registerDisposer(
          colorGroupState.segmentDefaultColor.changed.add(
            updateSkeletonShaderParameters,
          ),
        );
        updateSkeletonShaderParameters();
      }, this.displayState.segmentationColorGroupState),
    );
    this.registerDisposer(
      this.displayState.hoverHighlight.changed.add(
        updateSkeletonShaderParameters,
      ),
    );
    this.browsePassSkeletonShaderParameters = this.registerDisposer(
      makeCachedLazyDerivedWatchableValue(
        (params) => ({ ...params, spatialChunkCulling: true }),
        this.skeletonShaderParameters,
      ),
    );

    // Browse pass uses uniform-based dynamic segment color (not per-vertex attribute),
    // so segmentColorAttributeIndex is intentionally undefined here.
    this.browsePassLayerView = {
      vertexAttributes: this.source.vertexAttributes,
      segmentColorAttributeIndex: undefined,
      packedAttributeRange: this.packedAttributeRange,
      gl: this.gl,
      fallbackShaderParameters: this.fallbackShaderParameters,
      displayState: this.displayState,
      skeletonShaderParameters: this.browsePassSkeletonShaderParameters,
    };
    const selectedNodeIndex = this.vertexAttributes.findIndex(
      (x) => x.name === selectedNodeAttribute.name,
    );
    this.selectedNodeAttributeIndex =
      selectedNodeIndex >= 0 ? selectedNodeIndex : undefined;
    const requestRedraw = () => this.redrawNeeded.dispatch();
    // The mode changes which source each visible cell draws from, so a change
    // to it must repaint even though nothing about the camera moved.
    if (this.detailFocus.changed) {
      this.registerDisposer(this.detailFocus.changed.add(requestRedraw));
    }
    // A change to the admission fraction changes what each chunk CONTAINS, not
    // merely which chunk is drawn, so already-decoded chunks are wrong and must
    // go. Dropping them beats keying them by fraction: an invalidated chunk is
    // freed, whereas a re-keyed one lingers in the budget under a dead key until
    // something outbids it — the trap the `lod` suffix used to set.
    if (this.admissionFraction.changed) {
      this.registerDisposer(
        this.admissionFraction.changed.add(() => {
          for (const entries of [this.sources, this.sources2d]) {
            for (const entry of entries) entry.chunkSource.invalidateCache();
          }
          requestRedraw();
        }),
      );
    }
    const selectedNodeWatchable = this.selectedNodeId;
    if (selectedNodeWatchable?.changed) {
      this.registerDisposer(selectedNodeWatchable.changed.add(requestRedraw));
    }
    const pendingNodePositionVersion = options.pendingNodePositionVersion;
    if (pendingNodePositionVersion?.changed) {
      this.registerDisposer(
        pendingNodePositionVersion.changed.add(requestRedraw),
      );
    }
    const inspectionState = this.inspectionState;
    if (inspectionState !== undefined) {
      this.registerDisposer(
        inspectionState.nodeDataVersion.changed.add(() => {
          this.redrawNeeded.dispatch();
        }),
      );
    }
    // Create backend for perspective view chunk management
    const sharedObject = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    const rpc = chunkManager.rpc!;
    this.rpc = rpc;
    sharedObject.RPC_TYPE_ID = SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID;

    const renderScaleTargetWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(
        rpc,
        displayState.renderScaleTarget,
      ),
    );

    const skeletonLodWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.lod),
    );

    const skeletonGridLevelWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.gridLevel),
    );

    const skeletonLod2dWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.lod2d),
    );

    const skeletonGridLevel2dWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.gridLevel2d),
    );

    const skeletonDetailFocusWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.detailFocus),
    );

    const skeletonAdmissionFractionWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.admissionFraction),
    );

    // Per-level spacing in metres, indexed by grid level. The backend's
    // arbitration needs the same density-corrected figure this side uses; see
    // `getLevelSpacingsMeters`.
    const skeletonLevelSpacingsMetersWatchable = this.registerDisposer(
      SharedWatchableValue.make<number[]>(rpc, this.getLevelSpacingsMeters()),
    );
    const gridLevels = displayState.spatialSkeletonGridLevels;
    if (gridLevels?.changed !== undefined) {
      this.registerDisposer(
        gridLevels.changed.add(() => {
          skeletonLevelSpacingsMetersWatchable.value =
            this.getLevelSpacingsMeters();
        }),
      );
    }

    const skeletonGridResolutionTarget3dWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(
        rpc,
        displayState.spatialSkeletonGridResolutionTarget3d!,
      ),
    );

    const counterpartOptions: { [key: string]: any } = {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.localPosition),
      ).rpcId,
      renderScaleTarget: renderScaleTargetWatchable.rpcId,
      skeletonLod: skeletonLodWatchable.rpcId,
      skeletonGridLevel: skeletonGridLevelWatchable.rpcId,
      skeletonLod2d: skeletonLod2dWatchable.rpcId,
      skeletonGridLevel2d: skeletonGridLevel2dWatchable.rpcId,
      skeletonDetailFocus: skeletonDetailFocusWatchable.rpcId,
      skeletonAdmissionFraction: skeletonAdmissionFractionWatchable.rpcId,
      skeletonLevelSpacingsMeters: skeletonLevelSpacingsMetersWatchable.rpcId,
      skeletonGridResolutionTarget3d:
        skeletonGridResolutionTarget3dWatchable.rpcId,
    };

    // ROI streamline filter channel (zarr-vectors tract layers only): hand the
    // backend the ROI groups (so it recomputes the passing set) and the shared
    // passing set it mutates. `roiPassingSegments` is already a shared object
    // (it drives this layer's shader too); `roiGroups` is wrapped from the plain
    // display-state watchable the way `skeletonLod` is. The active flag and
    // ghost opacity are NOT sent to the backend — they only feed shader uniforms
    // here (the backend keeps the passing set current whenever ROIs exist, so
    // enabling the filter is instant). Redraw when the passing set, the active
    // flag, or the ghost opacity changes.
    if (
      displayState.roiPassingSegments !== undefined &&
      displayState.roiGroups !== undefined &&
      displayState.roiFilterActive !== undefined
    ) {
      counterpartOptions.roiPassingSegments =
        displayState.roiPassingSegments.rpcId;
      if (displayState.roiSegmentColors !== undefined) {
        counterpartOptions.roiSegmentColors =
          displayState.roiSegmentColors.rpcId;
      }
      counterpartOptions.roiGroups = this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, displayState.roiGroups),
      ).rpcId;
      if (displayState.roiObjectAttrColumns !== undefined) {
        counterpartOptions.roiObjectAttrColumns = this.registerDisposer(
          SharedWatchableValue.makeFromExisting(
            rpc,
            displayState.roiObjectAttrColumns,
          ),
        ).rpcId;
      }
      if (displayState.roiLabelField !== undefined) {
        // Shared, not snapshotted: the dense parcellation grid arrives
        // asynchronously (and can change if the linked layer changes), so the
        // backend must see the value settle rather than the initial undefined.
        counterpartOptions.roiLabelField = this.registerDisposer(
          SharedWatchableValue.makeFromExisting(
            rpc,
            displayState.roiLabelField,
          ),
        ).rpcId;
      }
      const roiRedraw = () => this.redrawNeeded.dispatch();
      this.registerDisposer(
        displayState.roiPassingSegments.changed.add(roiRedraw),
      );
      this.registerDisposer(
        displayState.roiFilterActive.changed.add(roiRedraw),
      );
      if (displayState.roiGhostAlpha !== undefined) {
        this.registerDisposer(
          displayState.roiGhostAlpha.changed.add(roiRedraw),
        );
      }
      if (displayState.roiSegmentColors !== undefined) {
        this.registerDisposer(
          displayState.roiSegmentColors.changed.add(roiRedraw),
        );
      }
      if (displayState.roiColorByGroup !== undefined) {
        this.registerDisposer(
          displayState.roiColorByGroup.changed.add(roiRedraw),
        );
      }
      if (displayState.roiObjectValues !== undefined) {
        this.registerDisposer(
          displayState.roiObjectValues.changed.add(roiRedraw),
        );
      }
      if (displayState.roiBackground !== undefined) {
        this.registerDisposer(
          displayState.roiBackground.changed.add(roiRedraw),
        );
      }
      if (displayState.roiHighDetailSegments !== undefined) {
        // Pass 1's hide tier reads this set, but it is not pass 1's visible set,
        // so no segmentation-state redraw covers it. Without this, tracts pass 2
        // has taken over keep drawing here until something else forces a frame --
        // which shows as every newly claimed tract briefly drawn twice.
        this.registerDisposer(
          displayState.roiHighDetailSegments.changed.add(roiRedraw),
        );
      }
    }

    sharedObject.initializeCounterpart(rpc, counterpartOptions);
    this.backend = sharedObject;
    this.gpuBrowseExcludedSegmentsHashTable = this.registerDisposer(
      GPUHashTable.get(this.gl, this.browseExcludedSegments.hashTable),
    );
  }

  /**
   * Ask the backend for each group's on-screen passing object ids -- the tract
   * export selection. Positional: `result[i]` corresponds to `groups[i]`. A
   * request/response RPC (the tab needs the ids before it can build the export
   * job); see the backend's `computeRoiExportIds`.
   */
  async computeRoiExportIds(
    groups: readonly RoiGroupConfig[],
  ): Promise<bigint[][]> {
    const rpc = this.rpc;
    if (rpc === undefined) {
      throw new Error("This tract layer is not connected to a worker.");
    }
    const { perGroup } = await rpc.promiseInvoke<{ perGroup: string[][] }>(
      SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_ROI_EXPORT_IDS_RPC_ID,
      { layer: this.backend.rpcId, groups },
    );
    return perGroup.map((ids) => ids.map((s) => BigInt(s)));
  }

  /**
   * Ask the backend for the observed range of each named per-vertex attribute
   * over the resident chunks -- what the Filter tab's attribute picker needs to
   * offer a meaningful control. Positional: `result[i]` ↔ `names[i]`. The
   * values exist only in the worker, hence the round trip.
   */
  async computeRoiVertexAttrStats(
    names: readonly string[],
  ): Promise<VertexAttrStats[]> {
    const rpc = this.rpc;
    if (rpc === undefined) {
      throw new Error("This geometry layer is not connected to a worker.");
    }
    const { stats } = await rpc.promiseInvoke<{ stats: VertexAttrStats[] }>(
      SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_VERTEX_ATTR_STATS_RPC_ID,
      { layer: this.backend.rpcId, names: [...names] },
    );
    return stats;
  }

  get gl() {
    return this.chunkManager.chunkQueueManager.gl;
  }

  getSources(view: SpatiallyIndexedSkeletonView) {
    return view === "2d" ? this.sources2d : this.sources;
  }

  /**
   * Whether OBJECT focus may draw the UNION of levels for this view.
   *
   * Object focus has two halves, and only one of them needs the store's help.
   * Drawing one level everywhere instead of arbitrating per cell -- so an
   * object is never cut off where the finer chunks ran out -- works on any
   * pyramid. Drawing several levels at once only works where they partition the
   * objects between them; on a resolution pyramid (mesh and point-cloud stores,
   * whose levels are decimated copies of every object) the union superimposes
   * those copies. So the union is gated here and the rest of object focus is
   * not.
   */
  objectPartitionAvailable(view: SpatiallyIndexedSkeletonView) {
    const sources = this.getSources(view);
    return (
      sources.length > 0 &&
      sources.every((entry) =>
        getSpatiallyIndexedSkeletonPartitionsObjects(entry.chunkSource),
      )
    );
  }

  private selectSourcesForViewAndGrid(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
  ) {
    return selectSpatiallyIndexedSkeletonEntriesForView(
      this.getSources(view),
      view,
      gridLevel,
      getSpatiallyIndexedSkeletonSourceView,
      getSpatiallyIndexedSkeletonGridIndex,
    );
  }

  private selectSourcesForViewAndGridWithFallback(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
  ) {
    return selectSpatiallyIndexedSkeletonEntriesForViewWithFallback(
      this.getSources(view),
      view,
      gridLevel,
      getSpatiallyIndexedSkeletonSourceView,
      getSpatiallyIndexedSkeletonGridIndex,
    );
  }

  /**
   * Per-level chunk spacing in METRES, indexed by grid level (coarsest first,
   * which is what `gridIndex` counts).
   *
   * Read from the same `spatialSkeletonGridLevels` list the resolution widget
   * and the camera-driven target use, so all three agree on how coarse a level
   * is. For an object-sparsity pyramid that figure is density-corrected -- mean
   * spacing between OBJECTS -- and is the only thing that distinguishes the
   * levels at all, since they share a chunk shape.
   *
   * `[]` where the display state publishes no levels, which sends every caller
   * back to the chunk-shape spacing they used before.
   */
  private getLevelSpacingsMeters(): number[] {
    const levels = this.displayState.spatialSkeletonGridLevels?.value;
    if (levels === undefined) return [];
    return levels.map(({ size }) =>
      Math.max(Math.min(size.x, size.y, size.z), 0),
    );
  }

  /**
   * How many grid cells the view covers, counted on the COARSEST level's grid.
   *
   * Deliberately not the level currently being drawn: the count feeds the
   * memory-aware resolution target, which then decides which level to draw, and
   * counting on the drawn level would close that loop into a feedback cycle. The
   * coarsest grid is a stable reference, and for an object-sparsity pyramid
   * (where every level shares one chunk_shape) it is the same count anyway.
   */
  countVisibleCells(
    view: SpatiallyIndexedSkeletonView,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
  ): number {
    let coarsest: TransformedSource | undefined;
    let coarsestGridIndex = Number.POSITIVE_INFINITY;
    const wanted = new Set(
      this.getSources(view).map((e) => getObjectId(e.chunkSource)),
    );
    for (const scales of transformedSources) {
      for (const tsource of scales) {
        if (!wanted.has(getObjectId(tsource.source))) continue;
        // Grid index 0 is the coarsest level, so the smallest wins.
        const gridIndex =
          getSpatiallyIndexedSkeletonGridIndex(tsource.source) ?? 0;
        if (gridIndex < coarsestGridIndex) {
          coarsestGridIndex = gridIndex;
          coarsest = tsource;
        }
      }
    }
    if (coarsest === undefined) return 0;
    let count = 0;
    forEachVisibleVolumetricChunk(
      projectionParameters,
      this.localPosition.value,
      coarsest,
      () => {
        ++count;
      },
    );
    return count;
  }

  private getChunkSpacing(chunkLayout: ChunkLayout): number {
    const { size } = chunkLayout;
    return Math.max(Math.min(size[0], size[1], size[2]), 1e-6);
  }

  private getChunkCenterWorld(
    chunkLayout: ChunkLayout,
    positionInChunks: Float32Array,
    out: Float32Array,
  ) {
    out[0] = (positionInChunks[0] + 0.5) * chunkLayout.size[0];
    out[1] = (positionInChunks[1] + 0.5) * chunkLayout.size[1];
    out[2] = (positionInChunks[2] + 0.5) * chunkLayout.size[2];
    vec3.transformMat4(out as vec3, out as vec3, chunkLayout.transform);
  }

  private getChunkGridPositionForWorldPoint(
    tsource: TransformedSource,
    worldPoint: Float32Array,
    out: Float32Array,
  ): boolean {
    const localPoint = vec3.create();
    tsource.chunkLayout.globalToLocalSpatial(localPoint, worldPoint as vec3);
    const { size } = tsource.chunkLayout;
    const { lowerChunkBound, upperChunkBound } = (
      tsource.source as SpatiallyIndexedSkeletonSource
    ).spec;
    for (let i = 0; i < 3; ++i) {
      const dimSize = size[i];
      if (!Number.isFinite(dimSize) || dimSize <= 0) {
        return false;
      }
      const chunkCoord = Math.floor(localPoint[i] / dimSize);
      if (
        Number.isFinite(lowerChunkBound[i]) &&
        Number.isFinite(upperChunkBound[i]) &&
        (chunkCoord < lowerChunkBound[i] || chunkCoord >= upperChunkBound[i])
      ) {
        return false;
      }
      out[i] = chunkCoord;
    }
    return true;
  }

  private getMetersPerUnit(projectionParameters: ProjectionParameters): number {
    const ddScales =
      projectionParameters.displayDimensionRenderInfo?.displayDimensionScales;
    if (ddScales === undefined || ddScales.length === 0) {
      return 1;
    }
    let metersPerUnit = Infinity;
    for (let i = 0; i < ddScales.length; ++i) {
      const s = ddScales[i];
      if (Number.isFinite(s) && s > 0) {
        metersPerUnit = Math.min(metersPerUnit, s);
      }
    }
    return Number.isFinite(metersPerUnit) ? metersPerUnit : 1;
  }

  private getReferencePixelSize(
    projectionParameters: ProjectionParameters,
  ): number {
    let reference: Float32Array | undefined;
    if (projectionParameters.globalPosition !== undefined) {
      reference = projectionParameters.globalPosition;
    } else if (this.localPosition.value.length >= 3) {
      reference = this.localPosition.value;
    } else {
      reference = new Float32Array(3);
    }
    const pixelSize = computePhysicalUnitsPerScreenPixel(
      projectionParameters.viewProjectionMat,
      projectionParameters.width,
      projectionParameters.height,
      reference,
      projectionParameters.displayDimensionRenderInfo?.displayDimensionScales,
    );
    return Number.isFinite(pixelSize) && pixelSize > 0 ? pixelSize : 1;
  }

  private getArbitrationTargetSpacingMeters3d(
    projectionParameters: ProjectionParameters,
  ): number {
    const target =
      this.displayState.spatialSkeletonGridResolutionTarget3d?.value;
    if (Number.isFinite(target) && target !== undefined && target > 0) {
      return target;
    }
    return this.getMetersPerUnit(projectionParameters);
  }

  // Quantize spacing to quarter-octave bins so tiny camera rotations do not
  // globally thrash chunk-level arbitration decisions.
  private quantizeSpacingForArbitration(spacing: number): number {
    const clamped = Math.max(spacing, 1e-12);
    const log2Spacing = Math.log2(clamped);
    const quantizedLog = Math.round(log2Spacing * 4) / 4;
    return 2 ** quantizedLog;
  }

  private getCachedNodeSnapshot(nodeId: number) {
    const cachedNode = this.getCachedNodeInfo?.(nodeId);
    if (cachedNode === undefined) {
      return undefined;
    }
    const pendingPosition =
      this.getPendingNodePositionOverride?.(cachedNode.nodeId) ??
      cachedNode.position;
    return {
      ...cachedNode,
      position: new Float32Array([
        Number(pendingPosition[0]),
        Number(pendingPosition[1]),
        Number(pendingPosition[2]),
      ]),
    };
  }

  invalidateSourceCellsForPositions(
    positions: Iterable<ArrayLike<number> | undefined>,
  ) {
    const positionList = [...positions].filter(
      (position): position is ArrayLike<number> => position !== undefined,
    );
    if (positionList.length === 0) {
      return false;
    }
    let invalidated = false;
    const seenSourceIds = new Set<string>();
    for (const sourceEntry of [...this.sources, ...this.sources2d]) {
      const chunkSource = sourceEntry.chunkSource;
      const sourceId = getObjectId(chunkSource);
      if (seenSourceIds.has(sourceId)) continue;
      seenSourceIds.add(sourceId);
      const keyPrefixes = new Set<string>();
      const { chunkDataSize } = chunkSource.spec;
      for (const position of positionList) {
        // Spatial skeleton node positions are already source/model coordinates;
        // render-layer transforms do not apply to CATMAID grid-cell keys.
        const keyPrefix = getSpatialSkeletonCellKeyPrefix(
          position,
          chunkDataSize,
        );
        if (keyPrefix !== undefined) {
          keyPrefixes.add(keyPrefix);
        }
      }
      if (keyPrefixes.size === 0) {
        continue;
      }
      chunkSource.invalidateCacheKeyPrefixes(keyPrefixes);
      invalidated = true;
    }
    if (!invalidated) {
      return false;
    }
    this.redrawNeeded.dispatch();
    return true;
  }

  private getChunkPositionAndSegmentArrays(
    chunk: SpatiallyIndexedSkeletonChunk,
  ) {
    const offsets = chunk.vertexAttributeOffsets;
    if (!offsets || offsets.length < 1) return undefined;
    const positions = new Float32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + offsets[0],
      chunk.numVertices * 3,
    );
    // Locate the "segment" column by its actual attribute index — the
    // zarr-vectors layout is [position, tangent, …, segment], so segment is
    // NOT necessarily offsets[1].  Count the uint32 it occupies per vertex:
    // a UINT64 attribute (zarr-vectors full uint64) is 2 uint32 [lo, hi]
    // despite numComponents===1; a UINT32 attribute (CATMAID) is 1 (high
    // half implicitly 0).
    const segIdx = this.vertexAttributes.findIndex((a) => a.name === "segment");
    if (segIdx < 0 || segIdx >= offsets.length) return undefined;
    const segInfo = this.vertexAttributes[segIdx];
    const segmentComponents =
      segInfo.dataType === DataType.UINT64
        ? 2 * segInfo.numComponents
        : segInfo.numComponents;
    const segmentIds = new Uint32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + offsets[segIdx],
      chunk.numVertices * segmentComponents,
    );
    return { positions, segmentIds, segmentComponents };
  }

  resolveSegmentPickFromChunk(
    chunk: SpatiallyIndexedSkeletonChunk,
    pickedOffset: number,
    kind: "node" | "edge" | "face",
  ) {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (data === undefined) {
      return undefined;
    }
    return resolveSpatiallyIndexedSkeletonSegmentPick(
      chunk,
      data.segmentIds,
      pickedOffset,
      kind,
      data.segmentComponents,
    );
  }

  resolveNodePickFromChunk(
    chunk: SpatiallyIndexedSkeletonChunk,
    pickedOffset: number,
  ) {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (
      data === undefined ||
      pickedOffset < 0 ||
      pickedOffset >= chunk.numVertices ||
      pickedOffset >= chunk.nodeIds.length
    ) {
      return undefined;
    }
    const nodeId = chunk.nodeIds[pickedOffset];
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) {
      return undefined;
    }
    const segmentId = resolveSpatiallyIndexedSkeletonSegmentPick(
      chunk,
      data.segmentIds,
      pickedOffset,
      "node",
      data.segmentComponents,
    );
    if (segmentId === undefined) {
      return undefined;
    }
    const baseOffset = pickedOffset * 3;
    return {
      nodeId,
      segmentId,
      position: data.positions.subarray(baseOffset, baseOffset + 3),
      sourceState: chunk.nodeSourceStates[pickedOffset],
    };
  }

  // Iterates every chunk slot in view for the given view/gridLevel.
  // Callback receives (chunkKey, chunkSource, chunkLayout); return false to stop early.
  //
  // Takes no `lod`: chunk keys are grid position alone (see `getChunk` in
  // `skeleton/backend.ts`). Callers still resolve their own `lod` first, but only
  // to decide whether there is a level to draw at all.
  private forEachVisibleChunkSlot(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    callback: (
      chunkKey: string,
      chunkSource: SpatiallyIndexedSkeletonSource,
      chunkLayout: ChunkLayout,
    ) => boolean | void,
  ) {
    const selectedSources =
      view === "3d"
        ? this.selectSourcesForViewAndGridWithFallback(view, gridLevel)
        : this.selectSourcesForViewAndGrid(view, gridLevel);
    // Which sources the non-arbitrating path below draws from.
    //
    // Normally exactly one: the preferred level. `...WithFallback` returns every
    // level in preference order for a caller to try in sequence, not a list to
    // draw simultaneously, and drawing the whole list would render every
    // resident level on top of itself.
    //
    // Under OBJECT focus it is every level from the drawn one up to the
    // coarsest, because there they do not overlap: each draws only the objects
    // that are new at it (see `admitObjects` in the zarr-vectors backend), so
    // together they cover the admitted set exactly once. Grid index counts from
    // the coarsest, so that is `gridIndex <= gridLevel`.
    //
    // ...and only where the levels really do partition the objects. Without
    // that the union draws every resident level's copy of the same object on
    // top of itself; object focus still means "one level everywhere", it just
    // means the SELECTED level rather than a coarse-to-fine stack.
    const objectFocus =
      this.detailFocus.value === SpatialSkeletonDetailFocus.OBJECT &&
      this.objectPartitionAvailable(view);
    const drawSourceIds = new Set<string>();
    if (objectFocus && gridLevel !== undefined) {
      for (const entry of this.getSources(view)) {
        const entryGrid = getSpatiallyIndexedSkeletonGridIndex(
          entry.chunkSource,
        );
        if (entryGrid !== undefined && entryGrid <= gridLevel) {
          drawSourceIds.add(getObjectId(entry.chunkSource));
        }
      }
    }
    if (drawSourceIds.size === 0 && selectedSources.length > 0) {
      drawSourceIds.add(getObjectId(selectedSources[0].chunkSource));
    }

    // Per-cell level arbitration -- the standard neuroglancer behaviour, where
    // a cell near the camera resolves finer than one far from it. OBJECT focus
    // opts out: it draws ONE level everywhere and spends the memory that leaves
    // on whole objects, which scattering levels across the view would undo.
    if (
      view === "3d" &&
      this.detailFocus.value === SpatialSkeletonDetailFocus.LOCAL &&
      selectedSources.length > 1
    ) {
      const transformedBySourceId = new Map<string, TransformedSource>();
      for (const scales of transformedSources) {
        for (const tsource of scales) {
          transformedBySourceId.set(getObjectId(tsource.source), tsource);
        }
      }
      const metersPerUnit = this.getMetersPerUnit(projectionParameters);
      const levelSpacings = this.getLevelSpacingsMeters();
      const arbitrationCandidates = selectedSources
        .map((source, fallbackRank) => {
          const sourceId = getObjectId(source.chunkSource);
          const transformed = transformedBySourceId.get(sourceId);
          if (transformed === undefined) return undefined;
          const spacing = this.getChunkSpacing(transformed.chunkLayout);
          // Published level spacing in preference to chunk shape, and for the
          // same reason the backend's copy of this does it: on an
          // object-sparsity pyramid every level shares one chunk_shape, so
          // chunk-derived spacings all tie and every cell falls through to the
          // selected level. The two sides must agree, or the view draws a level
          // the priorities never fetched.
          const gridIndex = getSpatiallyIndexedSkeletonGridIndex(
            source.chunkSource,
          );
          const published =
            gridIndex === undefined ? undefined : levelSpacings[gridIndex];
          return {
            fallbackRank,
            sourceId,
            transformed,
            spacing,
            spacingMeters:
              published !== undefined && published > 0
                ? published
                : spacing * metersPerUnit,
          };
        })
        .filter((x): x is NonNullable<typeof x> => x !== undefined);
      if (arbitrationCandidates.length > 1) {
        const worldCenter = new Float32Array(3);
        const candidateChunkPosition = new Float32Array(3);
        const targetSpacingMeters =
          this.getArbitrationTargetSpacingMeters3d(projectionParameters);
        const referencePixelSize =
          this.getReferencePixelSize(projectionParameters);
        // Anchor the position-enumeration grid on whichever candidate's
        // spacing is CLOSEST to the desired target, not unconditionally the
        // finest level -- same bug (and same fix) as
        // recomputeChunkPriorities's grid-anchor arbitration in
        // skeleton/backend.ts: anchoring on the finest level unconditionally
        // makes this walk the finest level's cell density across the whole
        // visible frustum even for a coarse/zoomed-out view.
        const anchor = arbitrationCandidates.reduce((best, candidate) =>
          Math.abs(candidate.spacingMeters - targetSpacingMeters) <
          Math.abs(best.spacingMeters - targetSpacingMeters)
            ? candidate
            : best,
        );
        const emittedChunkSlots = new Set<string>();
        let shouldContinue = true;
        forEachVisibleVolumetricChunk(
          projectionParameters,
          this.localPosition.value,
          anchor.transformed,
          (anchorPositionInChunks) => {
            if (!shouldContinue) return;
            this.getChunkCenterWorld(
              anchor.transformed.chunkLayout,
              anchorPositionInChunks,
              worldCenter,
            );
            const chunkPixelSize = computePhysicalUnitsPerScreenPixel(
              projectionParameters.viewProjectionMat,
              projectionParameters.width,
              projectionParameters.height,
              worldCenter,
              projectionParameters.displayDimensionRenderInfo
                ?.displayDimensionScales,
            );
            const desiredSpacingRaw =
              Number.isFinite(chunkPixelSize) && chunkPixelSize > 0
                ? targetSpacingMeters * (chunkPixelSize / referencePixelSize)
                : targetSpacingMeters;
            const desiredSpacing =
              this.quantizeSpacingForArbitration(desiredSpacingRaw);
            const orderedCandidates = [...arbitrationCandidates].sort(
              (a, b) => {
                const da = Math.abs(a.spacingMeters - desiredSpacing);
                const db = Math.abs(b.spacingMeters - desiredSpacing);
                if (da !== db) return da - db;
                return a.fallbackRank - b.fallbackRank;
              },
            );

            let selected:
              | {
                  candidate: (typeof orderedCandidates)[number];
                  chunkKey: string;
                  state: ChunkState | undefined;
                }
              | undefined;
            for (const candidate of orderedCandidates) {
              if (
                !this.getChunkGridPositionForWorldPoint(
                  candidate.transformed,
                  worldCenter,
                  candidateChunkPosition,
                )
              ) {
                continue;
              }
              const chunkKey = `${candidateChunkPosition.join()}${SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR}`;
              const chunk = (
                candidate.transformed.source as SpatiallyIndexedSkeletonSource
              ).chunks.get(chunkKey);
              const state = chunk?.state;
              if (state === ChunkState.GPU_MEMORY) {
                selected = { candidate, chunkKey, state };
                break;
              }
              if (selected === undefined) {
                selected = { candidate, chunkKey, state };
              }
            }
            if (selected === undefined) {
              return;
            }
            const emitKey = `${selected.candidate.sourceId}|${selected.chunkKey}`;
            if (emittedChunkSlots.has(emitKey)) {
              return;
            }
            emittedChunkSlots.add(emitKey);
            if (
              callback(
                selected.chunkKey,
                selected.candidate.transformed
                  .source as SpatiallyIndexedSkeletonSource,
                selected.candidate.transformed.chunkLayout,
              ) === false
            ) {
              shouldContinue = false;
            }
          },
        );
        return;
      }
    }

    let shouldContinue = true;
    for (const scales of transformedSources) {
      for (const tsource of scales) {
        if (!shouldContinue) return;
        if (!drawSourceIds.has(getObjectId(tsource.source))) continue;
        const emit = (positionInChunks: Float32Array) => {
          if (!shouldContinue) return;
          const chunkKey = `${positionInChunks.join()}${SPATIAL_SKELETON_CHUNK_KEY_TERMINATOR}`;
          if (
            callback(
              chunkKey,
              tsource.source as SpatiallyIndexedSkeletonSource,
              tsource.chunkLayout,
            ) === false
          ) {
            shouldContinue = false;
          }
        };
        // OBJECT focus draws the WHOLE VOLUME, not the frustum -- the same
        // enumeration the worker requests with, so the two agree cell for cell.
        // Drawing only what the frustum reaches would make a complete tract
        // appear to end at the edge of the view and reappear on panning, which
        // is the opposite of what loading whole objects is for.
        const source = tsource.source as SpatiallyIndexedSkeletonSource;
        const wholeVolume =
          objectFocus &&
          forEachSpatialSkeletonVolumeCell(
            source.spec.lowerChunkBound,
            source.spec.upperChunkBound,
            emit,
          ) >= 0;
        if (!wholeVolume) {
          forEachVisibleVolumetricChunk(
            projectionParameters,
            this.localPosition.value,
            tsource,
            emit,
          );
        }
      }
    }
  }

  getVisibleChunksInCurrentViewAndLod(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: any,
    lod: number | undefined,
  ): VisibleChunk[] {
    if (lod === undefined) {
      return [];
    }
    const result: VisibleChunk[] = [];
    this.forEachVisibleChunkSlot(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      (chunkKey, chunkSource, chunkLayout) => {
        const chunk = chunkSource.chunks.get(chunkKey);
        if (chunk?.state === ChunkState.GPU_MEMORY) {
          result.push({ chunk, chunkLayout });
        }
      },
    );
    return result;
  }

  /**
   * Populate the spatial-skeleton grid render-scale histogram using the
   * SAME visible-chunk enumeration the render path uses
   * (`forEachVisibleChunkSlot`), so the widget's bars reflect exactly the
   * chunks being drawn — present chunks fill the "present" (bright) colour
   * and grow the bar height as they load; not-yet-loaded slots contribute
   * to the dim "not-present" portion.
   *
   * This replaces an earlier implementation that ran a *parallel*
   * per-source `forEachVisibleVolumetricChunk` pass, which enumerated no
   * chunks in the perspective view (source arbitration is required) and so
   * left every bar an empty placeholder.  Placeholder bars are still added
   * for pyramid levels with no chunks in view, so all scales stay visible
   * in the widget.
   */
  updateGridRenderScaleHistogram(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    lod: number | undefined,
    levels:
      | ReadonlyArray<{
          size: { x: number; y: number; z: number };
          lod: number;
          objectCount?: number;
        }>
      | undefined,
    histogram: RenderScaleHistogram,
    frameNumber: number,
  ) {
    histogram.begin(frameNumber);
    if (lod === undefined || transformedSources.length === 0) return;
    const spacingOf = (size: { x: number; y: number; z: number }) =>
      Math.max(Math.min(size.x, size.y, size.z), 1e-6);
    const perSpacing = new Map<number, { present: number; missing: number }>();
    this.forEachVisibleChunkSlot(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      (chunkKey, chunkSource, chunkLayout) => {
        const gridIndex = getSpatiallyIndexedSkeletonGridIndex(chunkSource);
        // Prefer the level's physical (meters) spacing — the same value the
        // histogram axis is calibrated against — falling back to the raw
        // chunk-layout spacing only when the level lookup is unavailable.
        const levelSize =
          gridIndex !== undefined ? levels?.[gridIndex]?.size : undefined;
        const spacing =
          levelSize !== undefined
            ? spacingOf(levelSize)
            : this.getChunkSpacing(chunkLayout);
        let entry = perSpacing.get(spacing);
        if (entry === undefined) {
          entry = { present: 0, missing: 0 };
          perSpacing.set(spacing, entry);
        }
        const chunk = chunkSource.chunks.get(chunkKey);
        if (chunk?.state === ChunkState.GPU_MEMORY) {
          entry.present++;
        } else {
          entry.missing++;
        }
      },
    );
    // When the source can say how many objects each level holds, size the bars
    // by that instead of by chunk-slot demand. For a tractogram, "how big is
    // this level" means its streamline count -- a pyramid running ~503k / 50k /
    // 5k / 503 / 50 is describing sparsity, and that is what the user is
    // choosing between. Chunk counts answer a different question (how much grid
    // is in view) and only ever existed for the one level being drawn, which
    // left every other bar an identical stub.
    const objectCounts = levels?.some((l) => l.objectCount !== undefined)
      ? levels
      : undefined;
    if (objectCounts !== undefined) {
      for (const level of objectCounts) {
        const spacing = spacingOf(level.size);
        const count = level.objectCount ?? 0;
        // The drawn level counts as present, the rest as not-present, so the
        // bar for what is on screen reads differently from the alternatives.
        const drawn = perSpacing.has(spacing);
        histogram.add(spacing, spacing, drawn ? count : 0, drawn ? 0 : count);
      }
      return;
    }

    for (const [spacing, { present, missing }] of perSpacing) {
      histogram.add(spacing, spacing, present, missing);
    }
    // No per-level object counts: fall back to chunk slots. Only the selected
    // level's are enumerated, so the others are derived -- levels tile the same
    // viewed volume, so slots scale with the inverse cube of chunk spacing.
    // Scaling one measured count is O(levels) per frame, where enumerating each
    // level's slots would repeat the source selection and arbitration above for
    // every one of them. Flagged render-only so an estimate is never counted as
    // real chunk demand.
    let reference: { spacing: number; count: number } | undefined;
    for (const [spacing, { present, missing }] of perSpacing) {
      const count = present + missing;
      if (
        count > 0 &&
        (reference === undefined || spacing < reference.spacing)
      ) {
        reference = { spacing, count };
      }
    }
    for (const level of levels ?? []) {
      const spacing = spacingOf(level.size);
      if (histogram.spatialScales.has(spacing)) continue;
      const estimate =
        reference === undefined
          ? 1
          : Math.max(
              1,
              Math.round(reference.count * (reference.spacing / spacing) ** 3),
            );
      histogram.add(spacing, spacing, 0, estimate, true);
    }
  }

  private areVisibleChunksReady(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    lod: number | undefined,
  ) {
    if (
      this.displayState.objectAlpha.value <= 0.0 &&
      this.displayState.hiddenObjectAlpha.value <= 0.0
    ) {
      return true;
    }
    if (lod === undefined) {
      // No LOD configured — draw() renders nothing in this case, so nothing to wait for.
      return true;
    }
    if (transformedSources.length === 0) {
      return false;
    }
    let ready = true;
    this.forEachVisibleChunkSlot(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      (chunkKey, chunkSource, _) => {
        const chunk = chunkSource.chunks.get(chunkKey);
        if (chunk?.state !== ChunkState.GPU_MEMORY) {
          ready = false;
          return false;
        }
        return true;
      },
    );
    return ready;
  }

  getNode(
    nodeId: number,
    options: {
      lod?: number;
    } = {},
  ): SpatiallyIndexedSkeletonNode | undefined {
    void options.lod;
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return undefined;
    return this.getCachedNodeSnapshot(nodeId);
  }

  getNodes(
    options: {
      segmentId?: bigint;
      lod?: number;
    } = {},
  ): SpatiallyIndexedSkeletonNode[] {
    void options.lod;
    const normalizedSegmentFilter =
      options.segmentId === undefined
        ? undefined
        : Math.round(Number(options.segmentId));
    const useSegmentFilter =
      normalizedSegmentFilter !== undefined &&
      Number.isFinite(normalizedSegmentFilter);
    const segmentIds =
      normalizedSegmentFilter === undefined
        ? this.getActiveEditableSegmentIds()
        : [normalizedSegmentFilter];
    const nodes = new Map<number, SpatiallyIndexedSkeletonNode>();
    for (const segmentId of segmentIds) {
      const segmentNodes =
        this.inspectionState?.getCachedSegmentNodes(segmentId) ?? [];
      for (const node of segmentNodes) {
        if (nodes.has(node.nodeId)) continue;
        const cachedNode = this.getCachedNodeSnapshot(node.nodeId);
        if (cachedNode === undefined) continue;
        if (
          useSegmentFilter &&
          normalizedSegmentFilter !== undefined &&
          cachedNode.segmentId !== normalizedSegmentFilter
        ) {
          continue;
        }
        nodes.set(cachedNode.nodeId, cachedNode);
      }
    }
    return [...nodes.values()].sort((a, b) => a.nodeId - b.nodeId);
  }

  private beginSkeletonRenderPass(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    renderHelper: RenderHelper,
    modelMatrix: mat4,
    lineWidth: number,
    pointDiameter: number,
    excludedGPUTable?: GPUHashTable<HashSetUint64>,
  ):
    | {
        gl: GL;
        edgeShader: ShaderProgram;
        nodeShader: ShaderProgram;
        faceShader: ShaderProgram | undefined;
        skeletonParams: SkeletonShaderParameters;
      }
    | undefined {
    const { gl } = this;
    const edgeShaderResult = renderHelper.edgeShaderGetter(
      renderContext.emitter,
    );
    const nodeShaderResult = renderHelper.nodeShaderGetter(
      renderContext.emitter,
    );
    const {
      shader: edgeShader,
      parameters: edgeShaderParameters,
      extraParameters: skeletonParams,
    } = edgeShaderResult;
    const { shader: nodeShader, parameters: nodeShaderParameters } =
      nodeShaderResult;
    if (edgeShader === null || nodeShader === null) return undefined;

    const { shaderControlState } = this.displayState.skeletonRenderingOptions;

    edgeShader.bind();
    renderHelper.beginLayer(gl, edgeShader, renderContext, modelMatrix);
    gl.uniform1f(edgeShader.uniform("uLineWidth"), lineWidth);
    renderHelper.setPickInstanceStride(gl, edgeShader, 0);
    setControlsInShader(
      gl,
      edgeShader,
      shaderControlState,
      edgeShaderParameters.parseResult.controls,
    );
    renderHelper.setColor(gl, edgeShader, kOneVec4);
    renderHelper.maybeEnableDynamicSegmentAppearance(
      gl,
      edgeShader,
      skeletonParams,
      excludedGPUTable,
    );

    nodeShader.bind();
    renderHelper.beginLayer(gl, nodeShader, renderContext, modelMatrix);
    gl.uniform1f(nodeShader.uniform("uNodeDiameter"), pointDiameter);
    renderHelper.setPickInstanceStride(gl, nodeShader, 0);
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
    );
    renderHelper.setColor(gl, nodeShader, kOneVec4);
    renderHelper.maybeEnableDynamicSegmentAppearance(
      gl,
      nodeShader,
      skeletonParams,
      excludedGPUTable,
    );

    // Surface geometry only: a third program, compiled on first use.
    let faceShader: ShaderProgram | undefined;
    if (this.geometryPrimitive === "triangles") {
      const faceShaderResult = renderHelper.faceShaderGetter(
        renderContext.emitter,
      );
      const { shader, parameters: faceShaderParameters } = faceShaderResult;
      if (shader === null) return undefined;
      faceShader = shader;
      faceShader.bind();
      renderHelper.beginLayer(gl, faceShader, renderContext, modelMatrix);
      renderHelper.setPickInstanceStride(gl, faceShader, 0);
      setControlsInShader(
        gl,
        faceShader,
        shaderControlState,
        faceShaderParameters.parseResult.controls,
      );
      renderHelper.setColor(gl, faceShader, kOneVec4);
      renderHelper.setLightDirection(
        gl,
        faceShader,
        renderContext,
        modelMatrix,
      );
      renderHelper.maybeEnableDynamicSegmentAppearance(
        gl,
        faceShader,
        skeletonParams,
        excludedGPUTable,
      );
    }

    return { gl, edgeShader, nodeShader, faceShader, skeletonParams };
  }

  private endSkeletonRenderPass(
    renderHelper: RenderHelper,
    gl: GL,
    edgeShader: ShaderProgram,
    nodeShader: ShaderProgram,
    skeletonParams: SkeletonShaderParameters,
    faceShader?: ShaderProgram,
  ) {
    renderHelper.maybeDisableDynamicSegmentAppearance(
      gl,
      edgeShader,
      skeletonParams,
    );
    renderHelper.maybeDisableDynamicSegmentAppearance(
      gl,
      nodeShader,
      skeletonParams,
    );
    if (faceShader !== undefined) {
      renderHelper.maybeDisableDynamicSegmentAppearance(
        gl,
        faceShader,
        skeletonParams,
      );
    }
    renderHelper.endLayer(gl, edgeShader, nodeShader, faceShader ?? null);
  }

  private drawBrowsePass(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    modelMatrix: mat4,
    lineWidth: number,
    pointDiameter: number,
    visibleChunks: VisibleChunk[],
    renderMode: SkeletonRenderMode = SkeletonRenderMode.LINES_AND_POINTS,
  ) {
    if (visibleChunks.length === 0) return;
    const hasExcludedSegments =
      this.getBrowsePassExcludedSegments() !== undefined;
    const passState = this.beginSkeletonRenderPass(
      renderContext,
      renderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
      hasExcludedSegments ? this.gpuBrowseExcludedSegmentsHashTable : undefined,
    );
    if (passState === undefined) return;
    const { gl, edgeShader, nodeShader, faceShader, skeletonParams } =
      passState;

    const chunkOrigin = vec3.create();
    const chunkBound = vec3.create();
    for (const { chunk, chunkLayout } of visibleChunks) {
      if (skeletonParams.spatialChunkCulling) {
        vec3.mul(chunkOrigin, chunk.chunkGridPosition, chunkLayout.size);
        vec3.add(chunkBound, chunkOrigin, chunkLayout.size);
        if (faceShader !== undefined) {
          faceShader.bind();
          renderHelper.setChunkBounds(gl, faceShader, chunkOrigin, chunkBound);
        }
        edgeShader.bind();
        renderHelper.setChunkBounds(gl, edgeShader, chunkOrigin, chunkBound);
        nodeShader.bind();
        renderHelper.setChunkBounds(gl, nodeShader, chunkOrigin, chunkBound);
      }
      if (faceShader !== undefined) {
        // Surface geometry: `chunk.indices` holds triangles, so one instanced
        // triangle per face and one pick id per face -- `gl_InstanceID` is the
        // face number.
        if (renderContext.emitPickID) {
          let facePickId = 0;
          let facePickStride = 0;
          if (chunk.numIndices > 0) {
            facePickId = renderContext.pickIDs.register(
              layer,
              chunk.numIndices / 3,
              0n,
              {
                kind: "segment-face",
                chunk,
              } satisfies SpatiallyIndexedSkeletonPickData,
            );
            facePickStride = 1;
          }
          faceShader.bind();
          renderHelper.setPickID(gl, faceShader, facePickId);
          renderHelper.setPickInstanceStride(gl, faceShader, facePickStride);
        }
        renderHelper.drawTriangles(gl, faceShader, chunk);
        continue;
      }
      if (renderContext.emitPickID) {
        let edgePickId = 0;
        let edgePickStride = 0;
        let nodePickId = 0;
        let nodePickStride = 0;
        if (chunk.numIndices > 0) {
          edgePickId = renderContext.pickIDs.register(
            layer,
            chunk.numIndices / 2,
            0n,
            {
              kind: "segment-edge",
              chunk,
            } satisfies SpatiallyIndexedSkeletonPickData,
          );
          edgePickStride = 1;
        }
        if (chunk.numVertices > 0) {
          nodePickId = renderContext.pickIDs.register(
            layer,
            chunk.numVertices,
            0n,
            {
              kind: "segment-node",
              chunk,
            } satisfies SpatiallyIndexedSkeletonPickData,
          );
          nodePickStride = 1;
        }
        edgeShader.bind();
        renderHelper.setPickID(gl, edgeShader, edgePickId);
        renderHelper.setPickInstanceStride(gl, edgeShader, edgePickStride);
        nodeShader.bind();
        renderHelper.setPickID(gl, nodeShader, nodePickId);
        renderHelper.setPickInstanceStride(gl, nodeShader, nodePickStride);
      }
      // Render each chunk with different node/edge colors for debugging
      if (DEBUG_SPATIAL_SKELETON_CHUNKS) {
        const chunkKey = `${chunk.chunkGridPosition[0]},${chunk.chunkGridPosition[1]},${chunk.chunkGridPosition[2]}`;
        let randomColor = tempChunkKeyToColorMap.get(chunkKey);
        if (randomColor === undefined) {
          // Use same strategy as segment color hashing to be consistent
          // in colors across neuroglancer sessions
          randomColor = new Float32Array([0, 0, 0]);
          let h = hashCombine(0, chunk.chunkGridPosition[0]);
          h = hashCombine(h, chunk.chunkGridPosition[1]);
          h = hashCombine(h, chunk.chunkGridPosition[2]);
          const c0 = (h & 0xff) / 255;
          const c1 = ((h >> 8) & 0xff) / 255;
          hsvToRgb(randomColor, c0, 0.5 + 0.5 * c1, 1.0);
          tempChunkKeyToColorMap.set(chunkKey, randomColor);
        }
        if (skeletonParams.hasSegmentDefaultColor) {
          nodeShader.bind();
          gl.uniform3fv(
            nodeShader.uniform("uSegmentDefaultColor"),
            randomColor,
          );
          edgeShader.bind();
          gl.uniform3fv(
            edgeShader.uniform("uSegmentDefaultColor"),
            randomColor,
          );
        }
      }
      renderHelper.drawSkeletons(
        gl,
        edgeShader,
        nodeShader,
        chunk,
        renderContext.projectionParameters,
        renderMode,
        this.geometryPrimitive,
      );
    }
    this.endSkeletonRenderPass(
      renderHelper,
      gl,
      edgeShader,
      nodeShader,
      skeletonParams,
      faceShader,
    );
  }

  private drawInspectionOverlayPass(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    modelMatrix: mat4,
    lineWidth: number,
    pointDiameter: number,
    renderMode: SkeletonRenderMode = SkeletonRenderMode.LINES_AND_POINTS,
  ) {
    const overlayChunk = this.resolveSourceBackedOverlayChunk();
    if (overlayChunk === undefined) return;
    const passState = this.beginSkeletonRenderPass(
      renderContext,
      renderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
    );
    if (passState === undefined) return;
    const { gl, edgeShader, nodeShader, skeletonParams } = passState;

    if (renderContext.emitPickID) {
      const edgePickId =
        overlayChunk.numIndices > 0 &&
        overlayChunk.pickEdgeSegmentIds !== undefined &&
        overlayChunk.pickEdgeSegmentIds.length > 0
          ? renderContext.pickIDs.register(
              layer,
              overlayChunk.pickEdgeSegmentIds.length,
              0n,
              {
                kind: "edge",
                segmentIds: overlayChunk.pickEdgeSegmentIds,
              } satisfies SpatiallyIndexedSkeletonPickData,
            )
          : 0;
      edgeShader.bind();
      renderHelper.setPickID(gl, edgeShader, edgePickId);
      renderHelper.setPickInstanceStride(
        gl,
        edgeShader,
        edgePickId === 0 ? 0 : 1,
      );

      const nodePickId =
        overlayChunk.numVertices > 0 &&
        overlayChunk.pickNodeIds !== undefined &&
        overlayChunk.pickNodePositions !== undefined &&
        overlayChunk.pickSegmentIds !== undefined
          ? renderContext.pickIDs.register(
              layer,
              overlayChunk.numVertices,
              0n,
              {
                kind: "node",
                nodeIds: overlayChunk.pickNodeIds,
                nodePositions: overlayChunk.pickNodePositions,
                segmentIds: overlayChunk.pickSegmentIds,
              } satisfies SpatiallyIndexedSkeletonPickData,
            )
          : 0;
      nodeShader.bind();
      renderHelper.setPickID(gl, nodeShader, nodePickId);
      renderHelper.setPickInstanceStride(
        gl,
        nodeShader,
        nodePickId === 0 ? 0 : 1,
      );
    }

    renderHelper.drawSkeletons(
      gl,
      edgeShader,
      nodeShader,
      overlayChunk,
      renderContext.projectionParameters,
      renderMode,
      this.geometryPrimitive,
    );
    this.endSkeletonRenderPass(
      renderHelper,
      gl,
      edgeShader,
      nodeShader,
      skeletonParams,
    );
  }

  draw(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    overlayRenderHelper: RenderHelper,
    browseRenderHelper: RenderHelper,
    renderOptions: ViewSpecificSkeletonRenderingOptions,
    modelMatrix: mat4,
    visibleChunks: VisibleChunk[],
  ) {
    const { displayState } = this;
    if (
      displayState.objectAlpha.value <= 0.0 &&
      displayState.hiddenObjectAlpha.value <= 0.0
    ) {
      return;
    }

    const lineWidth = renderOptions.lineWidth.value;
    const renderMode = renderOptions.mode.value;
    // Point geometry always draws its vertices at the larger dot size: there is
    // no line for the "lines" preference to fall back to.
    const pointDiameter =
      this.geometryPrimitive === "points"
        ? Math.max(5, lineWidth * 2)
        : getSkeletonNodeDiameter(renderMode, lineWidth);

    this.drawBrowsePass(
      renderContext,
      layer,
      browseRenderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
      visibleChunks,
      renderMode,
    );
    this.drawInspectionOverlayPass(
      renderContext,
      layer,
      overlayRenderHelper,
      modelMatrix,
      lineWidth,
      pointDiameter,
      renderMode,
    );
  }

  isReady(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: ProjectionParameters,
    lod?: number,
  ) {
    return this.areVisibleChunksReady(
      view,
      gridLevel,
      transformedSources,
      projectionParameters,
      lod,
    );
  }
}

function transformSpatiallyIndexedSkeletonPickedValue(
  pickState: PickState,
): bigint | undefined {
  const u64 = pickState.pickedSpatialSkeleton?.segmentIdU64;
  return typeof u64 === "bigint" && u64 > 0n ? u64 : undefined;
}

const MAX_SAFE_SEGMENT_ID = BigInt(Number.MAX_SAFE_INTEGER);

/**
 * Split a full uint64 segment id into the dual representation stored on
 * `PickedSpatialSkeletonState`: the `bigint` for the selection widget, and a
 * safe-integer `number` for the legacy edit/overlay tooling (undefined when
 * the id exceeds 2⁵³, e.g. flywire ids).
 */
function pickedSegmentIdFields(u64: bigint): {
  segmentIdU64: bigint;
  segmentId?: number;
} {
  return {
    segmentIdU64: u64,
    segmentId: u64 <= MAX_SAFE_SEGMENT_ID ? Number(u64) : undefined,
  };
}

function updateSpatiallyIndexedSkeletonMouseState(
  base: SpatiallyIndexedSkeletonLayer,
  mouseState: MouseSelectionState,
  pickedOffset: number,
  data: SpatiallyIndexedSkeletonPickData | undefined,
): void {
  if (data === undefined) return;
  if (data.kind === "node") {
    if (
      pickedOffset < 0 ||
      pickedOffset >= data.nodeIds.length ||
      pickedOffset >= data.segmentIds.length
    ) {
      return;
    }
    const rawSegmentId = data.segmentIds[pickedOffset];
    if (!Number.isSafeInteger(rawSegmentId) || rawSegmentId <= 0) {
      return;
    }
    const segmentId = BigInt(rawSegmentId);
    mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(segmentId);
    if (
      !getVisibleSegments(base.displayState.segmentationGroupState.value).has(
        segmentId,
      )
    ) {
      return;
    }
    const nodeId = data.nodeIds[pickedOffset];
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return;
    const nodePosition = data.nodePositions.subarray(
      pickedOffset * 3,
      pickedOffset * 3 + 3,
    );
    mouseState.pickedSpatialSkeleton = {
      nodeId,
      ...pickedSegmentIdFields(segmentId),
      position: new Float32Array(nodePosition),
    };
    const transform = base.displayState.transform.value;
    if (transform.error === undefined) {
      setMouseStatePositionFromSpatialSkeletonNode(
        mouseState,
        nodePosition,
        transform,
      );
    }
    return;
  }
  if (data.kind === "edge") {
    if (pickedOffset < 0 || pickedOffset >= data.segmentIds.length) {
      return;
    }
    const rawSegmentId = data.segmentIds[pickedOffset];
    if (Number.isSafeInteger(rawSegmentId) && rawSegmentId > 0) {
      mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(
        BigInt(rawSegmentId),
      );
    }
    return;
  }
  if (
    data.kind === "segment-node" ||
    data.kind === "segment-edge" ||
    data.kind === "segment-face"
  ) {
    if (data.kind === "segment-node") {
      const pickedNode = base.resolveNodePickFromChunk(
        data.chunk,
        pickedOffset,
      );
      if (pickedNode !== undefined) {
        mouseState.pickedSpatialSkeleton = {
          nodeId: pickedNode.nodeId,
          ...pickedSegmentIdFields(pickedNode.segmentId),
          position: new Float32Array(pickedNode.position),
          sourceState: pickedNode.sourceState,
        };
        return;
      }
      // No node identity: `resolveNodePickFromChunk` needs `chunk.nodeIds`,
      // which only the CATMAID backend populates. Everything else -- notably a
      // zarr-vectors point cloud, whose vertices are ALL that is drawn and whose
      // decoder gives each one its own segment id -- would otherwise pick
      // nothing at all. The vertex still knows its segment, so fall through and
      // report that.
      const nodeSegmentId = base.resolveSegmentPickFromChunk(
        data.chunk,
        pickedOffset,
        "node",
      );
      if (nodeSegmentId !== undefined) {
        mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(nodeSegmentId);
      }
      return;
    }
    const segmentId = base.resolveSegmentPickFromChunk(
      data.chunk,
      pickedOffset,
      data.kind === "segment-face" ? "face" : "edge",
    );
    if (segmentId !== undefined) {
      mouseState.pickedSpatialSkeleton = pickedSegmentIdFields(segmentId);
    }
  }
}

function attachSpatiallyIndexedSkeletonLayer(
  base: SpatiallyIndexedSkeletonLayer,
  renderLayer: {
    transformedSources: TransformedSource[][];
    redrawNeeded: NullarySignal;
  },
  attachment: VisibleLayerInfo<
    LayerView,
    ThreeDimensionalRenderLayerAttachmentState
  >,
  view: "2d" | "3d",
): void {
  const { redrawNeeded } = renderLayer;
  attachment.registerDisposer(
    registerNested(
      (context, transform, displayDimensionRenderInfo) => {
        const transformedSources = getVolumetricTransformedSources(
          displayDimensionRenderInfo,
          transform,
          () => [
            base.getSources(view).map((sourceEntry) => ({
              chunkSource: sourceEntry.chunkSource,
              chunkToMultiscaleTransform:
                sourceEntry.chunkToMultiscaleTransform,
            })),
          ],
          attachment.messages,
          renderLayer,
        );
        for (const scales of transformedSources) {
          for (const tsource of scales) {
            context.registerDisposer(tsource.source);
          }
        }
        attachment.view.flushBackendProjectionParameters();
        renderLayer.transformedSources = transformedSources;
        base.rpc!.invoke(
          SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
          {
            layer: base.backend.rpcId,
            view: attachment.view.rpcId,
            displayDimensionRenderInfo,
            sources: serializeAllTransformedSources(transformedSources),
          },
        );
        redrawNeeded.dispatch();
        return transformedSources;
      },
      base.displayState.transform,
      attachment.view.displayDimensionRenderInfo,
    ),
  );
}

export class PerspectiveViewSpatiallyIndexedSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private browseRenderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  transformedSources: TransformedSource[][] = [];
  backend: ChunkRenderLayerFrontend;

  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.backend = base.backend;
    this.renderHelper = this.registerDisposer(new RenderHelper(base, false));
    this.browseRenderHelper = this.registerDisposer(
      new RenderHelper(base.browsePassLayerView, false),
    );
    this.renderOptions = base.displayState.skeletonRenderingOptions.params3d;

    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const histogram3d =
      base.displayState.spatialSkeletonGridRenderScaleHistogram3d;
    if (histogram3d !== undefined) {
      this.registerDisposer(histogram3d.visibility.add(this.visibility));
    }
  }

  attach(
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    super.attach(attachment);
    attachSpatiallyIndexedSkeletonLayer(this.base, this, attachment, "3d");
  }

  get gl() {
    return this.base.gl;
  }

  get isTransparent() {
    const { objectAlpha, hiddenObjectAlpha } = this.base.displayState;
    const opaque =
      (objectAlpha.value == 1.0 &&
        (hiddenObjectAlpha.value == 1.0 || hiddenObjectAlpha.value == 0.0)) ||
      (objectAlpha.value == 0.0 && hiddenObjectAlpha.value == 1.0);
    return !opaque;
  }

  getValueAt(_position: Float32Array) {
    return undefined;
  }

  transformPickedValue(pickState: PickState) {
    return transformSpatiallyIndexedSkeletonPickedValue(pickState);
  }

  updateMouseState(
    mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: unknown,
  ) {
    updateSpatiallyIndexedSkeletonMouseState(
      this.base,
      mouseState,
      pickedOffset,
      data as SpatiallyIndexedSkeletonPickData | undefined,
    );
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    if (!renderContext.emitColor && renderContext.alreadyEmittedPickID) {
      return;
    }
    const { displayState } = this.base;
    // Auto-LOD: refresh the resolution target from the current camera
    // projection before chunk selection, so the picker tracks zoom
    // (same path the manual slider takes).  Opt-in via
    // `autoSpatialSkeletonGridLevel3d` to preserve existing
    // user-driven behaviour for layers that don't want it.
    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      renderContext.projectionParameters,
      this.base.localPosition.value,
      "3d",
      // Only where this target decides the level. Under LOCAL focus it always
      // does; under OBJECT focus the memory budget decides instead -- but only
      // on a store that can be budgeted per object, which is the same store
      // property that lets the levels be drawn as a union. Everywhere else
      // OBJECT focus leaves level selection to the camera, and skipping this
      // would freeze the level at whatever was picked first.
      this.base.detailFocus.value === SpatialSkeletonDetailFocus.LOCAL ||
        !this.base.objectPartitionAvailable("3d")
        ? this.base.countVisibleCells(
            "3d",
            this.transformedSources,
            renderContext.projectionParameters,
          )
        : undefined,
    );
    const lodValue = displayState.skeletonLod?.value;
    const visibleChunks = this.base.getVisibleChunksInCurrentViewAndLod(
      "3d",
      displayState.spatialSkeletonGridLevel3d?.value,
      this.transformedSources,
      renderContext.projectionParameters,
      lodValue,
    );
    const levels = displayState.spatialSkeletonGridLevels?.value;
    const histogram = displayState.spatialSkeletonGridRenderScaleHistogram3d;
    if (histogram !== undefined) {
      const frameNumber =
        this.base.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
      this.base.updateGridRenderScaleHistogram(
        "3d",
        displayState.spatialSkeletonGridLevel3d?.value,
        this.transformedSources,
        renderContext.projectionParameters,
        lodValue,
        levels,
        histogram,
        frameNumber,
      );
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.browseRenderHelper,
      this.renderOptions,
      modelMatrix,
      visibleChunks,
    );
    if (renderContext.wireFrame) {
      this.drawChunkBoundsWireframe(renderContext, visibleChunks, modelMatrix);
    }
  }

  private drawChunkBoundsWireframe(
    renderContext: PerspectiveViewRenderContext,
    visibleChunks: VisibleChunk[],
    modelMatrix?: mat4,
  ) {
    if (
      visibleChunks.length === 0 ||
      !renderContext.emitColor ||
      modelMatrix === undefined
    )
      return;

    const { gl } = this.base;
    const wireframeHelper = ChunkWireframeHelper.get(gl);
    const shader = wireframeHelper.getShader(renderContext.emitter);
    shader.bind();
    const { viewProjectionMat } = renderContext.projectionParameters;

    mat4.multiply(tempMat4, viewProjectionMat, modelMatrix);
    gl.uniformMatrix4fv(shader.uniform("uChunkToClip"), false, tempMat4);

    for (const { chunk, chunkLayout } of visibleChunks) {
      wireframeHelper.setChunkUniforms(
        gl,
        shader,
        chunkLayout,
        chunk.chunkGridPosition,
      );
      drawBoxEdges(gl, 1, 1);
    }
  }

  isReady(
    renderContext: PerspectiveViewReadyRenderContext,
    _attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const { displayState } = this.base;
    return this.base.isReady(
      "3d",
      displayState.spatialSkeletonGridLevel3d?.value,
      this.transformedSources,
      renderContext.projectionParameters,
      displayState.skeletonLod?.value,
    );
  }
}

export class SliceViewPanelSpatiallyIndexedSkeletonLayer extends SliceViewPanelRenderLayer {
  private renderHelper: RenderHelper;
  private browseRenderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  transformedSources: TransformedSource[][] = [];
  backend: ChunkRenderLayerFrontend;
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.backend = base.backend;
    this.renderHelper = this.registerDisposer(new RenderHelper(base, true));
    this.browseRenderHelper = this.registerDisposer(
      new RenderHelper(base.browsePassLayerView, true),
    );
    this.renderOptions = base.displayState.skeletonRenderingOptions.params2d;
    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const { displayState: displayState2d } = base;
    const gridLevel2d = displayState2d.spatialSkeletonGridLevel2d;
    if (gridLevel2d?.changed) {
      this.registerDisposer(
        gridLevel2d.changed.add(this.redrawNeeded.dispatch),
      );
    }
    const lod2d = displayState2d.spatialSkeletonLod2d;
    if (lod2d?.changed) {
      this.registerDisposer(lod2d.changed.add(this.redrawNeeded.dispatch));
    }
    const histogram2d =
      displayState2d.spatialSkeletonGridRenderScaleHistogram2d;
    if (histogram2d !== undefined) {
      this.registerDisposer(histogram2d.visibility.add(this.visibility));
    }
  }

  get gl() {
    return this.base.gl;
  }

  getValueAt(_position: Float32Array) {
    return undefined;
  }

  transformPickedValue(pickState: PickState) {
    return transformSpatiallyIndexedSkeletonPickedValue(pickState);
  }

  updateMouseState(
    mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: unknown,
  ) {
    updateSpatiallyIndexedSkeletonMouseState(
      this.base,
      mouseState,
      pickedOffset,
      data as SpatiallyIndexedSkeletonPickData | undefined,
    );
  }

  attach(
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    super.attach(attachment);
    attachSpatiallyIndexedSkeletonLayer(this.base, this, attachment, "2d");
  }

  draw(
    renderContext: SliceViewPanelRenderContext,
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const { displayState } = this.base;
    maybeUpdateAutoSpatialSkeletonGridResolutionTarget(
      displayState,
      renderContext.sliceView.projectionParameters.value,
      this.base.localPosition.value,
      "2d",
      // Only where this target decides the level. Under LOCAL focus it always
      // does; under OBJECT focus the memory budget decides instead -- but only
      // on a store that can be budgeted per object, which is the same store
      // property that lets the levels be drawn as a union. Everywhere else
      // OBJECT focus leaves level selection to the camera, and skipping this
      // would freeze the level at whatever was picked first.
      this.base.detailFocus.value === SpatialSkeletonDetailFocus.LOCAL ||
        !this.base.objectPartitionAvailable("2d")
        ? this.base.countVisibleCells(
            "2d",
            this.transformedSources,
            renderContext.sliceView.projectionParameters.value,
          )
        : undefined,
    );
    const lodValue = displayState.spatialSkeletonLod2d?.value;
    const visibleChunks = this.base.getVisibleChunksInCurrentViewAndLod(
      "2d",
      displayState.spatialSkeletonGridLevel2d?.value,
      this.transformedSources,
      renderContext.sliceView.projectionParameters.value,
      lodValue,
    );
    const levels = displayState.spatialSkeletonGridLevels?.value;
    const histogram = displayState.spatialSkeletonGridRenderScaleHistogram2d;
    if (histogram !== undefined) {
      const frameNumber =
        this.base.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
      this.base.updateGridRenderScaleHistogram(
        "2d",
        displayState.spatialSkeletonGridLevel2d?.value,
        this.transformedSources,
        renderContext.sliceView.projectionParameters.value,
        lodValue,
        levels,
        histogram,
        frameNumber,
      );
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.browseRenderHelper,
      this.renderOptions,
      modelMatrix,
      visibleChunks,
    );
  }

  isReady(
    renderContext: SliceViewPanelReadyRenderContext,
    _attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const { displayState } = this.base;
    return this.base.isReady(
      "2d",
      displayState.spatialSkeletonGridLevel2d?.value,
      this.transformedSources,
      renderContext.projectionParameters,
      displayState.spatialSkeletonLod2d?.value,
    );
  }
}
