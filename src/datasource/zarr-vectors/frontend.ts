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

import type { AnnotationGeometryChunkSpecification } from "#src/annotation/base.js";
import {
  AnnotationGeometryChunkSource,
  MultiscaleAnnotationSource,
} from "#src/annotation/frontend_source.js";
import type { AnnotationPropertySpec } from "#src/annotation/index.js";
import {
  AnnotationType,
  parseAnnotationPropertySpecs,
} from "#src/annotation/index.js";
import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
  makeCoordinateSpace,
  makeIdentityTransform,
  makeIdentityTransformedBoundingBox,
} from "#src/coordinate_transform.js";
import {
  type DataSource,
  type DataSourceLookupResult,
  type GetKvStoreBasedDataSourceOptions,
  type KvStoreBasedDataSourceProvider,
} from "#src/datasource/index.js";
import { resolveAttributeSelection } from "#src/datasource/zarr-vectors/attribute_budget.js";
import type {
  ZarrVectorsAttributeDtype,
  ZarrVectorsPyramidMode,
} from "#src/datasource/zarr-vectors/base.js";
import {
  ZarrVectorsAnnotationSourceParameters,
  ZarrVectorsAnnotationSpatialIndexSourceParameters,
  ZarrVectorsObjectKeyedGeometrySourceParameters,
  ZarrVectorsSpatialGeometrySourceParameters,
} from "#src/datasource/zarr-vectors/base.js";
import {
  ZarrVectorsMultiscaleGeometrySource,
  ZarrVectorsObjectKeyedGeometrySource,
} from "#src/datasource/zarr-vectors/geometry_frontend.js";
import type { ZarrVectorsGeometryKind } from "#src/datasource/zarr-vectors/geometry_kind.js";
import { KIND_CAPABILITIES } from "#src/datasource/zarr-vectors/geometry_kind.js";
import {
  OBJECT_ATTR_DTYPE_TABLE,
  reinterpretObjectAttributeBytes,
  reinterpretWideToBigUint64,
  reinterpretWideToFloat32,
} from "#src/datasource/zarr-vectors/object_attribute_bytes.js";
import {
  levelsAreNested,
  objectDepths as computeObjectDepths,
 computePerLevelObjectCount } from "#src/datasource/zarr-vectors/object_budget.js";
import type { ObjectGroupMembership } from "#src/datasource/zarr-vectors/object_groups.js";
import {
  buildObjectGroupMembership,
  groupSegmentProperties,
  parseGroupCount,
} from "#src/datasource/zarr-vectors/object_groups.js";
import {
  formatAttributesFragment,
  parseAttributesFragment,
 resolveDeclaredGeometry , toAnnotationPropertyId } from "#src/datasource/zarr-vectors/store_metadata.js";
import { decodeVlenBytesChunk } from "#src/datasource/zarr-vectors/vlen_bytes.js";
import type { AutoDetectRegistry } from "#src/kvstore/auto_detect.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import {
  joinBaseUrlAndPath,
  kvstoreEnsureDirectoryPipelineUrl,
  parseUrlSuffix,
  pipelineUrlJoin,
} from "#src/kvstore/url.js";
import type {
  InlineSegmentNumericalProperty,
  InlineSegmentProperty,
  InlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import {
  normalizeInlineSegmentPropertyMap,
  SegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import { DataType } from "#src/util/data_type.js";
import * as matrix from "#src/util/matrix.js";
import type { ProgressOptions } from "#src/util/progress_listener.js";
import { ProgressSpan } from "#src/util/progress_listener.js";
import { allSiPrefixes, supportedUnits } from "#src/util/si_units.js";

// ---------------------------------------------------------------
// zstd decompression helper (object_attributes chunks may be zstd-compressed)

const ZSTD_MAGIC = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd]);

/** Keys already warned about, so one diagnostic does not repeat per level. */
const warnedOnceKeys = new Set<string>();
function warnOnceFe(key: string, message: string): void {
  if (warnedOnceKeys.has(key)) return;
  warnedOnceKeys.add(key);
  console.warn(message);
}

function looksLikeZstdFe(bytes: Uint8Array): boolean {
  return (
    bytes.byteLength >= 4 &&
    bytes[0] === ZSTD_MAGIC[0] &&
    bytes[1] === ZSTD_MAGIC[1] &&
    bytes[2] === ZSTD_MAGIC[2] &&
    bytes[3] === ZSTD_MAGIC[3]
  );
}

async function maybeDecompressObjAttr(
  bytes: Uint8Array<ArrayBuffer>,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!looksLikeZstdFe(bytes)) return bytes;
  return await requestAsyncComputation(
    decodeZstd,
    signal,
    [bytes.buffer],
    bytes,
  );
}

// ---------------------------------------------------------------
// Chunk source classes
// ---------------------------------------------------------------

class ZarrVectorsAnnotationSpatialIndexSource extends WithParameters(
  WithSharedKvStoreContext(AnnotationGeometryChunkSource),
  ZarrVectorsAnnotationSpatialIndexSourceParameters,
) {}

const MultiscaleAnnotationSourceBase = WithParameters(
  WithSharedKvStoreContext(MultiscaleAnnotationSource),
  ZarrVectorsAnnotationSourceParameters,
);

interface ZarrVectorsAnnotationSourceOptions {
  metadata: AnnotationMetadata;
  parameters: ZarrVectorsAnnotationSourceParameters;
  sharedKvStoreContext: SharedKvStoreContext;
}

export class ZarrVectorsAnnotationSource extends MultiscaleAnnotationSourceBase {
  declare key: unknown;
  metadata: AnnotationMetadata;
  declare OPTIONS: ZarrVectorsAnnotationSourceOptions;
  constructor(
    chunkManager: ChunkManager,
    options: ZarrVectorsAnnotationSourceOptions,
  ) {
    const { parameters } = options;
    super(chunkManager, {
      rank: parameters.rank,
      relationships: [],
      properties: parameters.properties,
      sharedKvStoreContext: options.sharedKvStoreContext,
      parameters,
    } as any);
    this.readonly = true;
    this.metadata = options.metadata;
  }

  getSources() {
    return [
      this.metadata.spatialIndices.map((level) => ({
        chunkSource: this.chunkManager.getChunkSource(
          ZarrVectorsAnnotationSpatialIndexSource,
          {
            sharedKvStoreContext: this.sharedKvStoreContext,
            parent: this,
            spec: level.spec,
            parameters: level.parameters,
          },
        ),
        chunkToMultiscaleTransform: level.spec.chunkToMultiscaleTransform,
      })),
    ];
  }
}

// ---------------------------------------------------------------
// Metadata parsing
// ---------------------------------------------------------------

// NGFF long-form unit strings → base SI letter + decimal exponent.
// Built from neuroglancer's known SI prefix table so any prefix
// understood elsewhere in the codebase round-trips correctly.
const OME_LONG_UNITS = (() => {
  const m = new Map<string, { unit: string; exponent: number }>();
  for (const baseUnit of ["meter", "second"]) {
    for (const p of allSiPrefixes) {
      if (p.longPrefix === undefined) continue;
      m.set(`${p.longPrefix}${baseUnit}`, {
        unit: baseUnit[0],
        exponent: p.exponent,
      });
    }
  }
  // Common irregular forms.
  m.set("micron", { unit: "m", exponent: -6 });
  m.set("microns", { unit: "m", exponent: -6 });
  return m;
})();

/**
 * Translate a (scale, unit) pair from user-facing form to the
 * normalised form neuroglancer's coordinate space expects: unit is one
 * of the base SI letters ("m", "s", "Hz", "rad/s", "") and any SI
 * prefix is folded into scale.  Returns {unit: "", scale} when the
 * unit string isn't recognised.
 */
function normalizeUnitScale(
  rawScale: number,
  rawUnit: unknown,
): { unit: string; scale: number } {
  if (typeof rawUnit !== "string" || rawUnit === "") {
    return { unit: "", scale: rawScale };
  }
  const longForm = OME_LONG_UNITS.get(rawUnit);
  if (longForm !== undefined) {
    return {
      unit: longForm.unit,
      scale: rawScale * 10 ** longForm.exponent,
    };
  }
  const shortForm = supportedUnits.get(rawUnit);
  if (shortForm !== undefined) {
    return {
      unit: shortForm.unit,
      scale: rawScale * 10 ** shortForm.exponent,
    };
  }
  return { unit: "", scale: rawScale };
}

/**
 * On-disk per-vertex dtypes the reader can decode. Everything here becomes a
 * float32 property (see `vertex_attribute_float.ts`); the set exists to reject
 * what it cannot decode at all -- a vlen-string column, say -- rather than to
 * choose a representation.
 *
 * The 64-bit members were missing until this list and the decoder were made to
 * agree: a store whose obs columns are float64 scores and int64 codes had them
 * all skipped without a word, which for `Zhuang-ABCA-1` meant nine of its
 * fourteen non-gene columns simply did not exist as far as the viewer was
 * concerned.
 */
const SUPPORTED_ATTR_DTYPES = new Set<string>([
  "float32",
  "uint8",
  "uint16",
  "uint32",
  "int8",
  "int16",
  "int32",
  "float64",
  "int64",
  "uint64",
]);

interface AnnotationSpatialIndexLevelMetadata {
  parameters: ZarrVectorsAnnotationSpatialIndexSourceParameters;
  spec: AnnotationGeometryChunkSpecification;
}

interface AnnotationMetadata {
  rank: number;
  coordinateSpace: ReturnType<typeof makeCoordinateSpace>;
  /** Stored→world offset, if the store declares one; see {@link readCoordinateOffset}. */
  coordinateOffset: Float64Array | undefined;
  parameters: ZarrVectorsAnnotationSourceParameters;
  spatialIndices: AnnotationSpatialIndexLevelMetadata[];
}

function buildCoordinateSpaceFromHints(
  hints: any,
  lowerBounds: Float64Array,
  upperBounds: Float64Array,
) {
  const names: string[] = hints.names.map((n: unknown) => String(n));
  const rawScales: number[] = hints.scales.map((s: unknown) => Number(s));
  const units: string[] = new Array(names.length);
  const scales = new Float64Array(names.length);
  for (let i = 0; i < names.length; ++i) {
    const normalized = normalizeUnitScale(rawScales[i], hints.units?.[i]);
    units[i] = normalized.unit;
    scales[i] = normalized.scale;
  }
  return makeCoordinateSpace({
    rank: names.length,
    names,
    units,
    scales,
    boundingBoxes: [
      makeIdentityTransformedBoundingBox({ lowerBounds, upperBounds }),
    ],
  });
}

function buildCoordinateSpaceFromMultiscales(
  multiscales: any,
  lowerBounds: Float64Array,
  upperBounds: Float64Array,
  rank: number,
) {
  const entry = Array.isArray(multiscales) ? multiscales[0] : undefined;
  const axes: any[] = entry?.axes ?? [];
  let names = axes.map((a, i) => (a?.name ? String(a.name) : `d${i}`));
  while (names.length < rank) names.push(`d${names.length}`);
  names = names.slice(0, rank);

  const dataset = entry?.datasets?.[0];
  const scaleXform = dataset?.coordinateTransformations?.find(
    (t: any) => t?.type === "scale",
  );
  const scaleArr: number[] = scaleXform?.scale ?? [];
  const units: string[] = new Array(rank);
  const scales = new Float64Array(rank);
  for (let i = 0; i < rank; ++i) {
    const rawScale = scaleArr[i] !== undefined ? Number(scaleArr[i]) : 1.0;
    const normalized = normalizeUnitScale(rawScale, axes[i]?.unit);
    units[i] = normalized.unit;
    scales[i] = normalized.scale;
  }
  return makeCoordinateSpace({
    rank,
    names,
    units,
    scales,
    boundingBoxes: [
      makeIdentityTransformedBoundingBox({ lowerBounds, upperBounds }),
    ],
  });
}

/**
 * Read `zarr_vectors.coordinate_offset` — the world position of the stored
 * coordinate origin.  The writer stores `world - coordinate_offset` so that
 * the spec's origin-0 chunk grid lines up with the source grid; zarr-vectors-py
 * states the contract as "World position = stored + coordinate_offset"
 * (`zarr_vectors/types/skeletons.py`) and its own reader adds it back when
 * returning positions.  Absent or all-zero offsets give `undefined`.
 */
function readCoordinateOffset(zv: any, rank: number): Float64Array | undefined {
  const raw = zv?.coordinate_offset;
  if (!Array.isArray(raw) || raw.length !== rank) return undefined;
  const offset = Float64Array.from(raw, Number);
  if (!offset.every((v) => Number.isFinite(v))) return undefined;
  if (offset.every((v) => v === 0)) return undefined;
  return offset;
}

/**
 * Model transform placing a store's raw coordinates into world space.
 *
 * Identity unless the store declares a `coordinate_offset`, in which case the
 * offset becomes the transform's translation column.  Expressing it as a model
 * transform (rather than shifting vertices or bounds) means the whole source
 * moves coherently — geometry, the declared bounds and the initial view
 * position neuroglancer derives from them — while every chunk-grid and
 * spatial-index calculation keeps running in the raw stored frame.
 *
 * Keyed on `coordinate_offset` and NOT on the NGFF per-level `translation`:
 * the writer overloads that field, mirroring the offset onto level 0 while
 * writing the bin-centre convention on the coarser levels, so it cannot be
 * read as a coordinate offset.  See the note in `buildGeometryMetadata`.
 */
function makeCoordinateOffsetTransform(
  coordinateSpace: ReturnType<typeof makeCoordinateSpace>,
  coordinateOffset: Float64Array | undefined,
) {
  const identity = makeIdentityTransform(coordinateSpace);
  if (coordinateOffset === undefined) return identity;
  const { rank } = coordinateSpace;
  const transform = matrix.createIdentity(Float64Array, rank + 1);
  for (let i = 0; i < rank; ++i) {
    transform[(rank + 1) * rank + i] = coordinateOffset[i];
  }
  return { ...identity, transform };
}

async function listAttributeNames(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  options: Partial<ProgressOptions>,
): Promise<string[]> {
  const attributesUrl = joinBaseUrlAndPath(levelUrl, "vertex_attributes/");
  let response;
  try {
    response = await sharedKvStoreContext.kvStoreContext.list(attributesUrl, {
      responseKeys: "suffix",
      ...options,
    });
  } catch (e) {
    // A genuinely-absent directory lists as empty (no throw); a throw here means
    // listing failed or is denied (e.g. a bucket without objects.list). Warn
    // rather than silently reporting "no attributes", which would drop
    // attribute-based colouring with no diagnostic.
    warnOnceFe(
      "vertex-attributes-list",
      "zarr-vectors: could not list vertex_attributes/ (object listing may be " +
        "denied on this store); per-vertex attribute-based colouring will be " +
        `unavailable even if attributes exist. ${e instanceof Error ? e.message : String(e)}`,
    );
    return [];
  }
  // Each subdirectory under vertex_attributes/ is one property.  Strip the
  // trailing "/" if present.
  return response.directories
    .map((d) => (d.endsWith("/") ? d.slice(0, -1) : d))
    .filter((d) => d.length > 0)
    .sort();
}

async function getJsonResource(
  sharedKvStoreContext: SharedKvStoreContext,
  url: string,
  description: string,
  options: Partial<ProgressOptions>,
): Promise<any | undefined> {
  return sharedKvStoreContext.chunkManager.memoize.getAsync(
    { type: "zarr-vectors:json", url },
    options,
    async (progressOptions) => {
      using _span = new ProgressSpan(progressOptions.progressListener, {
        message: `Reading ${description} from ${url}`,
      });
      const response = await sharedKvStoreContext.kvStoreContext.read(
        url,
        progressOptions,
      );
      if (response === undefined) return undefined;
      return await response.response.json();
    },
  );
}

/**
 * Read the per-chunk-array grid geometry from a level's `vertices/zarr.json`
 * (v0.9.0 single-array format): `chunk_grid_origin`, whether the array uses the
 * `sharding_indexed` codec, the shard shape, and the chunk-key separator. These
 * drive the backend cell reader (see `shard_cell_reader.ts`). Rejects the
 * pre-0.9.0 "Option G" layout, where the per-array node is a group rather than a
 * single array — those stores are unreadable and must be rewritten from source.
 */
async function readChunkGridParams(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  rank: number,
  options: Partial<ProgressOptions>,
): Promise<{
  chunkGridOrigin: number[];
  sharded: boolean;
  shardChunkShape: number[];
  cellSeparator: string;
}> {
  const json = await getJsonResource(
    sharedKvStoreContext,
    joinBaseUrlAndPath(levelUrl, "vertices/zarr.json"),
    "zarr-vectors vertices array metadata",
    options,
  );
  if (json === undefined || json.node_type !== "array") {
    throw new Error(
      `zarr-vectors: ${levelUrl}vertices is not a single zarr array ` +
        `(node_type=${JSON.stringify(json?.node_type)}). Pre-0.9.0 "Option G" ` +
        `stores are unreadable; rewrite from source with zarr-vectors >= 0.9.0.`,
    );
  }
  const attrs = json.attributes ?? {};
  const originRaw = attrs.chunk_grid_origin;
  const chunkGridOrigin =
    Array.isArray(originRaw) && originRaw.length === rank
      ? originRaw.map((x: any) => Number(x))
      : new Array<number>(rank).fill(0);
  const codecs = json.codecs;
  const sharded =
    Array.isArray(codecs) &&
    codecs.length > 0 &&
    codecs[0]?.name === "sharding_indexed";
  let shardChunkShape: number[] = [];
  if (sharded) {
    const cs = json.chunk_grid?.configuration?.chunk_shape;
    if (!Array.isArray(cs) || cs.length !== rank) {
      throw new Error(
        `zarr-vectors: sharded vertices array at ${levelUrl} is missing a ` +
          `chunk_grid.configuration.chunk_shape of rank ${rank}`,
      );
    }
    shardChunkShape = cs.map((x: any) => Number(x));
  }
  const sep = json.chunk_key_encoding?.configuration?.separator;
  return {
    chunkGridOrigin,
    sharded,
    shardChunkShape,
    cellSeparator: typeof sep === "string" ? sep : "/",
  };
}

/**
 * Number of concurrent reads used when opening a store's attribute metadata.
 * Stores can carry thousands of attributes (one per gene in a MERFISH panel):
 * reading them one at a time takes minutes, and firing all of them at once
 * just queues them in the browser's connection pool while starving the reads
 * the rest of the open depends on.
 */
const ATTRIBUTE_READ_CONCURRENCY = 16;

/**
 * Which `object_attributes/` column supplies the segment-property map's id
 * space, most authoritative first.
 *
 * `segment_id` is the spec's own name for the object id and is what the
 * per-fragment `fragment_attributes/segment_id` column — the ids the renderer
 * actually draws and selects with — is drawn from, so it must win. The others
 * are the CAVE/MICrONS spellings of the same root id, kept as fallbacks for
 * stores that ship the id under a table-specific name only.
 *
 * A column qualifies only if it is a 64-bit integer (see `AttrSpec.wideValues`);
 * a float column cannot represent a root id exactly and would reintroduce the
 * mismatch this list exists to fix.
 */
const OBJECT_ID_COLUMN_PREFERENCE = [
  "segment_id",
  "pt_root_id",
  "root_id_v1300",
] as const;

/**
 * `items.map(fn)` with at most `limit` calls in flight, preserving order.
 */
async function mapWithConcurrency<T, U>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<U>,
): Promise<U[]> {
  const results = new Array<U>(items.length);
  let next = 0;
  const worker = async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, worker),
  );
  return results;
}

/**
 * Read every attribute's `zarr.json` with bounded concurrency.  Attributes
 * whose metadata is unreadable are simply absent from the returned map; the
 * caller skips them (it cannot determine a dtype for them anyway).
 */
async function readAttributeMetadata(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  names: readonly string[],
  options: Partial<ProgressOptions>,
): Promise<Map<string, any>> {
  const metadataByName = new Map<string, any>();
  await mapWithConcurrency(names, ATTRIBUTE_READ_CONCURRENCY, async (name) => {
    try {
      metadataByName.set(
        name,
        await getJsonResource(
          sharedKvStoreContext,
          joinBaseUrlAndPath(levelUrl, `vertex_attributes/${name}/zarr.json`),
          `attribute ${JSON.stringify(name)} metadata`,
          options,
        ),
      );
    } catch {
      // Leave unset: the attribute is skipped.
    }
  });
  return metadataByName;
}

async function buildPropertySpecsAndDtypes(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  hints: any,
  options: Partial<ProgressOptions>,
  selectedAttributes?: readonly string[],
): Promise<{
  properties: AnnotationPropertySpec[];
  attributeNames: string[];
  attributeDtypes: ZarrVectorsAttributeDtype[];
}> {
  const declaredHints: any[] = Array.isArray(hints?.properties)
    ? hints.properties
    : [];
  const declaredByName = new Map<string, any>(
    declaredHints.map((p) => [String(p.identifier), p]),
  );

  const listedNames = await listAttributeNames(
    sharedKvStoreContext,
    levelUrl,
    options,
  );
  // `tangent` is reserved for the synthesised per-vertex direction
  // (vec3 float32), exposed as `prop_tangent()` for the default
  // colour-by-direction shader (see geometry_shader_bridge.ts). A store
  // that ALSO ships its own `tangent` vertex_attribute would redefine
  // the `prop_tangent` macro and — being multi-component (vec3) while the
  // reader packs user attributes as 1-component — make the skeleton
  // shader fail to compile ("illegal vector field selection"), rendering
  // nothing. The synthesised tangent wins; skip the store's copy. Filtered
  // here, before the budget, so it does not occupy a slot.
  const names = listedNames.filter((name) => {
    if (name !== "tangent") return true;
    console.warn(
      'zarr-vectors: ignoring reserved vertex attribute "tangent" — the ' +
        "reader synthesises prop_tangent() for colour-by-direction.",
    );
    return false;
  });

  // Stable order: declared properties first (in their declared order),
  // then any remaining listed attributes in alphabetical order.
  const orderedNames: string[] = [];
  for (const p of declaredHints) {
    const id = String(p.identifier);
    if (names.includes(id) && !orderedNames.includes(id)) {
      orderedNames.push(id);
    }
  }
  for (const n of names) {
    if (!orderedNames.includes(n)) orderedNames.push(n);
  }

  // Array metadata for the attributes actually kept, retained for the
  // dictionary/enum block below. Populated by the dtype reader the selection
  // resolver drives, so only the pages it reads are ever fetched.
  const metadataByName = new Map<string, any>();
  const { names: candidateNames, dtypes: dtypeByName } =
    await resolveAttributeSelection({
      orderedNames,
      availableNames: names,
      selectedAttributes,
      isSupported: (dtype) => SUPPORTED_ATTR_DTYPES.has(dtype),
      readDtypes: async (batch) => {
        const meta = await readAttributeMetadata(
          sharedKvStoreContext,
          levelUrl,
          batch,
          options,
        );
        const out = new Map<string, string | undefined>();
        for (const name of batch) {
          const arrayMeta = meta.get(name);
          metadataByName.set(name, arrayMeta);
          const dtype: unknown =
            arrayMeta?.attributes?.dtype ?? arrayMeta?.data_type;
          out.set(name, typeof dtype === "string" ? dtype : undefined);
        }
        return out;
      },
    });

  const attributeNames: string[] = [];
  const attributeDtypes: ZarrVectorsAttributeDtype[] = [];
  const rawPropertyJson: any[] = [];
  // `tangent` and `segment` are the synthesised per-vertex columns the
  // skeleton render layer adds around the store's own attributes; a store
  // attribute must not claim either name or it would shadow them in the
  // shader's `prop_<id>()` namespace.
  const usedPropertyIds = new Set<string>(["tangent", "segment"]);

  for (const name of candidateNames) {
    const arrayMeta = metadataByName.get(name);
    // Non-null by construction: `resolveAttributeSelection` only returns names
    // whose dtype it read and accepted.
    const dtype = dtypeByName.get(name)!;
    attributeNames.push(name);
    attributeDtypes.push(dtype as ZarrVectorsAttributeDtype);

    // Dictionary-encoded (categorical/enum) attributes: surface as a
    // numeric annotation property with enum_values + enum_labels so
    // shaders see category names, not raw codes.  The on-disk
    // convention is documented in zarr-vectors:
    // ``encoding: "dictionary"`` + ``categories: [...]`` lives in the
    // attribute array's metadata block.
    let enumValues: number[] | undefined;
    let enumLabels: string[] | undefined;
    if (arrayMeta?.attributes?.encoding === "dictionary") {
      const categories = arrayMeta.attributes.categories;
      if (Array.isArray(categories)) {
        enumLabels = categories.map((c: unknown) => String(c));
        enumValues = enumLabels.map((_, i) => i);
      }
    }

    // The property identifier is decoupled from the on-disk attribute name:
    // `attributeNames[i]` stays the directory to read, while `id` is the
    // GLSL-legal name the shader sees.  Both arrays remain index-parallel.
    const hint = declaredByName.get(name);
    const {
      identifier: _unusedIdentifier,
      id: _unusedId,
      ...hintRest
    } = hint ?? {};
    const declaredId = String(hint?.identifier ?? hint?.id ?? name);
    const id = toAnnotationPropertyId(declaredId, usedPropertyIds);
    rawPropertyJson.push({
      ...hintRest,
      id,
      description:
        hintRest.description ?? (id === declaredId ? undefined : declaredId),
      // float32 regardless of the on-disk dtype: every attribute is decoded to
      // float32 before it reaches the GPU, so the property type the shader UI
      // advertises has to say the same thing.
      type: hint?.type ?? "float32",
      // Hints win for enum metadata; only fill in from the on-disk
      // dictionary when the user didn't already specify it.
      enum_values: hint?.enum_values ?? enumValues,
      enum_labels: hint?.enum_labels ?? enumLabels,
    });
  }

  let properties: AnnotationPropertySpec[];
  try {
    properties = parseAnnotationPropertySpecs(rawPropertyJson);
  } catch (e) {
    throw new Error(
      `Failed to parse annotation property specs from zarr-vectors hints: ${(e as Error).message}`,
    );
  }
  return { properties, attributeNames, attributeDtypes };
}

/**
 * List per-object attribute names by enumerating subdirectories under
 * the level's `object_attributes/`.  Returns the empty list when the
 * directory is absent (older stores that don't carry per-object
 * attributes).
 */
async function listObjectAttributeNames(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  options: Partial<ProgressOptions>,
): Promise<string[]> {
  const dirUrl = joinBaseUrlAndPath(levelUrl, "object_attributes/");
  let response;
  try {
    response = await sharedKvStoreContext.kvStoreContext.list(dirUrl, {
      responseKeys: "suffix",
      ...options,
    });
  } catch (e) {
    // See listAttributeNames: a throw is a listing failure/denial, not an
    // absent directory, so warn rather than silently dropping the
    // segment-properties panel.
    warnOnceFe(
      "object-attributes-list",
      "zarr-vectors: could not list object_attributes/ (object listing may be " +
        "denied on this store); the segment-properties panel will be unavailable " +
        `even if object attributes exist. ${e instanceof Error ? e.message : String(e)}`,
    );
    return [];
  }
  return response.directories
    .map((d) => (d.endsWith("/") ? d.slice(0, -1) : d))
    .filter((d) => d.length > 0)
    .sort();
}

/**
 * Read a level's `groups/` array — the store's own named partition of the
 * objects (tract bundles, cell classes) — into a per-object group id.
 *
 * This is the I/O half: fetch the array metadata, read every chunk of the
 * group axis, decompress and decode it.  Turning the blobs into a per-object
 * group id is `buildObjectGroupMembership`.
 *
 * Returns `undefined` when the level has no `groups/` array, or when the array
 * is present but unreadable — a store whose bundles cannot be recovered is
 * still worth opening for its geometry.
 */
async function readObjectGroups(
  sharedKvStoreContext: SharedKvStoreContext,
  levelUrl: string,
  options: Partial<ProgressOptions>,
): Promise<ObjectGroupMembership | undefined> {
  const arrayUrl = joinBaseUrlAndPath(levelUrl, "groups/");
  const meta = await getJsonResource(
    sharedKvStoreContext,
    joinBaseUrlAndPath(arrayUrl, "zarr.json"),
    "groups metadata",
    options,
  );
  if (meta === undefined) return undefined;
  const attrs = meta?.attributes ?? {};
  if (attrs.zv_array !== undefined && attrs.zv_array !== "groups") {
    warnOnceFe(
      "groups-marker",
      `zarr-vectors: ${arrayUrl} is not a groups array ` +
        `(zv_array=${JSON.stringify(attrs.zv_array)}); ignoring it.`,
    );
    return undefined;
  }
  const numGroups = parseGroupCount(attrs, meta?.shape);
  if (numGroups === 0) return undefined;

  // The array is chunked along its single group axis, so a store with many
  // groups spreads them over several chunks.  Read every one — stopping at
  // `c/0` is exactly the bug this reader had for object attributes.
  const chunkShape = meta?.chunk_grid?.configuration?.chunk_shape;
  const rowsPerChunk = Array.isArray(chunkShape)
    ? Number(chunkShape[0])
    : numGroups;
  const numChunks =
    Number.isFinite(rowsPerChunk) && rowsPerChunk > 0
      ? Math.ceil(numGroups / rowsPerChunk)
      : 1;
  const blobs: Uint8Array[] = [];
  try {
    for (let chunk = 0; chunk < numChunks; ++chunk) {
      const response = await sharedKvStoreContext.kvStoreContext.read(
        joinBaseUrlAndPath(arrayUrl, `c/${chunk}`),
        options,
      );
      // An unwritten chunk is legal zarr (every group in it is empty); the
      // fill value for vlen-bytes is the empty blob, which is what the
      // membership builder sees when the blob list runs short.
      if (response === undefined) continue;
      const raw = new Uint8Array(
        (await response.response.arrayBuffer()) as ArrayBuffer,
      );
      const bytes = await maybeDecompressObjAttr(
        raw,
        options.signal ?? new AbortController().signal,
      );
      blobs.push(...decodeVlenBytesChunk(bytes));
    }
  } catch (e) {
    warnOnceFe(
      "groups-read",
      "zarr-vectors: could not read groups/; bundle names will be " +
        `unavailable. ${e instanceof Error ? e.message : String(e)}`,
    );
    return undefined;
  }

  const decoded = buildObjectGroupMembership(attrs, blobs, meta?.shape);
  if (decoded === undefined) return undefined;
  if (decoded.overlaps > 0) {
    // One tag per object is what the segment-properties encoding models here;
    // say so rather than silently showing the last writer's group.
    warnOnceFe(
      "groups-overlap",
      `zarr-vectors: ${decoded.overlaps} object(s) belong to more than one ` +
        "group; each is tagged with the highest-numbered group it belongs to.",
    );
  }
  return decoded.membership;
}

/**
 * Build a `SegmentPropertyMap` from level 0's `object_attributes/` and
 * `groups/`.  Each scalar (num_channels=1) attribute becomes one numerical
 * column in neuroglancer's segment-properties UI; the row order maps directly
 * to object_ids `[0, 1, ..., O-1]` (dense layout per spec §6).  Returns
 * `undefined` when the store has neither object attributes nor groups — the
 * caller skips the subsource entry in that case.
 *
 * The store's named groups become a `tags` property plus a `label`, which is
 * what makes them selectable BY NAME: the segment list filters on `#<name>`,
 * and its visibility-toggle-all control then shows or hides exactly that
 * group's objects.  A numerical column alone could only be filtered by the
 * group's id.
 *
 * Multi-channel (num_channels > 1) attributes are silently dropped:
 * neuroglancer's segment-properties UI is scalar-per-column, and
 * splitting an O×C attribute into C named columns would require a
 * naming convention writers don't currently follow.  (Colouring
 * streamlines by a 3-vector attribute as RGB is a separate path.)
 * Wide scalar dtypes (float64 / int64 / uint64) ARE accepted, downcast
 * to float32 so common computed attributes (tortuosity, counts) are not
 * lost; only genuinely unrepresentable dtypes are skipped.
 *
 * `present_mask` arrays are honoured: rows with mask=0 are dropped
 * from the segment-properties output so absent objects don't appear
 * with zero-padded values.
 */
async function buildSegmentPropertyMap(
  sharedKvStoreContext: SharedKvStoreContext,
  level0Url: string,
  options: Partial<ProgressOptions>,
): Promise<SegmentPropertyMap | undefined> {
  const [names, groups] = await Promise.all([
    listObjectAttributeNames(sharedKvStoreContext, level0Url, options),
    readObjectGroups(sharedKvStoreContext, level0Url, options),
  ]);
  if (names.length === 0 && groups === undefined) return undefined;

  // Read each attribute's metadata in parallel.  Skip attributes the
  // segment-properties UI can't represent rather than failing the whole
  // map — keeps the store openable even when one column is exotic.
  type AttrSpec = {
    name: string;
    dataType: DataType;
    values: ReturnType<typeof reinterpretObjectAttributeBytes>;
    /**
     * For a 64-bit integer column only: the same values before the lossy
     * float32 downcast, kept so an id column can key the map exactly. See
     * {@link OBJECT_ID_COLUMN_PREFERENCE}.
     */
    wideValues?: BigUint64Array;
    presentMask?: Uint8Array;
    numObjects: number;
  };
  const specs = await mapWithConcurrency(
    names,
    ATTRIBUTE_READ_CONCURRENCY,
    async (name): Promise<AttrSpec | undefined> => {
      const arrayUrl = joinBaseUrlAndPath(
        level0Url,
        `object_attributes/${name}/`,
      );
      const meta = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(arrayUrl, "zarr.json"),
        `object_attribute ${JSON.stringify(name)} metadata`,
        options,
      );
      if (meta === undefined) return undefined;
      const attrs = meta?.attributes ?? meta;
      // Semantic dtype rides the group attributes; fall back to the array's own
      // `data_type` for robustness.
      const dtype = String(attrs?.dtype ?? meta?.data_type ?? "");
      // Shape is `[O]` for a scalar column or `[O, C]` for a C-vector; the
      // trailing dim is the channel count. (There is no `num_channels` field.)
      const shape = Array.isArray(attrs?.shape)
        ? attrs.shape
        : Array.isArray(meta?.shape)
          ? meta.shape
          : undefined;
      if (shape === undefined || shape.length === 0) return undefined;
      const numObjects = Number(shape[0]);
      const numChannels =
        shape.length >= 2 ? Number(shape[shape.length - 1]) : 1;
      if (numObjects <= 0) return undefined;
      const entry = OBJECT_ATTR_DTYPE_TABLE[dtype];
      // Wide scalar dtypes are downcast to float32 (see reinterpretWideToFloat32)
      // so a float64 tortuosity or int64 count is still discovered.
      const wideKind =
        dtype === "float64"
          ? "float64"
          : dtype === "int64"
            ? "int64"
            : dtype === "uint64"
              ? "uint64"
              : undefined;
      if (
        numChannels !== 1 ||
        (entry === undefined && wideKind === undefined)
      ) {
        // Multi-channel attributes (e.g. a 3-vector orientation) are not scalar
        // columns; truly exotic dtypes can't be represented — skip both.
        return undefined;
      }
      const dataType = entry?.dataType ?? DataType.FLOAT32;
      // On-disk layout: `object_attributes/<name>/` is itself a zarr v3 array
      // (not a group with a nested `data/` array) whose grid holds the whole
      // column in ONE chunk, at `c/<0…>/` under the default chunk-key encoding
      // ("/" separator): `c/0` for a scalar `[O]` array, `c/0/0` for a `[O, C]`
      // vector. Read that blob directly; the semantic dtype is carried in the
      // array attributes.
      //
      // The chunk is full size even when the column is shorter — a writer that
      // picks a fixed `chunk_shape` (e.g. 65536) stores `shape[0]` real values
      // followed by fill-value padding — so decode `chunkElements` and trim.
      const chunkShape: number[] = Array.isArray(
        meta?.chunk_grid?.configuration?.chunk_shape,
      )
        ? (meta.chunk_grid.configuration.chunk_shape as number[]).map(Number)
        : shape.map(Number);
      if (chunkShape.length !== shape.length || chunkShape[0] < numObjects) {
        // More than one chunk along the object axis: this single-chunk read
        // would silently truncate the column, so skip it instead.
        warnOnceFe(
          "object-attributes-multi-chunk",
          `zarr-vectors: object attribute ${JSON.stringify(name)} spans ` +
            `multiple chunks (shape ${shape.join(",")}, chunk ` +
            `${chunkShape.join(",")}); skipping — only single-chunk object ` +
            "attribute columns are supported.",
        );
        return undefined;
      }
      const chunkElements = chunkShape.reduce((a, b) => a * b, 1);
      const chunkKey = `c/${shape.map(() => 0).join("/")}`;
      const dataResponse = await sharedKvStoreContext.kvStoreContext.read(
        joinBaseUrlAndPath(arrayUrl, chunkKey),
        options,
      );
      if (dataResponse === undefined) return undefined;
      const rawBytes = new Uint8Array(
        (await dataResponse.response.arrayBuffer()) as ArrayBuffer,
      );
      // Decompress if the writer applied zstd (the zarr codec pipeline writes
      // zstd by default; we bypass zarr's own chunk decoder here and handle
      // it manually so we can read the semantic dtype from the outer group).
      const bytes = await maybeDecompressObjAttr(
        rawBytes,
        options.signal ?? new AbortController().signal,
      );
      const values =
        entry !== undefined
          ? reinterpretObjectAttributeBytes(
              bytes,
              entry.ctor,
              entry.elementSize,
              numObjects,
              chunkElements,
            )
          : reinterpretWideToFloat32(
              bytes,
              wideKind!,
              numObjects,
              chunkElements,
            );
      let presentMask: Uint8Array | undefined;
      if (attrs?.has_present_mask === true) {
        const maskResponse = await sharedKvStoreContext.kvStoreContext.read(
          joinBaseUrlAndPath(arrayUrl, "present_mask/c/0"),
          options,
        );
        if (maskResponse !== undefined) {
          const rawMaskBytes = new Uint8Array(
            (await maskResponse.response.arrayBuffer()) as ArrayBuffer,
          );
          const fullMask = await maybeDecompressObjAttr(
            rawMaskBytes,
            options.signal ?? new AbortController().signal,
          );
          // Same padded-chunk trim as the values above.
          presentMask =
            fullMask.length > numObjects
              ? fullMask.subarray(0, numObjects)
              : fullMask;
        }
      }
      return {
        name,
        dataType,
        values,
        // Keep the undamaged 64-bit values for integer columns so one of them
        // can supply the id space below. Cheap: 8 bytes per object.
        wideValues:
          wideKind === "int64" || wideKind === "uint64"
            ? reinterpretWideToBigUint64(bytes, numObjects, chunkElements)
            : undefined,
        presentMask,
        numObjects,
      };
    },
  );

  // Reconcile object counts across attributes — they MUST agree because
  // every row is keyed by the same global object_id space.  Use the
  // first present-mask-aware count as the reference and union of
  // present masks for the id list.
  const present = specs.filter((s): s is AttrSpec => s !== undefined);
  if (present.length === 0 && groups === undefined) return undefined;
  // The attribute columns are authoritative on the object count when there are
  // any: `groups/` records only the ids it has members for, so a trailing run
  // of ungrouped objects would go missing if its length led here.
  const numObjects =
    present.length > 0 ? present[0].numObjects : groups!.groupByObject.length;
  for (const s of present) {
    if (s.numObjects !== numObjects) {
      throw new Error(
        `zarr-vectors object_attributes: row count mismatch ` +
          `(${s.name}=${s.numObjects} vs ${numObjects})`,
      );
    }
  }
  // Effective present-mask: AND of all attribute masks (a row is
  // emitted only if every attribute considers it real).  In the common
  // case where no attribute carries a mask, every row is kept.
  let effectiveMask: Uint8Array | undefined;
  for (const s of present) {
    if (s.presentMask === undefined) continue;
    if (effectiveMask === undefined) {
      effectiveMask = new Uint8Array(s.presentMask);
    } else {
      for (let i = 0; i < numObjects; ++i) {
        effectiveMask[i] = effectiveMask[i] && s.presentMask[i] ? 1 : 0;
      }
    }
  }
  const keepIndices: number[] = [];
  for (let i = 0; i < numObjects; ++i) {
    if (effectiveMask === undefined || effectiveMask[i] === 1) {
      keepIndices.push(i);
    }
  }
  if (keepIndices.length === 0) return undefined;

  // Key the map by the store's own object ids, not by row position.
  //
  // This used to be `ids[i] = BigInt(keepIndices[i])` — the dense row ordinal
  // 0,1,2,…  But the ids a segment-property map is looked up BY are the
  // segment ids the geometry carries: the per-fragment `segment_id` the chunk
  // downloader expands into the per-vertex segment column, which for a
  // connectomics store is a root id of order 1e17. Row ordinals and root ids
  // are disjoint id spaces, so every lookup missed and per-object colouring
  // and the attribute display silently did nothing — no error, because a
  // property map that answers "no such segment" is indistinguishable from one
  // that is simply sparse.
  //
  // The id column is whichever of these the store carries, in this order; the
  // fallback to row ordinals is kept for stores that carry none, where row
  // position genuinely is the id space.
  const idColumn = OBJECT_ID_COLUMN_PREFERENCE.map((wanted) =>
    present.find((s) => s.name === wanted && s.wideValues !== undefined),
  ).find((s) => s !== undefined);
  const ids = new BigUint64Array(keepIndices.length);
  if (idColumn !== undefined) {
    const wide = idColumn.wideValues!;
    for (let i = 0; i < keepIndices.length; ++i) {
      ids[i] = wide[keepIndices[i]];
    }
  } else {
    warnOnceFe(
      "object-attributes-no-id-column",
      "zarr-vectors: object_attributes/ carries none of " +
        `${OBJECT_ID_COLUMN_PREFERENCE.join(", ")} as a 64-bit integer ` +
        "column, so the segment-property map is keyed by row position. " +
        "Per-object colouring and attribute display will only work if the " +
        "store's segment ids really are 0,1,2,…",
    );
    for (let i = 0; i < keepIndices.length; ++i) {
      ids[i] = BigInt(keepIndices[i]);
    }
  }

  const properties: InlineSegmentProperty[] = [];
  for (const s of present) {
    const compactValues = new (s.values.constructor as new (
      n: number,
    ) => typeof s.values)(keepIndices.length);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < keepIndices.length; ++i) {
      const v = s.values[keepIndices[i]];
      (compactValues as unknown as { [k: number]: number })[i] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      min = 0;
      max = 0;
    }
    const numerical: InlineSegmentNumericalProperty = {
      id: s.name,
      type: "number",
      dataType: s.dataType,
      description: undefined,
      values:
        compactValues as unknown as InlineSegmentNumericalProperty["values"],
      bounds: [min, max],
    };
    properties.push(numerical);
  }

  if (groups !== undefined) {
    properties.push(
      ...groupSegmentProperties(
        groups,
        keepIndices,
        new Set(properties.map((p) => p.id)),
      ),
    );
  }

  const inline: InlineSegmentPropertyMap = normalizeInlineSegmentPropertyMap({
    ids,
    properties,
  });
  return new SegmentPropertyMap({ inlineProperties: inline });
}

const FALLBACK_LEVEL_LIMIT = 1_000_000;

function parsePyramidMode(raw: unknown): ZarrVectorsPyramidMode {
  if (raw === "additive" || raw === "replace") return raw;
  // Default: "replace" — see plan §"Context (v2)".  Safe for
  // metanode-style pyramids (no double-count) and matches the
  // image-style mental model of resolution alternatives.  A single
  // level falls through both branches identically.
  return "replace";
}

function enumerateLevelPaths(multiscales: any): string[] {
  const entry = Array.isArray(multiscales) ? multiscales[0] : undefined;
  const datasets = entry?.datasets;
  if (Array.isArray(datasets) && datasets.length > 0) {
    const paths: string[] = [];
    for (const d of datasets) {
      if (typeof d?.path === "string" && d.path.length > 0) {
        paths.push(d.path);
      }
    }
    if (paths.length > 0) return paths;
  }
  // No multiscales metadata → assume a single level "0".  This
  // preserves v1 behavior for stores written before pyramid support.
  return ["0"];
}

/**
 * Per-object vertex counts at every level, **finest-first**, with `0` marking an
 * object absent from that level.
 *
 * `object_attributes/vertex_count` is written at every level of a per-object
 * pyramid, dense over the shared object-id space, with the array's `fill_value`
 * standing in for "this level dropped that object". One ~2 MB read per level
 * therefore yields two things at once that nothing else in the store provides:
 * an **exact per-object cost**, and an **exact per-level membership mask**.
 * Together they are what lets a budget be spent on a chosen subset of objects
 * instead of on whichever pyramid rung happens to fit.
 *
 * Returns `undefined` if any level lacks the array or reports a different object
 * count, since a partial picture would silently under-admit whole levels.
 */
/**
 * Membership of level `level` as a bitset over the object-id space, or
 * `undefined` when the store carries no membership data.
 *
 * A level index past the coarsest yields an all-zero bitset rather than
 * `undefined`: "no coarser backbone" is a real answer, distinct from "cannot
 * tell", and the two must not collapse or the coarsest level would silently
 * lose per-object admission.
 */
function coarserMembershipBitset(
  perLevelObjectVertexCounts: Uint32Array[] | undefined,
  level: number,
): Uint8Array | undefined {
  if (perLevelObjectVertexCounts === undefined) return undefined;
  const numObjects = perLevelObjectVertexCounts[0].length;
  const bitset = new Uint8Array((numObjects + 7) >> 3);
  const counts = perLevelObjectVertexCounts[level];
  if (counts !== undefined) {
    for (let id = 0; id < numObjects; ++id) {
      if (counts[id] !== 0) bitset[id >> 3] |= 1 << (id & 7);
    }
  }
  return bitset;
}

let warnedNotNested = false;
function warnOnceNotNested() {
  if (warnedNotNested) return;
  warnedNotNested = true;
  console.warn(
    "zarr-vectors: pyramid levels are not nested subsets of one object id " +
      "space, so per-object budgeting is unavailable; falling back to " +
      "whole-level selection.",
  );
}

async function readPerLevelObjectVertexCounts(
  sharedKvStoreContext: SharedKvStoreContext,
  storeUrl: string,
  levelPaths: readonly string[],
  options: Partial<ProgressOptions>,
): Promise<Uint32Array[] | undefined> {
  const perLevel = await Promise.all(
    levelPaths.map(async (levelPath) => {
      const arrayUrl = joinBaseUrlAndPath(
        kvstoreEnsureDirectoryPipelineUrl(pipelineUrlJoin(storeUrl, levelPath)),
        "object_attributes/vertex_count/",
      );
      const meta = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(arrayUrl, "zarr.json"),
        `zarr-vectors level ${JSON.stringify(levelPath)} vertex_count metadata`,
        options,
      );
      if (meta === undefined) return undefined;
      const attrs = meta?.attributes ?? meta;
      const shape = Array.isArray(attrs?.shape) ? attrs.shape : meta?.shape;
      if (!Array.isArray(shape) || shape.length !== 1) return undefined;
      const numObjects = Number(shape[0]);
      if (!Number.isFinite(numObjects) || numObjects <= 0) return undefined;
      const response = await sharedKvStoreContext.kvStoreContext.read(
        joinBaseUrlAndPath(arrayUrl, "c/0"),
        options,
      );
      if (response === undefined) return undefined;
      const bytes = await maybeDecompressObjAttr(
        new Uint8Array((await response.response.arrayBuffer()) as ArrayBuffer),
        options.signal ?? new AbortController().signal,
      );
      if (bytes.byteLength < numObjects * 4) return undefined;
      // Copy rather than view: the decompressed buffer's offset is not
      // guaranteed 4-byte aligned, and this is retained for the session.
      const counts = new Uint32Array(numObjects);
      const view = new DataView(bytes.buffer, bytes.byteOffset, numObjects * 4);
      // The sentinel means "absent"; normalise it to 0 so membership is simply
      // `count !== 0` everywhere downstream.
      const sentinel = Number(
        meta?.fill_value ?? attrs?.fill_value ?? 0xffffffff,
      );
      for (let i = 0; i < numObjects; ++i) {
        const v = view.getUint32(i * 4, /*littleEndian=*/ true);
        counts[i] = v === sentinel ? 0 : v;
      }
      return counts;
    }),
  );
  if (perLevel.some((c) => c === undefined)) return undefined;
  const counts = perLevel as Uint32Array[];
  const numObjects = counts[0].length;
  if (counts.some((c) => c.length !== numObjects)) return undefined;
  return counts;
}

function computeLevelLimit(
  levelAttrs: any,
  numChunks: number,
  level0Limit: number,
  rank: number,
): number {
  const vertexCount = Number(levelAttrs?.vertex_count);
  if (Number.isFinite(vertexCount) && vertexCount > 0 && numChunks > 0) {
    return Math.max(1, Math.ceil(vertexCount / numChunks));
  }
  const binRatio = levelAttrs?.bin_ratio;
  if (Array.isArray(binRatio) && binRatio.length === rank) {
    let prod = 1;
    for (const v of binRatio) prod *= Math.max(1, Number(v) || 1);
    if (prod > 1) return Math.max(1, Math.round(level0Limit / prod));
  }
  return level0Limit;
}

/**
 * The on-disk format version this reader targets. ZVF 0.9.0 was a breaking
 * change (connectivity moved to the `links/<delta>/<offsets>/` family and the
 * single-array chunk layout landed); the spec states pre-0.9 stores are not
 * readable by 0.9+ readers.
 */
const SUPPORTED_ZV_MAJOR = 0;
const MIN_SUPPORTED_ZV_MINOR = 9;

/**
 * Gate on `zarr_vectors.zv_version` so a version-driven layout change surfaces a
 * diagnostic instead of failing silently downstream (as the 0.9 links move did
 * before this reader was migrated). Throws for a store older than this reader
 * supports; warns for a missing or newer-than-target version and proceeds.
 */
function checkZvVersion(zv: any): void {
  const raw = zv?.zv_version;
  if (raw === undefined) {
    console.warn(
      "zarr-vectors: store does not declare zv_version; assuming a ZVF " +
        `${SUPPORTED_ZV_MAJOR}.${MIN_SUPPORTED_ZV_MINOR}-compatible layout.`,
    );
    return;
  }
  const match = /^(\d+)\.(\d+)(?:\.\d+)?$/.exec(String(raw));
  if (match === null) {
    console.warn(
      `zarr-vectors: unrecognised zv_version ${JSON.stringify(raw)}; ` +
        "proceeding as if it were the supported version.",
    );
    return;
  }
  const major = Number(match[1]);
  const minor = Number(match[2]);
  if (
    major < SUPPORTED_ZV_MAJOR ||
    (major === SUPPORTED_ZV_MAJOR && minor < MIN_SUPPORTED_ZV_MINOR)
  ) {
    throw new Error(
      `zarr-vectors: store zv_version ${raw} predates the ${SUPPORTED_ZV_MAJOR}.` +
        `${MIN_SUPPORTED_ZV_MINOR}.0 layout this reader requires (connectivity and ` +
        "chunk layout changed in 0.9.0). Rewrite the store from source with a current " +
        "zarr-vectors writer.",
    );
  }
  if (major > SUPPORTED_ZV_MAJOR || minor > MIN_SUPPORTED_ZV_MINOR) {
    console.warn(
      `zarr-vectors: store zv_version ${raw} is newer than this reader's target ` +
        `${SUPPORTED_ZV_MAJOR}.${MIN_SUPPORTED_ZV_MINOR}; newer features may not render.`,
    );
  }
}

async function buildAnnotationMetadata(
  sharedKvStoreContext: SharedKvStoreContext,
  storeUrl: string,
  rootAttrs: any,
  options: Partial<ProgressOptions>,
  /**
   * When false (default), require `geometry_types` to declare
   * `"point_cloud"` — protects callers that came through the main
   * `zarr-vectors:` dispatcher.  When true, skip the check so the
   * `zarr-vectors-pointcloud:` alias can re-interpret any zarr-vectors
   * store (graph, skeleton, polyline, streamline) as a point cloud by
   * reading only its vertex / vertex_attribute arrays.
   */
  allowAnyGeometry: boolean = false,
): Promise<AnnotationMetadata> {
  const zv = rootAttrs?.zarr_vectors;
  if (zv === undefined) {
    throw new Error(
      "Not a zarr-vectors store: root attributes lack a 'zarr_vectors' block",
    );
  }
  checkZvVersion(zv);
  // Geometry-type validation lives at the dispatch layer
  // (`ZarrVectorsDataSource.get`) — by the time `buildAnnotationMetadata`
  // is called, the dispatcher has already verified that `geometry_types`
  // is `["point_cloud"]`.  Re-validate defensively here so a direct
  // caller (e.g. tests) gets a clear error.  The pointcloud-alias
  // provider passes `allowAnyGeometry=true` to skip this re-validation.
  const geometryTypes: string[] = Array.isArray(zv.geometry_types)
    ? zv.geometry_types
    : [];
  if (!allowAnyGeometry && !geometryTypes.includes("point_cloud")) {
    throw new Error(
      `buildAnnotationMetadata: called for a non-point_cloud store ` +
        `(geometry_types=${JSON.stringify(geometryTypes)})`,
    );
  }
  const bounds = zv.bounds;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 2 ||
    !Array.isArray(bounds[0]) ||
    !Array.isArray(bounds[1])
  ) {
    throw new Error(
      "zarr-vectors store: 'bounds' must be [[lower...], [upper...]]",
    );
  }
  const rank = bounds[0].length;
  if (bounds[1].length !== rank) {
    throw new Error(
      "zarr-vectors store: bounds[0] and bounds[1] have different rank",
    );
  }
  const chunkShape = zv.chunk_shape;
  if (!Array.isArray(chunkShape) || chunkShape.length !== rank) {
    throw new Error(`zarr-vectors store: 'chunk_shape' must have rank ${rank}`);
  }

  const lowerBounds = Float64Array.from(bounds[0], Number);
  const upperBounds = Float64Array.from(bounds[1], Number);

  // Build coordinate space: prefer neuroglancer hints, fall back to
  // NGFF multiscales.
  const ngHints = rootAttrs.neuroglancer ?? {};
  let coordinateSpace: ReturnType<typeof makeCoordinateSpace>;
  if (
    ngHints.coordinate_space &&
    Array.isArray(ngHints.coordinate_space.names) &&
    ngHints.coordinate_space.names.length === rank
  ) {
    coordinateSpace = buildCoordinateSpaceFromHints(
      ngHints.coordinate_space,
      lowerBounds,
      upperBounds,
    );
  } else {
    coordinateSpace = buildCoordinateSpaceFromMultiscales(
      rootAttrs.multiscales,
      lowerBounds,
      upperBounds,
      rank,
    );
  }

  const pyramidMode = parsePyramidMode(ngHints.pyramid_mode);
  const levelPaths = enumerateLevelPaths(rootAttrs.multiscales);

  // Property discovery runs once against level 0 — per the
  // zarr-vectors spec, attribute dtypes don't vary across levels.
  const level0Url = kvstoreEnsureDirectoryPipelineUrl(
    pipelineUrlJoin(storeUrl, levelPaths[0]),
  );
  const { properties, attributeNames, attributeDtypes } =
    await buildPropertySpecsAndDtypes(
      sharedKvStoreContext,
      level0Url,
      ngHints,
      options,
    );

  // Shared per-level geometry: all levels share the same physical
  // chunk grid on disk in zarr-vectors.
  const chunkShapeF32 = new Float32Array(rank);
  let numChunks = 1;
  for (let i = 0; i < rank; ++i) {
    const cs = Number(chunkShape[i]);
    chunkShapeF32[i] = cs;
    const extent = upperBounds[i] - lowerBounds[i];
    const g = Math.max(1, Math.ceil(extent / cs));
    numChunks *= g;
  }
  // zarr-vectors chunks are indexed around world origin (chunk
  // `(i, j, k)` covers world `[i*chunkShape, (i+1)*chunkShape]`) — see
  // `assign_chunks` in zarr-vectors-py: it computes the chunk index
  // as `floor(world / chunk_shape)` with no `lowerBounds` offset.  So
  // the chunk-to-multiscale transform must be identity (NOT translated
  // by `lowerBounds` like an image-style anchor-at-lower-corner store)
  // for the URL the reader fetches to match the URL the writer wrote.
  // The data-extent gating still works because
  // `makeSliceViewChunkSpecification` derives `lower/upperChunkBound`
  // from `lower/upperVoxelBound`.
  //
  // Matches `getSliceViewSources` in `geometry_frontend.ts` (also
  // identity) — the two paths agree about the chunk-coord convention.
  const chunkToMultiscaleTransform = matrix.createIdentity(
    Float32Array,
    rank + 1,
  );

  // Build one spatial-index level per zarr-vectors level.  Order:
  // finest first (level 0 first), which is the order
  // forEachVisibleAnnotationChunk expects (it iterates length-1 → 0).
  const spatialIndices: AnnotationSpatialIndexLevelMetadata[] = [];
  let level0Limit = FALLBACK_LEVEL_LIMIT;
  for (let k = 0; k < levelPaths.length; ++k) {
    const levelPath = levelPaths[k];
    const levelUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(storeUrl, levelPath),
    );
    let levelAttrs: any;
    try {
      const levelJson = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(levelUrl, "zarr.json"),
        `zarr-vectors level ${JSON.stringify(levelPath)} metadata`,
        options,
      );
      levelAttrs = levelJson?.attributes?.zarr_vectors_level;
    } catch {
      levelAttrs = undefined;
    }
    const limit = computeLevelLimit(levelAttrs, numChunks, level0Limit, rank);
    if (k === 0) level0Limit = limit;

    // Anchor the chunk grid at the world origin, NOT at `lowerBounds`,
    // to match the writer's `floor(world / chunkShape)` convention.
    // Pass the actual `[lowerBounds, upperBounds]` extent so
    // `makeSliceViewChunkSpecification` derives a chunk-index range
    // that can include negative indices when `lowerBounds < 0` (the
    // typical case for stores whose data straddles the world origin).
    const spec: AnnotationGeometryChunkSpecification = {
      limit,
      chunkToMultiscaleTransform,
      pyramidMode,
      ...makeSliceViewChunkSpecification({
        rank,
        chunkDataSize: chunkShapeF32,
        lowerVoxelBound: Float32Array.from(lowerBounds),
        upperVoxelBound: Float32Array.from(upperBounds),
      }),
    };

    const spatialParams =
      new ZarrVectorsAnnotationSpatialIndexSourceParameters();
    spatialParams.baseUrl = levelUrl;
    spatialParams.rank = rank;
    spatialParams.attributeNames = attributeNames;
    spatialParams.attributeDtypes = attributeDtypes;

    spatialIndices.push({ parameters: spatialParams, spec });
  }

  const parameters = new ZarrVectorsAnnotationSourceParameters();
  parameters.rank = rank;
  parameters.type = AnnotationType.POINT;
  parameters.properties = properties;
  parameters.pyramidMode = pyramidMode;

  const meta: AnnotationMetadata = {
    rank,
    coordinateSpace,
    coordinateOffset: readCoordinateOffset(zv, rank),
    parameters,
    spatialIndices,
  };
  return meta;
}

function getAnnotationDataSource(
  sharedKvStoreContext: SharedKvStoreContext,
  metadata: AnnotationMetadata,
): DataSource {
  return {
    modelTransform: makeCoordinateOffsetTransform(
      metadata.coordinateSpace,
      metadata.coordinateOffset,
    ),
    subsources: [
      {
        id: "default",
        default: true,
        subsource: {
          annotation: sharedKvStoreContext.chunkManager.getChunkSource(
            ZarrVectorsAnnotationSource,
            {
              sharedKvStoreContext,
              metadata,
              parameters: metadata.parameters,
            },
          ),
        },
      },
    ],
  };
}

// ---------------------------------------------------------------
// Skeleton / polyline / streamline path
// ---------------------------------------------------------------

interface GeometryMetadata {
  rank: number;
  coordinateSpace: ReturnType<typeof makeCoordinateSpace>;
  /** Stored→world offset, if the store declares one; see {@link readCoordinateOffset}. */
  coordinateOffset: Float64Array | undefined;
  /**
   * Whether the store's geometry kind has the discrete-object model.  False
   * for `point_cloud`, which has no `object_index/manifests` for the
   * per-segment (pass-2) source to resolve and no `object_attributes/` to
   * build a segment-property map from.
   */
  hasObjectModel: boolean;
  /** What primitive the store's geometry draws as; see {@link KIND_CAPABILITIES}. */
  primitive: "points" | "lines" | "triangles";
  /** Parameters for the per-segment (pass-2) chunk source. */
  pass2Params: ZarrVectorsObjectKeyedGeometrySourceParameters;
  /**
   * Per-object attributes assembled into a neuroglancer
   * `SegmentPropertyMap`.  `undefined` when the store has no
   * `object_attributes/` at level 0.  Exposed by
   * `getGeometryDataSource` as the opt-in `"properties"` subsource.
   */
  segmentPropertyMap?: SegmentPropertyMap;
  /**
   * Per-level parameter blobs for the spatially-indexed (pass-1) chunk
   * sources, finest-first.  Together with `spatialGrid` they let the
   * multiscale source build the per-level chunk specs.
   *
   * `undefined` when the store's rank is not 3 (neuroglancer's
   * spatially-indexed skeleton machinery hardcodes vec3 position
   * texture format) — the dispatch falls back to pass-2 only in that
   * case.
   */
  pass1Levels?: ReadonlyArray<{
    parameters: ZarrVectorsSpatialGeometrySourceParameters;
  }>;
  /** Grid info shared across all pass-1 levels.  Co-defined with `pass1Levels`. */
  spatialGrid?: {
    /**
     * Per-level chunk shape in world units.  Length == pass1Levels.length.
     * Each entry comes from the level's ``zarr_vectors_level.chunk_shape``
     * override if present, otherwise from root chunk_shape.  Writers
     * that want the spatial grid-resolution picker to expose multiple
     * LOD levels should set ``chunk_scale_factors`` so each level's
     * chunk_shape is monotonically distinct — that's the same pattern
     * CATMAID's per-level ``chunkSize`` follows
     * (`src/datasource/catmaid/frontend.ts:386-390`).  Sparsity-only
     * pyramids without per-level chunk-shape changes still load, but
     * adjacent levels with identical chunk_shape will collapse into a
     * single picker entry.
     */
    /**
     * Per-object vertex counts at each level, finest-first, `0` where the level
     * dropped the object. Both an exact per-object cost table and an exact
     * per-level membership mask; `undefined` when the store omits
     * `object_attributes/vertex_count` at any level.
     */
    perLevelObjectVertexCounts?: Uint32Array[];
    /**
     * Coarsest level containing each object (see `objectDepths`), or `undefined`
     * when the levels are not nested and the depth model does not apply.
     */
    objectDepths?: Uint8Array;
    perLevelChunkShape: Float32Array[];
    /**
     * Live object count per level, parallel to `perLevelChunkShape`;
     * `undefined` where it cannot be determined.  See
     * `computePerLevelObjectCount`.
     *
     * This is what makes an object-sparsity pyramid pickable: such a pyramid
     * keeps the same `chunk_shape` at every level and drops whole objects
     * instead, so chunk size — the framework's usual proxy for detail —
     * cannot tell the levels apart. Object count can. See
     * `getSpatialSkeletonGridSizes` in `geometry_frontend.ts`.
     */
    perLevelObjectCount: (number | undefined)[];
    /**
     * ``zarr_vectors_level.vertex_count`` per level -- the *cost* axis, as
     * distinct from `perLevelObjectCount`'s *detail* axis.  See
     * `getSpatialSkeletonLevelCostsBytes`.
     */
    perLevelVertexCount: (number | undefined)[];
    /**
     * World-space lower bound of the data.  Can be negative — zarr-vectors
     * chunks are indexed around world origin `(0,0,0)`, NOT around
     * `lowerBounds`.  `makeSliceViewChunkSpecification` consumes this as
     * `lowerVoxelBound` and computes negative chunk indices accordingly.
     */
    lowerBounds: Float32Array;
    /** World-space upper bound of the data. */
    upperBounds: Float32Array;
  };
}

/**
 * Geometry types that route through the spatially-indexed geometry path — i.e.
 * every kind {@link KIND_CAPABILITIES} knows about.  They share one on-disk
 * layout (`vertices/` + `vertex_fragments/` + optional `links/`); what differs
 * per kind is recorded in the capability table, not here.
 */
const SPATIAL_GEOM_KINDS = new Set<string>(Object.keys(KIND_CAPABILITIES));

/**
 * Read store metadata for a skeleton / polyline / streamline store and
 * assemble the parameter blob needed to construct chunk sources.
 *
 * Layout assumptions (per the zarr-vectors spec):
 *
 * - `zarr_vectors.geometry_types` contains exactly one of
 *   `"skeleton"`, `"polyline"`, `"streamline"`.
 * - `zarr_vectors.object_index_convention === "standard"` (the only
 *   value that maps to the segmentation-layer pathway).
 * - Level-0 metadata lives under `multiscales[0].datasets[0].path`.
 * - `links/0/.zattrs.dtype` (or `data_type` fallback) declares the
 *   on-disk link dtype; absent for `implicit_sequential` stores.
 */
/**
 * {@link getJsonResource} that treats an unreadable resource as absent.
 *
 * Used where absence is a legitimate answer -- probing whether a store has a
 * links family at all -- rather than an error to surface.
 */
async function getJsonResourceOrUndefined(
  sharedKvStoreContext: SharedKvStoreContext,
  url: string,
  description: string,
  options: Partial<ProgressOptions>,
): Promise<any | undefined> {
  try {
    return await getJsonResource(
      sharedKvStoreContext,
      url,
      description,
      options,
    );
  } catch {
    return undefined;
  }
}

async function buildGeometryMetadata(
  sharedKvStoreContext: SharedKvStoreContext,
  storeUrl: string,
  rootAttrs: any,
  options: Partial<ProgressOptions>,
  selectedAttributes?: readonly string[],
  editServiceUrl?: string,
): Promise<GeometryMetadata> {
  const zv = rootAttrs?.zarr_vectors;
  if (zv === undefined) {
    throw new Error(
      "Not a zarr-vectors store: root attributes lack a 'zarr_vectors' block",
    );
  }
  checkZvVersion(zv);
  const geometryTypes: string[] = Array.isArray(zv.geometry_types)
    ? zv.geometry_types
    : [];
  // A store may DECLARE several geometry types; only one of them can have
  // readable arrays (see `declared_geometry.ts`). Probe `links/0` first so the
  // choice is made from what is on disk rather than from declaration order.
  const linksFamilyJson = await getJsonResourceOrUndefined(
    sharedKvStoreContext,
    joinBaseUrlAndPath(
      kvstoreEnsureDirectoryPipelineUrl(
        pipelineUrlJoin(
          storeUrl,
          enumerateLevelPaths(rootAttrs.multiscales)[0],
        ),
      ),
      "links/0/zarr.json",
    ),
    "zarr-vectors links/0 metadata",
    options,
  );
  const declaredLinkWidth = Number(linksFamilyJson?.attributes?.link_width);
  const resolution = resolveDeclaredGeometry(geometryTypes, {
    hasLinks: linksFamilyJson !== undefined,
    linkWidth: Number.isInteger(declaredLinkWidth)
      ? declaredLinkWidth
      : undefined,
  });
  const geometryKind = resolution.kind;
  if (resolution.skipped.length > 0) {
    warnOnceFe(
      `multi-geometry-${geometryTypes.join(",")}`,
      `zarr-vectors: this store declares geometry types ` +
        `${JSON.stringify(geometryTypes)} but a store holds one set of ` +
        `vertices; reading them as ${JSON.stringify(geometryKind)}` +
        (resolution.ambiguous
          ? " (the arrays did not single one out, so declaration order decided)"
          : "") +
        `. ${JSON.stringify(resolution.skipped)} not rendered: the writer's ` +
        "multi-geometry support is unfinished, so no separate arrays exist " +
        "for them. A partly-written add_geometry() call leaves exactly this " +
        "metadata.",
    );
  }
  if (resolution.unsupported.length > 0) {
    warnOnceFe(
      `unsupported-geometry-${resolution.unsupported.join(",")}`,
      `zarr-vectors: ignoring unrecognised geometry type(s) ` +
        `${JSON.stringify(resolution.unsupported)}.`,
    );
  }
  const caps = KIND_CAPABILITIES[geometryKind];
  // Optional store hint naming a per-vertex attribute that already holds a
  // meaningful integer id (a cell label, a particle id).  Only consulted for
  // kinds without the object model, where each vertex is its own segment.
  const vertexIdAttribute =
    typeof zv.vertex_id_attribute === "string"
      ? zv.vertex_id_attribute
      : undefined;

  // Bounds + rank — identical idiom to the annotation path.
  const bounds = zv.bounds;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 2 ||
    !Array.isArray(bounds[0]) ||
    !Array.isArray(bounds[1])
  ) {
    throw new Error(
      "zarr-vectors store: 'bounds' must be [[lower...], [upper...]]",
    );
  }
  const rank = bounds[0].length;
  if (bounds[1].length !== rank) {
    throw new Error(
      "zarr-vectors store: bounds[0] and bounds[1] have different rank",
    );
  }
  const lowerBounds = Float64Array.from(bounds[0], Number);
  const upperBounds = Float64Array.from(bounds[1], Number);

  // Coordinate space — prefer NGFF multiscales axes / scales.
  const ngHints = rootAttrs.neuroglancer ?? {};
  const coordinateSpace =
    ngHints.coordinate_space &&
    Array.isArray(ngHints.coordinate_space.names) &&
    ngHints.coordinate_space.names.length === rank
      ? buildCoordinateSpaceFromHints(
          ngHints.coordinate_space,
          lowerBounds,
          upperBounds,
        )
      : buildCoordinateSpaceFromMultiscales(
          rootAttrs.multiscales,
          lowerBounds,
          upperBounds,
          rank,
        );

  // Resolve level 0 — the per-segment manifest lookup operates at one
  // fixed level (the v1 dispatch always uses level 0).  Multi-level
  // pass-1 spatial rendering will use `levelPaths` more broadly in
  // slice 4c-step2.
  const levelPaths = enumerateLevelPaths(rootAttrs.multiscales);
  const level0Url = kvstoreEnsureDirectoryPipelineUrl(
    pipelineUrlJoin(storeUrl, levelPaths[0]),
  );

  // Whether the store carries a per-fragment `segment_id` attribute. Probed by
  // a direct GET of its array metadata rather than trusted from each level's
  // `arrays_present`, which the 0.9 writer leaves incomplete at coarse levels
  // (it can omit arrays — links, vertex_fragments — that physically exist). The
  // fragment-attribute schema is store-wide, so level 0's answer holds for every
  // level. `arrays_present` is kept only as a fallback for pre-0.9 stores that
  // do not materialise this array.
  // Kinds without the discrete-object model never wrote this array, and each
  // vertex is its own segment anyway -- don't spend a request finding out.
  const segmentIdMeta = caps.hasObjectModel
    ? await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(
          level0Url,
          "fragment_attributes/segment_id/zarr.json",
        ),
        "fragment_attributes/segment_id metadata",
        options,
      )
    : undefined;
  const hasFragmentSegmentIds = segmentIdMeta !== undefined;

  // Per-vertex attribute discovery — reuse the annotation-path machinery
  // verbatim; the resulting (attributeNames, attributeDtypes) feed the
  // skeleton render layer's `prop_<name>()` shader bridge.
  const {
    properties: attributeProperties,
    attributeNames,
    attributeDtypes,
  } = await buildPropertySpecsAndDtypes(
    sharedKvStoreContext,
    level0Url,
    ngHints,
    options,
    selectedAttributes,
  );
  // Index-parallel to `attributeNames`; these are the GLSL-legal names the
  // `prop_<name>()` bridge exposes (the on-disk names may not be).
  const attributePropertyIds = attributeProperties.map((p) => p.identifier);

  // Links convention — drives whether explicit `links/0/<chunk>` edges
  // are read in addition to the implicit sequential ones synthesised
  // from fragment ranges.
  const linksConventionRaw = zv.links_convention;
  let linksConvention:
    | "implicit_sequential"
    | "implicit_sequential_with_branches"
    | "explicit";
  if (linksConventionRaw === undefined) {
    // Spec default per geometry: line / streamline / polyline →
    // implicit_sequential, skeleton → implicit_sequential_with_branches,
    // graph → explicit.  A point cloud declares none (it has no links at all);
    // the value is inert for it, since the capability table suppresses edges
    // regardless of what the convention says.
    if (geometryKind === "skeleton") {
      linksConvention = "implicit_sequential_with_branches";
    } else if (geometryKind === "graph") {
      linksConvention = "explicit";
    } else {
      linksConvention = "implicit_sequential";
    }
  } else if (
    linksConventionRaw === "implicit_sequential" ||
    linksConventionRaw === "implicit_sequential_with_branches" ||
    linksConventionRaw === "explicit"
  ) {
    linksConvention = linksConventionRaw;
  } else {
    throw new Error(
      `zarr-vectors links_convention=${JSON.stringify(linksConventionRaw)}: ` +
        `expected 'implicit_sequential', 'implicit_sequential_with_branches', or 'explicit'`,
    );
  }

  // Link dtype: read `links/0/zarr.json`'s declared dtype if the
  // convention has explicit edges.  Default to int64 (universally safe)
  // when no links array exists.
  let linkDtype:
    | "uint8"
    | "uint16"
    | "uint32"
    | "int8"
    | "int16"
    | "int32"
    | "int64";
  // The links family's record arity. 2 everywhere except a surface, where the
  // store declares its face arity on `links/0`. Read alongside the dtype below.
  let linkWidth = 2;
  if (caps.edgeSource === "none" || linksConvention === "implicit_sequential") {
    linkDtype = "int64";
  } else {
    let linksZarrJson: any | undefined;
    try {
      linksZarrJson = await getJsonResource(
        sharedKvStoreContext,
        joinBaseUrlAndPath(level0Url, "links/0/zarr.json"),
        "zarr-vectors links/0 metadata",
        options,
      );
    } catch {
      linksZarrJson = undefined;
    }
    const declaredWidth = Number(linksZarrJson?.attributes?.link_width);
    if (Number.isInteger(declaredWidth) && declaredWidth >= 2) {
      linkWidth = declaredWidth;
    } else if (caps.primitive === "triangles") {
      // A surface with no declared arity: assume triangles rather than reading
      // its faces as edges, which would silently produce garbage geometry.
      linkWidth = 3;
    }
    const raw =
      linksZarrJson?.attributes?.dtype ?? linksZarrJson?.data_type ?? "int64";
    if (
      raw !== "uint8" &&
      raw !== "uint16" &&
      raw !== "uint32" &&
      raw !== "int8" &&
      raw !== "int16" &&
      raw !== "int32" &&
      raw !== "int64"
    ) {
      throw new Error(
        `zarr-vectors links/0 dtype=${JSON.stringify(raw)}: expected an integer dtype`,
      );
    }
    linkDtype = raw;
  }

  const pass2Params = new ZarrVectorsObjectKeyedGeometrySourceParameters();
  pass2Params.baseUrl = level0Url;
  pass2Params.rank = rank;
  pass2Params.attributeNames = attributeNames;
  pass2Params.attributeDtypes = attributeDtypes;
  pass2Params.attributePropertyIds = attributePropertyIds;
  pass2Params.linksConvention = linksConvention;
  pass2Params.geometryKind = geometryKind;
  pass2Params.linkDtype = linkDtype;
  pass2Params.hasFragmentSegmentIds = hasFragmentSegmentIds;
  // Per-chunk-array grid geometry (origin + sharding) from level 0's
  // vertices/zarr.json. Needed by pass 2 regardless of rank (it reads the same
  // per-chunk arrays). Cached, so re-read per level below with no extra traffic.
  const level0Grid = await readChunkGridParams(
    sharedKvStoreContext,
    level0Url,
    rank,
    options,
  );
  pass2Params.chunkGridOrigin = level0Grid.chunkGridOrigin;
  pass2Params.sharded = level0Grid.sharded;
  pass2Params.shardChunkShape = level0Grid.shardChunkShape;
  pass2Params.cellSeparator = level0Grid.cellSeparator;

  // Pass-1 (spatially-indexed) wiring — only when the store is 3-D.
  // Neuroglancer's spatially-indexed skeleton machinery hardcodes a
  // vec3 position texture format (`skeleton/frontend.ts:1706-1709`); 2-D
  // or higher-rank stores fall back to pass-2 only.
  let pass1Levels:
    | ReadonlyArray<{
        parameters: ZarrVectorsSpatialGeometrySourceParameters;
      }>
    | undefined;
  let spatialGrid:
    | {
        perLevelChunkShape: Float32Array[];
        perLevelObjectCount: (number | undefined)[];
        perLevelVertexCount: (number | undefined)[];
        perLevelObjectVertexCounts?: Uint32Array[];
        objectDepths?: Uint8Array;
        lowerBounds: Float32Array;
        upperBounds: Float32Array;
      }
    | undefined;
  if (rank === 3) {
    const chunkShape = zv.chunk_shape;
    if (!Array.isArray(chunkShape) || chunkShape.length !== rank) {
      throw new Error(
        `zarr-vectors store: 'chunk_shape' must have rank ${rank}`,
      );
    }
    // Root chunk_shape is the default per-level chunk size.  When a
    // level stamps its own ``zarr_vectors_level.chunk_shape`` on disk
    // (writers using ``chunk_scale_factors`` to grow chunks at coarser
    // levels), that overrides the root for that level.
    const rootChunkShapeF32 = new Float32Array(rank);
    for (let i = 0; i < rank; ++i) {
      rootChunkShapeF32[i] = Number(chunkShape[i]);
    }
    const lowerBoundsF32 = Float32Array.from(lowerBounds);
    const upperBoundsF32 = Float32Array.from(upperBounds);

    // Fetch each level's zarr.json in parallel to read its optional
    // per-level chunk_shape override and arrays_present list.  Reuses
    // kvstore caching: the same files are read again below by the
    // chunk-source download path with zero net traffic.
    const perLevelMeta: {
      chunkShape: Float32Array;
      hasFragmentSegmentIds: boolean;
      vertexCount: number | undefined;
      objectSparsity: number | undefined;
      numObjects: number | undefined;
      grid: {
        chunkGridOrigin: number[];
        sharded: boolean;
        shardChunkShape: number[];
        cellSeparator: string;
      };
    }[] = await Promise.all(
      levelPaths.map(async (levelPath) => {
        const levelUrl = kvstoreEnsureDirectoryPipelineUrl(
          pipelineUrlJoin(storeUrl, levelPath),
        );
        // Required: per-level per-chunk-array grid geometry (throws on the
        // unreadable pre-0.9.0 "Option G" layout).
        const grid = await readChunkGridParams(
          sharedKvStoreContext,
          levelUrl,
          rank,
          options,
        );
        try {
          const levelJson = await getJsonResource(
            sharedKvStoreContext,
            joinBaseUrlAndPath(levelUrl, "zarr.json"),
            `zarr-vectors level ${JSON.stringify(levelPath)} metadata`,
            options,
          );
          const lvlAttrs = levelJson?.attributes?.zarr_vectors_level ?? {};
          const override = lvlAttrs?.chunk_shape;
          const chunkShape =
            Array.isArray(override) && override.length === rank
              ? (() => {
                  const arr = new Float32Array(rank);
                  for (let i = 0; i < rank; ++i) arr[i] = Number(override[i]);
                  return arr;
                })()
              : new Float32Array(rootChunkShapeF32);
          // `hasFragmentSegmentIds` comes from the store-wide direct probe
          // above, not this level's `arrays_present` (unreliable at coarse
          // levels); the fragment-attribute schema does not vary by level.
          const vc = Number(lvlAttrs?.vertex_count);
          const vertexCount = Number.isFinite(vc) && vc > 0 ? vc : undefined;
          const sp = Number(lvlAttrs?.object_sparsity);
          const objectSparsity = Number.isFinite(sp) && sp > 0 ? sp : undefined;
          const no = Number(lvlAttrs?.inherited_num_objects);
          const numObjects = Number.isFinite(no) && no > 0 ? no : undefined;
          return {
            chunkShape,
            hasFragmentSegmentIds,
            vertexCount,
            objectSparsity,
            numObjects,
            grid,
          };
        } catch {
          // fall through to defaults
        }
        return {
          chunkShape: new Float32Array(rootChunkShapeF32),
          hasFragmentSegmentIds: true,
          vertexCount: undefined,
          objectSparsity: undefined,
          numObjects: undefined,
          grid,
        };
      }),
    );
    const perLevelChunkShape = perLevelMeta.map((m) => m.chunkShape);
    const perLevelObjectCount = computePerLevelObjectCount(perLevelMeta);
    const perLevelVertexCount = perLevelMeta.map((m) => m.vertexCount);

    // Per-object costs and per-level membership, which together let a memory
    // budget be spent on a chosen SUBSET of a level's objects rather than only
    // on whichever whole rung fits. Optional: a store without these arrays, or
    // one whose levels are not nested subsets, keeps whole-level selection.
    const perLevelObjectVertexCounts = await readPerLevelObjectVertexCounts(
      sharedKvStoreContext,
      storeUrl,
      levelPaths,
      options,
    );
    let objectDepths: Uint8Array | undefined;
    if (perLevelObjectVertexCounts !== undefined) {
      if (levelsAreNested(perLevelObjectVertexCounts)) {
        objectDepths = computeObjectDepths(perLevelObjectVertexCounts);
      } else {
        // Not a coincidence worth papering over: the depth model would drop
        // every object a coarse level holds and a finer one does not.
        warnOnceNotNested();
      }
    }

    // Per-level parameter blobs.  Each level gets its own chunkShape
    // (may differ when the writer used ``chunk_scale_factors``).
    const levels: {
      parameters: ZarrVectorsSpatialGeometrySourceParameters;
    }[] = [];
    for (let k = 0; k < levelPaths.length; ++k) {
      const levelUrl = kvstoreEnsureDirectoryPipelineUrl(
        pipelineUrlJoin(storeUrl, levelPaths[k]),
      );
      const params = new ZarrVectorsSpatialGeometrySourceParameters();
      params.baseUrl = levelUrl;
      params.rank = rank;
      params.attributeNames = attributeNames;
      params.attributeDtypes = attributeDtypes;
      params.attributePropertyIds = attributePropertyIds;
      params.linksConvention = linksConvention;
      params.geometryKind = geometryKind;
      params.linkDtype = linkDtype;
      // gridIndex must match the framework's `spatialSkeletonGridLevels`
      // ordering, which is sorted DESCENDING by spacing (largest first).
      // Our `levelPaths[0]` is the FINEST pyramid level (smallest spacing),
      // so it should land at the END of the sorted list:
      //   levelPaths[0]  (finest)   → gridIndex = numLevels - 1
      //   levelPaths[N-1] (coarsest) → gridIndex = 0
      // See `findClosestSpatialSkeletonGridLevelBySpacing` in
      // `src/layer/segmentation/index.ts:588-603` for the picker that
      // then looks up sources by `gridIndex`.
      //
      // NOTE for anyone reading `spatialSkeletonGridLevel3d`/`2d` off a saved
      // state or URL: this inversion means that value is NOT the store's own
      // level directory number. A 3-level store (paths "0", "1", "2") with
      // `spatialSkeletonGridLevel3d === 2` is drawing the store's "0" (the
      // finest level, gridIndex numLevels-1), not its "2".
      params.gridIndex = levelPaths.length - 1 - k;
      params.hasFragmentSegmentIds = perLevelMeta[k].hasFragmentSegmentIds;
      params.vertexIdAttribute = vertexIdAttribute;
      // The edit target rides on every level's parameters, but only level 0's
      // source is the one the editing UI duck-types (`sources3d[0].chunkSource`,
      // `skeleton/frontend.ts`), so only it can act on this.
      params.editServiceUrl = editServiceUrl;
      params.editStore =
        editServiceUrl === undefined
          ? undefined
          : storeUrl.replace(/\/+$/, "").split("/").pop();
      params.linkWidth = linkWidth;
      params.chunkGridOrigin = perLevelMeta[k].grid.chunkGridOrigin;
      params.sharded = perLevelMeta[k].grid.sharded;
      params.shardChunkShape = perLevelMeta[k].grid.shardChunkShape;
      params.cellSeparator = perLevelMeta[k].grid.cellSeparator;
      // The next-coarser level's membership, for per-object admission. Static
      // per level (it is pure store metadata); only the rationing fraction is
      // supplied dynamically, at request time.
      //
      // Gated on `objectDepths`, i.e. on the levels being NESTED, and not
      // merely on the counts having been read. The partition rule is "an object
      // a coarser level also holds is that level's to draw"; on non-nested
      // levels that silently drops every object the coarse level holds and the
      // fine one does not -- and `canBudgetPerObject` has already declined to
      // budget there, so the partition would be in force with nothing choosing
      // what it admits. Tying both to one condition keeps "partitioned" and
      // "budgetable per object" the same store property.
      params.coarserMembership =
        objectDepths === undefined
          ? undefined
          : coarserMembershipBitset(perLevelObjectVertexCounts, k + 1);
      params.partitionsObjects = params.coarserMembership !== undefined;
      levels.push({ parameters: params });
    }

    pass1Levels = levels;
    spatialGrid = {
      perLevelChunkShape,
      perLevelObjectCount,
      perLevelVertexCount,
      perLevelObjectVertexCounts,
      objectDepths,
      lowerBounds: lowerBoundsF32,
      upperBounds: upperBoundsF32,
    };
    // Pass-2 always operates at level 0 — use its arrays_present to gate
    // the fragment_attributes/segment_id fetch.  Overrides the default-true
    // placeholder set above.
    pass2Params.hasFragmentSegmentIds = perLevelMeta[0].hasFragmentSegmentIds;
  }

  // Discover per-object attributes and named groups at level 0 and build the
  // segment-properties map.  Pinned to level 0 because object_ids are
  // global across the pyramid; coarser levels reuse the level-0 row
  // assignments via `present_mask` (handled inside the builder).
  // `object_attributes/` only exists for kinds with the discrete-object model;
  // listing it for a point cloud would 404-or-warn for nothing.
  const segmentPropertyMap = caps.hasObjectModel
    ? await buildSegmentPropertyMap(sharedKvStoreContext, level0Url, options)
    : undefined;

  // The NGFF per-level `translation` on `datasets[0]` is NOT read, because the
  // writer overloads that field and it cannot be told apart from a real offset:
  // on a store with no `coordinate_offset` it is the bin-centre convention
  // (`chunk_shape / 2`, written to round-trip `bin_shape` — the hcp1065 store
  // carries [15.5, 16, 17]), and applying that shifts every vertex half a chunk
  // off its true coordinate (verified by decoding `0/vertices/c/0/0/0`: chunk
  // (0,0,0)'s vertices already lie within its absolute [0,31)×[0,32)×[0,34)
  // range).  But on a store that DOES declare an offset the writer mirrors the
  // offset onto level 0 only, leaving the bin-centre value on the coarser
  // levels — so the same field means two different things at two levels of one
  // store.  The unambiguous `zarr_vectors.coordinate_offset` root attribute is
  // read instead (see `readCoordinateOffset`) and becomes the model transform's
  // translation, so the source moves into world coordinates as a whole while
  // all spatial-index calculations keep running in raw stored coordinates.
  return {
    rank,
    coordinateSpace,
    coordinateOffset: readCoordinateOffset(zv, rank),
    hasObjectModel: caps.hasObjectModel,
    primitive: caps.primitive,
    pass2Params,
    pass1Levels,
    spatialGrid,
    segmentPropertyMap,
  };
}

/**
 * Construct a segmentation-shaped `DataSource` from skeleton metadata.
 *
 * The data source exposes up to **two** chunk sources under
 * `subsource.zarrVectors`, each in its own subsource entry:
 *
 *   - `"zarr-vectors"` — the multiscale spatially-indexed source that drives
 *     **pass 1** (camera-relative chunk loading).  Only emitted when the store
 *     is 3-D (the spatially-indexed render layer assumes vec3 positions).
 *   - `"zarr-vectors-detail"` — the per-segment source that drives **pass 2**
 *     (user-typed object IDs in the segments-list UI).  Omitted entirely for
 *     geometry without the discrete-object model, and for surfaces.
 *
 * Both carry their former ids (`"skeleton-spatial"`, `"skeleton"`) as
 * `legacyIds`, so a saved link that enabled or disabled one still applies.
 *
 * The `zarrVectors` slot is this datasource's own; it is deliberately not the
 * `mesh` slot, which belongs to neuroglancer's `MeshSource` family.  The
 * segmentation layer accepts both and picks render layers by source class:
 *
 *   - `MultiscaleSpatiallyIndexedSkeletonSource` → mounts the
 *     spatially-indexed render layer.
 *   - `SkeletonSource` → mounts the per-segment render layer.
 *
 * Both render layers can coexist on the same segmentation layer, which
 * is how the two passes compose visually.
 */
function getGeometryDataSource(
  sharedKvStoreContext: SharedKvStoreContext,
  metadata: GeometryMetadata,
): DataSource {
  const subsources: DataSource["subsources"] = [];

  if (
    metadata.pass1Levels !== undefined &&
    metadata.spatialGrid !== undefined
  ) {
    subsources.push({
      // The bulk geometry: named for what it IS, not for the render layer it
      // happens to share with skeletons. `legacyIds` keeps every saved link
      // that toggled the old name working.
      id: "zarr-vectors",
      legacyIds: ["skeleton-spatial"],
      default: true,
      subsource: {
        zarrVectors: new ZarrVectorsMultiscaleGeometrySource(
          sharedKvStoreContext.chunkManager,
          sharedKvStoreContext,
          {
            levels: metadata.pass1Levels,
            perLevelChunkShape: metadata.spatialGrid.perLevelChunkShape,
            perLevelObjectCount: metadata.spatialGrid.perLevelObjectCount,
            perLevelVertexCount: metadata.spatialGrid.perLevelVertexCount,
            perLevelObjectVertexCounts:
              metadata.spatialGrid.perLevelObjectVertexCounts,
            objectDepths: metadata.spatialGrid.objectDepths,
            metersPerUnit: Float64Array.from(metadata.coordinateSpace.scales),
            lowerBounds: metadata.spatialGrid.lowerBounds,
            upperBounds: metadata.spatialGrid.upperBounds,
          },
        ),
      },
    });
  }

  // Pass-2 (per-segment, full-detail) source. When there is a pass-1 bulk to
  // draw it on top of, it defaults ON so the ROI filter's per-group "high
  // detail" toggle just works (the segmentation layer repurposes it as the
  // high-detail render layer, driven by roiHighDetailSegments; it renders
  // nothing until a group is marked high-detail). Without pass-1 (e.g. 2-d or
  // higher-rank stores) it stays the opt-in per-segment source.
  //
  // Skipped entirely for kinds without the discrete-object model: pass 2
  // resolves a selected id through `object_index/manifests`, which a point
  // cloud does not have, so the subsource could only ever fail.
  //
  // Skipped for surfaces too, for a different reason: the per-segment path is
  // line-only end to end -- it ignores link records of arity != 2 when
  // aggregating an object (`geometry_segment_download.ts`) and the plain
  // `SkeletonLayer` that draws it has no face pass -- so a mesh's faces would
  // arrive as edge pairs and draw as a scribble. Pass 1 already draws the whole
  // surface; the second pass exists to redraw SELECTED tracts at full detail,
  // which a surface has no equivalent of.
  if (metadata.hasObjectModel && metadata.primitive !== "triangles") {
    subsources.push({
      id: "zarr-vectors-detail",
      legacyIds: ["skeleton"],
      default: metadata.pass1Levels !== undefined,
      subsource: {
        zarrVectors: sharedKvStoreContext.chunkManager.getChunkSource(
          ZarrVectorsObjectKeyedGeometrySource,
          {
            sharedKvStoreContext,
            parameters: metadata.pass2Params,
          },
        ),
      },
    });
  }

  // Segment-properties subsource.  Beyond the sortable/filterable columns in
  // the segments-list panel, this map now BACKS the streamline filter's per-
  // group + background length filter and colour-by-object-attribute: those UIs
  // read `displayState.segmentPropertyMap.numericalProperties`, which is only
  // populated when this subsource is active.  So it defaults ON (the feature
  // must work without the user hunting for a source toggle); it carries no extra
  // geometry.  Only emitted when the store carries `object_attributes/` at level
  // 0; otherwise the subsource doesn't exist to toggle.
  if (metadata.segmentPropertyMap !== undefined) {
    subsources.push({
      id: "properties",
      default: true,
      subsource: { segmentPropertyMap: metadata.segmentPropertyMap },
    });
  }

  if (subsources.length === 0) {
    // Only reachable for a kind with no object model at rank != 3: pass 1 is
    // gated on vec3 positions and pass 2 does not exist for it. Say so, rather
    // than handing back a data source that silently renders nothing.
    throw new Error(
      `zarr-vectors datasource: a rank-${metadata.rank} store of this geometry ` +
        "has nothing to render — the spatially-indexed path requires 3-D " +
        "positions and this geometry has no per-object source to fall back on.",
    );
  }

  return {
    modelTransform: makeCoordinateOffsetTransform(
      metadata.coordinateSpace,
      metadata.coordinateOffset,
    ),
    subsources,
  };
}

// ---------------------------------------------------------------
// Provider
// ---------------------------------------------------------------

function resolveUrl(options: GetKvStoreBasedDataSourceOptions) {
  const { authorityAndPath, query, fragment } = parseUrlSuffix(
    options.url.suffix,
  );
  if (query) {
    throw new Error(
      `Invalid URL ${JSON.stringify(options.url.url)}: query parameters not supported`,
    );
  }
  // `#attributes=a,b,c` selects which per-vertex attributes to expose. A store
  // may declare far more than the GPU can hold (a MERFISH panel ships one
  // column per gene), and every exposed attribute costs a texture unit and a
  // read per chunk, so the user needs a way to name the handful they want
  // without rewriting the store. Nothing else is accepted as a fragment.
  // `#edit=<service url>` opts the layer into editing, alongside (and
  // independent of) the attribute selection. Both may appear, `&`-separated.
  let editServiceUrl: string | undefined;
  const fragmentParts = (fragment ?? "").split("&").filter((p) => p.length > 0);
  const remaining: string[] = [];
  for (const part of fragmentParts) {
    if (part.startsWith("edit=")) {
      editServiceUrl = decodeURIComponent(part.slice("edit=".length));
    } else {
      remaining.push(part);
    }
  }
  let selectedAttributes: string[] | undefined;
  try {
    selectedAttributes = parseAttributesFragment(
      remaining.length > 0 ? remaining.join("&") : undefined,
    );
  } catch (e) {
    throw new Error(
      `Invalid URL ${JSON.stringify(options.url.url)}: ${(e as Error).message}`,
    );
  }
  return {
    kvStoreUrl: kvstoreEnsureDirectoryPipelineUrl(options.kvStoreUrl),
    additionalPath: authorityAndPath ?? "",
    selectedAttributes,
    editServiceUrl,
  };
}

export class ZarrVectorsDataSource implements KvStoreBasedDataSourceProvider {
  get scheme() {
    return "zarr-vectors";
  }
  get expectsDirectory() {
    return true;
  }
  get description() {
    return "Zarr Vectors (experimental) data source";
  }

  async get(
    options: GetKvStoreBasedDataSourceOptions,
  ): Promise<DataSourceLookupResult> {
    let { kvStoreUrl, additionalPath, selectedAttributes, editServiceUrl } =
      resolveUrl(options);
    kvStoreUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(kvStoreUrl, additionalPath),
    );
    return options.registry.chunkManager.memoize.getAsync(
      // The attribute selection is part of the identity: two layers on the same
      // store with different `#attributes=` are different data sources.
      {
        type: "zarr-vectors:get",
        url: kvStoreUrl,
        attributes: selectedAttributes?.join(","),
        // A layer opened for editing is a different source from a read-only
        // one on the same store: it reports `readonly: false`.
        edit: editServiceUrl,
      },
      options,
      async (progressOptions) => {
        const { sharedKvStoreContext } = options.registry;
        const rootJson = await getJsonResource(
          sharedKvStoreContext,
          joinBaseUrlAndPath(kvStoreUrl, "zarr.json"),
          "zarr-vectors root metadata",
          progressOptions,
        );
        if (rootJson === undefined) {
          throw new Error(
            `No zarr.json found at ${kvStoreUrl} — is this a zarr v3 store?`,
          );
        }
        if (rootJson.node_type && rootJson.node_type !== "group") {
          throw new Error(
            `zarr-vectors expected a zarr v3 group, got node_type=${JSON.stringify(rootJson.node_type)}`,
          );
        }
        const attrs = rootJson.attributes ?? {};
        // Dispatch by geometry_types: point_cloud → annotation layer
        // (existing path); skeleton / polyline / streamline → segmentation
        // layer (slice 4c).  Validation (unknown types, mixed-geometry,
        // wrong object_index_convention) happens here so the failure
        // surface is consistent across geometries.
        const zv = attrs.zarr_vectors;
        if (zv === undefined) {
          throw new Error(
            "Not a zarr-vectors store: root attributes lack a 'zarr_vectors' block",
          );
        }
        const geometryTypes: string[] = Array.isArray(zv.geometry_types)
          ? zv.geometry_types
          : [];
        const renderable = geometryTypes.filter((g) =>
          SPATIAL_GEOM_KINDS.has(g),
        );
        // Unrecognised types alongside recognised ones are reported by
        // `resolveDeclaredGeometry` and skipped; only a store with NOTHING
        // recognisable is an error.
        if (renderable.length === 0) {
          throw new Error(
            `zarr-vectors datasource: no recognised geometry type in ` +
              `${JSON.stringify(geometryTypes)}; expected one of ` +
              `${JSON.stringify(Array.from(SPATIAL_GEOM_KINDS))}`,
          );
        }
        // The spec defines two object-index conventions. "standard" maps a
        // selected segment id to a dense object index via
        // `object_attributes/segment_id`; "identity" means the selected id IS
        // the dense index. The backend already degrades to identity when that
        // attribute is absent (see `resolveObjectIndex`), so both conventions
        // are supported here.  Point clouds have no object index at all, so the
        // key is not required of them.
        const objectIndexConvention = zv.object_index_convention;
        if (
          renderable.some(
            (g) =>
              KIND_CAPABILITIES[g as ZarrVectorsGeometryKind].hasObjectModel,
          ) &&
          objectIndexConvention !== "standard" &&
          objectIndexConvention !== "identity" &&
          objectIndexConvention !== undefined
        ) {
          throw new Error(
            `zarr-vectors datasource: ${renderable.join("/")} geometry requires ` +
              `object_index_convention 'standard' or 'identity' (got ` +
              `${JSON.stringify(objectIndexConvention)})`,
          );
        }
        const skelMeta = await buildGeometryMetadata(
          sharedKvStoreContext,
          kvStoreUrl,
          attrs,
          progressOptions,
          selectedAttributes,
          editServiceUrl,
        );
        const dataSource = getGeometryDataSource(
          sharedKvStoreContext,
          skelMeta,
        );
        // Keep the fragment: it selects which attributes exist on this layer,
        // so dropping it would canonicalise two different sources to one URL.
        // Re-encoded, so the canonical URL saved into the layer's JSON parses
        // back to the same names.
        dataSource.canonicalUrl =
          `${kvStoreUrl}|${options.url.scheme}:` +
          formatAttributesFragment(selectedAttributes);
        return dataSource;
      },
    );
  }
}

/**
 * Companion data source that re-interprets ANY zarr-vectors store
 * (graph, skeleton, polyline, streamline, or point_cloud) as a point
 * cloud.  Routes through the existing annotation-layer path: reads
 * `vertices/<level>/<chunk>` + `vertex_attributes/<name>/<level>/<chunk>`
 * (which exist for every geometry kind) and ignores everything edge-
 * or segment-related (links, manifests, cross_chunk_links,
 * object_index, object_attributes).
 *
 * Useful for spot-checking a graph store's vertex positions without
 * waiting for the segmentation-layer machinery, or for stripping a
 * heavyweight skeleton store down to a point cloud when the connectivity
 * isn't relevant.
 */
export class ZarrVectorsPointCloudDataSource
  implements KvStoreBasedDataSourceProvider
{
  get scheme() {
    return "zarr-vectors-pointcloud";
  }
  get expectsDirectory() {
    return true;
  }
  get description() {
    return "Zarr Vectors as a point cloud (ignores edges)";
  }

  async get(
    options: GetKvStoreBasedDataSourceOptions,
  ): Promise<DataSourceLookupResult> {
    let { kvStoreUrl, additionalPath } = resolveUrl(options);
    kvStoreUrl = kvstoreEnsureDirectoryPipelineUrl(
      pipelineUrlJoin(kvStoreUrl, additionalPath),
    );
    return options.registry.chunkManager.memoize.getAsync(
      { type: "zarr-vectors-pointcloud:get", url: kvStoreUrl },
      options,
      async (progressOptions) => {
        const { sharedKvStoreContext } = options.registry;
        const rootJson = await getJsonResource(
          sharedKvStoreContext,
          joinBaseUrlAndPath(kvStoreUrl, "zarr.json"),
          "zarr-vectors root metadata",
          progressOptions,
        );
        if (rootJson === undefined) {
          throw new Error(
            `No zarr.json found at ${kvStoreUrl} — is this a zarr v3 store?`,
          );
        }
        if (rootJson.node_type && rootJson.node_type !== "group") {
          throw new Error(
            `zarr-vectors expected a zarr v3 group, got node_type=${JSON.stringify(rootJson.node_type)}`,
          );
        }
        const attrs = rootJson.attributes ?? {};
        const meta = await buildAnnotationMetadata(
          sharedKvStoreContext,
          kvStoreUrl,
          attrs,
          progressOptions,
          /*allowAnyGeometry=*/ true,
        );
        const dataSource = getAnnotationDataSource(sharedKvStoreContext, meta);
        dataSource.canonicalUrl = `${kvStoreUrl}|${options.url.scheme}:`;
        return dataSource;
      },
    );
  }
}

export function registerAutoDetect(_registry: AutoDetectRegistry) {
  // Auto-detect is intentionally omitted in v1: a zarr-vectors store
  // is also a valid zarr v3 group, and we don't want to shadow the
  // existing zarr datasource by default.  Users opt in explicitly via
  // the "zarr-vectors://" scheme.
}
