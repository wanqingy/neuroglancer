/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Worker-side `SharedObject` chunk-source backends for zarr-vectors
 * skeleton / polyline / streamline rendering.  Provides:
 *
 * - `ZarrVectorsSpatialGeometrySourceBackend` — the **pass-1**
 *   backing store.  Subclasses neuroglancer's existing
 *   `SpatiallyIndexedSkeletonSourceBackend` and overrides `download()`
 *   to fetch + decode zarr-vectors chunks via the
 *   `downloadGeometryChunk()` orchestrator.
 *
 * - `ZarrVectorsObjectKeyedGeometrySourceBackend` — the **pass-2**
 *   backing store, intentionally **not implemented in this slice**.
 *   Will subclass `SkeletonSource` once the `object_index/manifests`
 *   zarr-vlen-bytes reader is in place (slice 4b).
 *
 * Mirrors the CATMAID pattern at
 * `src/datasource/catmaid/backend.ts:40-83`.
 */

import { decodeZstd } from "#src/async_computation/decode_zstd_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import { WithParameters } from "#src/chunk_manager/backend.js";
import {
  ZarrVectorsObjectKeyedGeometrySourceParameters,
  ZarrVectorsSpatialGeometrySourceParameters,
  type ZarrVectorsLinkDtype,
  type ZarrVectorsLinksConvention,
  type ZarrVectorsGeometryKind,
  ZARR_VECTORS_GET_OBJECT_NODES_RPC_ID,
} from "#src/datasource/zarr-vectors/base.js";
import { ChunkCoalescingCache } from "#src/datasource/zarr-vectors/chunk_coalescing_cache.js";
import {
  appendBoundaryFaces,
  appendGhostVertices,
  resolveBoundaryFaces,
  appendIntraChunkEdges,
  recomputeTangentsForBridges,
  type ResolvedBridge,
  type AttributeTypedArray,
  type SkeletonChunk as DecodedGeometryChunk,
} from "#src/datasource/zarr-vectors/geometry_chunk.js";
import {
  downloadGeometryChunk,
  fetchGhostVertices,
  type AttributeDtype,
  type GhostVertexRequest,
  type LinkDtype,
} from "#src/datasource/zarr-vectors/geometry_chunk_download.js";
import {
  hasSynthesisedTangent,
  isSurfaceGeometry,
  KIND_CAPABILITIES,
} from "#src/datasource/zarr-vectors/geometry_kind.js";
import { buildSpatialSkeletonNodes } from "#src/datasource/zarr-vectors/geometry_nodes.js";
import { downloadSegmentSkeleton } from "#src/datasource/zarr-vectors/geometry_segment_download.js";
import {
  createCrossChunkLinksCaches,
  readCrossChunkLinks,
  readCrossChunkLinksForChunk,
  type CrossChunkLinksCaches,
  type CrossChunkLinksTable,
} from "#src/datasource/zarr-vectors/links.js";
import {
  objectRank,
  filterChunkByAdmittedObjects,
} from "#src/datasource/zarr-vectors/object_budget.js";
import {
  ShardCellReader,
  type CellReader,
} from "#src/datasource/zarr-vectors/shard_cell_reader.js";
import { WithSharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import { joinBaseUrlAndPath } from "#src/kvstore/url.js";
import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import type { SkeletonChunk } from "#src/skeleton/backend.js";
import { SkeletonSource } from "#src/skeleton/backend.js";
import type { SpatiallyIndexedSkeletonChunk } from "#src/skeleton/spatial_backend.js";
import { SpatiallyIndexedSkeletonSourceBackend } from "#src/skeleton/spatial_backend.js";
import type { ProgressOptions } from "#src/util/progress_listener.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerPromiseRPC, registerSharedObject } from "#src/worker_rpc.js";

const ZSTD_MAGIC = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd]);

function looksLikeZstd(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 4) return false;
  return (
    bytes[0] === ZSTD_MAGIC[0] &&
    bytes[1] === ZSTD_MAGIC[1] &&
    bytes[2] === ZSTD_MAGIC[2] &&
    bytes[3] === ZSTD_MAGIC[3]
  );
}

/**
 * Decompress a zstd-framed byte buffer; pass through other formats.
 *
 * Duplicated from the point-cloud backend.ts intentionally — keeps this
 * slice's diff scoped to new files only.  If a third caller appears,
 * promote both copies to a shared helper module.
 */
async function maybeDecompress(
  bytes: Uint8Array<ArrayBuffer>,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!looksLikeZstd(bytes)) return bytes;
  return await requestAsyncComputation(
    decodeZstd,
    signal,
    [bytes.buffer],
    bytes,
  );
}

/**
 * Build a `kvStoreRead` callback bound to a base URL and the worker-side
 * shared kvstore context.  Resolves to a decompressed `Uint8Array` (or
 * `undefined` for a missing key).
 */
function makeKvStoreRead(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: { signal: AbortSignal }) => Promise<any>;
    };
  },
) {
  return async (
    subpath: string,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> => {
    const url = joinBaseUrlAndPath(baseUrl, subpath);
    const response = await sharedKvStoreContext.kvStoreContext.read(url, {
      signal,
    });
    if (response === undefined) return undefined;
    const bytes = new Uint8Array(
      (await response.response.arrayBuffer()) as ArrayBuffer,
    );
    return await maybeDecompress(bytes, signal);
  };
}

/**
 * Build a `kvStoreRead` callback like {@link makeKvStoreRead} but WITHOUT
 * the magic-byte-sniffing auto-decompress step.  Required by the ZVF 0.9
 * links reader: a `links/<delta>/<offsets>/c/...` cell is a `vlen-bytes`
 * container whose payload is not itself a zstd stream, so the reader must
 * see the raw stored bytes and do its own vlen framing / decode.
 */
function makeRawKvStoreRead(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: { signal: AbortSignal }) => Promise<any>;
    };
  },
) {
  return async (
    subpath: string,
    signal: AbortSignal,
  ): Promise<Uint8Array | undefined> => {
    const url = joinBaseUrlAndPath(baseUrl, subpath);
    const response = await sharedKvStoreContext.kvStoreContext.read(url, {
      signal,
    });
    if (response === undefined) return undefined;
    return new Uint8Array(
      (await response.response.arrayBuffer()) as ArrayBuffer,
    );
  };
}

/**
 * Build a `kvStoreList` callback bound to a base URL and the worker-side
 * shared kvstore context.  Used by the links reader to enumerate the
 * `<offsets>` arrays under `links/<delta>/` when the store permits object
 * listing (it falls back to a bounded GET-only probe otherwise).  Mirrors
 * ``listAttributeNames`` in `frontend.ts`, but returns bare
 * (no-trailing-slash) names for both directories and files.
 */
function makeKvStoreList(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      list: (
        urlPrefix: string,
        options: { responseKeys: "suffix"; signal: AbortSignal },
      ) => Promise<{ entries: { key: string }[]; directories: string[] }>;
    };
  },
) {
  return async (
    prefix: string,
    signal: AbortSignal,
  ): Promise<{ directories: string[]; files: string[] }> => {
    const url = joinBaseUrlAndPath(baseUrl, prefix);
    const response = await sharedKvStoreContext.kvStoreContext.list(url, {
      responseKeys: "suffix",
      signal,
    });
    const stripTrailingSlash = (s: string) =>
      s.endsWith("/") ? s.slice(0, -1) : s;
    return {
      directories: response.directories.map(stripTrailingSlash),
      files: response.entries.map((e) => stripTrailingSlash(e.key)),
    };
  };
}

/**
 * The parameter types in `base.ts` declare `ZarrVectorsLinkDtype` and
 * `ZarrVectorsAttributeDtype` as union subsets of the orchestrator's
 * dtypes.  The orchestrator's `LinkDtype` / `AttributeDtype` are
 * structurally identical at the value level — the two type names exist
 * separately so the parameter classes can carry semantically-named
 * unions while the orchestrator stays decoupled from the parameter
 * surface.  Cast through here.
 */
function asLinkDtype(d: ZarrVectorsLinkDtype): LinkDtype {
  return d as LinkDtype;
}
function asAttributeDtype(d: string): AttributeDtype {
  return d as AttributeDtype;
}

/**
 * Spatially-indexed skeleton chunk source — the **pass-1** backing
 * store.  One chunk per `(chunkGridPosition, lod)` pair.
 *
 * For each chunk, `download()` fetches the relevant byte blobs and
 * decodes them into a `SpatiallyIndexedSkeletonChunk` whose
 * `vertexPositions`, `indices`, and `vertexAttributes` fields the render
 * layer consumes.
 *
 * Streamline / polyline geometry kinds prepend a synthesised
 * `tangent` vec3 attribute to `vertexAttributes` so the default shader's
 * `prop_tangent()` resolves to the per-vertex unit direction.
 * Skeleton geometry skips this — branching breaks the
 * "direction at this vertex" abstraction.
 */
@registerSharedObject()
export class ZarrVectorsSpatialGeometrySourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(SpatiallyIndexedSkeletonSourceBackend),
  ZarrVectorsSpatialGeometrySourceParameters,
) {
  /**
   * Shared shard-discovery / shard-byte caches for this level's
   * ``cross_chunk_links/0/`` tree, reused across every chunk's
   * per-chunk-targeted query (see {@link getCrossChunkLinksForChunk}).
   *
   * ``false`` means "probed, store has no such table at all" (older
   * zarr-vectors stores written without
   * ``cross_chunk_strategy="explicit_links"``) — skip all future
   * queries.  ``undefined`` means "not yet probed".
   *
   * Deliberately NOT a cache of decoded records (that was the previous
   * design: a single ``CrossChunkLinksTable`` holding every record in
   * the level, read once and reused for every chunk).  For a real
   * dataset that whole-table decode can be tens of millions of records
   * — multiple gigabytes once V8's per-object overhead is counted for
   * the deeply-nested `record → endpoints[] → {chunkCoords[]}`
   * representation — and, being a plain field rather than a `Chunk`,
   * it was invisible to `ChunkState`'s GPU/system memory budget and
   * never evicted under memory pressure.  `CrossChunkLinksCaches` only
   * retains cheap shard-coordinate lists and byte-budgeted raw
   * (still-compressed) shard bytes; per-chunk queries decode only the
   * records actually needed.  See `cross_chunk_links.ts`'s
   * `CrossChunkLinksCaches` docstring for the full rationale.
   *
   * Mirror of the same field on
   * {@link ZarrVectorsObjectKeyedGeometrySourceBackend}; the two
   * backends share a parameter type's ``baseUrl`` for the same store
   * level, but each holds its own cache instance.
   */
  private crossChunkLinksCaches_: CrossChunkLinksCaches | false | undefined;

  private cellReader_: ShardCellReader | undefined;
  /**
   * One {@link ShardCellReader} per source so its shard-index cache is amortized
   * across all cells of a shard. Reads per-chunk array cells honoring
   * `chunk_grid_origin` and the optional `sharding_indexed` packing (grid
   * geometry threaded via source params).
   */
  protected get cellRead(): CellReader {
    let reader = this.cellReader_;
    if (reader === undefined) {
      const p = this.parameters;
      reader = this.cellReader_ = this.registerDisposer(
        new ShardCellReader(
          this.chunkManager,
          p.baseUrl,
          this.sharedKvStoreContext.kvStoreContext,
          makeRawKvStoreRead(p.baseUrl, this.sharedKvStoreContext),
          {
            origin: p.chunkGridOrigin,
            sharded: p.sharded,
            shardShape: p.shardChunkShape,
            separator: p.cellSeparator,
          },
        ),
      );
    }
    return reader.read;
  }

  /** See {@link ObjectIndexResolver}. */
  private objectIndex = new ObjectIndexResolver();

  /**
   * Shared across the editing UI's per-object reads, for the same reason the
   * pass-2 source has one: a tract's manifest names many spatial chunks, and
   * neighbouring tracts name the same ones.
   */
  private nodeChunkCache = new ChunkCoalescingCache<
    Awaited<ReturnType<typeof downloadGeometryChunk>>
  >();

  /**
   * Session-stable id bases for synthesised node ids, keyed by dense object
   * index.
   *
   * ZVF has no per-vertex identity -- a vertex is only `(level, chunk coords,
   * row index)`, and coarser levels replace vertices with bin centroids -- so
   * until a store carries a real id column these ids are minted here. They must
   * be unique ACROSS objects, because the edit overlay keys one `Map` by node id
   * for every segment it holds (`segment_overlay.ts`), so each object gets a
   * disjoint range rather than restarting at 1. Retaining the base means
   * re-reading an object after its chunks were evicted yields the same ids
   * within a session; across sessions it does not, which is exactly why an
   * editable store must supply its own ids.
   */
  private nodeIdBases = new Map<number, number>();
  private nextNodeIdBase = 0;

  /**
   * One object's geometry, as the rooted node tree the spatial-skeleton editing
   * UI consumes. Backs `getSkeleton` on the frontend source; see
   * `spatial_skeleton_manager.getFullSegmentNodes`, its only caller.
   *
   * Reads through the pass-2 aggregation path (`downloadSegmentSkeleton`)
   * rather than the resident pass-1 chunks: the UI wants the WHOLE object at
   * full detail, and pass 1 holds only what the camera admitted, decimated to
   * the level in view.
   */
  /**
   * Whole-level cross-chunk links table for the node-inspection path, cached
   * per source. `null` = absent, or a read that failed (see the catch).
   */
  private objectNodesCrossChunkLinks_: CrossChunkLinksTable | null | undefined;
  private objectNodesLinksWarned_ = false;

  /**
   * The cross-chunk links table {@link getObjectSkeletonNodes} needs to
   * reconnect one object across chunk boundaries.
   *
   * `implicit_sequential_with_branches` files every boundary-crossing edge in
   * the links family, so WITHOUT this table an object spanning several chunks
   * comes back as one disconnected component per chunk -- each crossing edge
   * silently absent. That matters because the spatially-indexed overlay is what
   * draws a SELECTED segment (pass 1 suppresses it from its browse pass), so
   * the omission shows up as a gap at every chunk boundary, even though pass 1
   * renders the same anatomy intact by stitching with ghost vertices.
   *
   * Reading the whole table is safe here: `explicit` -- the convention whose
   * table is the multi-gigabyte whole-level decode -- is refused before this
   * point, and the remaining conventions file one record per fragment
   * adjacency, not one per edge.
   */
  private async getObjectNodesCrossChunkLinks(
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    if (this.objectNodesCrossChunkLinks_ !== undefined) {
      return this.objectNodesCrossChunkLinks_ ?? undefined;
    }
    const { baseUrl } = this.parameters;
    try {
      const table = await readCrossChunkLinks(
        {
          kvStoreRead: makeRawKvStoreRead(baseUrl, this.sharedKvStoreContext),
          cellRead: this.cellRead,
          kvStoreList: makeKvStoreList(baseUrl, this.sharedKvStoreContext),
        },
        signal,
      );
      this.objectNodesCrossChunkLinks_ = table ?? null;
      return table;
    } catch (e) {
      // Degrade to the pre-fix behaviour (one component per chunk) rather than
      // rejecting: `requestOverlaySegmentLoad` records a rejected segment in
      // `failedOverlaySegmentLoads` and then draws NOTHING for it, which is
      // strictly worse than drawing it with the boundary edges missing.
      if (!this.objectNodesLinksWarned_) {
        this.objectNodesLinksWarned_ = true;
        console.warn(
          "zarr-vectors: failed to read the cross-chunk links table for node " +
            "inspection; a selected object will render disconnected at chunk " +
            "boundaries. " +
            (e instanceof Error ? e.message : String(e)),
        );
      }
      this.objectNodesCrossChunkLinks_ = null;
      return undefined;
    }
  }

  async getObjectSkeletonNodes(
    objectId: bigint,
    signal: AbortSignal,
  ): Promise<SpatiallyIndexedSkeletonNode[]> {
    const {
      baseUrl,
      rank,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      linkDtype,
      hasFragmentSegmentIds,
    } = this.parameters;
    if (
      !KIND_CAPABILITIES[geometryKind as ZarrVectorsGeometryKind].hasObjectModel
    ) {
      // A point cloud has no `object_index/manifests` to resolve an id
      // against: every vertex is its own segment. There are no nodes to
      // inspect, so answer empty rather than 404 on the manifest read.
      return [];
    }
    if (linksConvention === "explicit") {
      // `explicit` keeps EVERY edge in the links family, so reconstructing one
      // object means the whole-level decode documented on
      // `ZarrVectorsObjectKeyedGeometrySourceBackend.crossChunkLinks_` --
      // tens of millions of records, multiple gigabytes. Refuse rather than
      // hang the worker on the first node inspection.
      throw new Error(
        "zarr-vectors: reading a skeleton for editing is not supported for " +
          "`explicit` link stores; every edge would require a whole-level " +
          "links decode.",
      );
    }
    const kvStoreRead = makeKvStoreRead(baseUrl, this.sharedKvStoreContext);
    const { numObjects, chunkSize } = await readManifestArrayShape(
      baseUrl,
      this.sharedKvStoreContext,
      signal,
    );
    const oid = await this.objectIndex.resolve(
      objectId,
      numObjects,
      kvStoreRead,
      signal,
    );
    if (oid === undefined) return [];

    // `implicit_sequential` reconstructs its cross-chunk edges from manifest
    // block order inside `downloadSegmentSkeleton`, so it needs no table.
    const crossChunkLinks =
      linksConvention === "implicit_sequential"
        ? undefined
        : await this.getObjectNodesCrossChunkLinks(signal);

    const aggregated = await downloadSegmentSkeleton(
      oid,
      {
        manifestReader: { numObjects, chunkSize, sidNdim: rank, kvStoreRead },
        cellRead: this.cellRead,
        rank,
        linkDtype: asLinkDtype(linkDtype),
        attributeNames,
        attributeDtypes: attributeDtypes.map(asAttributeDtype),
        linksConvention: linksConvention as ZarrVectorsLinksConvention,
        geometryKind: geometryKind as ZarrVectorsGeometryKind,
        // See `getObjectNodesCrossChunkLinks`. Previously hardcoded
        // `undefined`, on the reasoning that fetching the table would mean the
        // whole-level decode this method refuses above -- but that cost is
        // specific to `explicit`, which is already refused, so every
        // convention that reaches here has a table of one record per fragment
        // adjacency. Passing it is what keeps an object spanning several
        // chunks from rendering as one disconnected component per chunk.
        crossChunkLinks,
        hasFragmentSegmentIds,
        chunkCache: this.nodeChunkCache,
      },
      signal,
    );
    if (aggregated === undefined) return [];

    const numVertices = Math.floor(aggregated.vertexPositions.length / rank);
    let idOffset = this.nodeIdBases.get(oid);
    if (idOffset === undefined) {
      idOffset = this.nextNodeIdBase;
      this.nodeIdBases.set(oid, idOffset);
      this.nextNodeIdBase += numVertices;
    }

    // The UI's segment ids are `number`-typed throughout the editing path
    // (`normalizeIdentifier` in `command_history.ts` rejects anything else), so
    // an id past 2^53 cannot round-trip. Say so rather than silently aliasing
    // two objects onto one segment.
    const segmentId = Number(objectId);
    if (!Number.isSafeInteger(segmentId) || segmentId <= 0) {
      throw new Error(
        `zarr-vectors: object id ${objectId} cannot be edited -- the editing ` +
          "path carries segment ids as JavaScript numbers, so ids must be " +
          "positive and below 2^53.",
      );
    }

    return buildSpatialSkeletonNodes({
      vertexPositions: aggregated.vertexPositions,
      indices: aggregated.indices,
      rank,
      segmentId,
      idOffset,
    });
  }

  private async getCrossChunkLinksForChunk(
    targetChunkCoords: readonly number[],
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    cellRead: CellReader,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    // `false` latches only the "this store has no links family" case, which the
    // reader signals by returning `undefined`. A links family that exists but
    // cannot be read *throws* — that must NOT permanently disable links for the
    // whole source (which is how a format-drift 404 previously masked the ZVF
    // 0.9 migration): warn once and render this chunk intra-only, retrying the
    // next chunk.
    if (this.crossChunkLinksCaches_ === false) return undefined;
    if (this.crossChunkLinksCaches_ === undefined) {
      this.crossChunkLinksCaches_ = createCrossChunkLinksCaches();
    }
    let table: CrossChunkLinksTable | undefined;
    try {
      table = await readCrossChunkLinksForChunk(
        { kvStoreRead, cellRead, kvStoreList },
        targetChunkCoords,
        this.crossChunkLinksCaches_,
        signal,
      );
    } catch (e) {
      if (!this.crossChunkLinksReadWarned_) {
        this.crossChunkLinksReadWarned_ = true;
        console.warn(
          "zarr-vectors: failed to read the cross-chunk links family; streamlines " +
            "may be broken at chunk boundaries. " +
            (e instanceof Error ? e.message : String(e)),
        );
      }
      return undefined;
    }
    if (table === undefined) {
      // Genuinely no links family: latch so we stop probing for it.
      this.crossChunkLinksCaches_ = false;
    }
    return table;
  }

  /** Whether the "links present but unreadable" warning has been emitted. */
  private crossChunkLinksReadWarned_ = false;

  /**
   * Plan the ghost fetches for a surface chunk's BOUNDARY faces.
   *
   * A face whose corners do not all sit in one chunk is an ordinary record in
   * the links family, filed under the offsets naming where its other corners
   * live (spec: geometry_types/mesh.md). Each foreign corner becomes a ghost
   * request; the face itself is held as a template of corner references --
   * a non-negative entry is a local vertex, a negative entry `-(r + 1)` is the
   * ghost answering request `r` -- because a request may be dropped (sparse
   * neighbour) and the face must then be dropped with it rather than drawn
   * against a wrong vertex.
   */
  private buildFaceBridgeRequests(
    table: CrossChunkLinksTable,
    selfChunkCoords: Float32Array,
    selfNumVertices: number,
  ): { ghostRequests: GhostVertexRequest[]; faceTemplates: Int32Array[] } {
    const selfCoords = Array.from(selfChunkCoords, (v) => Number(v));
    const ghostRequests: GhostVertexRequest[] = [];
    const faceTemplates: Int32Array[] = [];
    // One request per distinct foreign vertex: a corner is commonly shared by
    // several boundary faces, and refetching it per face would multiply the
    // ghost count (and the vertex texture) for nothing.
    const requestByForeignVertex = new Map<string, number>();
    for (const record of table.records) {
      const { endpoints } = record;
      if (endpoints.length < 3) continue;
      // Any local corner will do as the ghost's host: it is used only to
      // inherit a segment id, and every corner of a face is the same object.
      let hostLocalVertex = -1;
      for (const endpoint of endpoints) {
        if (
          endpointMatchesChunk(endpoint.chunkCoords, selfCoords) &&
          endpoint.vertexIndex >= 0 &&
          endpoint.vertexIndex < selfNumVertices
        ) {
          hostLocalVertex = endpoint.vertexIndex;
          break;
        }
      }
      // No corner here: the face belongs to another chunk, which will draw it.
      if (hostLocalVertex < 0) continue;

      const template = new Int32Array(endpoints.length);
      let usable = true;
      for (let i = 0; i < endpoints.length; ++i) {
        const endpoint = endpoints[i];
        if (endpointMatchesChunk(endpoint.chunkCoords, selfCoords)) {
          if (
            endpoint.vertexIndex < 0 ||
            endpoint.vertexIndex >= selfNumVertices
          ) {
            usable = false;
            break;
          }
          template[i] = endpoint.vertexIndex;
          continue;
        }
        const foreignKey = `${endpoint.chunkCoords.join(".")}/${endpoint.vertexIndex}`;
        let requestIndex = requestByForeignVertex.get(foreignKey);
        if (requestIndex === undefined) {
          requestIndex = ghostRequests.length;
          ghostRequests.push({
            hostLocalVertex,
            neighborChunkKey: endpoint.chunkCoords.join("."),
            neighborLocalVertex: endpoint.vertexIndex,
          });
          requestByForeignVertex.set(foreignKey, requestIndex);
        }
        template[i] = -(requestIndex + 1);
      }
      if (usable) faceTemplates.push(template);
    }
    return { ghostRequests, faceTemplates };
  }

  /**
   * `chunk` reduced to the objects THIS level is responsible for drawing.
   *
   * The levels of a per-object pyramid are nested, so drawing each of them
   * whole would draw the coarse backbone once per level. Instead each level
   * draws only the objects that are NEW at it -- present here and absent from
   * the next level up -- and the union across levels is exactly the admitted
   * set, with every object drawn exactly once, at exactly one level.
   *
   * That partition is what makes the load coarse-to-fine. The coarse levels are
   * tiny (tens of tracts, under a megabyte) so they appear at once and the whole
   * volume is populated immediately; each finer level then adds its own share on
   * top without redrawing what is already there. Picking a single level instead
   * means nothing at all is visible until that level's chunks arrive, which for
   * the finest level is gigabytes.
   *
   * `currentAdmissionFraction` carries both the mode and the ration: negative
   * disables the partition entirely (LOCAL focus, where one level is drawn whole
   * per cell), `1` keeps every object new at this level, and a value in between
   * keeps that share of them -- used only at the finest admitted level, where
   * the budget runs out partway.
   */
  private admitObjects(decoded: DecodedGeometryChunk): DecodedGeometryChunk {
    const fraction = this.currentAdmissionFraction;
    const coarser = this.parameters.coarserMembership;
    if (fraction < 0 || coarser === undefined) return decoded;
    const addressableIds = coarser.length * 8;
    const ration = fraction < 1;
    return filterChunkByAdmittedObjects(decoded, (low, high) => {
      // An id the membership bitset cannot address is KEPT. Rationing it would
      // be a decision made on no information, and dropping it could lose
      // geometry no other level is drawing. Note `low >>> 3`: ids at or above
      // 2^31 are negative as signed int32 and would index before the array.
      if (high !== 0 || low < 0 || low >= addressableIds) return true;
      // Present in the coarser level => that level draws it, not this one.
      if (((coarser[low >>> 3] >> (low & 7)) & 1) !== 0) return false;
      return ration ? objectRank(low) < fraction : true;
    });
  }

  async download(
    chunk: SpatiallyIndexedSkeletonChunk,
    signal: AbortSignal,
  ): Promise<void> {
    const {
      baseUrl,
      rank,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      linkDtype,
      hasFragmentSegmentIds,
      vertexIdAttribute,
      linkWidth,
    } = this.parameters;
    const { chunkGridPosition } = chunk;
    const chunkKey = Array.from(chunkGridPosition, (v) => String(v)).join(".");
    // `rawKvStoreRead` serves whole-file reads (links `zarr.json`); `cellRead`
    // serves the per-chunk array cells (origin + optional sharding resolved
    // inside it). The finest levels' large shards make the single shared
    // cellReader's per-shard index cache important — hence the per-source getter.
    const rawKvStoreRead = makeRawKvStoreRead(
      baseUrl,
      this.sharedKvStoreContext,
    );
    const kvStoreList = makeKvStoreList(baseUrl, this.sharedKvStoreContext);
    const cellRead = this.cellRead;

    const decoded = await downloadGeometryChunk(
      {
        chunkKey,
        rank,
        linkDtype: asLinkDtype(linkDtype),
        attributeNames,
        attributeDtypes: attributeDtypes.map(asAttributeDtype),
        linksConvention: linksConvention as ZarrVectorsLinksConvention,
        geometryKind: geometryKind as ZarrVectorsGeometryKind,
        hasFragmentSegmentIds,
        vertexIdAttribute,
        linkWidth,
        cellRead,
      },
      signal,
    );

    if (decoded === undefined) {
      // Sparse chunk presence — no vertices/<chunk> blob.  Set zero-
      // length buffers so the render layer's draw call short-circuits
      // safely.
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      return;
    }

    // Pass-1 cross-chunk bridge insertion.  For each cross_chunk_links
    // record incident on this chunk, fetch ONE boundary vertex from the
    // neighbor and append it as a ghost vertex + bridge edge.  Each
    // chunk renders independently with its existing per-chunk-isolated
    // GPU resources, but the visible line is continuous across chunk
    // boundaries.  See the design plan at
    // ~/.claude/plans/i-wanted-you-to-spicy-candy.md (option 3) for
    // the full rationale.
    let withBridges = decoded;
    const table = await this.getCrossChunkLinksForChunk(
      Array.from(chunkGridPosition, (v) => Number(v)),
      rawKvStoreRead,
      cellRead,
      kvStoreList,
      signal,
    );
    if (table !== undefined && isSurfaceGeometry(geometryKind)) {
      // Boundary faces: the surface analogue of the bridge logic below. Kept
      // separate because the two resolve different things -- a bridge is one
      // edge to one ghost with a walk direction, a boundary face is up to three
      // corners in three chunks with no direction at all.
      const { ghostRequests, faceTemplates } = this.buildFaceBridgeRequests(
        table,
        chunkGridPosition,
        decoded.numVertices,
      );
      if (faceTemplates.length > 0) {
        const ghosts = await fetchGhostVertices(
          ghostRequests,
          {
            rank,
            attributeNames,
            attributeDtypes: attributeDtypes.map(asAttributeDtype),
            cellRead,
          },
          signal,
        );
        const extraFaces = resolveBoundaryFaces(
          faceTemplates,
          ghosts,
          withBridges.numVertices,
        );
        withBridges = appendBoundaryFaces(withBridges, ghosts, extraFaces);
      }
    } else if (table !== undefined) {
      const {
        ghostRequests,
        intraChunkEdges,
        intraChunkBridges,
        ghostIsPredecessor,
      } = buildBridgeRequests(table, chunkGridPosition, decoded.numVertices);
      // Intra-chunk bridges first: both endpoints already live in the
      // chunk's vertex texture, so we just append flat (a, b) pairs to
      // the edges array.  Affects coarser pyramid levels where the
      // writer encodes metavertex-to-metavertex transitions inside one
      // chunk via cross_chunk_links records with same-chunk endpoints.
      if (intraChunkEdges.length > 0) {
        withBridges = appendIntraChunkEdges(withBridges, intraChunkEdges);
      }
      // Cross-chunk bridges next: fetch neighbor boundary data and
      // append ghost vertices with bridge edges.
      const resolvedBridges: ResolvedBridge[] = [...intraChunkBridges];
      if (ghostRequests.length > 0) {
        const ghosts = await fetchGhostVertices(
          ghostRequests,
          {
            rank,
            attributeNames,
            attributeDtypes: attributeDtypes.map(asAttributeDtype),
            cellRead,
          },
          signal,
        );
        if (ghosts.length > 0) {
          // Note: `fetchGhostVertices` drops requests whose neighbor
          // data is missing.  The remaining ghosts are appended in
          // order; ghost `g` lands at chunk-local index
          // `decodedNumVertices + intraChunkAppend + g`.
          //
          // We track each ghost's `isPredecessor` so the resolved
          // bridge points its predecessor/successor sides correctly
          // for tangent accumulation.  Drop alignment with the
          // original requests by matching `ghosts[g].bridgeFromLocalVertex`
          // back to `ghostRequests` — but in practice
          // `fetchGhostVertices` preserves request order; just skip
          // the dropped requests.
          const baseGhostIndex = withBridges.numVertices;
          withBridges = appendGhostVertices(withBridges, ghosts);
          // Walk ghosts and ghostRequests in parallel to build
          // resolved bridges.  Use bridgeFromLocalVertex to match
          // each surviving ghost back to its original request.
          let requestCursor = 0;
          for (let g = 0; g < ghosts.length; ++g) {
            const ghost = ghosts[g];
            // Advance requestCursor to the first request matching this
            // ghost's host vertex.  fetchGhostVertices is order-
            // preserving, so this is monotonic.
            while (
              requestCursor < ghostRequests.length &&
              ghostRequests[requestCursor].hostLocalVertex !==
                ghost.bridgeFromLocalVertex
            ) {
              requestCursor++;
            }
            if (requestCursor >= ghostRequests.length) break;
            const isPredecessor = ghostIsPredecessor[requestCursor];
            requestCursor++;
            const ghostIdx = baseGhostIndex + g;
            const hostIdx = ghost.bridgeFromLocalVertex;
            resolvedBridges.push({
              predecessorLocalIdx: isPredecessor ? ghostIdx : hostIdx,
              successorLocalIdx: isPredecessor ? hostIdx : ghostIdx,
            });
          }
        }
      }
      // Finally: refresh per-vertex tangents so the default RGB-by-
      // tangent shader gives non-black colors on metavertex centroids
      // at coarser pyramid levels (where `computeTangents` would
      // otherwise leave single-vertex-fragment tangents at zero).
      // Vertices not touched by any bridge keep their existing
      // tangent (correct at level 0 for multi-vertex fragments).
      if (resolvedBridges.length > 0 && withBridges.tangents !== undefined) {
        withBridges = recomputeTangentsForBridges(withBridges, resolvedBridges);
      }
    }

    // Per-object admission: drop the objects the memory budget did not buy.
    //
    // Last, on the fully assembled chunk, because a ghost vertex inherits its
    // host's segment id -- so a cross-chunk bridge is kept or dropped together
    // with the tract it belongs to, and admitted tracts stay continuous across
    // cell boundaries with no separate reasoning about bridges. Inert unless a
    // fraction below 1 is in force, so every other store and mode decodes
    // byte-identically.
    const drawn = this.admitObjects(withBridges);

    // Node identity survives only while the vertex array is the one it was
    // decoded against. The bridge and admission transforms above may append or
    // drop vertices, and each drops `nodeIds` when it can no longer vouch for
    // the alignment, so this assignment is either correct or absent.
    if (
      drawn.nodeIds !== undefined &&
      drawn.nodeIds.length === drawn.numVertices
    ) {
      chunk.nodeIds = drawn.nodeIds;
    }
    chunk.vertexPositions = drawn.positions;
    // `indices` carries whatever primitive this geometry draws: vertex pairs
    // for lines, vertex triples for a surface. The render layer knows which
    // from the source's `geometryPrimitive`, so one field serves both.
    chunk.indices = drawn.faces ?? drawn.edges;
    // Order: synthesised tangent first (streamline/polyline only), then
    // user-declared attributes in declaration order, then the synthesised
    // per-vertex `"segment"` column last.  The frontend
    // (`buildZvSpatialVertexAttributes`) mirrors this exact ordering so
    // texture-unit indices line up and the render layer's
    // `findIndex(name === "segment")` resolves `segmentAttributeIndex`.
    const attrs: (
      | Float32Array
      | Uint8Array
      | Uint16Array
      | Uint32Array
      | Int8Array
      | Int16Array
      | Int32Array
    )[] = [];
    if (drawn.tangents !== undefined) {
      attrs.push(drawn.tangents);
    }
    for (const a of drawn.vertexAttributes) attrs.push(a);
    if (drawn.segmentIds !== undefined) {
      attrs.push(drawn.segmentIds);
    }
    chunk.vertexAttributes = attrs;

    // Retain a slim view of this chunk's geometry for the ROI filter: the
    // render-layer backend re-tests it whenever the ROI list changes, so the
    // filter runs within memory without refetching. Only when a segment column
    // exists — otherwise geometry cannot be attributed to an object and there
    // is nothing to test. The decoded `SkeletonChunk` already has exactly the
    // `RoiFilterableChunk` shape, and these references stay valid after
    // serialize (see `roiFilterableChunk`'s docstring).
    if (drawn.segmentIds !== undefined) {
      const caps = KIND_CAPABILITIES[geometryKind as ZarrVectorsGeometryKind];
      chunk.roiFilterableChunk = {
        rank,
        numVertices: drawn.numVertices,
        positions: drawn.positions,
        segmentIds: drawn.segmentIds,
        fragmentIndex: drawn.fragmentIndex,
        // Without the object model the segment column holds one id per VERTEX,
        // so the fold must run per vertex; see `perVertexObjects`.
        perVertexObjects: !caps.hasObjectModel,
        surfaceVertices: caps.primitive === "triangles",
        // Per-vertex attribute values, for attribute predicates. Retained only
        // for the kinds with no per-OBJECT attribute tier to read instead: the
        // arrays are otherwise dropped after `serialize` transfers its packed
        // copy, so keeping them is real worker RAM (`downloadSucceeded` charges
        // it) and a store that can answer from `object_attributes/` must not
        // pay it. Keyed by the on-disk attribute name — what the filter state
        // persists, and stable across the pyramid.
        ...(caps.hasObjectModel
          ? {}
          : {
              vertexAttributes: retainedVertexAttributes(
                attributeNames,
                drawn.vertexAttributes,
              ),
            }),
      };
    }
  }
}

/**
 * The per-vertex attribute columns to retain for attribute predicates, keyed by
 * on-disk attribute name.
 *
 * Everything the reader decodes is already float32 (`vertex_attribute_float.ts`),
 * so the common case aliases the decoded array rather than copying it -- the
 * retention then costs nothing beyond keeping it reachable past `serialize`.
 * A narrower array (a store read by an older path) is widened once here so the
 * filter has one representation to test against.
 */
function retainedVertexAttributes(
  attributeNames: readonly string[],
  columns: readonly AttributeTypedArray[],
): Map<string, Float32Array> {
  const retained = new Map<string, Float32Array>();
  for (let i = 0; i < attributeNames.length && i < columns.length; ++i) {
    const column = columns[i];
    retained.set(
      attributeNames[i],
      column instanceof Float32Array ? column : Float32Array.from(column),
    );
  }
  return retained;
}

/**
 * Compare a cross-chunk endpoint's chunk-coordinates array to a
 * spatial chunk's grid position.  Both inputs are arrays of length
 * ``sid_ndim`` (3 for streamlines).  Returns ``true`` when the two
 * point at the same chunk.
 */
function endpointMatchesChunk(
  endpointCoords: readonly number[],
  selfCoords: readonly number[],
): boolean {
  if (endpointCoords.length !== selfCoords.length) return false;
  for (let i = 0; i < endpointCoords.length; ++i) {
    if (endpointCoords[i] !== selfCoords[i]) return false;
  }
  return true;
}

/**
 * Filter the cross-chunk-link table down to records incident on the
 * named chunk.  Output has two kinds of items:
 *
 * - `ghostRequests`: cross-chunk records where one endpoint is in
 *   this chunk and the other is in a different chunk.  Each becomes
 *   a {@link GhostVertexRequest} so the backend can fetch the
 *   neighbor's boundary vertex and append it as a ghost.
 * - `intraChunkEdges`: records where BOTH endpoints are in this
 *   chunk.  At coarser pyramid levels the writer encodes intra-chunk
 *   metavertex-to-metavertex bridges (consecutive same-chunk
 *   fragments of one streamline) here too — no ghost vertex needed
 *   because both endpoints already live in this chunk's vertex
 *   texture.  Each record becomes one flat `(a, b)` pair of chunk-
 *   local vertex indices appended directly to the chunk's edge list.
 *
 * Triangle / metanode records (``linkWidth !== 2``) are skipped —
 * those describe mesh-style face stitching, not line edges.
 *
 * Module-level and exported rather than a method on
 * {@link ZarrVectorsSpatialGeometrySourceBackend}: it touches no instance
 * state, and pass-1's cross-chunk continuity is only verifiable end-to-end by
 * driving this together with `downloadGeometryChunk`,
 * `readCrossChunkLinksForChunk` and `appendGhostVertices` over real store
 * bytes.  Keeping it private forced such a harness to re-implement it, which
 * is precisely the copy that would stop reflecting this code.
 */
export function buildBridgeRequests(
  table: CrossChunkLinksTable,
  selfChunkCoords: Float32Array | readonly number[],
  selfNumVertices: number,
): {
  ghostRequests: GhostVertexRequest[];
  intraChunkEdges: Uint32Array;
  /**
   * Intra-chunk bridges resolved to chunk-local predecessor/successor
   * indices.  Cross-chunk bridges (one endpoint is a future ghost)
   * are appended later in `download()` once we know each ghost's
   * chunk-local index — see the comment on the ghost-append step.
   */
  intraChunkBridges: ResolvedBridge[];
  /**
   * Per-ghost-request side info needed to extend `intraChunkBridges`
   * after ghosts are appended.  `ghostIsPredecessor[i]` mirrors
   * `ghostRequests[i].isGhostPredecessor`.
   */
  ghostIsPredecessor: boolean[];
} {
  if (table.linkWidth !== 2) {
    return {
      ghostRequests: [],
      intraChunkEdges: new Uint32Array(0),
      intraChunkBridges: [],
      ghostIsPredecessor: [],
    };
  }
  const selfCoords = Array.from(selfChunkCoords, (v) => Number(v));
  const ghostRequests: GhostVertexRequest[] = [];
  const intraEdges: number[] = [];
  const intraChunkBridges: ResolvedBridge[] = [];
  const ghostIsPredecessor: boolean[] = [];
  for (const record of table.records) {
    // Cross-chunk records encode walk order: endpoint[0] is the
    // PREDECESSOR (last vertex of fragment A) and endpoint[1] is
    // the SUCCESSOR (first vertex of fragment B).  See the polyline
    // writer at `zarr_vectors/types/polylines.py` and the boundary
    // helper at `zarr_vectors/spatial/boundary.py:75-114`.
    //
    // For cross-chunk records, the ghost's tangent must point in the
    // forward walk direction.  When this chunk matches endpoint[0],
    // the ghost is the successor — it sits AFTER the host.  When
    // this chunk matches endpoint[1], the ghost is the predecessor.
    // `appendGhostVertices` flips the synthesised ghost-tangent
    // sign based on `isGhostPredecessor` so both sides of the
    // bridge interpolate the same forward walk direction.
    const a = record.endpoints[0];
    const b = record.endpoints[1];
    const aMatches = endpointMatchesChunk(a.chunkCoords, selfCoords);
    const bMatches = endpointMatchesChunk(b.chunkCoords, selfCoords);
    if (aMatches && bMatches) {
      // Intra-chunk bridge: writer-emitted record connecting two
      // fragments inside the SAME chunk (coarser-pyramid-level
      // metavertex-to-metavertex transition).  Drop the record if
      // either endpoint is out of range.
      if (
        a.vertexIndex < 0 ||
        a.vertexIndex >= selfNumVertices ||
        b.vertexIndex < 0 ||
        b.vertexIndex >= selfNumVertices
      ) {
        continue;
      }
      intraEdges.push(a.vertexIndex, b.vertexIndex);
      intraChunkBridges.push({
        predecessorLocalIdx: a.vertexIndex,
        successorLocalIdx: b.vertexIndex,
      });
    } else if (aMatches) {
      ghostRequests.push({
        hostLocalVertex: a.vertexIndex,
        neighborChunkKey: b.chunkCoords.join("."),
        neighborLocalVertex: b.vertexIndex,
        isGhostPredecessor: false, // ghost is endpoint[1] = successor
      });
      ghostIsPredecessor.push(false);
    } else if (bMatches) {
      ghostRequests.push({
        hostLocalVertex: b.vertexIndex,
        neighborChunkKey: a.chunkCoords.join("."),
        neighborLocalVertex: a.vertexIndex,
        isGhostPredecessor: true, // ghost is endpoint[0] = predecessor
      });
      ghostIsPredecessor.push(true);
    }
    // Neither-match records (not ours) are silently ignored.
  }
  return {
    ghostRequests,
    intraChunkEdges: Uint32Array.from(intraEdges),
    intraChunkBridges,
    ghostIsPredecessor,
  };
}

/**
 * Resolves a selected segment id to the dense object index the manifests are
 * keyed by, caching the store's ``object_attributes/segment_id`` column.
 *
 * Shared by both backends: the per-object (pass-2) source resolves an id per
 * download, and the spatially-indexed (pass-1) source resolves one per
 * `getSkeleton` call for the editing UI. Each backend owns an instance, so the
 * cached column is amortized within a source but never shared across levels.
 */
class ObjectIndexResolver {
  /**
   * Cached ``object_attributes/segment_id`` array (uint64, dense-OID
   * order → sorted ascending) mapping the dense object index ↔ the
   * original (e.g. flywire) segment id.  ``null`` = no such attribute
   * (selected id IS the dense index — identity); ``undefined`` = not yet
   * probed.
   */
  private segmentIds_: BigUint64Array | null | undefined;

  /**
   * Resolve a selected segment id to its dense object index.  When the
   * store carries ``object_attributes/segment_id`` (standard object-index
   * convention) binary-search the sorted ids; otherwise the id IS the
   * dense index.  Returns ``undefined`` when the id is absent.
   */
  async resolve(
    objectId: number | bigint,
    numObjects: number,
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    signal: AbortSignal,
  ): Promise<number | undefined> {
    if (this.segmentIds_ === undefined) {
      // `object_attributes/<name>/` is itself a zarr v3 array, NOT a group
      // wrapping a nested `data/` array — the same layout the frontend reads
      // and documents in `buildSegmentPropertyMap`. The old
      // `.../segment_id/data/c/0` spelling matched nothing, so every read
      // returned undefined, `segmentIds_` latched to null, and the identity
      // fallback below took over. For a connectomics store that fallback
      // cannot work: the ids are root ids of order 1e17 while `numObjects` is
      // a few thousand, so `target < numObjects` is false and EVERY object
      // lookup resolved to undefined.
      //
      // `kvStoreRead` already zstd-decompresses (see `makeKvStoreRead`), and
      // the chunk is writer-padded to its full `chunk_shape` (65536), which
      // the length check and `slice` below trim to the real object count.
      const bytes = await kvStoreRead(
        "object_attributes/segment_id/c/0",
        signal,
      );
      if (bytes === undefined || bytes.byteLength < numObjects * 8) {
        this.segmentIds_ = null;
      } else {
        const copy = bytes.slice(0, numObjects * 8);
        this.segmentIds_ = new BigUint64Array(copy.buffer);
      }
    }
    const ids = this.segmentIds_;
    const target = BigInt(objectId);
    if (ids === null) {
      return target >= 0n && target < BigInt(numObjects)
        ? Number(target)
        : undefined;
    }
    let lo = 0;
    let hi = ids.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const v = ids[mid];
      if (v === target) return mid;
      if (v < target) lo = mid + 1;
      else hi = mid - 1;
    }
    return undefined;
  }
}

/**
 * Per-segment (object-keyed) skeleton chunk source — the **pass-2**
 * backing store.  One chunk per `objectId`.  The render layer (a
 * subclass of `SkeletonLayer`) iterates `forEachVisibleSegment` and
 * requests one chunk per visible object_id; this backend resolves each
 * `objectId` against the store's `object_index/manifests` array and
 * aggregates the named fragments across spatial chunks into one
 * merged geometry.
 *
 * Configuration of `numObjects` and `manifestChunkSize` is supplied
 * via the parameter object — the frontend dispatch (slice 4c) reads
 * `object_index/.zattrs` and `object_index/manifests/zarr.json` to fill
 * these fields before constructing the source.
 */
@registerSharedObject()
export class ZarrVectorsObjectKeyedGeometrySourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(SkeletonSource),
  ZarrVectorsObjectKeyedGeometrySourceParameters,
) {
  /**
   * This source draws ON TOP OF a coarse pass-1 bulk that shares the same chunk
   * budget, so its chunks must never outbid that bulk. `SkeletonLayer` reads
   * this to anchor their priority; see `getSkeletonChunkPriority`.
   */
  readonly drawsOverSpatialBulk = true;

  /**
   * Shared across this source's concurrent object downloads.
   *
   * Tracts in a dissection are spatially clustered by construction -- they pass
   * through the same regions -- so without this the same spatial chunk is
   * fetched and decoded once per tract crossing it. That redundancy, not the
   * transfer, is what made a large full-detail set unusable.
   */
  private chunkCoalescingCache = new ChunkCoalescingCache<
    Awaited<ReturnType<typeof downloadGeometryChunk>>
  >();
  /**
   * Cached decoded ``cross_chunk_links/0/`` table for this level.  Read
   * lazily on the first ``download()`` and reused across all subsequent
   * object downloads — the table is per-level, not per-object.
   *
   * ``null`` means "checked, store has no such table" (older
   * zarr-vectors stores written without ``cross_chunk_strategy =
   * "explicit_links"``).  ``undefined`` means "not yet probed".
   *
   * Only fetched for ``explicit`` / ``implicit_sequential_with_branches``
   * geometry (graphs / skeletons) — see the ``download()`` call site.
   * For ``implicit_sequential`` (streamline/polyline) stores,
   * `downloadSegmentSkeleton` reconstructs cross-chunk edges purely from
   * manifest order (`deriveImplicitSequentialCrossChunkEdges`) and never
   * reads this table at all (its `vi` fields are literal `0`
   * placeholders for that convention — see the comment at
   * `geometry_segment_download.ts`'s call site), so fetching it
   * unconditionally was pure waste: a real dataset's whole-level
   * decode can be tens of millions of records / multiple gigabytes
   * (the same issue fixed for the spatially-indexed pass-1 backend via
   * `readCrossChunkLinksForChunk` — see that class's
   * `crossChunkLinksCaches_` docstring).
   *
   * TODO: graphs/skeletons still pay the whole-table decode here since
   * one object's manifest can span many chunks (not the single target
   * a per-chunk query assumes) — a proper fix needs a multi-chunk-
   * scoped query (or deferring the query until `ownedChunks` is known
   * inside `downloadSegmentSkeleton`), out of scope for this pass since
   * it doesn't affect the reported streamline crash.
   */
  private crossChunkLinks_: CrossChunkLinksTable | null | undefined;

  /** See {@link ObjectIndexResolver}. */
  private objectIndex = new ObjectIndexResolver();

  private cellReader_: ShardCellReader | undefined;
  /**
   * One {@link ShardCellReader} per source (amortizes the per-shard index
   * cache). Reads the per-chunk geometry arrays honoring `chunk_grid_origin`
   * and the optional `sharding_indexed` packing. The manifest / object-attribute
   * arrays are 1-D object-indexed and stay on the whole-file `kvStoreRead` path.
   */
  protected get cellRead(): CellReader {
    let reader = this.cellReader_;
    if (reader === undefined) {
      const p = this.parameters;
      reader = this.cellReader_ = this.registerDisposer(
        new ShardCellReader(
          this.chunkManager,
          p.baseUrl,
          this.sharedKvStoreContext.kvStoreContext,
          makeRawKvStoreRead(p.baseUrl, this.sharedKvStoreContext),
          {
            origin: p.chunkGridOrigin,
            sharded: p.sharded,
            shardShape: p.shardChunkShape,
            separator: p.cellSeparator,
          },
        ),
      );
    }
    return reader.read;
  }

  private async getCrossChunkLinks(
    kvStoreRead: (
      subpath: string,
      signal: AbortSignal,
    ) => Promise<Uint8Array | undefined>,
    cellRead: CellReader,
    kvStoreList: (
      prefix: string,
      signal: AbortSignal,
    ) => Promise<{ directories: string[]; files: string[] }>,
    signal: AbortSignal,
  ): Promise<CrossChunkLinksTable | undefined> {
    if (this.crossChunkLinks_ !== undefined) {
      return this.crossChunkLinks_ ?? undefined;
    }
    const table = await readCrossChunkLinks(
      { kvStoreRead, cellRead, kvStoreList },
      signal,
    );
    this.crossChunkLinks_ = table ?? null;
    return table;
  }

  async download(chunk: SkeletonChunk, signal: AbortSignal): Promise<void> {
    const {
      baseUrl,
      rank,
      attributeNames,
      attributeDtypes,
      linksConvention,
      geometryKind,
      linkDtype,
      hasFragmentSegmentIds,
    } = this.parameters;
    const kvStoreRead = makeKvStoreRead(baseUrl, this.sharedKvStoreContext);
    const rawKvStoreRead = makeRawKvStoreRead(
      baseUrl,
      this.sharedKvStoreContext,
    );
    const kvStoreList = makeKvStoreList(baseUrl, this.sharedKvStoreContext);
    const cellRead = this.cellRead;

    // The manifests array's `numObjects` / `chunkSize` aren't carried
    // on the parameter blob (slice 4c will plumb them through from
    // `object_index/.zattrs.num_objects` and the array's `zarr.json`).
    // For now the backend reads them on each download — cheap because
    // the kvstore caches the metadata after the first fetch.
    const { numObjects, chunkSize } = await readManifestArrayShape(
      baseUrl,
      this.sharedKvStoreContext,
      signal,
    );

    // See `crossChunkLinks_`'s docstring: `implicit_sequential` streamline
    // stores never consult this table (cross-chunk edges come from
    // manifest order instead), so skip the potentially multi-gigabyte
    // whole-level decode entirely for that convention.
    const crossChunkLinks =
      linksConvention === "implicit_sequential"
        ? undefined
        : await this.getCrossChunkLinks(
            rawKvStoreRead,
            cellRead,
            kvStoreList,
            signal,
          );

    // Map the selected segment id (e.g. a flywire uint64) to the dense
    // object index via object_attributes/segment_id before the manifest
    // lookup.  Out-of-store ids yield an empty skeleton.
    const resolvedOid = await this.objectIndex.resolve(
      chunk.objectId,
      numObjects,
      kvStoreRead,
      signal,
    );
    if (resolvedOid === undefined) {
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      if (hasSynthesisedTangent(geometryKind as ZarrVectorsGeometryKind)) {
        chunk.vertexAttributes = [
          new Float32Array(0),
          ...chunk.vertexAttributes,
        ];
      }
      return;
    }

    const aggregated = await downloadSegmentSkeleton(
      resolvedOid,
      {
        manifestReader: {
          numObjects,
          chunkSize,
          sidNdim: rank,
          kvStoreRead,
        },
        cellRead,
        rank,
        linkDtype: asLinkDtype(linkDtype),
        attributeNames,
        attributeDtypes: attributeDtypes.map(asAttributeDtype),
        linksConvention: linksConvention as ZarrVectorsLinksConvention,
        geometryKind: geometryKind as ZarrVectorsGeometryKind,
        crossChunkLinks,
        hasFragmentSegmentIds,
        chunkCache: this.chunkCoalescingCache,
      },
      signal,
    );

    if (aggregated === undefined) {
      // OID not in the manifest, or every named chunk is missing.
      chunk.vertexPositions = new Float32Array(0);
      chunk.indices = new Uint32Array(0);
      chunk.vertexAttributes = attributeNames.map(() => new Float32Array(0));
      // Every geometry kind with synthesised tangents (streamline,
      // polyline, graph) reserves a tangent slot at index 0 — mirror
      // that here so the render layer's attribute count is consistent
      // across passes even when an OID has no geometry.  See
      // `hasSynthesisedTangent` in `geometry_kind.ts` for the canonical
      // per-kind capability table.
      if (hasSynthesisedTangent(geometryKind as ZarrVectorsGeometryKind)) {
        chunk.vertexAttributes = [
          new Float32Array(0),
          ...chunk.vertexAttributes,
        ];
      }
      return;
    }

    chunk.vertexPositions = aggregated.vertexPositions;
    chunk.indices = aggregated.indices;
    chunk.vertexAttributes = aggregated.vertexAttributes;
  }
}

/**
 * Read `numObjects` and the manifests array's chunk shape from
 * the store's `object_index/.zattrs` and `object_index/manifests/zarr.json`.
 *
 * Centralised here so the per-segment backend has one read path; slice
 * 4c will move this into the frontend dispatch so the values arrive
 * pre-resolved on the parameter blob.
 */
async function readManifestArrayShape(
  baseUrl: string,
  sharedKvStoreContext: {
    kvStoreContext: {
      read: (url: string, options: { signal: AbortSignal }) => Promise<any>;
    };
  },
  signal: AbortSignal,
): Promise<{ numObjects: number; chunkSize: number }> {
  const arrayMetaUrl = joinBaseUrlAndPath(
    baseUrl,
    "object_index/manifests/zarr.json",
  );
  const response = await sharedKvStoreContext.kvStoreContext.read(
    arrayMetaUrl,
    {
      signal,
    },
  );
  if (response === undefined) {
    throw new Error(
      "zarr-vectors object-keyed skeleton: missing object_index/manifests/zarr.json",
    );
  }
  const text = new TextDecoder().decode(
    new Uint8Array((await response.response.arrayBuffer()) as ArrayBuffer),
  );
  const meta = JSON.parse(text);
  const shape = meta.shape;
  const chunkGrid = meta?.chunk_grid;
  if (
    !Array.isArray(shape) ||
    shape.length !== 1 ||
    typeof shape[0] !== "number"
  ) {
    throw new Error(
      "zarr-vectors object_index/manifests: shape must be a 1-D array of one integer",
    );
  }
  const numObjects = shape[0];
  // Zarr v3 regular chunk grid: chunk_grid.configuration.chunk_shape
  const chunkShape =
    chunkGrid?.configuration?.chunk_shape ?? chunkGrid?.chunk_shape;
  if (!Array.isArray(chunkShape) || chunkShape.length !== 1) {
    throw new Error(
      "zarr-vectors object_index/manifests: missing or non-1-D chunk_shape",
    );
  }
  const chunkSize = chunkShape[0];
  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new Error(
      `zarr-vectors object_index/manifests: invalid chunk_shape ${JSON.stringify(chunkShape)}`,
    );
  }
  return { numObjects, chunkSize };
}

/**
 * Request/response RPC backing `ZarrVectorsSpatialGeometrySource.getSkeleton`.
 *
 * A round trip rather than a chunk: the store lives in the worker, and the
 * editing UI needs one named object's whole node list on demand -- not
 * whatever the camera happens to have admitted.
 */
registerPromiseRPC(
  ZARR_VECTORS_GET_OBJECT_NODES_RPC_ID,
  async function (
    this: RPC,
    x: { source: number; objectId: string },
    progressOptions: Partial<ProgressOptions>,
  ) {
    const source = this.get(
      x.source,
    ) as ZarrVectorsSpatialGeometrySourceBackend;
    const nodes = await source.getObjectSkeletonNodes(
      BigInt(x.objectId),
      progressOptions.signal ?? new AbortController().signal,
    );
    return { value: { nodes }, transfers: [] };
  },
);
