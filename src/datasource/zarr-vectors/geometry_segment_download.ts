/**
 * @license
 * Copyright 2026 Allen Institute for Brain Science
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/**
 * Aggregate one object's skeleton geometry across the chunks that the
 * `object_index/manifests` reader reports for it.
 *
 * The pass-2 chunk-source backend calls `downloadSegmentSkeleton(oid,
 * ...)`: this resolves the manifest, fetches every named source chunk,
 * filters each one to just the fragments the object owns, and emits a
 * single merged geometry — `vertexPositions`, `indices`, and
 * `vertexAttributes` — ready to drop into a per-segment `SkeletonChunk`.
 */

import type { ChunkCoalescingCache } from "#src/datasource/zarr-vectors/chunk_coalescing_cache.js";
import type {
  AttributeTypedArray,
  LinksConvention,
  SkeletonChunk,
  GeometryKind,
} from "#src/datasource/zarr-vectors/geometry_chunk.js";
import {
  downloadGeometryChunk,
  type AttributeDtype,
  type LinkDtype,
} from "#src/datasource/zarr-vectors/geometry_chunk_download.js";
import { hasSynthesisedTangent } from "#src/datasource/zarr-vectors/geometry_kind.js";
import type { CrossChunkLinksTable } from "#src/datasource/zarr-vectors/links.js";
import { resolveFragmentRef } from "#src/datasource/zarr-vectors/object_manifest.js";
import {
  readObjectManifest,
  type ObjectManifestReaderOptions,
} from "#src/datasource/zarr-vectors/object_manifest_reader.js";
import type { CellReader } from "#src/datasource/zarr-vectors/shard_cell_reader.js";

/**
 * The merged geometry for one object.  Shapes match the per-segment
 * `SkeletonChunk` fields that the render layer consumes.
 */
export interface AggregatedSegmentSkeleton {
  /** `(numVertices * rank)` floats. */
  readonly vertexPositions: Float32Array;
  /** `(numEdges * 2)` chunk-local-then-global vertex indices. */
  readonly indices: Uint32Array;
  /**
   * Per-vertex attributes, in the order the render layer will reference.
   * For streamline / polyline geometry kinds, index 0 is the synthesised
   * `tangent` (vec3); subsequent entries are the user-declared
   * attributes from `attributeNames` in declaration order.  For
   * skeleton geometry, only user-declared attributes are present.
   */
  readonly vertexAttributes: AttributeTypedArray[];
}

/**
 * Pure function: filter one decoded `SkeletonChunk` to just the
 * vertices and edges named by `fragmentIndices`.  Returns chunk-local
 * geometry (positions in float-flat layout, edges as chunk-local
 * vertex indices into the filtered output), plus the filtered attribute
 * arrays parallel to the positions.
 *
 * The render layer's vertex-attribute ordering convention is mirrored
 * here: when `chunk.tangents` is present (streamline/polyline) it is
 * emitted as the first attribute; user attributes follow.
 *
 * Vertices that don't belong to any of the named fragments are
 * dropped; edges with at least one endpoint dropped are also dropped
 * (no dangling references).
 */
export function filterChunkByFragments(
  chunk: SkeletonChunk,
  fragmentIndices: Uint32Array,
): {
  positions: Float32Array;
  edges: Uint32Array;
  attributes: AttributeTypedArray[];
  /** Map from source chunk-local vertex index → position in the filtered output. */
  vertexRemap: Int32Array;
} {
  const { rank, numVertices, positions, edges, vertexAttributes, tangents } =
    chunk;

  // Collect chunk-local vertex indices owned by the named fragments.
  // Use a `seen` mask to dedupe (the same vertex can be referenced by
  // multiple fragments — e.g. a branch point at level 0).  Walk-order
  // is preserved for the first occurrence so attribute lookups stay
  // deterministic.
  const seen = new Uint8Array(numVertices);
  const owned: number[] = [];
  for (let i = 0; i < fragmentIndices.length; ++i) {
    const f = fragmentIndices[i];
    const fragVerts = chunk.fragmentIndex.indices(f);
    for (let j = 0; j < fragVerts.length; ++j) {
      const v = fragVerts[j];
      if (seen[v] === 0) {
        seen[v] = 1;
        owned.push(v);
      }
    }
  }
  const numOwned = owned.length;

  // Build the source→filtered vertex remap.  -1 means "not in output".
  const vertexRemap = new Int32Array(numVertices).fill(-1);
  for (let i = 0; i < numOwned; ++i) vertexRemap[owned[i]] = i;

  // Gather positions for owned vertices, in walk order.
  const filteredPositions = new Float32Array(numOwned * rank);
  for (let i = 0; i < numOwned; ++i) {
    const v = owned[i];
    for (let d = 0; d < rank; ++d) {
      filteredPositions[i * rank + d] = positions[v * rank + d];
    }
  }

  // Filter edges: keep only those whose endpoints are both in the owned
  // set, and remap to filtered-output indices.
  const keptEdges: number[] = [];
  for (let e = 0; e < edges.length; e += 2) {
    const a = edges[e];
    const b = edges[e + 1];
    if (seen[a] === 1 && seen[b] === 1) {
      keptEdges.push(vertexRemap[a]);
      keptEdges.push(vertexRemap[b]);
    }
  }
  const filteredEdges = new Uint32Array(keptEdges);

  // Filter attribute arrays (tangents first if present, then user attrs)
  // in the same conventional order the spatially-indexed backend uses.
  const filteredAttrs: AttributeTypedArray[] = [];
  if (tangents !== undefined) {
    const t = new Float32Array(numOwned * 3);
    for (let i = 0; i < numOwned; ++i) {
      const v = owned[i];
      t[i * 3] = tangents[v * 3];
      t[i * 3 + 1] = tangents[v * 3 + 1];
      t[i * 3 + 2] = tangents[v * 3 + 2];
    }
    filteredAttrs.push(t);
  }
  for (const src of vertexAttributes) {
    // Each `src` is a per-vertex array of length `numVertices` (scalar
    // attribute) — the higher-level zarr-vectors writer paths don't
    // currently emit multi-component vertex attributes via the
    // ZarrVectorsAttributeDtype enum, so a 1:1 element copy suffices.
    const Ctor = src.constructor as new (n: number) => AttributeTypedArray;
    const dst = new Ctor(numOwned);
    for (let i = 0; i < numOwned; ++i) dst[i] = src[owned[i]] as never;
    filteredAttrs.push(dst);
  }

  return {
    positions: filteredPositions,
    edges: filteredEdges,
    attributes: filteredAttrs,
    vertexRemap,
  };
}

export interface DownloadSegmentSkeletonOptions {
  /** Manifest reader configuration (numObjects, chunkSize, sidNdim, kvStoreRead).
   * Its `kvStoreRead` serves the 1-D, object-indexed `object_index/manifests`
   * array (non-sharded, origin 0) — NOT the per-chunk geometry, which goes
   * through `cellRead`. */
  readonly manifestReader: ObjectManifestReaderOptions;
  /** Reads the per-chunk geometry array cells (vertices, vertex_fragments, …),
   * resolving `chunk_grid_origin` and the optional `sharding_indexed` packing. */
  readonly cellRead: CellReader;
  /**
   * Shared across concurrent calls, so a chunk several tracts pass through is
   * decoded once rather than once per tract. Optional: omit it and each call
   * decodes independently, as before.
   */
  readonly chunkCache?: ChunkCoalescingCache<
    Awaited<ReturnType<typeof downloadGeometryChunk>>
  >;
  /** Spatial-chunk download parameters (rank, dtypes, links convention, etc.). */
  readonly rank: number;
  readonly linkDtype: LinkDtype;
  readonly attributeNames: readonly string[];
  readonly attributeDtypes: readonly AttributeDtype[];
  readonly linksConvention: LinksConvention;
  readonly geometryKind: GeometryKind;
  /**
   * Optional decoded ``cross_chunk_links/0/`` table for the level.  When
   * present, ``downloadSegmentSkeleton`` appends one edge per record
   * whose two endpoints both land on vertices the current object owns
   * (i.e. survived the per-block fragment filter).  Records of
   * ``linkWidth !== 2`` are ignored — they're for meshes / metanode
   * pyramids, not streamlines.
   *
   * Pass-2 callers (the segment-keyed backend) should fetch this table
   * once per level via {@link readCrossChunkLinks} and share it across
   * objects.
   */
  readonly crossChunkLinks?: CrossChunkLinksTable;
  /** Whether to fetch `fragment_attributes/segment_id` per chunk. */
  readonly hasFragmentSegmentIds?: boolean;
}

/**
 * Per-block bookkeeping kept while {@link downloadSegmentSkeleton}
 * processes a manifest.  Used to build {@link OrderedManifestBlock}s for
 * the `implicit_sequential` cross-chunk path, which needs one entry PER
 * BLOCK (a chunk visited more than once yields more than one).  The
 * skeleton cross-chunk path uses `chunkVertexGlobal` instead (accumulated
 * per CHUNK, across every block that touches it) — see its own doc
 * comment for why a per-block map like this one can't serve that path.
 */
interface OwnedChunkInfo {
  /** Map from chunk-local vertex index → filtered-output position (-1 = dropped). */
  readonly vertexRemap: Int32Array;
  /** Cumulative merged-output index at which this chunk's vertices start. */
  readonly vertexOffset: number;
}

/**
 * Per-block bookkeeping carried by {@link downloadSegmentSkeleton}'s
 * manifest walk.  Exposed via the helper signatures so unit tests can
 * drive {@link deriveImplicitSequentialCrossChunkEdges} without
 * staging a whole download.
 */
export interface OrderedManifestBlock {
  /** Joined chunk-coordinate string, e.g. ``"0.-1.2"``. */
  readonly chunkKey: string;
  /** Maps chunk-local vertex index → filtered-output position (-1 = dropped). */
  readonly vertexRemap: Int32Array;
  /** Cumulative merged-output index at which this block's vertices start. */
  readonly vertexOffset: number;
  /** Chunk-local first vertex of this block's single fragment (-1 if N/A). */
  readonly firstFragmentLocalVert: number;
  /** Chunk-local last vertex of this block's single fragment (-1 if N/A). */
  readonly lastFragmentLocalVert: number;
}

/**
 * Pure helper for the ``implicit_sequential`` inter-fragment bridge
 * path.  Walks the manifest-ordered blocks pairwise; for **every**
 * consecutive pair emits one edge bridging fragment k's last vertex
 * with fragment k+1's first vertex (both translated to the merged-
 * output vertex index space).
 *
 * Bridges connect *consecutive fragments*, not just *cross-chunk*
 * transitions.  Streamlines are partitioned by zarr-vectors' bin grid
 * (writer default: ``bin_shape`` = chunk_shape / 4), so a polyline can
 * generate multiple fragments **inside one chunk**.  Each fragment is
 * its own implicit-sequential edge run, so adjacent fragments — same
 * chunk or not — need an explicit bridge between fragment k's last
 * vertex and fragment k+1's first vertex.  Skipping same-chunk
 * transitions would leave intra-chunk bin-boundary gaps visible.
 *
 * Skips pairs where either side isn't a single-fragment block
 * (``firstFragmentLocalVert`` / ``lastFragmentLocalVert`` are -1) — the
 * endpoint identity becomes ambiguous in that case.  Also skips pairs
 * where the relevant chunk-local vertex was filtered out (remap < 0) —
 * same no-dangling rule the per-chunk filter applies.
 *
 * Exported so unit tests can drive it with hand-crafted block sequences
 * without staging a whole download.
 */
export function deriveImplicitSequentialCrossChunkEdges(
  orderedBlocks: readonly OrderedManifestBlock[],
): Uint32Array {
  const out: number[] = [];
  for (let i = 0; i + 1 < orderedBlocks.length; ++i) {
    const a = orderedBlocks[i];
    const b = orderedBlocks[i + 1];
    if (a.lastFragmentLocalVert < 0 || b.firstFragmentLocalVert < 0) continue;
    if (a.lastFragmentLocalVert >= a.vertexRemap.length) continue;
    if (b.firstFragmentLocalVert >= b.vertexRemap.length) continue;
    const aRemap = a.vertexRemap[a.lastFragmentLocalVert];
    const bRemap = b.vertexRemap[b.firstFragmentLocalVert];
    if (aRemap < 0 || bRemap < 0) continue;
    out.push(aRemap + a.vertexOffset);
    out.push(bRemap + b.vertexOffset);
  }
  return new Uint32Array(out);
}

/**
 * Pure helper: given a decoded cross-chunk table and each touched
 * chunk's chunk-local-vertex → merged-output-index map, emit the subset
 * of edges whose endpoints both land on owned vertices.
 *
 * Takes `chunkVertexGlobal` (accumulated per CHUNK, across every
 * manifest block that touches it), NOT the per-BLOCK `OwnedChunkInfo`
 * map. A chunk hosting more than one of this object's fragments (an
 * ordinary occurrence -- a branch that revisits a chunk, or simply two
 * separate fragments sharing a chunk) produces more than one block for
 * that chunk key; a per-block map can only remember the last one
 * written, silently losing any cross-chunk edge whose endpoint falls in
 * an earlier block. `chunkVertexGlobal` has no such gap: it is one
 * array per chunk, written into (never replaced) as each block touching
 * that chunk is processed, so every fragment's vertices resolve
 * correctly regardless of how many blocks the chunk was split into.
 * Confirmed against a real 158-vertex neuron with four multi-fragment
 * chunks: the per-block map recovered only 1 of 8 real cross-chunk
 * edges (the one whose endpoints happened to both land in each chunk's
 * LAST block); every other real edge was silently dropped.
 *
 * Exported so unit tests can drive it with hand-crafted fixtures
 * without staging a whole manifest/chunk pipeline.
 */
export function collectOwnedCrossChunkEdges(
  table: CrossChunkLinksTable,
  chunkVertexGlobal: Map<string, Int32Array>,
): Uint32Array {
  // Only line-arity (linkWidth=2) records describe cross-chunk edges.
  // Triangle / metanode records aren't relevant to streamline rendering.
  if (table.linkWidth !== 2) return new Uint32Array(0);
  const out: number[] = [];
  for (const record of table.records) {
    const [a, b] = record.endpoints;
    const aGlobalOf = chunkVertexGlobal.get(a.chunkCoords.join("."));
    const bGlobalOf = chunkVertexGlobal.get(b.chunkCoords.join("."));
    if (aGlobalOf === undefined || bGlobalOf === undefined) continue;
    if (a.vertexIndex < 0 || a.vertexIndex >= aGlobalOf.length) continue;
    if (b.vertexIndex < 0 || b.vertexIndex >= bGlobalOf.length) continue;
    const aGlobal = aGlobalOf[a.vertexIndex];
    const bGlobal = bGlobalOf[b.vertexIndex];
    if (aGlobal < 0 || bGlobal < 0) continue;
    out.push(aGlobal);
    out.push(bGlobal);
  }
  return new Uint32Array(out);
}

/**
 * Download and aggregate one object's skeleton geometry across all the
 * chunks the manifest reports for it.  Returns `undefined` when the
 * object is absent (no manifest, or every fragment chunk missing).
 *
 * Algorithm:
 *
 * 1. Resolve `oid` → `ManifestBlock[]` via `readObjectManifest`.
 * 2. For each block:
 *    a. Fetch + decode the spatial chunk via `downloadGeometryChunk`.
 *    b. Resolve `block.fragmentRef` to a flat list of fragment indices
 *       within that chunk.
 *    c. Call `filterChunkByFragments` to extract just those fragments'
 *       vertices/edges/attributes.
 * 3. Concatenate the per-chunk filtered outputs, re-offsetting the edge
 *    indices so they reference the merged vertex array.
 */
export async function downloadSegmentSkeleton(
  oid: number | bigint,
  options: DownloadSegmentSkeletonOptions,
  signal: AbortSignal,
): Promise<AggregatedSegmentSkeleton | undefined> {
  const {
    manifestReader,
    cellRead,
    rank,
    linkDtype,
    attributeNames,
    attributeDtypes,
    linksConvention,
    geometryKind,
    crossChunkLinks,
    hasFragmentSegmentIds,
    chunkCache,
  } = options;
  const manifest = await readObjectManifest(oid, manifestReader, signal);
  if (manifest === undefined || manifest.length === 0) return undefined;

  const perChunkPositions: Float32Array[] = [];
  const perChunkEdges: Uint32Array[] = [];
  // Outer array: one slot per attribute.  Inner: one entry per source
  // chunk.  Every geometry kind with synthesised tangents (streamline,
  // polyline, graph) carries the tangent in slot 0 of
  // `filterChunkByFragments`'s output — see `hasSynthesisedTangent` in
  // `geometry_kind.ts` for the canonical per-kind capability table.
  const numAttrsExpected =
    (hasSynthesisedTangent(geometryKind) ? 1 : 0) + attributeNames.length;
  const perChunkAttrs: AttributeTypedArray[][] = Array.from(
    { length: numAttrsExpected },
    () => [] as AttributeTypedArray[],
  );
  // Per-block bookkeeping, in manifest order.  Drives the
  // implicit_sequential cross-chunk path: consecutive blocks in
  // different chunks emit one bridging edge (last vertex of fragment k →
  // first vertex of fragment k+1).  See zarr-vectors-py
  // ``polylines.py:325``: the on-disk cross_chunk_links table only
  // records ``((cc_a, 0), (cc_b, 0))`` placeholders for streamlines, so
  // it carries no fragment-specific endpoint info — we have to
  // reconstruct edges from manifest order.
  interface OrderedBlock extends OwnedChunkInfo {
    readonly chunkKey: string;
    /** Chunk-local index of the first vertex of this block's single
     * fragment; -1 if the block has 0 or >1 fragments (cross-chunk edge
     * reconstruction skips those). */
    readonly firstFragmentLocalVert: number;
    /** Chunk-local index of the last vertex of this block's single
     * fragment; -1 if not single-fragment. */
    readonly lastFragmentLocalVert: number;
  }
  const orderedBlocks: OrderedBlock[] = [];

  let runningVertexOffset = 0;

  // Fetch every chunk this manifest touches UP FRONT, in parallel, and (when a
  // cache is supplied) shared with the other tracts being fetched alongside.
  //
  // The loop below stays strictly ordered -- `runningVertexOffset` accumulates
  // and `orderedBlocks` drives the implicit_sequential cross-chunk edge
  // reconstruction, both of which depend on manifest order -- but it no longer
  // pays a round trip per block. Previously each block was a serial `await`, so
  // a tract spanning 40 chunks cost 40 sequential decodes.
  const uniqueChunkKeys = [
    ...new Set(manifest.map((block) => block.chunkCoords.join("."))),
  ];
  const loadChunk = (chunkKey: string, chunkSignal: AbortSignal) =>
    downloadGeometryChunk(
      {
        chunkKey,
        rank,
        linkDtype,
        attributeNames,
        attributeDtypes,
        linksConvention,
        geometryKind,
        hasFragmentSegmentIds,
        cellRead,
      },
      chunkSignal,
    );
  const fetchedChunks = new Map<
    string,
    Awaited<ReturnType<typeof downloadGeometryChunk>>
  >();
  await Promise.all(
    uniqueChunkKeys.map(async (chunkKey) => {
      const skel =
        chunkCache === undefined
          ? await loadChunk(chunkKey, signal)
          : // A cached entry is shared, so it must not carry THIS caller's
            // signal: aborting one tract would cancel a decode the others are
            // still waiting on. See `ChunkCoalescingCache`.
            await chunkCache.get(chunkKey, () =>
              loadChunk(chunkKey, new AbortController().signal),
            );
      fetchedChunks.set(chunkKey, skel);
    }),
  );
  if (signal.aborted) return undefined;

  // Chunk-local vertex -> merged-output index, accumulated across every
  // block of that chunk (a per-block map can only remember the last
  // block written for a given chunk key, silently losing any earlier
  // fragment's vertices -- see collectOwnedCrossChunkEdges's doc
  // comment). Drives both the intra-chunk branch-link lookup below and
  // the skeleton cross-chunk edge path further down.
  const chunkVertexGlobal = new Map<string, Int32Array>();

  for (const block of manifest) {
    const chunkKey = block.chunkCoords.join(".");
    const skel = fetchedChunks.get(chunkKey);
    if (skel === undefined) continue;

    const fragmentIds = resolveFragmentRef(block.fragmentRef);
    const filtered = filterChunkByFragments(skel, fragmentIds);
    if (filtered.positions.length === 0) continue;

    // For the implicit_sequential cross-chunk reconstruction, capture
    // the first and last chunk-local vertex of this block's single
    // fragment.  Blocks with 0 or >1 fragments don't participate
    // (cross-chunk endpoint identity becomes ambiguous).
    let firstFragmentLocalVert = -1;
    let lastFragmentLocalVert = -1;
    if (fragmentIds.length === 1) {
      const fragVerts = skel.fragmentIndex.indices(fragmentIds[0]);
      if (fragVerts.length > 0) {
        firstFragmentLocalVert = fragVerts[0];
        lastFragmentLocalVert = fragVerts[fragVerts.length - 1];
      }
    }

    const info: OwnedChunkInfo = {
      vertexRemap: filtered.vertexRemap,
      vertexOffset: runningVertexOffset,
    };
    // `orderedBlocks` intentionally keeps one entry PER BLOCK (a chunk
    // visited more than once yields more than one) -- the
    // implicit_sequential path needs each visit's own identity. The
    // skeleton cross-chunk path instead accumulates into
    // `chunkVertexGlobal` below, per CHUNK rather than per block.
    orderedBlocks.push({
      ...info,
      chunkKey,
      firstFragmentLocalVert,
      lastFragmentLocalVert,
    });

    perChunkPositions.push(filtered.positions);
    // Shift edge indices into the merged-output coordinate space.
    if (filtered.edges.length > 0) {
      if (runningVertexOffset === 0) {
        perChunkEdges.push(filtered.edges);
      } else {
        const shifted = new Uint32Array(filtered.edges.length);
        for (let i = 0; i < filtered.edges.length; ++i) {
          shifted[i] = filtered.edges[i] + runningVertexOffset;
        }
        perChunkEdges.push(shifted);
      }
    }
    if (filtered.attributes.length !== numAttrsExpected) {
      throw new Error(
        `downloadSegmentSkeleton: chunk ${chunkKey} returned ` +
          `${filtered.attributes.length} attributes; expected ${numAttrsExpected}`,
      );
    }
    for (let i = 0; i < numAttrsExpected; ++i) {
      perChunkAttrs[i].push(filtered.attributes[i]);
    }

    {
      let globalOf = chunkVertexGlobal.get(chunkKey);
      if (globalOf === undefined) {
        globalOf = new Int32Array(skel.numVertices).fill(-1);
        chunkVertexGlobal.set(chunkKey, globalOf);
      }
      const { vertexRemap } = filtered;
      for (let v = 0; v < vertexRemap.length; ++v) {
        const r = vertexRemap[v];
        if (r >= 0) globalOf[v] = r + runningVertexOffset;
      }
    }

    runningVertexOffset += filtered.positions.length / rank;
  }

  // Branch links joining two fragments of the SAME chunk.
  //
  // Each manifest block is filtered on its own fragment list, and
  // `filterChunkByFragments` keeps only edges with both endpoints inside that
  // list -- so an edge between two fragments is dropped by both blocks and
  // never appears in the output. For `implicit_sequential` that is harmless
  // (the bridges are re-derived from block order below), but a skeleton's
  // branch links live exactly there: dropping them returns one path per
  // fragment instead of one tree, and every path start then looks like a root.
  // Measured on a 1075-vertex axon: 36 fragments came back as 36 roots with
  // zero branch points.
  const intraChunkBranchEdges: number[] = [];
  if (linksConvention !== "implicit_sequential") {
    const emitted = new Set<number>();
    const edgeKey = (a: number, b: number) =>
      a < b ? a * runningVertexOffset + b : b * runningVertexOffset + a;
    for (const chunkEdges of perChunkEdges) {
      for (let i = 0; i < chunkEdges.length; i += 2) {
        emitted.add(edgeKey(chunkEdges[i], chunkEdges[i + 1]));
      }
    }
    for (const [chunkKey, globalOf] of chunkVertexGlobal) {
      const skel = fetchedChunks.get(chunkKey);
      if (skel === undefined) continue;
      const { edges } = skel;
      for (let e = 0; e + 1 < edges.length; e += 2) {
        const a = globalOf[edges[e]];
        const b = globalOf[edges[e + 1]];
        if (a < 0 || b < 0 || a === b) continue;
        const key = edgeKey(a, b);
        if (emitted.has(key)) continue;
        emitted.add(key);
        intraChunkBranchEdges.push(a, b);
      }
    }
  }

  // Inter-fragment bridge reconstruction.  Two strategies:
  //
  //  - implicit_sequential (polylines / streamlines): walk
  //    `orderedBlocks` pairwise; emit one edge per consecutive pair,
  //    connecting fragment k's last vertex to fragment k+1's first
  //    vertex.  Bridges are needed for both chunk-to-chunk transitions
  //    AND for bin-to-bin transitions within one chunk — zarr-vectors
  //    partitions streamlines by `bin_shape`, so one streamline can
  //    produce multiple same-chunk fragments.  The on-disk
  //    cross_chunk_links blob for these stores carries no usable
  //    endpoint info (its `vi` values are literal `0` placeholders;
  //    see zarr-vectors-py polylines.py:325).
  //
  //  - explicit / implicit_sequential_with_branches (graphs, skeletons):
  //    the on-disk cross_chunk_links blob carries real chunk-local
  //    vertex indices for each endpoint.  Use the blob-based filter on
  //    `chunkVertexGlobal`.
  let crossChunkEdges: Uint32Array | undefined;
  if (linksConvention === "implicit_sequential") {
    const edges = deriveImplicitSequentialCrossChunkEdges(orderedBlocks);
    if (edges.length > 0) crossChunkEdges = edges;
  } else if (crossChunkLinks !== undefined && chunkVertexGlobal.size > 0) {
    const edges = collectOwnedCrossChunkEdges(
      crossChunkLinks,
      chunkVertexGlobal,
    );
    if (edges.length > 0) crossChunkEdges = edges;
  }

  if (runningVertexOffset === 0) return undefined;

  // Concatenate per-chunk arrays.
  const totalFloats = runningVertexOffset * rank;
  const vertexPositions = new Float32Array(totalFloats);
  {
    let cursor = 0;
    for (const p of perChunkPositions) {
      vertexPositions.set(p, cursor);
      cursor += p.length;
    }
  }
  let totalEdgeEntries = intraChunkBranchEdges.length;
  for (const e of perChunkEdges) totalEdgeEntries += e.length;
  if (crossChunkEdges !== undefined) totalEdgeEntries += crossChunkEdges.length;
  const indices = new Uint32Array(totalEdgeEntries);
  {
    let cursor = 0;
    for (const e of perChunkEdges) {
      indices.set(e, cursor);
      cursor += e.length;
    }
    if (intraChunkBranchEdges.length > 0) {
      indices.set(intraChunkBranchEdges, cursor);
      cursor += intraChunkBranchEdges.length;
    }
    if (crossChunkEdges !== undefined) {
      indices.set(crossChunkEdges, cursor);
    }
  }
  const vertexAttributes: AttributeTypedArray[] = [];
  for (let a = 0; a < numAttrsExpected; ++a) {
    const parts = perChunkAttrs[a];
    // Each attribute keeps its dtype consistent across chunks because
    // the dtypes come from per-array zarr metadata, not per-chunk.
    // Use the first non-empty part's constructor to allocate the merged
    // buffer (fallback to Float32Array if all parts are zero-length).
    let totalLen = 0;
    for (const p of parts) totalLen += p.length;
    let merged: AttributeTypedArray;
    if (parts.length === 0 || totalLen === 0) {
      merged = new Float32Array(0);
    } else {
      const Ctor = parts[0].constructor as new (
        n: number,
      ) => AttributeTypedArray;
      merged = new Ctor(totalLen);
      let cursor = 0;
      for (const p of parts) {
        (
          merged as unknown as {
            set: (a: ArrayLike<number>, o: number) => void;
          }
        ).set(p as unknown as ArrayLike<number>, cursor);
        cursor += p.length;
      }
    }
    vertexAttributes.push(merged);
  }

  return { vertexPositions, indices, vertexAttributes };
}
