import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import * as json_keys from "#src/layer/segmentation/json_keys.js";
import { makeCachedDerivedWatchableValue } from "#src/trackable_value.js";
import { formatScaleWithUnitAsString } from "#src/util/si_units.js";
import type { LayerControlDefinition } from "#src/widget/layer_control.js";
import { registerLayerControl } from "#src/widget/layer_control.js";
import { checkboxLayerControl } from "#src/widget/layer_control_checkbox.js";
import { enumLayerControl } from "#src/widget/layer_control_enum.js";
import { rangeLayerControl } from "#src/widget/layer_control_range.js";
import {
  renderScaleLayerControl,
  SpatialSkeletonGridRenderScaleWidget,
} from "#src/widget/render_scale_widget.js";
import {
  colorSeedLayerControl,
  fixedColorLayerControl,
} from "#src/widget/segmentation_color_mode.js";

/**
 * Best-effort unit label for the layer's spatial coordinate space,
 * formatted with the appropriate SI prefix (e.g. ``"mm"`` from
 * (scale=1e-3, unit="m"); ``"nm"`` from (scale=1e-9, unit="m")).
 *
 * Used to label the spatial-skeleton grid-resolution widget so a
 * millimetre-units dataset doesn't read as ``"339 nm"`` because of the
 * widget's class-level ``unitOfTarget = "nm"`` fallback.  Falls back
 * to ``"nm"`` (the widget default) when no usable unit information is
 * available, matching legacy behaviour.
 *
 * Picks the first dimension's unit — for typical 3-D spatial layers
 * (x, y, z all in the same unit), all three would resolve identically.
 */
function getLayerSpatialUnit(layer: SegmentationUserLayer): string {
  const cs = layer.manager.root.coordinateSpace.value;
  if (!cs.valid || cs.rank === 0) return "nm";
  const unit = cs.units[0];
  const scale = cs.scales[0];
  if (typeof unit !== "string" || unit === "") return "nm";
  if (!Number.isFinite(scale) || scale <= 0) return unit;
  return formatScaleWithUnitAsString(scale, unit, {
    elide1: true,
    precision: 0,
  });
}

export const LAYER_CONTROLS: LayerControlDefinition<SegmentationUserLayer>[] = [
  {
    label: "Color seed",
    title: "Color segments based on a hash of their id",
    toolJson: json_keys.COLOR_SEED_JSON_KEY,
    ...colorSeedLayerControl(),
  },
  {
    label: "Fixed color",
    title:
      "Use a fixed color for all segments without an explicitly-specified color",
    toolJson: json_keys.SEGMENT_DEFAULT_COLOR_JSON_KEY,
    ...fixedColorLayerControl(),
  },
  {
    label: "Saturation",
    toolJson: json_keys.SATURATION_JSON_KEY,
    title: "Saturation of segment colors",
    ...rangeLayerControl((layer) => ({ value: layer.displayState.saturation })),
  },
  {
    label: "Opacity (on)",
    toolJson: json_keys.SELECTED_ALPHA_JSON_KEY,
    // Shown for anything with cross-section geometry.
    //
    // This was previously hidden whenever `hasSpatiallyIndexedSkeletonsLayer`
    // was set, on the reasoning that a tract layer's "on" opacity is set
    // per-group in the Filter tab. That equates "spatially indexed" with
    // "tractogram", which zarr-vectors broke: point clouds, graphs, skeletons
    // and meshes all draw through the same spatially-indexed render layers and
    // have no per-group opacity to fall back on, so they simply lost the only
    // control over `selectedAlpha` (which defaults to 0.5). See the
    // "Opacity (3d)" control below for the same mistake in the 3-d path.
    isValid: (layer) => layer.has2dLayer,
    title:
      "Opacity in cross-section views of selected segments and of dense " +
      "(spatially-indexed) skeletons",
    ...rangeLayerControl((layer) => ({
      value: layer.displayState.selectedAlpha,
    })),
  },
  {
    label: "Opacity (off)",
    toolJson: json_keys.NOT_SELECTED_ALPHA_JSON_KEY,
    // Hidden for spatially-indexed (tract) layers: those fold cross-section off
    // opacity into the unified "Off opacity" control below, so users see one
    // background-opacity slider instead of three.
    isValid: (layer) =>
      makeCachedDerivedWatchableValue(
        (has2d, hasSpatialSkeletons) => has2d && !hasSpatialSkeletons,
        [layer.has2dLayer, layer.hasSpatiallyIndexedSkeletonsLayer],
      ),
    title:
      "Opacity in cross-section views of non-selected segments and of dense " +
      "(spatially-indexed) skeletons",
    ...rangeLayerControl((layer) => ({
      value: layer.displayState.notSelectedAlpha,
    })),
  },
  {
    label: "Opacity (off)",
    toolJson: json_keys.ROI_NONPASSING_ALPHA_JSON_KEY,
    isValid: (layer) => layer.hasSpatiallyIndexedSkeletonsLayer,
    title:
      "Opacity of the background tractogram — one control for the 3-d hidden " +
      "opacity, the cross-section 'off' opacity, and (when the ROI filter is " +
      "active) the non-passing 'ghost' streamlines, kept in sync. The per-group " +
      "'on' opacity is set in the Filter tab.",
    ...rangeLayerControl((layer) => ({
      // A single control driving the three background-opacity trackables at
      // once. Reads the 3-d hidden alpha (the main perspective tractogram) and
      // writes all three so they stay equal. Each is a plain live trackable, so
      // no stale-closure snap-back.
      value: {
        get value() {
          return layer.displayState.hiddenObjectAlpha.value;
        },
        set value(v: number) {
          layer.displayState.hiddenObjectAlpha.value = v;
          layer.displayState.notSelectedAlpha.value = v;
          layer.displayState.roiFilter.ghostAlpha = v;
        },
        changed: layer.displayState.hiddenObjectAlpha.changed,
      },
      options: { min: 0, max: 1, step: 0.01 },
    })),
  },
  {
    label: "Resolution (slice)",
    toolJson: json_keys.CROSS_SECTION_RENDER_SCALE_JSON_KEY,
    isValid: (layer) => layer.has2dLayer,
    ...renderScaleLayerControl((layer) => ({
      histogram: layer.sliceViewRenderScaleHistogram,
      target: layer.sliceViewRenderScaleTarget,
    })),
  },
  {
    label: "Resolution (mesh)",
    toolJson: json_keys.MESH_RENDER_SCALE_JSON_KEY,
    isValid: (layer) => layer.has3dLayer,
    ...renderScaleLayerControl((layer) => ({
      histogram: layer.displayState.renderScaleHistogram,
      target: layer.displayState.renderScaleTarget,
    })),
  },
  {
    label: "Resolution (skeleton grid 2D)",
    toolJson: json_keys.SKELETON_CROSS_SECTION_RENDER_SCALE_JSON_KEY,
    isValid: (layer) =>
      makeCachedDerivedWatchableValue(
        (levels, hasSpatialSkeletons) =>
          hasSpatialSkeletons && levels.length > 0,
        [
          layer.displayState.spatialSkeletonGridLevels,
          layer.hasSpatiallyIndexedSkeletonsLayer,
        ],
      ),
    title:
      "Select the grid size level for spatially indexed skeletons in 2D views",
    ...renderScaleLayerControl(
      (layer) => ({
        histogram: layer.displayState.spatialSkeletonGridRenderScaleHistogram2d,
        target: layer.displayState.spatialSkeletonGridResolutionTarget2d,
        unitOfTarget: getLayerSpatialUnit(layer),
        // A deliberate drag takes over from the camera. Without this the
        // auto-level derivation rewrites the target on the very next frame
        // and the handle springs back, which reads as a broken control.
        onManualTarget: () => {
          layer.displayState.autoSpatialSkeletonGridLevel2d.value = false;
        },
        // ...and reset hands it back. Taking over used to be one-way: nothing
        // in the UI could re-enable the camera-driven level once a drag had
        // switched it off, so a single stray scroll over the control disabled
        // automatic level selection for the rest of the session.
        onResetTarget: () => {
          layer.displayState.autoSpatialSkeletonGridLevel2d.value = true;
          // Restore the default calibration too. Without this, a bias baked
          // in before a fix to the auto-resume path (or by any future bug in
          // it) survives every future reset -- a double-click should return
          // to pure camera-driven behaviour, not to a stale multiplier.
          layer.displayState.spatialSkeletonGridResolutionBias2d.reset();
        },
      }),
      SpatialSkeletonGridRenderScaleWidget,
    ),
  },
  {
    label: "Resolution (skeleton grid 3D)",
    toolJson: json_keys.SKELETON_PERSPECTIVE_RENDER_SCALE_JSON_KEY,
    isValid: (layer) =>
      makeCachedDerivedWatchableValue(
        (levels, hasSpatialSkeletons) =>
          hasSpatialSkeletons && levels.length > 0,
        [
          layer.displayState.spatialSkeletonGridLevels,
          layer.hasSpatiallyIndexedSkeletonsLayer,
        ],
      ),
    title:
      "Select the grid size level for spatially indexed skeletons in 3D views",
    ...renderScaleLayerControl(
      (layer) => ({
        histogram: layer.displayState.spatialSkeletonGridRenderScaleHistogram3d,
        target: layer.displayState.spatialSkeletonGridResolutionTarget3d,
        unitOfTarget: getLayerSpatialUnit(layer),
        // A deliberate drag takes over from the camera. Without this the
        // auto-level derivation rewrites the target on the very next frame
        // and the handle springs back, which reads as a broken control.
        onManualTarget: () => {
          layer.displayState.autoSpatialSkeletonGridLevel3d.value = false;
        },
        // ...and reset hands it back. Taking over used to be one-way: nothing
        // in the UI could re-enable the camera-driven level once a drag had
        // switched it off, so a single stray scroll over the control disabled
        // automatic level selection for the rest of the session.
        onResetTarget: () => {
          layer.displayState.autoSpatialSkeletonGridLevel3d.value = true;
          // Restore the default calibration too. Without this, a bias baked
          // in before a fix to the auto-resume path (or by any future bug in
          // it) survives every future reset -- a double-click should return
          // to pure camera-driven behaviour, not to a stale multiplier.
          layer.displayState.spatialSkeletonGridResolutionBias3d.reset();
        },
      }),
      SpatialSkeletonGridRenderScaleWidget,
    ),
  },
  {
    label: "Opacity (3d)",
    toolJson: json_keys.OBJECT_ALPHA_JSON_KEY,
    // Shown for anything with 3-d geometry.
    //
    // This was previously hidden whenever `hasSpatiallyIndexedSkeletonsLayer`
    // was set and no `MeshLayer` was present, on the reasoning that a tract
    // layer's 3-d opacity is already covered by the unified "Opacity (off)"
    // (bulk) plus the per-group "on" opacity. That gate was wrong twice over:
    //
    //  - `hasMeshLayer` is true only for MeshLayer/MultiscaleMeshLayer. A
    //    zarr-vectors store whose geometry_types is ["mesh"] draws through
    //    PerspectiveViewSpatiallyIndexedSkeletonLayer with
    //    geometryPrimitive === "triangles", so it read as "spatial skeletons,
    //    no mesh" and lost the control -- as did every non-tract skeleton
    //    store, which has no per-group opacity to fall back on.
    //  - `objectAlpha` is not merely cosmetic here. It is half of the test in
    //    PerspectiveViewSpatiallyIndexedSkeletonLayer.isTransparent that
    //    decides whether the layer draws in the opaque pass or in the
    //    weighted-blended OIT pass. With it pinned below 1 and unreachable,
    //    the geometry renders additively -- overlapping surfaces of a single
    //    object summing instead of occluding -- and no visible control could
    //    restore solid rendering.
    //
    // The cost of showing it on a pure tract layer is one extra slider; the
    // cost of hiding it was an unreachable render mode.
    isValid: (layer) => layer.has3dLayer,
    title: "Opacity of meshes and skeletons",
    ...rangeLayerControl((layer) => ({
      value: layer.displayState.objectAlpha,
    })),
  },
  {
    label: "Silhouette (3d)",
    toolJson: json_keys.MESH_SILHOUETTE_RENDERING_JSON_KEY,
    // Silhouette shades mesh faces by their angle to the view; it does nothing
    // for skeletons/streamlines (no faces), so gate it on an actual mesh.
    isValid: (layer) => layer.hasMeshLayer,
    title:
      "Set to a non-zero value to increase transparency of object faces perpendicular to view direction",
    ...rangeLayerControl((layer) => ({
      value: layer.displayState.silhouetteRendering,
      options: { min: 0, max: maxSilhouettePower, step: 0.1 },
    })),
  },
  {
    label: "Hide segment ID 0",
    toolJson: json_keys.HIDE_SEGMENT_ZERO_JSON_KEY,
    title: "Disallow selection and display of segment id 0",
    ...checkboxLayerControl((layer) => layer.displayState.hideSegmentZero),
  },
  {
    label: "Detail focus",
    toolJson: json_keys.SPATIAL_SKELETON_DETAIL_FOCUS_JSON_KEY,
    // Only meaningful where a pyramid is being budgeted at all.
    isValid: (layer) => layer.hasSpatiallyIndexedSkeletonsLayer,
    title:
      "What the memory left over by the pyramid level being drawn is spent " +
      "on. LOCAL is the standard behaviour: more detail near the camera, " +
      "which for long objects means streamlines cut off wherever the finer " +
      "chunks ran out. OBJECT spends it on whole objects instead, fetched by " +
      "id and spread across the entire volume, so what it adds is complete " +
      "tracts rather than fragments. On a store whose levels each hold every " +
      "object -- meshes and point clouds -- OBJECT still draws one level " +
      "everywhere and keeps its whole volume resident, but the level is then " +
      "chosen by the memory ceiling rather than per object.",
    ...enumLayerControl(
      (layer) => layer.displayState.spatialSkeletonDetailFocus,
    ),
  },
  {
    label: "Ignore memory ceiling",
    toolJson: json_keys.IGNORE_SKELETON_MEMORY_CEILING_JSON_KEY,
    // Only meaningful where a pyramid is being budgeted at all.
    isValid: (layer) => layer.hasSpatiallyIndexedSkeletonsLayer,
    title:
      "Load as if memory were unlimited. Under object focus that admits every " +
      "object at the finest level; otherwise it allows finer pyramid levels " +
      "than the budget nominally permits, whose whole-level estimate refuses a " +
      "zoomed-in view detail it would never actually fetch. Either way, a wide " +
      "view of a dense level can then exhaust GPU memory.",
    ...checkboxLayerControl(
      (layer) => layer.displayState.ignoreSpatialSkeletonMemoryCeiling,
    ),
  },
  {
    label: "Base segment coloring",
    toolJson: json_keys.BASE_SEGMENT_COLORING_JSON_KEY,
    title: "Color base segments individually",
    ...checkboxLayerControl((layer) => layer.displayState.baseSegmentColoring),
  },
  {
    label: "Show all by default",
    title: "Show all segments if none are selected",
    toolJson: json_keys.IGNORE_NULL_VISIBLE_SET_JSON_KEY,
    ...checkboxLayerControl((layer) => layer.displayState.ignoreNullVisibleSet),
  },
  {
    label: "Highlight on hover",
    toolJson: json_keys.HOVER_HIGHLIGHT_JSON_KEY,
    title: "Highlight the segment under the mouse pointer",
    ...checkboxLayerControl((layer) => layer.displayState.hoverHighlight),
  },
  ...getViewSpecificSkeletonRenderingControl("2d"),
  ...getViewSpecificSkeletonRenderingControl("3d"),
];

const maxSilhouettePower = 10;

function getViewSpecificSkeletonRenderingControl(
  viewName: "2d" | "3d",
): LayerControlDefinition<SegmentationUserLayer>[] {
  return [
    {
      label: `Skeleton mode (${viewName})`,
      toolJson: `${json_keys.SKELETON_RENDERING_JSON_KEY}.mode${viewName}`,
      // Lines-only: a point cloud and a mesh have no lines-versus-points choice
      // to make, and offering one that does nothing is worse than not offering it.
      isValid: (layer) => layer.hasLineGeometryLayer,
      ...enumLayerControl(
        (layer) =>
          layer.displayState.skeletonRenderingOptions[
            `params${viewName}` as const
          ].mode,
      ),
    },
    {
      label: `Line width (${viewName})`,
      toolJson: `${json_keys.SKELETON_RENDERING_JSON_KEY}.lineWidth${viewName}`,
      isValid: (layer) => layer.hasSkeletonsLayer,
      toolDescription: `Skeleton line width (${viewName})`,
      title: `Skeleton line width (${viewName})`,
      ...rangeLayerControl((layer) => ({
        value:
          layer.displayState.skeletonRenderingOptions[
            `params${viewName}` as const
          ].lineWidth,
        options: { min: 1, max: 40, step: 1 },
      })),
    },
  ];
}

export function registerLayerControls(layerType: typeof SegmentationUserLayer) {
  for (const control of LAYER_CONTROLS) {
    registerLayerControl(layerType, control);
  }
}
