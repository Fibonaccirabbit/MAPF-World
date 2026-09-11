"""Generate MAPF city grids from Overpass or local OSM PBF data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

__version__ = "0.3.0"
LOG = logging.getLogger("mapf_generator")


class ProgressLogHandler(logging.StreamHandler):
    """Write log messages alongside terminal progress bars."""

    def emit(self, record):
        from tqdm import tqdm

        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


DEFAULT_WIDTHS_M = {
    "motorway": 12.0,
    "trunk": 10.0,
    "primary": 9.0,
    "secondary": 8.0,
    "tertiary": 7.0,
    "residential": 6.0,
    "service": 4.0,
    "living_street": 5.0,
    "pedestrian": 5.0,
    "footway": 2.5,
    "path": 2.0,
    "cycleway": 2.5,
    "steps": 1.5,
    "track": 3.0,
}
NETWORKS = ("all", "all_public", "walk", "bike", "drive", "drive_service")


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str
    source: str
    output_dir: str
    tile_size_m: float = 3072.0
    tile_stride_m: float = 1536.0
    grid_resolution: int = 256
    resolution_m: float | None = None
    min_free_ratio: float = 0.25
    max_free_ratio: float = 0.70
    min_largest_component: float = 0.90
    min_spatial_coverage: float = 0.75
    max_tiles: int = 128
    max_candidates: int = 100000
    min_path_length_m: float = 20.0
    default_width_m: float = 3.0
    width_scale: float = 1.0
    morphology_radius: int = 0
    keep_largest_component: bool = False
    network_type: str = "all"
    max_api_area_km2: float = 250.0
    max_pbf_size_mb: float = 512.0
    max_extent_km: float = 500.0
    bbox: tuple[float, float, float, float] | None = None
    country: str = "UnknownCountry"
    city: str = "UnknownCity"
    cache_dir: str | None = None
    progress: str = "auto"
    log_level: str = "INFO"
    preview_count: int = 9
    request_timeout: float = 180.0
    http_retries: int = 2
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    user_agent: str = "MAPF-World-Generator/0.3 (research map generation)"

    @property
    def grid_size(self):
        return (
            round(self.tile_size_m / self.resolution_m)
            if self.resolution_m
            else self.grid_resolution
        )

    def validate(self):
        if self.backend not in ("pbf", "overpass"):
            raise ValueError("backend must be pbf or overpass")
        for name in (
            "tile_size_m",
            "tile_stride_m",
            "default_width_m",
            "width_scale",
            "max_api_area_km2",
            "max_pbf_size_mb",
            "max_extent_km",
            "request_timeout",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.resolution_m is not None and (
            not math.isfinite(self.resolution_m) or self.resolution_m <= 0
        ):
            raise ValueError("resolution_m must be finite and > 0")
        if not 8 <= self.grid_size <= 4096 or self.grid_size % 8:
            raise ValueError("effective grid size must be a multiple of 8 between 8 and 4096")
        if not 0 <= self.min_free_ratio <= self.max_free_ratio <= 1:
            raise ValueError("require 0 <= min_free_ratio <= max_free_ratio <= 1")
        for name in ("min_largest_component", "min_spatial_coverage"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not math.isfinite(self.min_path_length_m) or self.min_path_length_m < 0:
            raise ValueError("min_path_length_m must be finite and >= 0")
        if self.max_candidates < 1 or self.max_tiles < 0 or self.preview_count < 0:
            raise ValueError("max_candidates must be > 0; max_tiles and preview_count must be >= 0")
        if (
            not 0 <= self.morphology_radius <= self.grid_size // 8
            or not 0 <= self.http_retries <= 5
        ):
            raise ValueError("morphology_radius must be 0..grid_size/8; http_retries must be 0..5")
        if self.network_type not in NETWORKS:
            raise ValueError(f"network_type must be one of {NETWORKS}")
        if self.progress not in ("auto", "always", "never") or self.log_level not in (
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ):
            raise ValueError("invalid progress mode or log level")
        for name in ("country", "city"):
            if not re.fullmatch(r"[\w][\w .-]*", getattr(self, name), re.UNICODE):
                raise ValueError(
                    f"{name} must contain only letters, digits, spaces, '.', '_' or '-'"
                )
        if not self.source.strip() or not self.output_dir.strip():
            raise ValueError("source and output_dir cannot be empty")
        if self.bbox is not None:
            if len(self.bbox) != 4 or not all(math.isfinite(x) for x in self.bbox):
                raise ValueError("bbox requires WEST SOUTH EAST NORTH")
            w, s, e, n = self.bbox
            if not (-180 <= w < e <= 180 and -80 <= s < n <= 84):
                raise ValueError(
                    "bbox requires west < east, south < north, lon in [-180,180], lat in [-80,84]"
                )
        if self.backend == "pbf":
            p = Path(self.source)
            if not p.is_file() or not p.name.lower().endswith(".osm.pbf"):
                raise ValueError(f"source must be an existing .osm.pbf file: {p}")
            if not p.stat().st_size:
                raise ValueError("PBF file is empty")
            if p.stat().st_size > self.max_pbf_size_mb * 1024**2:
                raise ValueError(
                    "PBF exceeds max_pbf_size_mb; pre-extract a city with osmium extract, or explicitly raise the limit on a suitable machine. --bbox does not bound decoder RAM."
                )
        if not self.overpass_url.startswith("https://"):
            raise ValueError("overpass_url must use HTTPS")


@contextmanager
def stage(name, timings):
    start = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            LOG.info("%s: still working (%.0fs elapsed)", name, time.monotonic() - start)

    LOG.info("%s: started", name)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        timings[name] = round(time.monotonic() - start, 3)
        LOG.info("%s: finished in %.2fs", name, timings[name])


def progress(items, total, name, cfg):
    from tqdm import tqdm

    enabled = cfg.progress == "always" or (cfg.progress == "auto" and sys.stderr.isatty())
    last = time.monotonic()
    for i, item in enumerate(
        tqdm(items, total=total, desc=name, unit="item", disable=not enabled), 1
    ):
        yield item
        now = time.monotonic()
        if not enabled and (i == 1 or i == total or now - last >= 10):
            LOG.info("%s: %d/%d (%.1f%%)", name, i, total, 100 * i / max(total, 1))
            last = now


def _first(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) else None
    return value


def infer_width_m(row, default_width_m):
    raw = str(_first(row.get("width"))).lower().strip().split(";")[0]
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(m|meters?|metres?|ft|feet|')?", raw)
    if match:
        width = float(match[1]) * (0.3048 if match[2] in ("ft", "feet", "'") else 1)
        if 0.5 <= width <= 60:
            return width
    highway = str(_first(row.get("highway")))
    base = DEFAULT_WIDTHS_M.get(highway.removesuffix("_link"), default_width_m)
    try:
        lanes = float(str(_first(row.get("lanes"))).split(";")[0])
        if math.isfinite(lanes) and 0 < lanes <= 16:
            return max(base, lanes * 3.2)
    except (TypeError, ValueError):
        pass
    return base


def _cache(cfg):
    p = Path(cfg.cache_dir) if cfg.cache_dir else Path(cfg.output_dir) / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _request_json(url, payload, cfg, *, post=False):
    """Cache only validated JSON; retry transient failures a bounded number of times."""
    import requests

    key = hashlib.sha256(json.dumps([url, payload], sort_keys=True).encode()).hexdigest()
    cache = _cache(cfg) / f"{key}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if not isinstance(data, dict) or "remark" in data:
                raise ValueError("invalid cached response")
            LOG.info("API cache hit: %s", key[:12])
            return data
        except (ValueError, OSError):
            LOG.warning("Ignoring invalid cache entry %s", key[:12])
    for attempt in range(cfg.http_retries + 1):
        try:
            LOG.info(
                "HTTP %s attempt %d/%d",
                "POST" if post else "GET",
                attempt + 1,
                cfg.http_retries + 1,
            )
            response = requests.request(
                "POST" if post else "GET",
                url,
                **({"data": payload} if post else {"params": payload}),
                headers={"User-Agent": cfg.user_agent},
                timeout=(15, cfg.request_timeout),
            )
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
            else:
                raise requests.ConnectionError(f"HTTP {response.status_code}")
            data = response.json()
            if not isinstance(data, dict) or "remark" in data:
                raise ValueError(
                    f"Incomplete API response: {str(data.get('remark', 'invalid JSON object'))[:300]}"
                    if isinstance(data, dict)
                    else "Invalid API JSON"
                )
            with tempfile.NamedTemporaryFile(
                mode="w", dir=cache.parent, delete=False, suffix=".partial"
            ) as f:
                json.dump(data, f)
                temp = Path(f.name)
            temp.replace(cache)
            return data
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == cfg.http_retries:
                raise RuntimeError(f"API unavailable after {attempt + 1} attempts: {exc}") from exc
            delay = min(2**attempt * 5, 60)
            LOG.warning("Transient API failure; retrying in %ss: %s", delay, exc)
            time.sleep(delay)
    raise RuntimeError("API request did not complete")


def load_edges(cfg):
    import geopandas as gpd
    from shapely.geometry import LineString, box, shape

    if cfg.backend == "pbf":
        from pyrosm import OSM

        LOG.info(
            "PBF: %.1f MiB, network=%s, bbox=%s",
            Path(cfg.source).stat().st_size / 1024**2,
            cfg.network_type,
            cfg.bbox,
        )
        edges = OSM(cfg.source, bounding_box=list(cfg.bbox) if cfg.bbox else None).get_network(
            network_type=cfg.network_type
        )
        if edges is None or edges.empty:
            raise ValueError("No roads found in PBF for this network/bbox")
        if cfg.bbox:
            edges = edges.copy()
            edges.geometry = edges.geometry.intersection(box(*cfg.bbox))
        return edges.reset_index(drop=True)
    if cfg.bbox:
        polygon = box(*cfg.bbox)
        display_name = "explicit bbox (source is a label)"
    else:
        result = _request_json(
            "https://nominatim.openstreetmap.org/search",
            {"q": cfg.source, "format": "geojson", "polygon_geojson": 1, "limit": 1},
            cfg,
        )
        features = result.get("features", [])
        if not features:
            raise ValueError("Place was not found; use an unambiguous place name or --bbox")
        polygon = shape(features[0]["geometry"])
        display_name = features[0].get("properties", {}).get("display_name", cfg.source)
        if polygon.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("Geocoder returned a point, not an area; supply --bbox")
    w, s, e, n = polygon.bounds
    if not (-80 <= s < n <= 84) or e - w > 6:
        raise ValueError("API region outside supported local projection; choose a smaller bbox")
    # The actual request uses the bounding rectangle, so guard that rectangle.
    from pyproj import Geod

    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(box(w, s, e, n))[0]) / 1e6
    if area > cfg.max_api_area_km2:
        raise ValueError(
            f"API query bbox is {area:.1f} km^2, above {cfg.max_api_area_km2:g}; use PBF or a smaller --bbox"
        )
    LOG.info("Resolved %s; query bbox=%s, area=%.1f km^2", display_name, polygon.bounds, area)
    # Rasterization uses road geometries directly.
    filters = '["highway"]["area"!="yes"]["highway"!~"^(abandoned|construction|proposed|platform|raceway)$"]'
    if cfg.network_type != "all":
        filters += '["access"!~"^(private|no)$"]'
    if cfg.network_type == "walk":
        filters += '["foot"!="no"]["highway"!~"^(motorway|motorway_link|trunk|trunk_link)$"]'
    elif cfg.network_type == "bike":
        filters += '["bicycle"!="no"]["highway"!~"^(motorway|motorway_link|steps)$"]'
    elif cfg.network_type in ("drive", "drive_service"):
        filters += (
            '["motor_vehicle"!="no"]["motorcar"!="no"]["highway"!~"^(footway|pedestrian|path|cycleway|steps|track|bridleway'
            + ("|service" if cfg.network_type == "drive" else "")
            + ')$"]'
        )
    query = f"[out:json][timeout:{max(1, int(cfg.request_timeout) - 10)}];way{filters}({s},{w},{n},{e});out geom;"
    result = _request_json(cfg.overpass_url, {"data": query}, cfg, post=True)
    records = []
    for way in result.get("elements", []):
        coords = [
            (p["lon"], p["lat"]) for p in way.get("geometry", []) if "lon" in p and "lat" in p
        ]
        if way.get("type") == "way" and len(coords) >= 2:
            geom = LineString(coords).intersection(polygon)
            if not geom.is_empty:
                tags = way.get("tags", {})
                records.append(
                    {**{k: tags.get(k) for k in ("highway", "width", "lanes")}, "geometry": geom}
                )
    if not records:
        raise ValueError("API returned no roads in the selected area")
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    frame.attrs["source_info"] = {
        "resolved_name": display_name,
        "query_bbox": list(polygon.bounds),
        "query_area_km2": area,
        "boundary_sha256": hashlib.sha256(polygon.wkb).hexdigest(),
        "response_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
        "osm_base_timestamp": result.get("osm3s", {}).get("timestamp_osm_base"),
    }
    return frame


def prepare_edges(edges, cfg):
    import geopandas as gpd
    import shapely

    edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty].copy()
    if edges.crs is None:
        raise ValueError("Input roads have no CRS")
    if edges.empty:
        raise ValueError("No geometries remain after clipping")
    bounds = edges.to_crs(4326).total_bounds
    if bounds[0] > bounds[2] or bounds[2] - bounds[0] > 6 or bounds[1] < -80 or bounds[3] > 84:
        raise ValueError(
            "Input spans too many longitudes or lies outside UTM; select a local --bbox"
        )
    crs = edges.estimate_utm_crs()
    edges = edges.to_crs(crs).explode(index_parts=False, ignore_index=True)
    edges = edges[edges.geom_type.isin(["LineString", "MultiLineString"])].copy()
    edges = edges[edges.is_valid & (edges.length > 0)].copy()
    if edges.empty:
        raise ValueError("No valid road lines remain")
    minx, miny, maxx, maxy = edges.total_bounds
    if max(maxx - minx, maxy - miny) > cfg.max_extent_km * 1000:
        raise ValueError("Projected extent exceeds max_extent_km; extract or crop a city first")
    tag_names = ("width", "lanes", "highway")
    tag_rows = edges.reindex(columns=tag_names).itertuples(index=False, name=None)
    widths = np.fromiter(
        (infer_width_m(dict(zip(tag_names, row)), cfg.default_width_m) for row in tag_rows),
        dtype=float,
        count=len(edges),
    )
    lines = edges.geometry.to_numpy()
    chunks = []
    for i in progress(
        range(0, len(lines), 10000), math.ceil(len(lines) / 10000), "Buffer roads", cfg
    ):
        chunks.extend(
            shapely.buffer(
                lines[i : i + 10000],
                widths[i : i + 10000] * cfg.width_scale / 2,
                cap_style=2,
                join_style=2,
            )
        )
    prepared = gpd.GeoDataFrame(
        {"width_m": widths, "centerline": lines, "geometry": chunks}, crs=crs
    )
    LOG.info("Prepared %d road features in %s", len(prepared), crs)
    return prepared


def _clean_grid(free, radius):
    if radius == 0:
        return free
    from scipy.ndimage import binary_closing, binary_opening

    structure = np.ones((2 * radius + 1,) * 2, dtype=bool)
    return binary_opening(binary_closing(free, structure=structure), structure=structure)


def _quality_metrics(free):
    from scipy import ndimage
    from skimage.morphology import skeletonize

    labels, components = ndimage.label(free)  # Four-neighbor MAPF connectivity.
    sizes = np.bincount(labels.ravel())[1:]
    free_cells = int(free.sum())
    largest = float(sizes.max() / free_cells) if free_cells else 0.0
    skeleton = skeletonize(free)
    neighbors = (
        ndimage.convolve(
            skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
        )
        - skeleton
    )
    branch = skeleton & (neighbors >= 3)
    # Count connected junction regions, not every branch pixel as a junction.
    junctions = int(ndimage.label(branch, structure=np.ones((3, 3)))[1])
    endpoints = int(np.sum(skeleton & (neighbors == 1)))
    block = free.reshape(8, free.shape[0] // 8, 8, free.shape[1] // 8).mean(axis=(1, 3))
    coverage = float(np.mean(block > 0.02))
    ratio = float(free.mean())
    balance = max(0.0, 1 - abs(ratio - 0.45) / 0.25)
    score = (
        4 * largest
        + 3 * coverage
        + 2 * balance
        + 0.8 * math.log1p(junctions)
        - 0.05 * max(0, components - 1)
    )
    return dict(
        free_ratio=ratio,
        components=int(components),
        largest_component=largest,
        junctions=junctions,
        skeleton_branch_pixels=int(branch.sum()),
        endpoints=endpoints,
        spatial_coverage=coverage,
        quality_score=float(score),
    )


def iter_tiles(edges, cfg, stats=None) -> Iterable:
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from scipy import ndimage
    from shapely.geometry import box

    stats = stats if stats is not None else Counter()
    minx, miny, maxx, maxy = edges.total_bounds
    # Include partial tiles along the boundary.
    cols = max(1, math.ceil((maxx - minx - cfg.tile_size_m) / cfg.tile_stride_m) + 1)
    rows = max(1, math.ceil((maxy - miny - cfg.tile_size_m) / cfg.tile_stride_m) + 1)
    stats["total_windows"] = rows * cols
    if rows * cols > cfg.max_candidates:
        raise ValueError(
            f"{rows * cols} windows exceeds --max-candidates {cfg.max_candidates}; crop with --bbox or increase --tile-stride-m"
        )
    selected_cap = min(rows * cols, cfg.max_tiles) if cfg.max_tiles else rows * cols
    estimated_bytes = rows * cols * (cfg.grid_size**2 / 8 + 2048) + selected_cap * (
        cfg.grid_size**2 * 1.1 + 2048
    )
    space_path = Path(cfg.output_dir).resolve()
    while not space_path.exists():
        space_path = space_path.parent
    if estimated_bytes + 64 * 1024**2 > shutil.disk_usage(space_path).free:
        raise ValueError(
            "Insufficient free output space for the worst-case candidate spool and outputs; reduce candidates/resolution/max-tiles or use a larger volume"
        )
    sindex = edges.sindex
    for idx in progress(range(rows * cols), rows * cols, "Rasterize / score", cfg):
        iy, ix = divmod(idx, cols)
        x0, y0 = minx + ix * cfg.tile_stride_m, miny + iy * cfg.tile_stride_m
        bounds = (x0, y0, x0 + cfg.tile_size_m, y0 + cfg.tile_size_m)
        tile = box(*bounds)
        indices = sindex.query(tile, predicate="intersects")
        candidates = edges.iloc[indices]
        stats["processed_windows"] += 1
        if candidates.empty:
            stats["rejected_no_roads"] += 1
            continue
        length = float(
            shapely.length(shapely.intersection(candidates.centerline.to_numpy(), tile)).sum()
        )
        if length < cfg.min_path_length_m:
            stats["rejected_path_length"] += 1
            continue
        # GDAL clips polygons to the raster extent; avoid expensive polygon
        # intersections for every overlapping tile.
        free = rasterize(
            ((geom, 1) for geom in candidates.geometry),
            out_shape=(cfg.grid_size,) * 2,
            transform=from_bounds(*bounds, cfg.grid_size, cfg.grid_size),
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        free = _clean_grid(free, cfg.morphology_radius)
        if cfg.keep_largest_component and free.any():
            labels, _ = ndimage.label(free)
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            free = labels == sizes.argmax()
        if not free.any() or not cfg.min_free_ratio <= free.mean() <= cfg.max_free_ratio:
            stats["rejected_free_ratio"] += 1
            continue
        metrics = _quality_metrics(free)
        if metrics["largest_component"] < cfg.min_largest_component:
            stats["rejected_connectivity"] += 1
            continue
        if metrics["spatial_coverage"] < cfg.min_spatial_coverage:
            stats["rejected_coverage"] += 1
            continue
        stats["accepted_windows"] += 1
        key = f"{cfg.country}-{cfg.city}-tile_{ix:03d}_{iy:03d}"
        yield key, free, dict(x=ix, y=iy, bounds=bounds, path_length_m=length, **metrics)


def _select_informative_tiles(candidates, cfg):
    ranked = sorted(candidates, key=lambda item: (-item[2]["quality_score"], item[0]))
    if cfg.max_tiles == 0 or len(ranked) <= cfg.max_tiles:
        return ranked
    selected, centers = [], []
    for item in ranked:
        b = item[2]["bounds"]
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if all(math.dist(c, other) >= cfg.tile_size_m * 0.60 for other in centers):
            selected.append(item)
            centers.append(c)
            if len(selected) == cfg.max_tiles:
                break
    names = {x[0] for x in selected}
    for item in ranked:
        if len(selected) >= cfg.max_tiles:
            break
        if item[0] not in names:
            selected.append(item)
    return sorted(selected, key=lambda item: (-item[2]["quality_score"], item[0]))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_grid(spool, offset, size):
    spool.seek(offset)
    return (
        np.unpackbits(np.frombuffer(spool.read(size * size // 8), dtype=np.uint8))
        .reshape(size, size)
        .astype(bool)
    )


def _write_preview(previews, path):
    from PIL import Image, ImageDraw

    cols = min(3, len(previews))
    rows = math.ceil(len(previews) / cols)
    image = Image.new("RGB", (cols * 320, rows * 352), "white")
    draw = ImageDraw.Draw(image)
    for i, (key, grid, meta) in enumerate(previews):
        x, y = i % cols * 320, i // cols * 352
        tile = Image.fromarray(grid.astype(np.uint8) * 255).resize(
            (304, 304), Image.Resampling.NEAREST
        )
        image.paste(tile, (x + 8, y + 40))
        # Coordinates avoid unsupported font glyphs for non-Latin city names.
        draw.text(
            (x + 8, y + 4),
            f"tile {meta['x']},{meta['y']} | free {meta['free_ratio']:.2f}",
            fill="black",
        )
        draw.text(
            (x + 8, y + 20),
            f"LCC {meta['largest_component']:.3f} | junctions {meta['junctions']}",
            fill="black",
        )
    image.save(path)


class NoTilesError(RuntimeError):
    pass


def write_outputs(edges, cfg, timings=None, source_info=None):
    timings = timings if timings is not None else {}
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    candidates = []
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=output) as tmp:
        tmp = Path(tmp)
        with (tmp / "grids.bin").open("w+b") as spool:
            with stage("Rasterize and score", timings):
                for name, grid, meta in iter_tiles(edges, cfg, stats):
                    offset = spool.tell()
                    spool.write(np.packbits(grid).tobytes())
                    candidates.append((name, offset, meta))
            LOG.info("Window statistics: %s", dict(stats))
            if not candidates:
                raise NoTilesError(
                    f"No maps passed filters: {dict(stats)}. Check bbox/network/viewport and thresholds."
                )
            selected = _select_informative_tiles(candidates, cfg)
            LOG.info(
                "Selected %d/%d accepted windows (spooled grids: %.2f MiB)",
                len(selected),
                len(candidates),
                spool.tell() / 1024**2,
            )
            metadata = dict(
                schema_version=2,
                generator_version=__version__,
                config=asdict(cfg),
                effective_grid_size=cfg.grid_size,
                effective_resolution_m=cfg.tile_size_m / cfg.grid_size,
                crs=str(edges.crs),
                source=source_info or {},
                statistics=dict(stats),
                candidate_count=len(candidates),
                selected_count=len(selected),
                metric_definition="skeleton-junction-regions-v2; LCC uses 4-neighbor grid connectivity",
                attribution="© OpenStreetMap contributors; https://www.openstreetmap.org/copyright",
                tiles=[],
            )
            from ruamel.yaml import YAML
            from ruamel.yaml.scalarstring import LiteralScalarString

            yaml = YAML()
            yaml.default_flow_style = False
            previews = []
            with (
                stage("Write outputs", timings),
                (tmp / "maps.yaml").open("w", encoding="utf-8") as stream,
            ):
                for key, offset, meta in progress(selected, len(selected), "Write maps", cfg):
                    grid = _read_grid(spool, offset, cfg.grid_size)
                    lines = ["".join(row) for row in np.where(grid, ".", "#")]
                    yaml.dump({key: LiteralScalarString("\n".join(lines))}, stream)
                    metadata["tiles"].append({"name": key, **meta})
                    if len(previews) < cfg.preview_count:
                        previews.append((key, grid, meta))
                if previews:
                    _write_preview(previews, tmp / "preview.png")
        metadata["timings_seconds"] = dict(timings)
        (tmp / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        names = ["maps.yaml", "metadata.json"] + (["preview.png"] if previews else [])
        manifest = dict(
            status="complete",
            generator_version=__version__,
            selected_count=len(selected),
            files={name: sha256_file(tmp / name) for name in names},
        )
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Publish the manifest last to mark a completed run.
        for name in names + ["manifest.json"]:
            (tmp / name).replace(output / name)
    return len(selected)


def run(cfg):
    cfg.validate()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".generator.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"Output is locked: {lock}. Use a new directory; inspect any previous process before removing a stale lock."
        ) from exc
    os.close(fd)
    handlers = []
    try:
        for name in (
            "maps.yaml",
            "metadata.json",
            "manifest.json",
            "run.json",
            "generator.log",
            "preview.png",
        ):
            if (output / name).exists():
                raise ValueError(
                    f"Output already contains {name}; choose a new --output-dir. A shared --cache-dir can reuse downloads."
                )
        LOG.setLevel(cfg.log_level)
        LOG.propagate = False
        handlers = [
            ProgressLogHandler(sys.stderr),
            logging.FileHandler(output / "generator.log", encoding="utf-8"),
        ]
        for handler in handlers:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
            )
            LOG.addHandler(handler)
        start = time.monotonic()
        timings = {}
        status = "failed"
        error = None
        count = 0
        try:
            LOG.info(
                "MAPF-World generator %s | %s | %dx%d | %.2f m/cell",
                __version__,
                cfg.backend,
                cfg.grid_size,
                cfg.grid_size,
                cfg.tile_size_m / cfg.grid_size,
            )
            LOG.info("Config: %s", json.dumps(asdict(cfg), ensure_ascii=False))
            if cfg.resolution_m is not None:
                LOG.warning(
                    "--resolution-m is deprecated and overrides --grid-resolution; effective size=%d",
                    cfg.grid_size,
                )
            source = {"backend": cfg.backend, "input": cfg.source, "bbox": cfg.bbox}
            if cfg.backend == "pbf":
                with stage("Fingerprint input", timings):
                    source.update(
                        sha256=sha256_file(cfg.source), size_bytes=Path(cfg.source).stat().st_size
                    )
            with stage("Load roads", timings):
                raw = load_edges(cfg)
                source.update(raw.attrs.get("source_info", {}))
            with stage("Project and buffer", timings):
                edges = prepare_edges(raw, cfg)
            del raw
            count = write_outputs(edges, cfg, timings, source)
            status = "complete"
            LOG.info("SUCCESS: %d maps in %s (%.2fs)", count, output, time.monotonic() - start)
            return count
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            LOG.error("%s: %s", status.upper(), error, exc_info=cfg.log_level == "DEBUG")
            raise
        finally:
            versions = {}
            for package in (
                "numpy",
                "pyrosm",
                "geopandas",
                "shapely",
                "rasterio",
                "scipy",
                "scikit-image",
                "requests",
                "ruamel.yaml",
                "Pillow",
                "tqdm",
            ):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            report = dict(
                status=status,
                error=error,
                selected_count=count,
                config=asdict(cfg),
                elapsed_seconds=round(time.monotonic() - start, 3),
                stages_seconds=timings,
                python=sys.version.split()[0],
                dependencies=versions,
                generator_version=__version__,
            )
            (output / "run.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    finally:
        for handler in handlers:
            LOG.removeHandler(handler)
            handler.close()
        lock.unlink(missing_ok=True)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument(
        "--backend",
        choices=("pbf", "overpass"),
        required=True,
        help="Local PBF decoder or read-only Overpass API",
    )
    p.add_argument(
        "--source",
        required=True,
        help="PBF path; otherwise geocodable place name (label when --bbox is supplied)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="New output directory; existing run artifacts are never overwritten",
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="WGS84 degrees; crop PBF roads or query this API rectangle",
    )
    p.add_argument(
        "--network-type",
        choices=NETWORKS,
        default="all",
        help="Road-access filter; all includes private roads, not just walking roads",
    )
    options = [
        ("tile-size-m", float, 3072.0, "Physical width and height of each viewport, metres"),
        (
            "tile-stride-m",
            float,
            1536.0,
            "Spacing of viewport origins, metres; smaller means more overlapping candidates",
        ),
        ("grid-resolution", int, 256, "Square output size in cells; multiple of 8, from 8 to 4096"),
        (
            "resolution-m",
            float,
            None,
            "Deprecated metres/cell override; output size is round(tile-size-m / resolution-m)",
        ),
        ("min-free-ratio", float, 0.25, "Minimum traversable-cell fraction after optional cleanup"),
        ("max-free-ratio", float, 0.70, "Maximum traversable-cell fraction"),
        (
            "min-largest-component",
            float,
            0.90,
            "Minimum fraction of free cells in the largest 4-neighbor component",
        ),
        (
            "min-spatial-coverage",
            float,
            0.75,
            "Minimum fraction of 8x8 viewport blocks having >2%% free cells",
        ),
        (
            "max-tiles",
            int,
            128,
            "Output cap after quality ranking and diversity selection; 0 keeps all accepted tiles",
        ),
        (
            "max-candidates",
            int,
            100000,
            "Hard cap on all candidate windows, checked before rasterization",
        ),
        (
            "min-path-length-m",
            float,
            20.0,
            "Minimum summed clipped source-road centerline length per tile, metres",
        ),
        (
            "default-width-m",
            float,
            3.0,
            "Fallback full road width when tags and road-class defaults are absent",
        ),
        (
            "width-scale",
            float,
            1.0,
            "Multiply inferred full road widths before buffering; changes connectivity",
        ),
        (
            "morphology-radius",
            int,
            0,
            "Closing then opening radius in cells; 0 disables topology-changing cleanup",
        ),
        (
            "max-api-area-km2",
            float,
            250.0,
            "Maximum API query rectangle area; use PBF for larger areas",
        ),
        (
            "max-pbf-size-mb",
            float,
            512.0,
            "Maximum compressed input MiB; size guard, NOT a RAM limit",
        ),
        (
            "max-extent-km",
            float,
            500.0,
            "Maximum projected width/height; use a local bbox for regional files",
        ),
        (
            "preview-count",
            int,
            9,
            "Number of highest-ranked selected maps in preview.png; 0 disables preview",
        ),
        (
            "request-timeout",
            float,
            180.0,
            "HTTP read timeout in seconds; connection timeout is 15s",
        ),
        ("http-retries", int, 2, "Retries for connection, timeout, HTTP 429/5xx failures; 0..5"),
    ]
    for name, kind, default, help_text in options:
        p.add_argument("--" + name, type=kind, default=default, help=help_text)
    p.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Remove disconnected free islands before quality scoring",
    )
    p.add_argument(
        "--country",
        default="UnknownCountry",
        help="Output-name label only; does not geocode or crop",
    )
    p.add_argument(
        "--city", default="UnknownCity", help="Output-name label only; does not geocode or crop"
    )
    p.add_argument("--cache-dir", help="Persistent API cache directory; default: OUTPUT/.cache")
    p.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Terminal progress bars; non-terminal mode logs periodic counts",
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="stderr and generator.log verbosity; DEBUG includes tracebacks",
    )
    p.add_argument(
        "--overpass-url",
        default=GeneratorConfig.overpass_url,
        help="HTTPS Overpass interpreter endpoint",
    )
    p.add_argument(
        "--user-agent",
        default=GeneratorConfig.user_agent,
        help="HTTP identity; set project/contact for repeated public API use",
    )
    return p


def parse_args(argv=None):
    p = build_parser()
    cfg = GeneratorConfig(**vars(p.parse_args(argv)))
    try:
        cfg.validate()
    except ValueError as exc:
        p.error(str(exc))
    return cfg


def verify_outputs(directory):
    """Verify hashes, the occupancy-grid contract and reported grid metrics."""
    from ruamel.yaml import YAML

    output = Path(directory)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Output has no complete manifest")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or name not in ("maps.yaml", "metadata.json", "preview.png"):
            raise ValueError("Unexpected manifest filename")
        if sha256_file(output / name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    if not {"maps.yaml", "metadata.json"} <= set(manifest["files"]):
        raise ValueError("Manifest omits required outputs")
    metadata = json.loads((output / "metadata.json").read_text())
    maps = YAML(typ="safe").load((output / "maps.yaml").read_text())
    size = metadata["effective_grid_size"]
    if not maps or len(maps) != manifest["selected_count"] or len(maps) != len(metadata["tiles"]):
        raise ValueError("Map count does not match manifest/metadata")
    if set(maps) != {t["name"] for t in metadata["tiles"]}:
        raise ValueError("Map names do not match metadata")
    for tile in metadata["tiles"]:
        lines = maps[tile["name"]].splitlines()
        if len(lines) != size or any(len(row) != size or set(row) - {".", "#"} for row in lines):
            raise ValueError(f"Invalid grid shape/alphabet: {tile['name']}")
        grid = np.array([[x == "." for x in row] for row in lines])
        for key, value in _quality_metrics(grid).items():
            if not math.isclose(value, tile[key], rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Metric mismatch in {tile['name']}: {key}")
    if "preview.png" in manifest["files"]:
        from PIL import Image

        with Image.open(output / "preview.png") as image:
            image.verify()
    return len(maps)


def validate_main(argv=None):
    p = argparse.ArgumentParser(
        description="Verify generated maps, hashes and connectivity metrics"
    )
    p.add_argument("output_dir", help="Completed generator output directory")
    args = p.parse_args(argv)
    try:
        count = verify_outputs(args.output_dir)
        print(f"VALID: {count} maps; hashes, shape, alphabet and quality metrics match")
        return 0
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "validate":
        return validate_main(argv[1:])
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; see run.json and generator.log.", file=sys.stderr)
        return 130
    except NoTilesError as exc:
        print(f"No maps: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"Configuration/data error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"Generation failed: {exc}. See generator.log; use --log-level DEBUG for traceback.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
