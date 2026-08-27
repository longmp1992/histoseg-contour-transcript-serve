from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def bootstrap_runtime_env() -> None:
    """Point caches to writable paths before importing matplotlib or Gradio."""
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/gradio")

    for key in ("HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "GRADIO_TEMP_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


bootstrap_runtime_env()

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional runtime helper
    pq = None

try:
    from shapely import contains_xy
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    SHAPELY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - startup fallback only
    contains_xy = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]
    SHAPELY_IMPORT_ERROR = str(exc)


APP_NAME = "HistoSeg Contour Transcript Explorer"
APP_DESCRIPTION = (
    "A standalone SciLifeLab Serve Gradio app that reads an already generated HistoSeg contour bundle "
    "plus a Xenium transcript.parquet file, computes signed contour-distance curves "
    "with negative values inside selected contours and Voronoi-style positive values outside, "
    "and ranks the most spatially variant genes."
)
INPUT_MODE_UPLOAD = "upload_files"
INPUT_MODE_STORAGE = "mounted_storage"
CONTOUR_PATTERN = re.compile(r"(?:^|/|\\)structure_(\d+)_contour_(\d+)\.npy$", re.IGNORECASE)
PARTITION_MEMBER_CANDIDATES = (
    "cells_with_structure_partition.parquet",
    "cells_with_structure_partition.csv",
)
CELL_COORDINATE_MEMBER_CANDIDATES = (
    "cells_with_structure_partition.parquet",
    "cells_with_structure_partition.csv",
    "cells.parquet",
    "cells.csv",
)
PARTITION_STRUCTURE_ID_CANDIDATES = (
    "isoline_structure_id",
    "structure_id",
    "selected_structure_id",
)
PARTITION_CONTOUR_ID_CANDIDATES = (
    "isoline_contour_id",
    "contour_id",
    "polygon_id",
    "contour_index",
    "isoline_polygon_id",
)
CELL_X_COORD_CANDIDATES = (
    "x_centroid",
    "x_location",
    "x",
    "X",
)
CELL_Y_COORD_CANDIDATES = (
    "y_centroid",
    "y_location",
    "y",
    "Y",
)
DEFAULT_BATCH_SIZE = 250000
DEFAULT_BACKGROUND_SAMPLE = 60000
DEFAULT_TOP_GENE_SAMPLE = 50000
FALLBACK_WORK_DIR = Path("/tmp/project-vol")
WORK_DIR_CANDIDATE = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()


@dataclass(frozen=True)
class ResolvedInputFile:
    path: Path
    source: str
    original: str


@dataclass(frozen=True)
class TranscriptInputSchema:
    gene_col: str
    x_col: str
    y_col: str
    qv_col: str | None
    is_gene_col: str | None


def resolve_work_dir() -> Path:
    for candidate in (WORK_DIR_CANDIDATE, FALLBACK_WORK_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"Could not find a writable work directory. Tried: {WORK_DIR_CANDIDATE} and {FALLBACK_WORK_DIR}"
    )


WORK_DIR = resolve_work_dir()
RUNS_DIR = WORK_DIR / "runs"
PREVIEWS_DIR = WORK_DIR / "previews"


def configured_input_roots() -> tuple[Path, ...]:
    raw_env = os.environ.get("APP_ALLOWED_INPUT_ROOTS", "")
    candidates = [item.strip() for item in raw_env.split(os.pathsep) if item.strip()]
    if not candidates:
        candidates = ["/home/data", "/srv/shiny-server/data"]

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return tuple(roots)


ALLOWED_INPUT_ROOTS = configured_input_roots()
PRIMARY_INPUT_ROOT = ALLOWED_INPUT_ROOTS[0] if ALLOWED_INPUT_ROOTS else Path("/home/data")
INPUT_ROOTS_LABEL = ", ".join(str(path) for path in ALLOWED_INPUT_ROOTS)


def ensure_workdirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def build_run_dir() -> Path:
    ensure_workdirs()
    token = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"run-{token}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = RUNS_DIR / f"run-{token}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_preview_dir() -> Path:
    ensure_workdirs()
    token = datetime.now().strftime("%Y%m%d-%H%M%S")
    preview_dir = PREVIEWS_DIR / f"preview-{token}"
    suffix = 1
    while preview_dir.exists():
        suffix += 1
        preview_dir = PREVIEWS_DIR / f"preview-{token}-{suffix}"
    preview_dir.mkdir(parents=True, exist_ok=False)
    return preview_dir


def cleanup_old_directories(base_dir: Path, *, max_keep: int) -> list[str]:
    if not base_dir.exists():
        return []
    directories = [item for item in base_dir.iterdir() if item.is_dir()]
    directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    removed: list[str] = []
    for stale in directories[max_keep:]:
        try:
            shutil.rmtree(stale)
            removed.append(stale.name)
        except OSError:
            continue
    return removed


def cleanup_old_runs(max_keep: int = 3) -> list[str]:
    return cleanup_old_directories(RUNS_DIR, max_keep=max_keep)


def cleanup_old_previews(max_keep: int = 4) -> list[str]:
    return cleanup_old_directories(PREVIEWS_DIR, max_keep=max_keep)


def safe_count_parquet_rows(parquet_path: Path) -> int | None:
    if pq is None:
        return None
    try:
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return None


def path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_storage_path(
    storage_path: str | None,
    *,
    label: str,
    allowed_suffixes: tuple[str, ...] | None = None,
    allow_directory: bool = False,
) -> Path | None:
    raw = str(storage_path or "").strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PRIMARY_INPUT_ROOT / candidate

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"Could not resolve the mounted-storage path for {label}: {exc}") from exc

    resolved_roots = tuple(root.expanduser().resolve(strict=False) for root in ALLOWED_INPUT_ROOTS)
    if not any(path_is_within(resolved, root) for root in resolved_roots):
        raise ValueError(f"{label} must stay inside the configured mounted storage roots: {INPUT_ROOTS_LABEL}")
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found at mounted-storage path: {resolved}")
    if resolved.is_dir():
        if allow_directory:
            return resolved
        raise ValueError(f"{label} must point to a file, not a directory: {resolved}")
    if allowed_suffixes and resolved.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"{label} must use one of these suffixes: {', '.join(allowed_suffixes)}")
    return resolved


def stage_uploaded_file(uploaded: object | None, target_dir: Path) -> Path | None:
    if uploaded is None:
        return None
    source = Path(str(uploaded))
    if not source.exists():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    destination = target_dir / source.name
    shutil.copy2(source, destination)
    return destination


def resolve_contour_bundle_source(
    *,
    input_mode: str,
    contour_bundle_upload: object | None,
    contour_bundle_storage_path: str | None,
    target_dir: Path | None = None,
) -> ResolvedInputFile:
    if input_mode == INPUT_MODE_STORAGE:
        resolved = resolve_storage_path(
            contour_bundle_storage_path,
            label="HistoSeg contour bundle",
            allowed_suffixes=(".zip",),
            allow_directory=True,
        )
        if resolved is None:
            raise ValueError("Missing HistoSeg contour bundle path.")
        return ResolvedInputFile(path=resolved, source=INPUT_MODE_STORAGE, original=str(resolved))

    staged = stage_uploaded_file(contour_bundle_upload, target_dir or WORK_DIR)
    if staged is None:
        raise ValueError("Missing HistoSeg contour bundle upload.")
    return ResolvedInputFile(path=staged, source=INPUT_MODE_UPLOAD, original=str(contour_bundle_upload))


def resolve_transcript_source(
    *,
    input_mode: str,
    transcript_upload: object | None,
    transcript_storage_path: str | None,
    target_dir: Path | None = None,
) -> ResolvedInputFile:
    if input_mode == INPUT_MODE_STORAGE:
        resolved = resolve_storage_path(
            transcript_storage_path,
            label="transcript.parquet",
            allowed_suffixes=(".parquet",),
            allow_directory=False,
        )
        if resolved is None:
            raise ValueError("Missing transcript.parquet path.")
        return ResolvedInputFile(path=resolved, source=INPUT_MODE_STORAGE, original=str(resolved))

    staged = stage_uploaded_file(transcript_upload, target_dir or WORK_DIR)
    if staged is None:
        raise ValueError("Missing transcript.parquet upload.")
    return ResolvedInputFile(path=staged, source=INPUT_MODE_UPLOAD, original=str(transcript_upload))


def bundle_is_zip(bundle_path: Path) -> bool:
    return bundle_path.is_file() and bundle_path.suffix.lower() == ".zip"


def list_bundle_members(bundle_path: Path) -> list[str]:
    if bundle_is_zip(bundle_path):
        with zipfile.ZipFile(bundle_path) as handle:
            return handle.namelist()
    members: list[str] = []
    for item in bundle_path.rglob("*"):
        if item.is_file():
            members.append(str(item.relative_to(bundle_path)).replace("\\", "/"))
    return members


def read_bundle_bytes(bundle_path: Path, member_name: str) -> bytes:
    if bundle_is_zip(bundle_path):
        with zipfile.ZipFile(bundle_path) as handle:
            with handle.open(member_name) as fp:
                return fp.read()
    return (bundle_path / member_name).read_bytes()


def bundle_member_exists(bundle_path: Path, member_name: str) -> bool:
    if bundle_is_zip(bundle_path):
        with zipfile.ZipFile(bundle_path) as handle:
            return member_name in handle.namelist()
    return (bundle_path / member_name).exists()


def extract_bundle_member_to_path(bundle_path: Path, member_name: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(read_bundle_bytes(bundle_path, member_name))
    return target_path


def first_present_optional_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def first_existing_bundle_member(bundle_path: Path, candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if bundle_member_exists(bundle_path, candidate)), None)


def bundle_table_columns(bundle_path: Path, member_name: str) -> list[str]:
    if member_name.lower().endswith(".parquet"):
        if pq is None:
            raise ValueError("pyarrow.parquet is required to inspect cells_with_structure_partition.parquet.")
        source: object
        if bundle_is_zip(bundle_path):
            source = io.BytesIO(read_bundle_bytes(bundle_path, member_name))
        else:
            source = bundle_path / member_name
        return [str(name) for name in pq.ParquetFile(source).schema.names]

    csv_text = read_bundle_bytes(bundle_path, member_name).decode("utf-8", errors="replace")
    return [str(name) for name in pd.read_csv(io.StringIO(csv_text), nrows=0).columns]


def read_bundle_table_subset(bundle_path: Path, member_name: str, columns: list[str]) -> pd.DataFrame:
    unique_columns = list(dict.fromkeys(columns))
    if member_name.lower().endswith(".parquet"):
        if pq is None:
            raise ValueError("pyarrow.parquet is required to read cells_with_structure_partition.parquet.")
        source: object
        if bundle_is_zip(bundle_path):
            source = io.BytesIO(read_bundle_bytes(bundle_path, member_name))
        else:
            source = bundle_path / member_name
        return pq.read_table(source, columns=unique_columns).to_pandas()

    csv_text = read_bundle_bytes(bundle_path, member_name).decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(csv_text), usecols=lambda column_name: column_name in unique_columns)


def geometric_contour_counts_from_member(
    bundle_path: Path,
    member_name: str,
    contour_entries: list[dict[str, object]],
    *,
    structure_col: str | None,
) -> tuple[dict[tuple[int, int], int], dict[int, int], str] | None:
    columns = bundle_table_columns(bundle_path, member_name)
    x_col = first_present_optional_column(columns, CELL_X_COORD_CANDIDATES)
    y_col = first_present_optional_column(columns, CELL_Y_COORD_CANDIDATES)
    if x_col is None or y_col is None:
        return None

    requested_columns = [x_col, y_col]
    use_structure_col = structure_col if structure_col in columns else None
    if use_structure_col is not None:
        requested_columns.append(use_structure_col)
    coord_df = read_bundle_table_subset(bundle_path, member_name, requested_columns)
    coord_df = coord_df.dropna(subset=[x_col, y_col]).copy()
    if coord_df.empty:
        return None
    coord_df[x_col] = pd.to_numeric(coord_df[x_col], errors="coerce")
    coord_df[y_col] = pd.to_numeric(coord_df[y_col], errors="coerce")
    coord_df = coord_df.loc[coord_df[x_col].notna() & coord_df[y_col].notna()].copy()
    if coord_df.empty:
        return None

    points_by_structure: dict[int | None, tuple[np.ndarray, np.ndarray]] = {}
    if use_structure_col is not None:
        coord_df[use_structure_col] = pd.to_numeric(coord_df[use_structure_col], errors="coerce")
        coord_df = coord_df.loc[coord_df[use_structure_col].notna()].copy()
        if coord_df.empty:
            return None
        coord_df.loc[:, use_structure_col] = coord_df[use_structure_col].astype(int)
        for structure_id, group in coord_df.groupby(use_structure_col, observed=True):
            points_by_structure[int(structure_id)] = (
                group[x_col].to_numpy(dtype=float),
                group[y_col].to_numpy(dtype=float),
            )
    else:
        points_by_structure[None] = (
            coord_df[x_col].to_numpy(dtype=float),
            coord_df[y_col].to_numpy(dtype=float),
        )

    contour_counts: dict[tuple[int, int], int] = {}
    structure_counts: dict[int, int] = {}
    for entry in contour_entries:
        structure_id = int(entry["structure_id"])
        contour_index = int(entry["contour_index"])
        vertices = np.asarray(entry["vertices"], dtype=float)
        if use_structure_col is not None:
            x_values, y_values = points_by_structure.get(structure_id, (np.empty(0), np.empty(0)))
        else:
            x_values, y_values = points_by_structure[None]
        if len(x_values) == 0:
            contour_counts[(structure_id, contour_index)] = 0
            continue

        minx = float(np.min(vertices[:, 0]))
        maxx = float(np.max(vertices[:, 0]))
        miny = float(np.min(vertices[:, 1]))
        maxy = float(np.max(vertices[:, 1]))
        bbox_mask = (x_values >= minx) & (x_values <= maxx) & (y_values >= miny) & (y_values <= maxy)
        if not bbox_mask.any():
            contour_counts[(structure_id, contour_index)] = 0
            continue

        polygon = polygon_from_vertices(vertices)
        if polygon is None or polygon.is_empty:
            contour_counts[(structure_id, contour_index)] = 0
            continue

        inside_mask = contains_xy(polygon, x_values[bbox_mask], y_values[bbox_mask])
        assigned_cell_count = int(np.count_nonzero(inside_mask))
        contour_counts[(structure_id, contour_index)] = assigned_cell_count
        structure_counts[structure_id] = structure_counts.get(structure_id, 0) + assigned_cell_count

    basis = (
        f"geometry counts from {member_name} restricted to the assigned structure column {use_structure_col}"
        if use_structure_col is not None
        else f"geometry counts from {member_name} using all cell coordinates"
    )
    return contour_counts, structure_counts, basis


def load_partition_cell_counts(
    bundle_path: Path,
    contour_entries: list[dict[str, object]],
) -> dict[str, object]:
    member_name = first_existing_bundle_member(bundle_path, PARTITION_MEMBER_CANDIDATES)
    if member_name is None:
        member_name = first_existing_bundle_member(bundle_path, CELL_COORDINATE_MEMBER_CANDIDATES)
        if member_name is None:
            return {
                "available": False,
                "reason": (
                    "No cells_with_structure_partition or cells.parquet file was found in the contour bundle, "
                    "so contour cell counts could not be estimated."
                ),
                "member_name": None,
                "scope": "none",
                "structure_counts": {},
                "contour_counts": {},
                "notes": [],
            }
        geometric_result = geometric_contour_counts_from_member(
            bundle_path,
            member_name,
            contour_entries,
            structure_col=None,
        )
        if geometric_result is None:
            return {
                "available": False,
                "reason": (
                    f"{member_name} was found, but it did not expose usable cell coordinate columns. "
                    f"Looked for x columns {CELL_X_COORD_CANDIDATES} and y columns {CELL_Y_COORD_CANDIDATES}."
                ),
                "member_name": member_name,
                "scope": "none",
                "structure_counts": {},
                "contour_counts": {},
                "notes": [],
            }
        contour_counts, structure_counts, basis = geometric_result
        return {
            "available": True,
            "reason": None,
            "member_name": member_name,
            "scope": "contour",
            "structure_counts": structure_counts,
            "contour_counts": contour_counts,
            "notes": [
                "No structure-partition table was available, so contour counts were estimated directly from cell coordinates.",
                f"Count basis: {basis}.",
            ],
        }

    try:
        columns = bundle_table_columns(bundle_path, member_name)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Could not inspect {member_name}: {exc}",
            "member_name": member_name,
            "scope": "none",
            "structure_counts": {},
            "contour_counts": {},
            "notes": [],
        }

    structure_col = first_present_optional_column(columns, PARTITION_STRUCTURE_ID_CANDIDATES)
    contour_col = first_present_optional_column(columns, PARTITION_CONTOUR_ID_CANDIDATES)
    if structure_col is None:
        return {
            "available": False,
            "reason": (
                f"{member_name} did not contain any recognized structure assignment column. "
                f"Looked for: {', '.join(PARTITION_STRUCTURE_ID_CANDIDATES)}."
            ),
            "member_name": member_name,
            "scope": "none",
            "structure_counts": {},
            "contour_counts": {},
            "notes": [],
        }

    requested_columns = [structure_col]
    if contour_col is not None:
        requested_columns.append(contour_col)
    try:
        partition_df = read_bundle_table_subset(bundle_path, member_name, requested_columns)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Could not read {member_name}: {exc}",
            "member_name": member_name,
            "scope": "none",
            "structure_counts": {},
            "contour_counts": {},
            "notes": [],
        }

    partition_df = partition_df.dropna(subset=[structure_col]).copy()
    partition_df[structure_col] = pd.to_numeric(partition_df[structure_col], errors="coerce")
    partition_df = partition_df.loc[partition_df[structure_col].notna()].copy()
    if partition_df.empty:
        return {
            "available": False,
            "reason": f"{member_name} did not contain any usable structure assignments after cleaning.",
            "member_name": member_name,
            "scope": "none",
            "structure_counts": {},
            "contour_counts": {},
            "notes": [],
        }
    partition_df.loc[:, structure_col] = partition_df[structure_col].astype(int)
    structure_counts = {
        int(structure_id): int(cell_count)
        for structure_id, cell_count in partition_df.groupby(structure_col, observed=True).size().items()
    }

    if contour_col is None:
        geometric_result = geometric_contour_counts_from_member(
            bundle_path,
            member_name,
            contour_entries,
            structure_col=structure_col,
        )
        if geometric_result is not None:
            contour_counts, geometric_structure_counts, basis = geometric_result
            return {
                "available": True,
                "reason": None,
                "member_name": member_name,
                "scope": "contour",
                "structure_counts": geometric_structure_counts,
                "contour_counts": contour_counts,
                "notes": [
                    (
                        f"{member_name} had {structure_col} but no contour-level column, "
                        "so contour counts were estimated geometrically instead."
                    ),
                    f"Count basis: {basis}.",
                ],
            }
        return {
            "available": True,
            "reason": None,
            "member_name": member_name,
            "scope": "structure",
            "structure_counts": structure_counts,
            "contour_counts": {},
            "notes": [
                (
                    f"{member_name} had {structure_col} but no contour-level column. "
                    "The filter will fall back to structure-level assigned-cell counts."
                )
            ],
        }

    contour_df = partition_df.dropna(subset=[contour_col]).copy()
    contour_df[contour_col] = pd.to_numeric(contour_df[contour_col], errors="coerce")
    contour_df = contour_df.loc[contour_df[contour_col].notna()].copy()
    if contour_df.empty:
        return {
            "available": True,
            "reason": None,
            "member_name": member_name,
            "scope": "structure",
            "structure_counts": structure_counts,
            "contour_counts": {},
            "notes": [
                (
                    f"{member_name} had {contour_col}, but it became empty after numeric cleanup. "
                    "The filter will fall back to structure-level assigned-cell counts."
                )
            ],
        }
    contour_df.loc[:, contour_col] = contour_df[contour_col].astype(int)

    raw_contour_counts: dict[int, dict[int, int]] = {}
    grouped = (
        contour_df.groupby([structure_col, contour_col], observed=True)
        .size()
        .reset_index(name="assigned_cell_count")
    )
    for row in grouped.itertuples(index=False):
        structure_id = int(getattr(row, structure_col))
        contour_id = int(getattr(row, contour_col))
        raw_contour_counts.setdefault(structure_id, {})[contour_id] = int(row.assigned_cell_count)

    available_contours_by_structure: dict[int, set[int]] = {}
    for entry in contour_entries:
        structure_id = int(entry["structure_id"])
        contour_index = int(entry["contour_index"])
        available_contours_by_structure.setdefault(structure_id, set()).add(contour_index)

    normalized_contour_counts: dict[tuple[int, int], int] = {}
    notes: list[str] = []
    for structure_id, contour_map in raw_contour_counts.items():
        available_indices = available_contours_by_structure.get(structure_id, set())
        if not available_indices:
            continue

        direct_matches = sum(1 for contour_id in contour_map if contour_id in available_indices)
        shifted_matches = sum(1 for contour_id in contour_map if (contour_id - 1) in available_indices)
        subtract_one = shifted_matches > direct_matches
        if subtract_one and shifted_matches > 0:
            notes.append(
                f"Matched contour assignments for structure S{structure_id} after shifting {contour_col} from 1-based to 0-based indexing."
            )

        for contour_id, assigned_cell_count in contour_map.items():
            mapped_contour_index = contour_id - 1 if subtract_one else contour_id
            if mapped_contour_index in available_indices:
                normalized_contour_counts[(structure_id, mapped_contour_index)] = int(assigned_cell_count)

    if not normalized_contour_counts:
        geometric_result = geometric_contour_counts_from_member(
            bundle_path,
            member_name,
            contour_entries,
            structure_col=structure_col,
        )
        if geometric_result is not None:
            contour_counts, geometric_structure_counts, basis = geometric_result
            return {
                "available": True,
                "reason": None,
                "member_name": member_name,
                "scope": "contour",
                "structure_counts": geometric_structure_counts,
                "contour_counts": contour_counts,
                "notes": notes
                + [
                    (
                        f"Could not align {contour_col} values in {member_name} to contour filenames, "
                        "so contour counts were estimated geometrically instead."
                    ),
                    f"Count basis: {basis}.",
                ],
            }
        return {
            "available": True,
            "reason": None,
            "member_name": member_name,
            "scope": "structure",
            "structure_counts": structure_counts,
            "contour_counts": {},
            "notes": notes
            + [
                (
                    f"Could not align {contour_col} values in {member_name} to the contour filenames. "
                    "The filter will fall back to structure-level assigned-cell counts."
                )
            ],
        }

    matched_contours = len(normalized_contour_counts)
    total_contours = len(contour_entries)
    if matched_contours < total_contours:
        notes.append(
            f"Resolved contour-level cell counts for {matched_contours} of {total_contours} contour files from {member_name}. Unmatched contours are treated as having 0 assigned cells."
        )

    return {
        "available": True,
        "reason": None,
        "member_name": member_name,
        "scope": "contour",
        "structure_counts": structure_counts,
        "contour_counts": normalized_contour_counts,
        "notes": notes,
    }


def safe_filename_component(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(raw).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "output"


def polygon_from_vertices(vertices: np.ndarray) -> Any | None:
    if SHAPELY_IMPORT_ERROR is not None:
        raise RuntimeError(f"Shapely is required for contour analysis but could not be imported: {SHAPELY_IMPORT_ERROR}")
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    coords = arr[:, :2]
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    poly = Polygon(coords)
    if poly.is_empty:
        return None
    poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def build_contour_polygon_records(structure_records: list[dict[str, object]]) -> list[dict[str, object]]:
    contour_records: list[dict[str, object]] = []
    for record in structure_records:
        structure_id = int(record["structure_id"])
        structure_name = str(record["structure_name"])
        for contour_index, contour in enumerate(record.get("polygons") or []):
            polygon = polygon_from_vertices(contour)
            if polygon is None or polygon.is_empty:
                continue
            contour_records.append(
                {
                    "structure_id": structure_id,
                    "structure_name": structure_name,
                    "contour_index": int(contour_index),
                    "polygon": polygon,
                }
            )
    return contour_records


def grid_slice_for_bounds(
    *,
    bounds: tuple[float, float, float, float],
    x0: float,
    y0: float,
    grid_resolution_um: float,
    shape: tuple[int, int],
    pad_cells: int = 1,
) -> tuple[slice, slice]:
    minx, miny, maxx, maxy = bounds
    row_start = max(0, int(math.floor((miny - y0) / grid_resolution_um)) - int(pad_cells))
    row_stop = min(shape[0], int(math.ceil((maxy - y0) / grid_resolution_um)) + int(pad_cells) + 1)
    col_start = max(0, int(math.floor((minx - x0) / grid_resolution_um)) - int(pad_cells))
    col_stop = min(shape[1], int(math.ceil((maxx - x0) / grid_resolution_um)) + int(pad_cells) + 1)
    return slice(row_start, row_stop), slice(col_start, col_stop)


def rasterize_contour_boundary_owners(
    *,
    gx: np.ndarray,
    gy: np.ndarray,
    contour_records: list[dict[str, object]],
    x0: float,
    y0: float,
    grid_resolution_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundary_mask = np.zeros(gx.shape, dtype=bool)
    boundary_owner = np.full(gx.shape, -1, dtype=np.int32)
    erosion_kernel = np.ones((3, 3), dtype=bool)

    for owner_index, contour_record in enumerate(contour_records):
        polygon = contour_record["polygon"]
        row_slice, col_slice = grid_slice_for_bounds(
            bounds=polygon.bounds,
            x0=float(x0),
            y0=float(y0),
            grid_resolution_um=float(grid_resolution_um),
            shape=gx.shape,
            pad_cells=1,
        )
        sub_gx = gx[row_slice, col_slice]
        sub_gy = gy[row_slice, col_slice]
        if sub_gx.size == 0:
            continue

        inside_mask = contains_xy(polygon, sub_gx.ravel(), sub_gy.ravel()).reshape(sub_gx.shape)
        if not inside_mask.any():
            continue
        boundary_pixels = inside_mask & ~binary_erosion(inside_mask, structure=erosion_kernel, border_value=0)
        if not boundary_pixels.any():
            boundary_pixels = inside_mask

        sub_boundary_mask = boundary_mask[row_slice, col_slice]
        sub_boundary_owner = boundary_owner[row_slice, col_slice]
        claim_mask = boundary_pixels & ~sub_boundary_mask
        sub_boundary_owner[claim_mask] = int(owner_index)
        sub_boundary_mask |= boundary_pixels

    return boundary_mask, boundary_owner


def load_contour_bundle(
    bundle_path: Path,
    *,
    include_polygons: bool,
    filter_contours_by_assigned_cells: bool = False,
    min_assigned_cells_threshold: int = 10,
) -> dict[str, object]:
    member_names = list_bundle_members(bundle_path)
    contour_matches = [CONTOUR_PATTERN.match(name) for name in member_names]
    contour_entries = [
        {
            "member_name": name,
            "structure_id": int(match.group(1)),
            "contour_index": int(match.group(2)),
        }
        for name, match in zip(member_names, contour_matches)
        if match is not None
    ]
    if not contour_entries:
        raise ValueError(
            "Could not find any HistoSeg contour files in the bundle. Expected files like "
            "'structure_1_contour_0.npy'."
        )

    structure_name_map: dict[int, str] = {}
    if bundle_member_exists(bundle_path, "structure_contour_metrics.json"):
        try:
            metrics_json = json.loads(read_bundle_bytes(bundle_path, "structure_contour_metrics.json").decode("utf-8"))
            for item in metrics_json.get("selected_structures", []):
                structure_id = int(item["structure_id"])
                structure_name_map[structure_id] = str(item.get("structure_name", f"Structure {structure_id}"))
        except Exception:
            structure_name_map = {}

    for entry in contour_entries:
        member_name = str(entry["member_name"])
        try:
            contour = np.load(io.BytesIO(read_bundle_bytes(bundle_path, member_name)), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not read contour file {member_name}: {exc}") from exc
        contour_arr = np.asarray(contour, dtype=float)
        if contour_arr.ndim != 2 or contour_arr.shape[0] < 3 or contour_arr.shape[1] < 2:
            entry["vertices"] = None
            continue
        entry["vertices"] = contour_arr[:, :2]

    contour_entries = [entry for entry in contour_entries if entry.get("vertices") is not None]
    if not contour_entries:
        raise ValueError("Contour files were found, but none contained valid Nx2 polygon vertices.")

    total_contours_before_filter = int(len(contour_entries))
    contours_before_filter_by_structure: dict[int, int] = {}
    for entry in contour_entries:
        structure_id = int(entry["structure_id"])
        contours_before_filter_by_structure[structure_id] = contours_before_filter_by_structure.get(structure_id, 0) + 1

    partition_counts = load_partition_cell_counts(bundle_path, contour_entries)
    contour_filter = {
        "requested": bool(filter_contours_by_assigned_cells),
        "applied": False,
        "scope": str(partition_counts.get("scope", "none")),
        "member_name": partition_counts.get("member_name"),
        "min_assigned_cells_threshold": int(min_assigned_cells_threshold),
        "total_contours_before_filter": total_contours_before_filter,
        "kept_contours_after_filter": total_contours_before_filter,
        "removed_contours": 0,
        "reason": partition_counts.get("reason"),
        "notes": list(partition_counts.get("notes", [])),
    }

    if partition_counts.get("available"):
        scope = str(partition_counts["scope"])
        structure_counts_lookup = {
            int(structure_id): int(cell_count)
            for structure_id, cell_count in dict(partition_counts.get("structure_counts", {})).items()
        }
        contour_counts_lookup = {
            (int(structure_id), int(contour_index)): int(cell_count)
            for (structure_id, contour_index), cell_count in dict(partition_counts.get("contour_counts", {})).items()
        }

        for entry in contour_entries:
            structure_id = int(entry["structure_id"])
            contour_index = int(entry["contour_index"])
            assigned_cell_count: int | None
            if scope == "contour":
                assigned_cell_count = contour_counts_lookup.get((structure_id, contour_index), 0)
            else:
                assigned_cell_count = structure_counts_lookup.get(structure_id, 0)
            entry["assigned_cell_count"] = assigned_cell_count
    else:
        for entry in contour_entries:
            entry["assigned_cell_count"] = None

    if filter_contours_by_assigned_cells:
        if not partition_counts.get("available"):
            contour_filter["reason"] = partition_counts.get("reason")
        else:
            kept_entries = [
                entry
                for entry in contour_entries
                if int(entry.get("assigned_cell_count") or 0) > int(min_assigned_cells_threshold)
            ]
            if not kept_entries:
                raise ValueError(
                    "The contour-cell filter removed every contour in the bundle. "
                    "Lower the threshold or disable the filter."
                )
            contour_entries = kept_entries
            contour_filter["applied"] = True
            contour_filter["kept_contours_after_filter"] = int(len(contour_entries))
            contour_filter["removed_contours"] = int(total_contours_before_filter - len(contour_entries))
            contour_filter["reason"] = None

    contours_by_structure: dict[int, list[np.ndarray]] = {}
    contour_cell_counts_by_structure: dict[int, list[int | None]] = {}
    for entry in contour_entries:
        structure_id = int(entry["structure_id"])
        contours_by_structure.setdefault(structure_id, []).append(np.asarray(entry["vertices"], dtype=float))
        contour_cell_counts_by_structure.setdefault(structure_id, []).append(entry.get("assigned_cell_count"))

    structures: list[dict[str, object]] = []
    for structure_id in sorted(contours_by_structure):
        contours = contours_by_structure[structure_id]
        stacked = np.vstack(contours)
        contour_assigned_counts = contour_cell_counts_by_structure.get(structure_id, [])
        assigned_cell_count = None
        if any(count is not None for count in contour_assigned_counts):
            if str(partition_counts.get("scope")) == "contour":
                assigned_cell_count = int(sum(int(count or 0) for count in contour_assigned_counts))
            else:
                assigned_cell_count = int(contour_assigned_counts[0] or 0)
        structures.append(
            {
                "structure_id": int(structure_id),
                "structure_name": structure_name_map.get(structure_id, f"Structure {structure_id}"),
                "n_contours": int(len(contours)),
                "n_contours_before_filter": int(contours_before_filter_by_structure.get(structure_id, len(contours))),
                "assigned_cell_count": assigned_cell_count,
                "bbox_xmin": float(np.min(stacked[:, 0])),
                "bbox_xmax": float(np.max(stacked[:, 0])),
                "bbox_ymin": float(np.min(stacked[:, 1])),
                "bbox_ymax": float(np.max(stacked[:, 1])),
                "polygons": contours if include_polygons else None,
            }
        )

    return {
        "bundle_path": str(bundle_path),
        "bundle_kind": "zip" if bundle_is_zip(bundle_path) else "directory",
        "has_metrics_json": bundle_member_exists(bundle_path, "structure_contour_metrics.json"),
        "has_partition_file": bundle_member_exists(bundle_path, "cells_with_structure_partition.parquet")
        or bundle_member_exists(bundle_path, "cells_with_structure_partition.csv"),
        "contour_filter": contour_filter,
        "structure_count": int(len(structures)),
        "structures": structures,
    }


def build_structure_choice_label(record: dict[str, object]) -> str:
    structure_id = int(record["structure_id"])
    structure_name = str(record["structure_name"])
    contour_count = int(record["n_contours"])
    label = f"S{structure_id} | {structure_name} | contours={contour_count}"
    assigned_cell_count = record.get("assigned_cell_count")
    if assigned_cell_count is not None:
        label += f" | cells={int(assigned_cell_count)}"
    return label


def build_structure_table(bundle_meta: dict[str, object]) -> pd.DataFrame:
    show_assigned_cell_count = any(record.get("assigned_cell_count") is not None for record in bundle_meta["structures"])
    show_pre_filter_contours = any(
        int(record.get("n_contours_before_filter", record["n_contours"])) != int(record["n_contours"])
        for record in bundle_meta["structures"]
    )
    rows: list[dict[str, object]] = []
    for record in bundle_meta["structures"]:
        row = {
            "structure_id": int(record["structure_id"]),
            "structure_name": str(record["structure_name"]),
            "contour_count": int(record["n_contours"]),
        }
        if show_pre_filter_contours:
            row["contours_before_filter"] = int(record.get("n_contours_before_filter", record["n_contours"]))
        if show_assigned_cell_count:
            assigned_cell_count = record.get("assigned_cell_count")
            row["assigned_cell_count"] = int(assigned_cell_count) if assigned_cell_count is not None else None
        row.update(
            {
                "bbox_xmin": round(float(record["bbox_xmin"]), 2),
                "bbox_xmax": round(float(record["bbox_xmax"]), 2),
                "bbox_ymin": round(float(record["bbox_ymin"]), 2),
                "bbox_ymax": round(float(record["bbox_ymax"]), 2),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def render_structure_context_preview(
    *,
    bundle_meta: dict[str, object],
    selected_ids: set[int],
    output_path: Path,
) -> Path:
    structure_records = bundle_meta["structures"]
    legend_columns = 1 if len(structure_records) <= 16 else 2 if len(structure_records) <= 32 else 3
    fig_width = 12.4 + 2.0 * (legend_columns - 1)
    fig, ax = plt.subplots(figsize=(fig_width, 10.5))
    fig.patch.set_facecolor("#08111B")
    ax.set_facecolor("#08111B")

    color_cycle = plt.cm.tab20(np.linspace(0, 1, max(1, len(structure_records))))
    legend_handles: list[Line2D] = []
    for idx, record in enumerate(structure_records):
        color = color_cycle[idx]
        structure_id = int(record["structure_id"])
        selected = structure_id in selected_ids
        line_width = 2.4 if selected else 1.2
        alpha = 0.92 if selected else 0.35
        label = f"S{structure_id}: {record['structure_name']}"
        for contour in record["polygons"] or []:
            contour_arr = np.asarray(contour, dtype=float)
            ax.plot(contour_arr[:, 0], contour_arr[:, 1], color=color, linewidth=line_width, alpha=alpha)
        legend_handles.append(Line2D([], [], color=color, linewidth=line_width, alpha=alpha, label=label))

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X (um)", color="#B9CBDE")
    ax.set_ylabel("Y (um)", color="#B9CBDE")
    ax.tick_params(colors="#7F96AC", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#294057")
    ax.set_title("Uploaded HistoSeg contours", color="#EAF2FA", fontsize=14)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            ncol=legend_columns,
            frameon=False,
            labelcolor="#EAF2FA",
            borderaxespad=0.0,
        )

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def parquet_column_names(parquet_path: Path) -> list[str]:
    if pq is not None:
        return [str(name) for name in pq.ParquetFile(parquet_path).schema.names]
    return [str(col) for col in pd.read_parquet(parquet_path).columns]


def first_present_column(columns: list[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a {label} column. Available columns: {columns}")


def detect_transcript_input_schema(transcript_parquet: Path) -> TranscriptInputSchema:
    columns = parquet_column_names(transcript_parquet)
    gene_col = first_present_column(
        columns,
        ("feature_name", "gene", "gene_name", "feature_id"),
        "transcript gene",
    )
    x_col = first_present_column(
        columns,
        ("x_location", "x", "X", "x_centroid"),
        "transcript x-coordinate",
    )
    y_col = first_present_column(
        columns,
        ("y_location", "y", "Y", "y_centroid"),
        "transcript y-coordinate",
    )
    qv_col = next((candidate for candidate in ("qv", "quality_score") if candidate in columns), None)
    is_gene_col = next((candidate for candidate in ("is_gene", "is_transcript_gene") if candidate in columns), None)
    return TranscriptInputSchema(
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        qv_col=qv_col,
        is_gene_col=is_gene_col,
    )


def read_parquet_subset(parquet_path: Path, columns: list[str]) -> pd.DataFrame:
    unique_columns = list(dict.fromkeys(columns))
    if pq is not None:
        return pq.read_table(parquet_path, columns=unique_columns).to_pandas()
    return pd.read_parquet(parquet_path, columns=unique_columns)


def iter_transcript_batches(parquet_path: Path, columns: list[str], batch_size: int = DEFAULT_BATCH_SIZE):
    unique_columns = list(dict.fromkeys(columns))
    if pq is None:
        yield pd.read_parquet(parquet_path, columns=unique_columns)
        return
    parquet_file = pq.ParquetFile(parquet_path)
    for record_batch in parquet_file.iter_batches(columns=unique_columns, batch_size=int(batch_size)):
        yield record_batch.to_pandas()


def build_analysis_grid(
    *,
    selected_geometry: Any,
    tissue_geometry: Any,
    contour_records: list[dict[str, object]],
    grid_resolution_um: float,
    max_distance_um: float,
) -> dict[str, np.ndarray | float]:
    minx, miny, maxx, maxy = tissue_geometry.bounds
    padding = float(max_distance_um) + float(grid_resolution_um)
    x0 = math.floor((minx - padding) / grid_resolution_um) * grid_resolution_um
    y0 = math.floor((miny - padding) / grid_resolution_um) * grid_resolution_um
    x1 = math.ceil((maxx + padding) / grid_resolution_um) * grid_resolution_um
    y1 = math.ceil((maxy + padding) / grid_resolution_um) * grid_resolution_um

    xs = np.arange(x0, x1 + grid_resolution_um, grid_resolution_um, dtype=float)
    ys = np.arange(y0, y1 + grid_resolution_um, grid_resolution_um, dtype=float)
    gx, gy = np.meshgrid(xs + grid_resolution_um / 2.0, ys + grid_resolution_um / 2.0)

    tissue_mask = contains_xy(tissue_geometry, gx.ravel(), gy.ravel()).reshape(gx.shape)
    target_mask = contains_xy(selected_geometry, gx.ravel(), gy.ravel()).reshape(gx.shape)

    boundary_mask, boundary_owner = rasterize_contour_boundary_owners(
        gx=gx,
        gy=gy,
        contour_records=contour_records,
        x0=float(x0),
        y0=float(y0),
        grid_resolution_um=float(grid_resolution_um),
    )
    if not boundary_mask.any():
        raise ValueError("Could not rasterize any contour boundary pixels for Voronoi-style outward expansion.")

    inside_dist = distance_transform_edt(target_mask) * float(grid_resolution_um)
    outside_dist = distance_transform_edt(~target_mask) * float(grid_resolution_um)
    _, nearest_indices = distance_transform_edt(~boundary_mask, return_indices=True)
    nearest_owner = boundary_owner[nearest_indices[0], nearest_indices[1]]
    outward_voronoi_mask = (~target_mask) & tissue_mask & (nearest_owner >= 0)

    signed_distance = outside_dist.astype(float)
    signed_distance[target_mask] = -inside_dist[target_mask]
    analysis_mask = (
        (target_mask & (inside_dist <= float(max_distance_um)))
        | (outward_voronoi_mask & (outside_dist <= float(max_distance_um)))
    )

    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "xs": xs,
        "ys": ys,
        "gx": gx,
        "gy": gy,
        "tissue_mask": tissue_mask,
        "target_mask": target_mask,
        "outward_voronoi_mask": outward_voronoi_mask,
        "signed_distance": signed_distance,
        "analysis_mask": analysis_mask,
    }


def aggregate_transcript_distance_counts(
    *,
    transcript_parquet: Path,
    transcript_schema: TranscriptInputSchema,
    grid_state: dict[str, np.ndarray | float],
    qv_min: float,
    max_distance_um: float,
    bin_width_um: float,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, object], np.ndarray, np.ndarray]:
    bin_edges = np.arange(-float(max_distance_um), float(max_distance_um) + float(bin_width_um), float(bin_width_um))
    if len(bin_edges) < 2:
        raise ValueError("Distance bin edges could not be constructed. Increase max_distance_um or lower bin_width_um.")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)

    signed_distance = np.asarray(grid_state["signed_distance"], dtype=float)
    tissue_mask = np.asarray(grid_state["tissue_mask"], dtype=bool)
    analysis_mask = np.asarray(grid_state["analysis_mask"], dtype=bool)
    x0 = float(grid_state["x0"])
    y0 = float(grid_state["y0"])
    x1 = float(grid_state["x1"])
    y1 = float(grid_state["y1"])
    grid_resolution_um = float(np.asarray(grid_state["xs"])[1] - np.asarray(grid_state["xs"])[0]) if len(np.asarray(grid_state["xs"])) > 1 else 10.0

    area_bin_ids = np.floor((signed_distance - bin_edges[0]) / float(bin_width_um)).astype(int)
    valid_area_mask = analysis_mask & (area_bin_ids >= 0) & (area_bin_ids < n_bins)
    area_mm2 = np.array(
        [float(np.sum(valid_area_mask & (area_bin_ids == index))) * grid_resolution_um * grid_resolution_um / 1e6 for index in range(n_bins)],
        dtype=float,
    )

    columns = [transcript_schema.gene_col, transcript_schema.x_col, transcript_schema.y_col]
    if transcript_schema.qv_col is not None:
        columns.append(transcript_schema.qv_col)
    if transcript_schema.is_gene_col is not None:
        columns.append(transcript_schema.is_gene_col)

    gene_counts: dict[str, np.ndarray] = {}
    stats = {
        "rows_seen": 0,
        "rows_after_quality": 0,
        "rows_in_bbox": 0,
        "rows_in_tissue": 0,
        "rows_in_analysis_region": 0,
        "rows_in_distance_window": 0,
        "rows_counted": 0,
        "gene_count_pre_filter": 0,
    }

    for batch_df in iter_transcript_batches(transcript_parquet, columns=columns, batch_size=batch_size):
        stats["rows_seen"] += int(len(batch_df))
        if batch_df.empty:
            continue

        batch = batch_df.dropna(subset=[transcript_schema.gene_col, transcript_schema.x_col, transcript_schema.y_col]).copy()
        if batch.empty:
            continue

        gene_series = batch[transcript_schema.gene_col].astype(str).str.strip()
        batch = batch.loc[gene_series != ""].copy()
        batch.loc[:, transcript_schema.gene_col] = gene_series.loc[gene_series != ""]
        if batch.empty:
            continue

        if transcript_schema.qv_col is not None and transcript_schema.qv_col in batch.columns:
            batch = batch.loc[pd.to_numeric(batch[transcript_schema.qv_col], errors="coerce").fillna(-np.inf) >= float(qv_min)].copy()
        if transcript_schema.is_gene_col is not None and transcript_schema.is_gene_col in batch.columns:
            batch = batch.loc[batch[transcript_schema.is_gene_col].astype(bool)].copy()
        stats["rows_after_quality"] += int(len(batch))
        if batch.empty:
            continue

        x_values = pd.to_numeric(batch[transcript_schema.x_col], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(batch[transcript_schema.y_col], errors="coerce").to_numpy(dtype=float)
        gene_values = batch[transcript_schema.gene_col].astype(str).to_numpy()
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not finite_mask.any():
            continue
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]
        gene_values = gene_values[finite_mask]

        in_bbox = (x_values >= x0) & (x_values <= x1) & (y_values >= y0) & (y_values <= y1)
        stats["rows_in_bbox"] += int(in_bbox.sum())
        if not in_bbox.any():
            continue
        x_values = x_values[in_bbox]
        y_values = y_values[in_bbox]
        gene_values = gene_values[in_bbox]

        x_idx = np.floor((x_values - x0) / grid_resolution_um).astype(np.int64)
        y_idx = np.floor((y_values - y0) / grid_resolution_um).astype(np.int64)
        x_idx = np.clip(x_idx, 0, signed_distance.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, signed_distance.shape[0] - 1)

        in_tissue = tissue_mask[y_idx, x_idx]
        stats["rows_in_tissue"] += int(in_tissue.sum())
        if not in_tissue.any():
            continue
        x_idx = x_idx[in_tissue]
        y_idx = y_idx[in_tissue]
        gene_values = gene_values[in_tissue]

        in_analysis_region = analysis_mask[y_idx, x_idx]
        stats["rows_in_analysis_region"] += int(in_analysis_region.sum())
        if not in_analysis_region.any():
            continue
        x_idx = x_idx[in_analysis_region]
        y_idx = y_idx[in_analysis_region]
        gene_values = gene_values[in_analysis_region]

        signed_values = signed_distance[y_idx, x_idx]
        in_window = np.abs(signed_values) <= float(max_distance_um)
        stats["rows_in_distance_window"] += int(in_window.sum())
        if not in_window.any():
            continue
        signed_values = signed_values[in_window]
        gene_values = gene_values[in_window]

        bin_ids = np.floor((signed_values - bin_edges[0]) / float(bin_width_um)).astype(np.int64)
        valid_bins = (bin_ids >= 0) & (bin_ids < n_bins)
        if not valid_bins.any():
            continue
        bin_ids = bin_ids[valid_bins]
        gene_values = gene_values[valid_bins]
        stats["rows_counted"] += int(valid_bins.sum())

        group_df = pd.DataFrame({"gene": gene_values, "bin_id": bin_ids})
        grouped = (
            group_df.groupby(["gene", "bin_id"], observed=True)
            .size()
            .reset_index(name="count")
        )
        for row in grouped.itertuples(index=False):
            gene_name = str(row.gene)
            counts = gene_counts.get(gene_name)
            if counts is None:
                counts = np.zeros(n_bins, dtype=np.int64)
                gene_counts[gene_name] = counts
            counts[int(row.bin_id)] += int(row.count)

    stats["gene_count_pre_filter"] = int(len(gene_counts))
    return gene_counts, stats, bin_edges, area_mm2


def collect_gene_points_for_overlay(
    *,
    transcript_parquet: Path,
    transcript_schema: TranscriptInputSchema,
    target_gene: str,
    grid_state: dict[str, np.ndarray | float],
    qv_min: float,
    max_points: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    signed_distance = np.asarray(grid_state["signed_distance"], dtype=float)
    analysis_mask = np.asarray(grid_state["analysis_mask"], dtype=bool)
    x0 = float(grid_state["x0"])
    y0 = float(grid_state["y0"])
    x1 = float(grid_state["x1"])
    y1 = float(grid_state["y1"])
    grid_resolution_um = float(np.asarray(grid_state["xs"])[1] - np.asarray(grid_state["xs"])[0]) if len(np.asarray(grid_state["xs"])) > 1 else 10.0
    columns = [transcript_schema.gene_col, transcript_schema.x_col, transcript_schema.y_col]
    if transcript_schema.qv_col is not None:
        columns.append(transcript_schema.qv_col)
    if transcript_schema.is_gene_col is not None:
        columns.append(transcript_schema.is_gene_col)

    collected: list[np.ndarray] = []
    total_points = 0
    for batch_df in iter_transcript_batches(transcript_parquet, columns=columns, batch_size=batch_size):
        if batch_df.empty:
            continue
        batch = batch_df.dropna(subset=[transcript_schema.gene_col, transcript_schema.x_col, transcript_schema.y_col]).copy()
        if batch.empty:
            continue
        gene_series = batch[transcript_schema.gene_col].astype(str).str.strip()
        batch = batch.loc[gene_series == str(target_gene)].copy()
        if batch.empty:
            continue
        if transcript_schema.qv_col is not None and transcript_schema.qv_col in batch.columns:
            batch = batch.loc[pd.to_numeric(batch[transcript_schema.qv_col], errors="coerce").fillna(-np.inf) >= float(qv_min)].copy()
        if transcript_schema.is_gene_col is not None and transcript_schema.is_gene_col in batch.columns:
            batch = batch.loc[batch[transcript_schema.is_gene_col].astype(bool)].copy()
        if batch.empty:
            continue

        x_values = pd.to_numeric(batch[transcript_schema.x_col], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(batch[transcript_schema.y_col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not finite_mask.any():
            continue
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]

        in_bbox = (x_values >= x0) & (x_values <= x1) & (y_values >= y0) & (y_values <= y1)
        if not in_bbox.any():
            continue
        x_values = x_values[in_bbox]
        y_values = y_values[in_bbox]
        x_idx = np.floor((x_values - x0) / grid_resolution_um).astype(np.int64)
        y_idx = np.floor((y_values - y0) / grid_resolution_um).astype(np.int64)
        x_idx = np.clip(x_idx, 0, signed_distance.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, signed_distance.shape[0] - 1)
        keep = analysis_mask[y_idx, x_idx]
        if not keep.any():
            continue
        points = np.column_stack([x_values[keep], y_values[keep]])
        if len(points):
            collected.append(points)
            total_points += int(len(points))
        if total_points >= int(max_points * 3):
            break

    if not collected:
        return np.empty((0, 2), dtype=float)
    stacked = np.vstack(collected)
    if len(stacked) > int(max_points):
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(len(stacked), size=int(max_points), replace=False)
        stacked = stacked[keep_idx]
    return stacked


def build_curve_outputs(
    *,
    gene_counts: dict[str, np.ndarray],
    area_mm2: np.ndarray,
    bin_edges: np.ndarray,
    min_transcripts_per_gene: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid_area_mask = area_mm2 > 0
    rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    density_records: dict[str, np.ndarray] = {}

    for gene_name, counts in sorted(gene_counts.items()):
        counts_arr = np.asarray(counts, dtype=np.int64)
        total_count = int(counts_arr.sum())
        if total_count < int(min_transcripts_per_gene):
            continue

        area_safe = np.where(area_mm2 > 0, area_mm2, np.nan)
        density = counts_arr / area_safe
        density_records[gene_name] = density

        counts_valid = counts_arr[valid_area_mask]
        centers_valid = bin_centers[valid_area_mask]
        if counts_valid.sum() <= 0:
            continue

        profile = counts_valid / counts_valid.sum()
        nonzero = profile > 0
        if np.count_nonzero(valid_area_mask) > 1:
            entropy = float(-(profile[nonzero] * np.log(profile[nonzero])).sum())
            entropy_norm = entropy / math.log(len(profile)) if len(profile) > 1 else 0.0
        else:
            entropy_norm = 0.0
        variation_score = float(max(0.0, 1.0 - entropy_norm))
        signed_com_um = float(np.sum(centers_valid * profile))
        inward_count = int(counts_arr[bin_centers < 0].sum())
        outward_count = int(counts_arr[bin_centers >= 0].sum())
        peak_index = int(np.nanargmax(np.nan_to_num(density, nan=-1.0)))
        peak_distance_um = float(bin_centers[peak_index])
        peak_density = float(np.nanmax(np.nan_to_num(density, nan=0.0)))

        rows.append(
            {
                "gene": gene_name,
                "distance_profile_variation_score": variation_score,
                "total_transcripts": total_count,
                "inward_transcripts": inward_count,
                "outward_transcripts": outward_count,
                "inside_fraction": inward_count / max(total_count, 1),
                "signed_center_of_mass_um": signed_com_um,
                "peak_distance_um": peak_distance_um,
                "peak_density_per_mm2": peak_density,
            }
        )

        for bin_center, transcript_count, density_value, area_value in zip(bin_centers, counts_arr, density, area_mm2):
            long_rows.append(
                {
                    "gene": gene_name,
                    "bin_center_um": float(bin_center),
                    "side": "inside" if float(bin_center) < 0 else "outside",
                    "transcript_count": int(transcript_count),
                    "ring_area_mm2": float(area_value),
                    "density_per_mm2": float(density_value) if np.isfinite(density_value) else np.nan,
                }
            )

    ranking_df = pd.DataFrame(rows)
    if ranking_df.empty:
        raise ValueError(
            "No genes passed the minimum transcript count threshold after distance filtering. "
            "Try lowering min_transcripts_per_gene or increasing max_distance_um."
        )
    ranking_df = ranking_df.sort_values(
        ["distance_profile_variation_score", "total_transcripts"],
        ascending=[False, False],
    ).reset_index(drop=True)

    density_wide_df = pd.DataFrame({"bin_center_um": bin_centers, "ring_area_mm2": area_mm2})
    for gene_name, density in density_records.items():
        density_wide_df[gene_name] = density

    long_curves_df = pd.DataFrame(long_rows)
    return ranking_df, density_wide_df, long_curves_df


def render_variation_curve_plot(
    *,
    ranking_df: pd.DataFrame,
    density_wide_df: pd.DataFrame,
    top_n_genes: int,
    output_path: Path,
) -> Path:
    top_genes = ranking_df.head(int(top_n_genes))["gene"].tolist()
    if not top_genes:
        raise ValueError("No top genes were available for the curve plot.")

    fig, ax = plt.subplots(figsize=(12.4, 8.6))
    fig.patch.set_facecolor("#08111B")
    ax.set_facecolor("#08111B")
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(top_genes))))
    bin_centers = density_wide_df["bin_center_um"].to_numpy(dtype=float)

    for color, gene_name in zip(palette, top_genes):
        density = density_wide_df[gene_name].to_numpy(dtype=float)
        density = np.nan_to_num(density, nan=0.0)
        normalized = density / density.sum() if density.sum() > 0 else density
        ax.plot(
            bin_centers,
            normalized,
            color=color,
            linewidth=2.0,
            label=gene_name,
            alpha=0.92,
        )

    ax.axvline(0.0, color="#8AA2B8", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Signed distance from contour (um)  [<0 inside, >0 outside]", color="#C7D7E7")
    ax.set_ylabel("Normalized transcript density", color="#C7D7E7")
    ax.tick_params(colors="#8096AA")
    for spine in ax.spines.values():
        spine.set_color("#294057")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False, labelcolor="#EAF2FA")
    ax.set_title("Top spatially variant genes: normalized distance curves", color="#EAF2FA", fontsize=14)

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_variation_heatmap(
    *,
    ranking_df: pd.DataFrame,
    density_wide_df: pd.DataFrame,
    top_n_genes: int,
    output_path: Path,
) -> Path:
    top_genes = ranking_df.head(int(top_n_genes))["gene"].tolist()
    if not top_genes:
        raise ValueError("No top genes were available for the heatmap.")

    matrix_rows: list[np.ndarray] = []
    for gene_name in top_genes:
        density = density_wide_df[gene_name].to_numpy(dtype=float)
        values = np.log1p(np.nan_to_num(density, nan=0.0))
        std = float(np.std(values))
        if std > 0:
            values = (values - float(np.mean(values))) / std
        else:
            values = values - float(np.mean(values))
        matrix_rows.append(values)
    heatmap = np.vstack(matrix_rows)
    clustered_genes = list(top_genes)
    if len(top_genes) > 1:
        linkage_matrix = linkage(heatmap, method="average", metric="euclidean")
        cluster_order = leaves_list(linkage_matrix).astype(int)
        heatmap = heatmap[cluster_order, :]
        clustered_genes = [top_genes[idx] for idx in cluster_order]

    fig, ax = plt.subplots(figsize=(12.6, max(5.4, 0.42 * len(top_genes) + 2.0)))
    fig.patch.set_facecolor("#08111B")
    ax.set_facecolor("#08111B")
    image = ax.imshow(heatmap, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_yticks(np.arange(len(top_genes)))
    ax.set_yticklabels(clustered_genes, color="#EAF2FA", fontsize=8)

    bin_centers = density_wide_df["bin_center_um"].to_numpy(dtype=float)
    tick_step = max(1, len(bin_centers) // 10)
    tick_idx = np.arange(0, len(bin_centers), tick_step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{int(round(bin_centers[idx]))}" for idx in tick_idx], color="#C7D7E7", fontsize=8)
    if len(bin_centers) == 1:
        zero_position = 0.0
    else:
        zero_position = float(np.interp(0.0, bin_centers, np.arange(len(bin_centers), dtype=float)))
    ax.axvline(zero_position, color="#EAF2FA", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_xlabel("Signed distance from contour (um)", color="#C7D7E7")
    ax.set_title("Top spatially variant genes: hierarchically clustered z-scored log-density heatmap", color="#EAF2FA", fontsize=14)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.ax.tick_params(colors="#C7D7E7", labelsize=8)
    colorbar.set_label("z-scored log1p density", color="#C7D7E7")

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_top_gene_overlay(
    *,
    bundle_meta: dict[str, object],
    selected_ids: set[int],
    top_gene: str,
    top_gene_points: np.ndarray,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.8, 10.8))
    fig.patch.set_facecolor("#08111B")
    ax.set_facecolor("#08111B")
    color_cycle = plt.cm.tab20(np.linspace(0, 1, max(1, len(bundle_meta["structures"]))))

    for idx, record in enumerate(bundle_meta["structures"]):
        structure_id = int(record["structure_id"])
        if structure_id not in selected_ids:
            continue
        color = color_cycle[idx]
        for contour in record["polygons"] or []:
            contour_arr = np.asarray(contour, dtype=float)
            ax.plot(contour_arr[:, 0], contour_arr[:, 1], color=color, linewidth=2.0, alpha=0.95)
            ax.fill(contour_arr[:, 0], contour_arr[:, 1], color=color, alpha=0.10)

    if len(top_gene_points):
        ax.scatter(
            top_gene_points[:, 0],
            top_gene_points[:, 1],
            s=3.0,
            color="#FF7E6B",
            alpha=0.65,
            rasterized=True,
            label=f"{top_gene} transcripts",
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X (um)", color="#B9CBDE")
    ax.set_ylabel("Y (um)", color="#B9CBDE")
    ax.tick_params(colors="#7F96AC", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#294057")
    ax.set_title(f"Top spatially variant gene overlay: {top_gene}", color="#EAF2FA", fontsize=14)
    if len(top_gene_points):
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#EAF2FA")

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def zip_outputs(output_dir: Path, *, archive_dir: Path) -> tuple[Path | None, str | None]:
    archive_base = archive_dir / "contour_transcript_outputs"
    try:
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
        return Path(archive_path), None
    except Exception as exc:
        return None, f"ZIP archive was skipped: {exc}"


def update_visibility(input_mode: str):
    use_upload_mode = input_mode != INPUT_MODE_STORAGE
    return (
        gr.update(visible=use_upload_mode),
        gr.update(visible=not use_upload_mode),
        gr.update(visible=use_upload_mode),
        gr.update(visible=not use_upload_mode),
    )


def load_contour_bundle_metadata(
    input_mode: str,
    contour_bundle_upload: object | None,
    contour_bundle_storage_path: str | None,
    filter_contours_by_assigned_cells: bool,
    min_assigned_cells_threshold: float,
):
    if SHAPELY_IMPORT_ERROR is not None:
        raise gr.Error(
            "Shapely could not be imported inside the app container. "
            f"Import error: {SHAPELY_IMPORT_ERROR}"
        )

    removed_previews = cleanup_old_previews(max_keep=4)
    preview_dir = build_preview_dir()

    bundle_input = resolve_contour_bundle_source(
        input_mode=input_mode,
        contour_bundle_upload=contour_bundle_upload,
        contour_bundle_storage_path=contour_bundle_storage_path,
        target_dir=preview_dir,
    )
    bundle_meta = load_contour_bundle(
        bundle_input.path,
        include_polygons=True,
        filter_contours_by_assigned_cells=bool(filter_contours_by_assigned_cells),
        min_assigned_cells_threshold=int(min_assigned_cells_threshold),
    )
    structure_table = build_structure_table(bundle_meta)
    choices = [build_structure_choice_label(record) for record in bundle_meta["structures"]]
    choice_to_id = {label: int(record["structure_id"]) for label, record in zip(choices, bundle_meta["structures"])}
    selected_ids = {choice_to_id[choices[0]]} if choices else set()

    preview_path = preview_dir / "contour_bundle_preview.png"
    render_structure_context_preview(bundle_meta=bundle_meta, selected_ids=selected_ids, output_path=preview_path)

    status_lines = [
        f"Loaded contour bundle: {bundle_input.original}",
        f"Bundle type: {bundle_meta['bundle_kind']}",
        f"Structures discovered: {bundle_meta['structure_count']}",
        f"Has structure_contour_metrics.json: {bundle_meta['has_metrics_json']}",
        f"Has cells_with_structure_partition file: {bundle_meta['has_partition_file']}",
        "Signed distance convention for analysis: negative inside, positive outside.",
        "Outside expansion rule: Voronoi-style among selected contours only; unselected structures do not block expansion.",
        (
            "Contour-cell filter requested: "
            f"{'yes' if filter_contours_by_assigned_cells else 'no'}"
        ),
        "Select one or more structures below, then upload transcript.parquet and run the transcript analysis.",
    ]
    contour_filter = dict(bundle_meta.get("contour_filter", {}))
    if contour_filter.get("requested"):
        if contour_filter.get("applied"):
            basis = "contour-level" if contour_filter.get("scope") == "contour" else "structure-level"
            status_lines.append(
                "Contour-cell filter applied: kept "
                f"{contour_filter.get('kept_contours_after_filter')} of "
                f"{contour_filter.get('total_contours_before_filter')} contours "
                f"with more than {contour_filter.get('min_assigned_cells_threshold')} assigned cells "
                f"using {basis} counts from {contour_filter.get('member_name')}."
            )
        else:
            status_lines.append(
                "Contour-cell filter was not applied: "
                f"{contour_filter.get('reason') or 'the bundle did not expose usable assigned-cell counts.'}"
            )
    for note in contour_filter.get("notes", []):
        status_lines.append(f"Filter note: {note}")
    if removed_previews:
        status_lines.append(f"Cleaned old preview directories: {', '.join(removed_previews)}")

    state = {
        "bundle_source": {
            "input_mode": input_mode,
            "bundle_path": str(bundle_input.path),
            "original": bundle_input.original,
        },
        "choice_to_id": choice_to_id,
        "structure_records": [
            {
                "structure_id": int(record["structure_id"]),
                "structure_name": str(record["structure_name"]),
                "n_contours": int(record["n_contours"]),
                "n_contours_before_filter": int(record.get("n_contours_before_filter", record["n_contours"])),
                "assigned_cell_count": (
                    int(record["assigned_cell_count"]) if record.get("assigned_cell_count") is not None else None
                ),
            }
            for record in bundle_meta["structures"]
        ],
        "contour_filter": contour_filter,
    }
    return (
        "\n".join(status_lines),
        str(preview_path),
        structure_table,
        gr.update(choices=choices, value=choices[:1]),
        state,
    )


def run_contour_transcript_analysis(
    input_mode: str,
    contour_bundle_upload: object | None,
    contour_bundle_storage_path: str | None,
    transcript_parquet: object | None,
    transcript_storage_path: str | None,
    selected_structure_labels: list[str] | None,
    bundle_state: dict[str, object] | None,
    qv_min: float,
    grid_resolution_um: float,
    bin_width_um: float,
    max_distance_um: float,
    min_transcripts_per_gene: int,
    top_n_genes: int,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if SHAPELY_IMPORT_ERROR is not None:
        raise gr.Error(
            "Shapely could not be imported inside the app container. "
            f"Import error: {SHAPELY_IMPORT_ERROR}"
        )

    removed_runs = cleanup_old_runs(max_keep=3)
    start_time = time.perf_counter()
    run_dir = build_run_dir()
    input_dir = run_dir / "inputs"
    output_dir = run_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.05, desc="Staging contour bundle and transcript file")
        contour_input = resolve_contour_bundle_source(
            input_mode=input_mode,
            contour_bundle_upload=contour_bundle_upload,
            contour_bundle_storage_path=contour_bundle_storage_path,
            target_dir=input_dir,
        )
        transcript_input = resolve_transcript_source(
            input_mode=input_mode,
            transcript_upload=transcript_parquet,
            transcript_storage_path=transcript_storage_path,
            target_dir=input_dir,
        )

        progress(0.12, desc="Parsing contour structures")
        contour_filter_state = dict((bundle_state or {}).get("contour_filter", {}))
        bundle_meta = load_contour_bundle(
            contour_input.path,
            include_polygons=True,
            filter_contours_by_assigned_cells=bool(contour_filter_state.get("requested", False)),
            min_assigned_cells_threshold=int(contour_filter_state.get("min_assigned_cells_threshold", 10)),
        )
        label_to_id = dict(bundle_state.get("choice_to_id", {})) if bundle_state else {}
        if not label_to_id:
            label_to_id = {
                build_structure_choice_label(record): int(record["structure_id"])
                for record in bundle_meta["structures"]
            }

        selected_labels = list(selected_structure_labels or [])
        if not selected_labels:
            raise ValueError("Please select at least one structure from the contour bundle before running the analysis.")

        selected_ids = {int(label_to_id[label]) for label in selected_labels if label in label_to_id}
        if not selected_ids:
            raise ValueError("The selected structure labels could not be resolved back to contour IDs.")

        structure_lookup = {int(record["structure_id"]): record for record in bundle_meta["structures"]}
        selected_records = [structure_lookup[structure_id] for structure_id in sorted(selected_ids)]
        all_contour_records = build_contour_polygon_records(bundle_meta["structures"])
        all_polygons = [record["polygon"] for record in all_contour_records]
        if not all_polygons:
            raise ValueError("The contour bundle did not yield any valid polygons after parsing.")
        selected_contour_records = [
            contour_record
            for contour_record in all_contour_records
            if int(contour_record["structure_id"]) in selected_ids
        ]
        selected_polygons = [record["polygon"] for record in selected_contour_records]
        if not selected_polygons:
            raise ValueError("The selected structures did not yield any valid polygons after parsing.")

        selected_geometry = unary_union(selected_polygons)
        tissue_geometry = unary_union(all_polygons)

        selected_preview_path = output_dir / "selected_structure_context.png"
        render_structure_context_preview(
            bundle_meta=bundle_meta,
            selected_ids=selected_ids,
            output_path=selected_preview_path,
        )

        progress(0.22, desc="Inspecting transcript.parquet schema")
        transcript_schema = detect_transcript_input_schema(transcript_input.path)
        estimated_rows = safe_count_parquet_rows(transcript_input.path)

        progress(0.35, desc="Rasterizing selected contour and tissue region")
        grid_state = build_analysis_grid(
            selected_geometry=selected_geometry,
            tissue_geometry=tissue_geometry,
            contour_records=selected_contour_records,
            grid_resolution_um=float(grid_resolution_um),
            max_distance_um=float(max_distance_um),
        )

        progress(0.58, desc="Aggregating all transcript distance curves")
        gene_counts, aggregation_stats, bin_edges, area_mm2 = aggregate_transcript_distance_counts(
            transcript_parquet=transcript_input.path,
            transcript_schema=transcript_schema,
            grid_state=grid_state,
            qv_min=float(qv_min),
            max_distance_um=float(max_distance_um),
            bin_width_um=float(bin_width_um),
        )

        progress(0.72, desc="Ranking spatially variant genes")
        ranking_df, density_wide_df, long_curves_df = build_curve_outputs(
            gene_counts=gene_counts,
            area_mm2=area_mm2,
            bin_edges=bin_edges,
            min_transcripts_per_gene=int(min_transcripts_per_gene),
        )
        top_gene = str(ranking_df.iloc[0]["gene"])

        progress(0.82, desc="Rendering plots")
        top_curve_path = output_dir / "top_spatially_variant_gene_curves.png"
        top_heatmap_path = output_dir / "top_spatially_variant_gene_heatmap.png"
        render_variation_curve_plot(
            ranking_df=ranking_df,
            density_wide_df=density_wide_df,
            top_n_genes=int(top_n_genes),
            output_path=top_curve_path,
        )
        render_variation_heatmap(
            ranking_df=ranking_df,
            density_wide_df=density_wide_df,
            top_n_genes=int(top_n_genes),
            output_path=top_heatmap_path,
        )
        top_gene_points = collect_gene_points_for_overlay(
            transcript_parquet=transcript_input.path,
            transcript_schema=transcript_schema,
            target_gene=top_gene,
            grid_state=grid_state,
            qv_min=float(qv_min),
            max_points=DEFAULT_TOP_GENE_SAMPLE,
        )
        top_overlay_path = output_dir / f"top_gene_{safe_filename_component(top_gene)}_overlay.png"
        render_top_gene_overlay(
            bundle_meta=bundle_meta,
            selected_ids=selected_ids,
            top_gene=top_gene,
            top_gene_points=top_gene_points,
            output_path=top_overlay_path,
        )

        progress(0.90, desc="Writing output tables")
        ranking_path = output_dir / "gene_spatial_variation_ranking.csv"
        density_wide_path = output_dir / "gene_distance_density_curves_wide.csv"
        long_curves_path = output_dir / "gene_distance_curves_long.csv"
        bin_summary_path = output_dir / "distance_bin_summary.csv"
        ranking_df.to_csv(ranking_path, index=False)
        density_wide_df.to_csv(density_wide_path, index=False)
        long_curves_df.to_csv(long_curves_path, index=False)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        pd.DataFrame(
            {
                "bin_center_um": bin_centers,
                "side": np.where(bin_centers < 0, "inside", "outside"),
                "ring_area_mm2": area_mm2,
            }
        ).to_csv(bin_summary_path, index=False)

        elapsed = round(time.perf_counter() - start_time, 2)
        summary = {
            "app_name": APP_NAME,
            "contour_bundle": {
                "path": str(contour_input.path),
                "original": contour_input.original,
                "source": contour_input.source,
            },
            "transcript_parquet": {
                "path": str(transcript_input.path),
                "original": transcript_input.original,
                "source": transcript_input.source,
                "estimated_rows": estimated_rows,
                "gene_col": transcript_schema.gene_col,
                "x_col": transcript_schema.x_col,
                "y_col": transcript_schema.y_col,
                "qv_col": transcript_schema.qv_col,
                "is_gene_col": transcript_schema.is_gene_col,
            },
            "selected_structures": [
                {
                    "structure_id": int(record["structure_id"]),
                    "structure_name": str(record["structure_name"]),
                    "contour_count": int(record["n_contours"]),
                    "contours_before_filter": int(record.get("n_contours_before_filter", record["n_contours"])),
                    "assigned_cell_count": (
                        int(record["assigned_cell_count"]) if record.get("assigned_cell_count") is not None else None
                    ),
                }
                for record in selected_records
            ],
            "contour_filter": bundle_meta.get("contour_filter", {}),
            "parameters": {
                "qv_min": float(qv_min),
                "grid_resolution_um": float(grid_resolution_um),
                "bin_width_um": float(bin_width_um),
                "max_distance_um": float(max_distance_um),
                "min_transcripts_per_gene": int(min_transcripts_per_gene),
                "top_n_genes": int(top_n_genes),
                "signed_distance_convention": "negative_inside_positive_outside",
                "outward_assignment_mode": "voronoi_among_selected_contours_only",
            },
            "aggregation_stats": aggregation_stats,
            "top_gene": top_gene,
            "top_gene_variation_score": float(ranking_df.iloc[0]["distance_profile_variation_score"]),
            "top_gene_total_transcripts": int(ranking_df.iloc[0]["total_transcripts"]),
            "genes_ranked": int(len(ranking_df)),
            "elapsed_seconds": elapsed,
        }
        params_path = output_dir / "params.json"
        params_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        output_files: list[str] = [
            str(selected_preview_path),
            str(top_curve_path),
            str(top_heatmap_path),
            str(top_overlay_path),
            str(ranking_path),
            str(density_wide_path),
            str(long_curves_path),
            str(bin_summary_path),
            str(params_path),
        ]
        archive_path, archive_note = zip_outputs(output_dir, archive_dir=run_dir)
        if archive_path is not None:
            output_files.append(str(archive_path))

        status_lines = [
            f"{APP_NAME} finished successfully.",
            f"Run directory: {run_dir}",
            f"Selected structures analyzed: {len(selected_records)}",
            "Signed distance convention: negative inside, positive outside.",
            "Outward expansion mode: Voronoi-style among selected contours only; unselected structures do not block expansion.",
            f"Transcript rows seen: {aggregation_stats['rows_seen']}",
            f"Transcript rows counted after filtering: {aggregation_stats['rows_counted']}",
            f"Genes ranked: {len(ranking_df)}",
            f"Top spatially variant gene: {top_gene}",
            f"Top score: {float(ranking_df.iloc[0]['distance_profile_variation_score']):.4f}",
            f"Elapsed time: {elapsed} seconds",
        ]
        active_contour_filter = dict(bundle_meta.get("contour_filter", {}))
        if active_contour_filter.get("requested"):
            if active_contour_filter.get("applied"):
                status_lines.append(
                    "Contour-cell filter was active during analysis: kept "
                    f"{active_contour_filter.get('kept_contours_after_filter')} of "
                    f"{active_contour_filter.get('total_contours_before_filter')} contours."
                )
            else:
                status_lines.append(
                    "Contour-cell filter was requested but not applied: "
                    f"{active_contour_filter.get('reason') or 'usable assigned-cell counts were unavailable.'}"
                )
        if removed_runs:
            status_lines.append(f"Cleaned old run directories: {', '.join(removed_runs)}")
        if archive_note:
            status_lines.append(archive_note)

        progress(1.0, desc="Finished")
        return (
            "\n".join(status_lines),
            str(selected_preview_path),
            str(top_curve_path),
            str(top_heatmap_path),
            str(top_overlay_path),
            ranking_df.head(200),
            summary,
            str(archive_path) if archive_path is not None else None,
            output_files,
        )
    except Exception as exc:
        log_event(f"Contour transcript run failed: {exc}")
        print(traceback.format_exc(), flush=True)
        raise gr.Error(str(exc))


CUSTOM_CSS = """
:root {
  --app-bg: #07111d;
  --app-bg-soft: #0b1523;
  --panel-bg: rgba(12, 23, 38, 0.96);
  --panel-border: #1f3850;
  --text-main: #edf5ff;
  --text-dim: #b5c7d9;
  --accent: #78b9ff;
  --accent-strong: #6ef0d4;
}
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(120, 185, 255, 0.16), transparent 32%),
    radial-gradient(circle at top right, rgba(110, 240, 212, 0.12), transparent 24%),
    linear-gradient(180deg, #07111d 0%, #091523 100%);
  color: var(--text-main);
}
#app-shell {
  max-width: 1500px;
  margin: 0 auto;
  padding-bottom: 20px;
}
.hero-card, .micro-guide, .app-note {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 22px;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
}
.hero-card {
  padding: 24px 28px;
  margin-bottom: 18px;
}
.hero-kicker {
  color: var(--accent-strong);
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.hero-card h1 {
  margin: 0 0 10px 0;
  font-size: 32px;
}
.hero-card p {
  color: var(--text-dim);
  margin: 0;
  line-height: 1.55;
}
.micro-guide {
  padding: 18px 20px;
  margin-bottom: 18px;
}
.micro-guide-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.micro-step {
  background: rgba(8, 17, 29, 0.78);
  border: 1px solid #173049;
  border-radius: 18px;
  padding: 14px;
}
.micro-step strong {
  display: block;
  color: var(--accent-strong);
  font-size: 12px;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.micro-step h3 {
  margin: 0 0 6px 0;
  font-size: 17px;
}
.micro-step p {
  margin: 0;
  color: var(--text-dim);
  line-height: 1.45;
  font-size: 14px;
}
.app-note {
  padding: 16px 18px;
  margin-bottom: 18px;
  color: var(--text-dim);
  line-height: 1.5;
}
@media (max-width: 980px) {
  .micro-guide-grid {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 640px) {
  .micro-guide-grid {
    grid-template-columns: 1fr;
  }
}
"""


ensure_workdirs()

with gr.Blocks(title=APP_NAME, css=CUSTOM_CSS, fill_width=True) as demo:
    gr.HTML(
        f"""
        <div id="app-shell">
          <div class="hero-card">
            <div class="hero-kicker">SciLifeLab Serve app | HistoSeg contour transcript analysis</div>
            <h1>{APP_NAME}</h1>
            <p>{APP_DESCRIPTION}</p>
          </div>
          <div class="micro-guide">
            <div class="micro-guide-grid">
              <div class="micro-step">
                <strong>Step 1</strong>
                <h3>Load contours</h3>
                <p>Upload a HistoSeg output zip or point to a mounted folder/zip that already contains <code>structure_*_contour_*.npy</code>.</p>
              </div>
              <div class="micro-step">
                <strong>Step 2</strong>
                <h3>Select structures</h3>
                <p>Choose one or more structures whose contours define the signed distance reference; only selected contours compete for Voronoi-style outward ownership.</p>
              </div>
              <div class="micro-step">
                <strong>Step 3</strong>
                <h3>Analyze transcripts</h3>
                <p>Upload <code>transcript.parquet</code>, then compute signed-distance curves for all genes with negative values inside and positive values outside.</p>
              </div>
              <div class="micro-step">
                <strong>Step 4</strong>
                <h3>Rank top genes</h3>
                <p>The app saves all gene curves and ranks the most spatially variant genes using the distance-profile concentration score.</p>
              </div>
            </div>
          </div>
          <div class="app-note">
            <strong>Input expectation.</strong> This app does not run HistoSeg itself. It expects a contour bundle that HistoSeg has already produced,
            plus a fresh <code>transcript.parquet</code>. Upload mode is convenient for smaller files; mounted-storage mode is safer for large Xenium files.
          </div>
        </div>
        """
    )

    bundle_state = gr.State(value={})

    with gr.Row():
        input_mode = gr.Radio(
            label="Input source",
            choices=[
                ("Upload files in browser", INPUT_MODE_UPLOAD),
                ("Use mounted project storage paths", INPUT_MODE_STORAGE),
            ],
            value=INPUT_MODE_UPLOAD,
        )

    with gr.Row():
        contour_bundle_upload = gr.File(
            label="HistoSeg contour bundle (.zip)",
            file_count="single",
            file_types=[".zip"],
            visible=True,
        )
        contour_bundle_storage_path = gr.Textbox(
            label="Mounted path: HistoSeg contour bundle (.zip or folder)",
            placeholder=f"{PRIMARY_INPUT_ROOT}/my-run/histoseg_output.zip",
            visible=False,
        )

    with gr.Row():
        transcript_parquet = gr.File(
            label="Xenium transcripts (transcript.parquet)",
            file_count="single",
            file_types=[".parquet"],
            visible=True,
        )
        transcript_storage_path = gr.Textbox(
            label="Mounted path: transcript.parquet",
            placeholder=f"{PRIMARY_INPUT_ROOT}/my-run/outs/transcript.parquet",
            visible=False,
        )

    with gr.Row():
        load_bundle_button = gr.Button("1. Load contour bundle and list structures", variant="secondary")

    with gr.Row():
        filter_contours_by_assigned_cells = gr.Checkbox(
            label="Filter contour bundle by assigned cells before structure selection",
            value=False,
            info=(
                "When the bundle contains cells_with_structure_partition, keep only contours with more than the threshold "
                "below. If only structure-level assignments are available, the filter falls back to structure-level counts."
            ),
        )
        min_assigned_cells_threshold = gr.Slider(
            label="Keep only contours with more than this many assigned cells",
            minimum=0,
            maximum=500,
            step=1,
            value=10,
        )

    with gr.Row():
        structure_selector = gr.CheckboxGroup(
            label="Structures to use as the contour reference",
            choices=[],
            value=[],
            info=(
                "Select one or more uploaded structures. Inside each selected contour is treated as negative distance, "
                "and outward positive-distance regions are assigned Voronoi-style only among selected contours. "
                "Unselected structures do not block expansion."
            ),
        )

    with gr.Row():
        qv_min = gr.Slider(label="Minimum transcript qv", minimum=0, maximum=40, step=1, value=20)
        grid_resolution_um = gr.Slider(label="Grid resolution (um)", minimum=2, maximum=30, step=1, value=10)
        bin_width_um = gr.Slider(label="Distance bin width (um)", minimum=10, maximum=200, step=5, value=50)
        max_distance_um = gr.Slider(label="Maximum inward/outward distance (um)", minimum=100, maximum=5000, step=50, value=1000)

    with gr.Row():
        min_transcripts_per_gene = gr.Slider(
            label="Minimum transcripts per gene for ranking",
            minimum=1,
            maximum=1000,
            step=1,
            value=30,
        )
        top_n_genes = gr.Slider(
            label="Top genes to plot",
            minimum=3,
            maximum=30,
            step=1,
            value=12,
        )

    with gr.Row():
        run_button = gr.Button("2. Run contour transcript analysis", variant="primary")

    with gr.Row():
        contour_status = gr.Textbox(label="Status", lines=10)

    with gr.Row():
        contour_preview = gr.Image(label="Contour bundle preview", type="filepath")
        selected_structure_preview = gr.Image(label="Selected structure context", type="filepath")

    with gr.Row():
        structure_table = gr.Dataframe(label="Discovered structures", interactive=False, wrap=True)
        ranking_table = gr.Dataframe(label="Top ranked spatially variant genes", interactive=False, wrap=True)

    with gr.Row():
        top_curve_image = gr.Image(label="Top spatially variant gene curves", type="filepath")
        top_heatmap_image = gr.Image(label="Top spatially variant gene heatmap", type="filepath")

    with gr.Row():
        top_overlay_image = gr.Image(label="Top gene spatial overlay", type="filepath")
        run_summary = gr.JSON(label="Run summary")

    with gr.Row():
        archive_file = gr.File(label="ZIP archive of outputs")
        output_files = gr.File(label="Output files", file_count="multiple")

    input_mode.change(
        fn=update_visibility,
        inputs=[input_mode],
        outputs=[
            contour_bundle_upload,
            contour_bundle_storage_path,
            transcript_parquet,
            transcript_storage_path,
        ],
    )

    load_bundle_button.click(
        fn=load_contour_bundle_metadata,
        inputs=[
            input_mode,
            contour_bundle_upload,
            contour_bundle_storage_path,
            filter_contours_by_assigned_cells,
            min_assigned_cells_threshold,
        ],
        outputs=[
            contour_status,
            contour_preview,
            structure_table,
            structure_selector,
            bundle_state,
        ],
    )

    run_button.click(
        fn=run_contour_transcript_analysis,
        inputs=[
            input_mode,
            contour_bundle_upload,
            contour_bundle_storage_path,
            transcript_parquet,
            transcript_storage_path,
            structure_selector,
            bundle_state,
            qv_min,
            grid_resolution_um,
            bin_width_um,
            max_distance_um,
            min_transcripts_per_gene,
            top_n_genes,
        ],
        outputs=[
            contour_status,
            selected_structure_preview,
            top_curve_image,
            top_heatmap_image,
            top_overlay_image,
            ranking_table,
            run_summary,
            archive_file,
            output_files,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
