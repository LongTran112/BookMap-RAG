"""Shared UI helpers for BookMap dashboards (Streamlit, Gradio). Pure Python — no framework imports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import html
from io import BytesIO
import json
import math
import os
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional runtime dependency
    go = None
try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfium = None
try:
    from ebooklib import epub
except ImportError:  # pragma: no cover - optional runtime dependency
    epub = None

NOTEBOOKLM_URL = "https://notebooklm.google.com"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"
RAG_RETRIEVAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "Definition Q&A": {
        "top_k_chunks": 4,
        "min_similarity": 0.15,
        "use_hybrid": True,
        "dense_weight": 0.6,
        "lexical_weight": 0.4,
        "candidate_pool_size": 32,
        "reranker_enabled": True,
        "reranker_top_n": 16,
    },
    "Concept Compare": {
        "top_k_chunks": 6,
        "min_similarity": 0.1,
        "use_hybrid": True,
        "dense_weight": 0.7,
        "lexical_weight": 0.3,
        "candidate_pool_size": 48,
        "reranker_enabled": True,
        "reranker_top_n": 24,
    },
    "Learning Path": {
        "top_k_chunks": 8,
        "min_similarity": 0.05,
        "use_hybrid": True,
        "dense_weight": 0.75,
        "lexical_weight": 0.25,
        "candidate_pool_size": 64,
        "reranker_enabled": False,
        "reranker_top_n": 24,
    },
}
RAG_PERFORMANCE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Fast": {
        "generation_mode": "ollama",
        "ollama_model": "granite3.3:8b",
        "ollama_num_ctx": 4096,
        "ollama_temp": 0.15,
        "ollama_top_p": 0.85,
        "ollama_timeout_sec": 180,
        "top_k_chunks": 4,
        "max_citations": 3,
        "candidate_pool_size": 32,
        "min_similarity": 0.16,
        "reranker_enabled": True,
        "reranker_top_n": 24,
    },
    "Balanced": {
        "generation_mode": "ollama",
        "ollama_model": "granite3.3:8b",
        "ollama_num_ctx": 6144,
        "ollama_temp": 0.2,
        "ollama_top_p": 0.9,
        "ollama_timeout_sec": 240,
        "top_k_chunks": 6,
        "max_citations": 4,
        "candidate_pool_size": 40,
        "min_similarity": 0.12,
        "reranker_enabled": True,
        "reranker_top_n": 32,
    },
    "Quality": {
        "generation_mode": "ollama",
        "ollama_model": "qwen3.5:27b",
        "ollama_num_ctx": 8192,
        "ollama_temp": 0.2,
        "ollama_top_p": 0.9,
        "ollama_timeout_sec": 360,
        "top_k_chunks": 8,
        "max_citations": 6,
        "candidate_pool_size": 64,
        "min_similarity": 0.08,
        "reranker_enabled": True,
        "reranker_top_n": 40,
    },
}
RAG_LATENCY_TARGET_MS = 25000.0
RAG_AUTO_FAST_MAX_QUERY_CHARS = 120
RAG_AUTO_BALANCED_MAX_QUERY_CHARS = 260
RERANKER_MODEL_OPTIONS: List[str] = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "Custom",
]


@dataclass(frozen=True)
class BookmapPaths:
    """Default artifact locations relative to repository root."""

    repo_root: Path

    @property
    def output_dir(self) -> Path:
        return self.repo_root / "output"

    @property
    def semantic_index(self) -> Path:
        return self.output_dir / "semantic_index"

    @property
    def semantic_index_chunks(self) -> Path:
        return self.output_dir / "semantic_index_chunks"

    @property
    def semantic_index_chunks_gte_large(self) -> Path:
        return self.output_dir / "semantic_index_chunks_gte_large"

    @property
    def cover_cache(self) -> Path:
        return self.output_dir / "covers"

    @property
    def reading_list(self) -> Path:
        return self.output_dir / "currently_reading.json"

    @property
    def daily_recommendations(self) -> Path:
        return self.output_dir / "daily_recommendations.json"

    @property
    def current_read_books_dir(self) -> Path:
        return self.repo_root / "current-read-books"


def rag_chunk_index_directory_options(paths: BookmapPaths) -> List[str]:
    gte = paths.semantic_index_chunks_gte_large
    default_chunks = paths.semantic_index_chunks
    preferred_default = gte if gte.exists() else default_chunks
    options: List[str] = [str(preferred_default)]
    if str(default_chunks) not in options:
        options.append(str(default_chunks))
    out_dir = paths.output_dir
    if out_dir.exists() and out_dir.is_dir():
        for child in sorted(out_dir.iterdir(), key=lambda p: p.name):
            if child.is_dir() and child.name.startswith("semantic_index_chunks"):
                candidate = str(child)
                if candidate not in options:
                    options.append(candidate)
    options.append("Custom")
    return options


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_currently_reading(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def save_currently_reading(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def days_reading(added_at: str) -> int:
    try:
        started = datetime.fromisoformat(added_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return 1
    delta_days = (datetime.now(timezone.utc).date() - started.date()).days
    return max(1, delta_days + 1)


def make_reading_entry(item: dict) -> Dict[str, Any]:
    return {
        "book_id": item.get("book_id"),
        "title": item.get("title"),
        "category": item.get("category"),
        "learning_mode": item.get("learning_mode"),
        "absolute_path": item.get("absolute_path"),
        "added_at": utc_now_iso(),
        "progress_pct": 0,
        "reading_copy_path": "",
    }


def build_reading_copy_path(absolute_path: str, book_id: str, paths: BookmapPaths) -> Path:
    source = Path(absolute_path)
    safe_book_id = (book_id or "").replace("/", "_").replace("\\", "_").replace(":", "_")
    prefix = f"{safe_book_id}__" if safe_book_id else ""
    return paths.current_read_books_dir / f"{prefix}{source.name}"


def copy_book_to_current_read_folder(entry: Dict[str, Any], paths: BookmapPaths) -> Tuple[bool, str]:
    source = Path(str(entry.get("absolute_path", ""))).expanduser()
    if not source.exists() or not source.is_file():
        return False, f"Source file not found: {source}"
    copy_path = build_reading_copy_path(str(source), str(entry.get("book_id", "")), paths)
    try:
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy_path)
        entry["reading_copy_path"] = str(copy_path)
        return True, f"Copied to {copy_path}"
    except Exception as exc:
        return False, f"Could not copy book to current-read-books: {exc}"


def remove_book_copy_from_current_read_folder(entry: Dict[str, Any], paths: BookmapPaths) -> Tuple[bool, str]:
    stored = str(entry.get("reading_copy_path", "")).strip()
    if stored:
        copy_path = Path(stored).expanduser()
    else:
        copy_path = build_reading_copy_path(
            str(entry.get("absolute_path", "")),
            str(entry.get("book_id", "")),
            paths,
        )
    if not copy_path.exists():
        return True, "No copied file to remove."
    try:
        copy_path.unlink()
        return True, f"Removed copied file: {copy_path.name}"
    except Exception as exc:
        return False, f"Could not remove copied file: {exc}"


def get_file_dates(file_path: str) -> Tuple[Optional[date], Optional[date]]:
    path = Path(file_path)
    if not path.exists():
        return None, None
    try:
        stat = path.stat()
    except Exception:
        return None, None

    created_ts = getattr(stat, "st_birthtime", stat.st_ctime)
    updated_ts = stat.st_mtime
    created = datetime.fromtimestamp(created_ts).date()
    updated = datetime.fromtimestamp(updated_ts).date()
    return created, updated


def build_cover_thumbnail(source_path: str, cache_dir: str, max_width: int = 260) -> Optional[str]:
    source = Path(source_path)
    if not source.exists():
        return None

    suffix = source.suffix.lower()
    if suffix not in {".pdf", ".epub"}:
        return None

    cover_dir = Path(cache_dir)
    cover_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtime_key = str(int(source.stat().st_mtime))
    except Exception:
        mtime_key = "0"
    cache_key = hashlib.sha1(
        f"{source.as_posix()}::{suffix}::{mtime_key}::{max_width}".encode("utf-8")
    ).hexdigest()[:16]
    cover_path = cover_dir / f"{cache_key}.jpg"
    if cover_path.exists():
        return str(cover_path)

    if suffix == ".pdf":
        return _build_pdf_cover_thumbnail(source, cover_path, max_width=max_width)
    if suffix == ".epub":
        return _build_epub_cover_thumbnail(source, cover_path, max_width=max_width)
    return None


def _fit_cover_to_canvas(image: Image.Image, cover_path: Path) -> str:
    target_width = 300
    target_height = 420
    fitted = image.copy()
    fitted.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), color=(245, 245, 245))
    paste_x = (target_width - fitted.width) // 2
    paste_y = (target_height - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))
    canvas.save(cover_path, format="JPEG", quality=84)
    return str(cover_path)


def _build_pdf_cover_thumbnail(source: Path, cover_path: Path, max_width: int) -> Optional[str]:
    if pdfium is None:
        return None
    try:
        pdf = pdfium.PdfDocument(str(source))
        page = pdf[0]
        page_width = page.get_width() or max_width
        scale = max(max_width / float(page_width), 0.1)
        image = page.render(scale=scale).to_pil().convert("RGB")
        return _fit_cover_to_canvas(image, cover_path)
    except Exception:
        return None


def _build_epub_cover_thumbnail(source: Path, cover_path: Path, max_width: int) -> Optional[str]:
    if epub is None:
        return None
    try:
        book = epub.read_epub(str(source))
    except Exception:
        return None

    cover_bytes: Optional[bytes] = None
    try:
        for cover_id, _ in book.get_metadata("OPF", "cover"):
            item = book.get_item_with_id(str(cover_id))
            if item is not None and hasattr(item, "get_content"):
                payload = item.get_content()
                if isinstance(payload, bytes) and payload:
                    cover_bytes = payload
                    break
    except Exception:
        cover_bytes = None

    if not cover_bytes:
        image_items = []
        for item in book.get_items():
            media_type = str(getattr(item, "media_type", "")).lower()
            if media_type.startswith("image/"):
                image_items.append(item)

        preferred = [it for it in image_items if "cover" in str(getattr(it, "file_name", "")).lower()]
        candidates = preferred if preferred else image_items
        for item in candidates:
            try:
                payload = item.get_content()
            except Exception:
                continue
            if isinstance(payload, bytes) and payload:
                cover_bytes = payload
                break

    if not cover_bytes:
        return None

    try:
        image = Image.open(BytesIO(cover_bytes)).convert("RGB")
        if image.width > 0:
            scale = max(max_width / float(image.width), 0.1)
            resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        else:
            resized = image
        return _fit_cover_to_canvas(resized, cover_path)
    except Exception:
        return None


def card_title(text: str) -> str:
    clean = " ".join(str(text or "Untitled").split())
    if len(clean) > 76:
        return clean[:73].rstrip() + "..."
    return clean


def coerce_progress(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except Exception:
        return 0


def build_book_summary(book: dict) -> str:
    title = str(book.get("title", "This book")).strip() or "This book"
    category = str(book.get("category", "Other"))
    mode = str(book.get("learning_mode", "unknown"))
    confidence = float(book.get("confidence", 0.0) or 0.0)
    keywords = book.get("matched_keywords") or []
    keyword_text = ", ".join(str(item) for item in keywords[:5]) if keywords else "general concepts"

    metadata_text = " ".join(str(book.get("metadata_text", "")).split())
    body_preview = " ".join(str(book.get("body_preview", "")).split())
    source_text = metadata_text if metadata_text else body_preview

    detail_sentence = ""
    if source_text:
        source_text = source_text[:320]
        if len(source_text) == 320:
            source_text = source_text.rstrip() + "..."
        detail_sentence = f" It covers topics such as {source_text}."

    return (
        f"{title} is categorized under {category} and looks mostly {mode} in learning style "
        f"(confidence {confidence:.3f}). Core signals include {keyword_text}.{detail_sentence}"
    )


def get_book_format(item: Dict[str, Any]) -> str:
    path = Path(str(item.get("absolute_path", "")))
    return (path.suffix.lower().lstrip(".") or "unknown").upper()


def blend_results_to_surface_epubs(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not items:
        return items
    target = len(items) if top_k <= 0 else max(1, top_k)
    epubs = [item for item in items if get_book_format(item) == "EPUB"]
    non_epubs = [item for item in items if get_book_format(item) != "EPUB"]
    if not epubs or target <= 2:
        return items[:target]

    desired_epubs = min(len(epubs), max(2, target // 4))
    picked_epubs = epubs[:desired_epubs]
    picked_non_epubs = non_epubs[: max(0, target - desired_epubs)]
    merged = picked_epubs + picked_non_epubs
    merged.sort(key=lambda item: float(item.get("score", item.get("similarity", 0.0)) or 0.0), reverse=True)
    return merged[:target]


def open_pdf_in_file_manager(pdf_path: str) -> Tuple[bool, str]:
    path = Path(pdf_path).expanduser()
    if not path.exists():
        return False, f"File not found: {path}"

    system_name = platform.system()
    try:
        if system_name == "Darwin":
            subprocess.Popen(["open", "-R", str(path)])
            return True, "Opened in Finder."

        if system_name == "Windows":
            subprocess.Popen(["explorer", f"/select,{path}"])
            return True, "Opened in File Explorer."

        if shutil.which("nautilus"):
            subprocess.Popen(["nautilus", "--select", str(path)])
            return True, "Opened in file manager."
        if shutil.which("dolphin"):
            subprocess.Popen(["dolphin", "--select", str(path)])
            return True, "Opened in file manager."
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path.parent)])
            return True, "Opened parent folder."
        return False, "No supported file manager command found."
    except Exception as exc:
        return False, f"Could not open file location: {exc}"


def open_notebooklm_in_browser() -> Tuple[bool, str]:
    try:
        opened = webbrowser.open_new_tab(NOTEBOOKLM_URL)
        if opened:
            return True, "Opened NotebookLM."
        return False, f"Could not open browser tab for {NOTEBOOKLM_URL}"
    except Exception as exc:
        return False, f"Could not open NotebookLM: {exc}"


def graph_palette() -> List[str]:
    return [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]


def build_color_map(values: List[str]) -> Dict[str, str]:
    palette = graph_palette()
    unique = sorted({v for v in values if v})
    return {value: palette[idx % len(palette)] for idx, value in enumerate(unique)}


def compute_force_layout(
    node_ids: List[str],
    edges: List[Dict[str, Any]],
    iterations: int = 100,
) -> Dict[str, Tuple[float, float]]:
    if not node_ids:
        return {}
    n = len(node_ids)
    if n == 1:
        return {node_ids[0]: (0.0, 0.0)}

    if n > 420:
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return {node_ids[i]: (float(np.cos(angles[i])), float(np.sin(angles[i]))) for i in range(n)}

    rng = np.random.default_rng(42)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float64)
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    edge_pairs: List[Tuple[int, int, float]] = []
    for edge in edges:
        src = id_to_idx.get(str(edge.get("source", "")))
        dst = id_to_idx.get(str(edge.get("target", "")))
        if src is None or dst is None or src == dst:
            continue
        weight = float(edge.get("weight", 0.0) or 0.0)
        edge_pairs.append((src, dst, max(0.02, weight)))

    k = 1.2 / math.sqrt(n)
    temp = 0.25
    iter_count = max(40, min(160, int(iterations)))

    for _ in range(iter_count):
        disp = np.zeros_like(pos)
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1) + 1e-6
            force = (k * k) / dist
            force[i] = 0.0
            disp[i] += np.sum((delta / dist[:, None]) * force[:, None], axis=0)

        for src, dst, weight in edge_pairs:
            delta = pos[src] - pos[dst]
            dist = float(np.linalg.norm(delta) + 1e-6)
            attract = ((dist * dist) / k) * weight * 0.35
            direction = delta / dist
            disp[src] -= direction * attract
            disp[dst] += direction * attract

        norms = np.linalg.norm(disp, axis=1)
        norms[norms == 0] = 1.0
        pos += (disp / norms[:, None]) * np.minimum(norms, temp)[:, None]
        temp *= 0.95

    pos -= np.mean(pos, axis=0)
    max_abs = float(np.max(np.abs(pos)))
    if max_abs > 0:
        pos = pos / max_abs
    return {node_ids[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def extract_selected_graph_node_id(plot_state: Any) -> Optional[str]:
    if not isinstance(plot_state, dict):
        return None
    selection = plot_state.get("selection", {})
    if not isinstance(selection, dict):
        return None
    points = selection.get("points", [])
    if not isinstance(points, list) or not points:
        return None

    first = points[0]
    if not isinstance(first, dict):
        return None
    custom = first.get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        return str(custom[0])
    if isinstance(custom, str):
        return custom
    return None


def build_relationship_figure(
    graph_payload: Dict[str, Any],
    color_by: str,
    seed_ids: Optional[set[str]] = None,
    selected_node_id: Optional[str] = None,
):
    if go is None:
        return None

    nodes = graph_payload.get("nodes", []) or []
    edges = graph_payload.get("edges", []) or []
    if not nodes:
        return None

    node_by_id = {str(node.get("id", "")): node for node in nodes}
    node_ids = [node_id for node_id in node_by_id if node_id]
    positions = compute_force_layout(node_ids=node_ids, edges=edges)
    if not positions:
        return None

    degree_map = {node_id: 0 for node_id in node_ids}
    edge_x: List[float] = []
    edge_y: List[float] = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in positions or target not in positions:
            continue
        degree_map[source] = degree_map.get(source, 0) + 1
        degree_map[target] = degree_map.get(target, 0) + 1
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    color_field = "category" if color_by == "Category" else "learning_mode"
    color_values = [str(node_by_id[node_id].get(color_field, "unknown")) for node_id in node_ids]
    color_lookup = build_color_map(color_values)
    node_colors = [color_lookup.get(value, "#888") for value in color_values]

    marker_sizes = [10 + min(22, degree_map.get(node_id, 0)) for node_id in node_ids]
    marker_symbols = []
    seed_ids = seed_ids or set()
    for node_id in node_ids:
        if node_id == selected_node_id:
            marker_symbols.append("star")
        elif node_id in seed_ids:
            marker_symbols.append("diamond")
        else:
            marker_symbols.append("circle")

    hover_text = []
    for node_id in node_ids:
        node = node_by_id[node_id]
        hover_text.append(
            "<br>".join(
                [
                    f"<b>{html.escape(str(node.get('title', 'Untitled')))}</b>",
                    f"Category: {html.escape(str(node.get('category', 'Other')))}",
                    f"Mode: {html.escape(str(node.get('learning_mode', 'unknown')))}",
                    f"Degree: {degree_map.get(node_id, 0)}",
                ]
            )
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(140, 140, 140, 0.35)"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=[positions[node_id][0] for node_id in node_ids],
        y=[positions[node_id][1] for node_id in node_ids],
        mode="markers",
        customdata=[[node_id] for node_id in node_ids],
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            size=marker_sizes,
            color=node_colors,
            line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            symbol=marker_symbols,
            opacity=0.95,
        ),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",
        ),
    )
    return fig


def apply_rag_retrieval_preset_to_session(preset_name: str, session_state: Any) -> None:
    preset = RAG_RETRIEVAL_PRESETS.get(preset_name)
    if not preset:
        return
    session_state["rag-top-k-chunks"] = int(preset["top_k_chunks"])
    session_state["rag-min-similarity"] = float(preset["min_similarity"])
    session_state["rag-hybrid-enabled"] = bool(preset["use_hybrid"])
    session_state["rag-dense-weight"] = float(preset["dense_weight"])
    session_state["rag-lexical-weight"] = float(preset["lexical_weight"])
    session_state["rag-candidate-pool"] = int(preset["candidate_pool_size"])
    session_state["rag-reranker-enabled"] = bool(preset["reranker_enabled"])
    session_state["rag-reranker-topn"] = int(preset["reranker_top_n"])


def apply_rag_performance_profile_to_session(profile_name: str, session_state: Any) -> None:
    profile = RAG_PERFORMANCE_PROFILES.get(profile_name)
    if not profile:
        return
    session_state["rag-generation-mode"] = str(profile["generation_mode"])
    session_state["rag-ollama-model"] = str(profile["ollama_model"])
    session_state["rag-ollama-num-ctx"] = int(profile["ollama_num_ctx"])
    session_state["rag-ollama-temp"] = float(profile["ollama_temp"])
    session_state["rag-ollama-top-p"] = float(profile["ollama_top_p"])
    session_state["rag-ollama-timeout"] = int(profile["ollama_timeout_sec"])
    session_state["rag-top-k-chunks"] = int(profile["top_k_chunks"])
    session_state["rag-max-citations"] = int(profile["max_citations"])
    session_state["rag-candidate-pool"] = int(profile["candidate_pool_size"])
    session_state["rag-min-similarity"] = float(profile["min_similarity"])
    session_state["rag-reranker-enabled"] = bool(profile.get("reranker_enabled", True))
    session_state["rag-reranker-topn"] = int(profile.get("reranker_top_n", 32))


def select_rag_auto_profile(query: str) -> str:
    q = str(query or "").strip().lower()
    if not q:
        return "Balanced"
    definition_markers = ("what is ", "define ", "definition of ", "meaning of ")
    complex_markers = (
        "compare ",
        "difference",
        "tradeoff",
        "pros and cons",
        "architecture",
        "design",
        "deep dive",
        "comprehensive",
    )
    if any(marker in q for marker in definition_markers):
        return "Balanced"
    if any(marker in q for marker in complex_markers) or len(q) > RAG_AUTO_BALANCED_MAX_QUERY_CHARS:
        return "Quality"
    return "Balanced"


def build_rag_answer_payload(
    query: str,
    top_k_chunks: int,
    max_citations: int,
    selected_categories: List[str],
    selected_modes: List[str],
    min_similarity: float,
    use_hybrid: bool,
    dense_weight: float,
    lexical_weight: float,
    candidate_pool_size: int,
    reranker_enabled: bool,
    reranker_model: str,
    reranker_top_n: int,
    generation_mode: str,
    llama_model_path: str,
    llama_n_ctx: int,
    llama_max_tokens: int,
    llama_temp: float,
    llama_top_p: float,
    llama_threads: int,
    llama_gpu_layers: int,
    ollama_base_url: str,
    ollama_model: str,
    ollama_temp: float,
    ollama_top_p: float,
    ollama_num_ctx: int,
    ollama_timeout_sec: int,
    allow_fallback: bool,
) -> Dict[str, Any]:
    return {
        "query": query.strip(),
        "top_k": int(top_k_chunks),
        "max_citations": int(max_citations),
        "allow_fallback": bool(allow_fallback),
        "filters": {
            "categories": selected_categories or None,
            "learning_modes": selected_modes or None,
            "min_similarity": float(min_similarity),
        },
        "retrieval": {
            "hybrid_enabled": bool(use_hybrid),
            "dense_weight": float(dense_weight),
            "lexical_weight": float(lexical_weight),
            "candidate_pool_size": int(candidate_pool_size),
            "final_top_k": int(top_k_chunks),
            "reranker_enabled": bool(reranker_enabled),
            "reranker_model_name": str(reranker_model).strip() if reranker_enabled else None,
            "reranker_top_n": int(reranker_top_n),
        },
        "llm": {
            "enabled": generation_mode == "llama.cpp",
            "model_path": str(llama_model_path).strip(),
            "n_ctx": int(llama_n_ctx),
            "max_tokens": int(llama_max_tokens),
            "temperature": float(llama_temp),
            "top_p": float(llama_top_p),
            "n_threads": int(llama_threads),
            "n_gpu_layers": int(llama_gpu_layers),
        },
        "ollama": {
            "enabled": generation_mode == "ollama",
            "base_url": str(ollama_base_url).strip(),
            "model": str(ollama_model).strip(),
            "temperature": float(ollama_temp),
            "top_p": float(ollama_top_p),
            "num_ctx": int(ollama_num_ctx),
            "timeout_sec": int(ollama_timeout_sec),
        },
    }


def call_rag_api_answer(
    api_url: str,
    payload: Dict[str, Any],
    timeout_sec: int = 30,
    api_key: str = "",
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["X-API-Key"] = key
    request = urllib.request.Request(
        api_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(3, int(timeout_sec))) as response:
            content = response.read().decode("utf-8")
            data = json.loads(content) if content.strip() else {}
            if not isinstance(data, dict):
                raise ValueError("API returned non-object JSON response.")
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach API endpoint: {exc}") from exc


def split_answer_display_sections(answer_text: str) -> Tuple[str, str]:
    text = str(answer_text or "").replace("\r\n", "\n")
    if not text.strip():
        return "", ""

    lines = text.split("\n")
    final_markers = (
        "answer:",
        "formal definition:",
        "plain-language intuition:",
        "practical use-case:",
    )
    blur_prefixes = (
        "sourcesused:",
        "summary",
        "grounded from",
        "suggested follow-ups",
        "insufficient sources:",
        "thinking process:",
        "generating grounded answer",
    )
    reasoning_markers = (
        "so, i need to",
        "let me",
        "alright, i need",
        "okay, so i need",
        "the user wants",
    )

    answer_start_idx = -1
    for idx, line in enumerate(lines):
        if str(line or "").strip().lower().startswith(final_markers):
            answer_start_idx = idx
            break

    think_close_idx = -1
    for idx, line in enumerate(lines):
        if "</think>" in str(line or "").lower():
            think_close_idx = idx
            break

    blur_lines: List[str] = []
    normal_lines: List[str] = []
    for idx, raw_line in enumerate(lines):
        stripped = str(raw_line or "").strip()
        lowered = stripped.lower()
        is_blur = False
        if think_close_idx >= 0 and idx <= think_close_idx:
            is_blur = True
        elif answer_start_idx >= 0 and idx < answer_start_idx:
            is_blur = True
        elif lowered.startswith(blur_prefixes):
            is_blur = True
        elif any(marker in lowered for marker in reasoning_markers):
            is_blur = True

        if is_blur:
            if stripped:
                blur_lines.append(raw_line)
        else:
            if stripped:
                normal_lines.append(raw_line)

    return "\n".join(normal_lines).strip(), "\n".join(blur_lines).strip()


def normalize_answer_markdown(text: str) -> str:
    clean = str(text or "").replace("\r\n", "\n").strip()
    if not clean:
        return ""

    clean = re.sub(r"\*\*\s*([^*]+?)\s*\*\*", r"\1", clean)

    clean = re.sub(r"\bSources\s*Used\s*:", "SourcesUsed:", clean, flags=re.IGNORECASE)

    section_labels = [
        (r"answer\s*:", "Answer"),
        (r"formal\s*definition\s*:", "Formal Definition"),
        (r"plain\s*[-\s]?language\s*intuition\s*:", "Plain-language Intuition"),
        (r"practical\s*use\s*[-\s]?case\s*:", "Practical Use-Case"),
        (r"sourcesused\s*:", "SourcesUsed"),
        (r"summary\s*:", "Summary"),
    ]
    for raw_pattern, label in section_labels:
        clean = re.sub(
            rf"(?is)\s*{raw_pattern}\s*",
            f"\n\n### {label}\n\n",
            clean,
        )

    clean = re.sub(r"(?<!\n)(\d+\.)\s+", r"\n\n\1 ", clean)

    if "\n" not in clean and len(clean) > 280:
        clean = re.sub(r"(?<=[.!?])\s+(?=[A-Z0-9*])", "\n\n", clean)

    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean


def format_answer_markdown_for_display(answer_text: str, *, hide_meta_text: bool, show_cursor: bool = False) -> str:
    render_text = normalize_answer_markdown(str(answer_text or ""))
    if hide_meta_text:
        normal_text, _blurred = split_answer_display_sections(render_text)
        render_text = normal_text
    if show_cursor:
        render_text = (render_text.rstrip() + "\n\n▌").strip()
    if not render_text.strip():
        render_text = " "
    return render_text


def metrics_row_from_response(
    response: Dict[str, Any],
    *,
    answer_idx: int,
    query: str = "",
) -> Optional[Dict[str, Any]]:
    metrics = response.get("metrics", {}) if isinstance(response, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        return None
    return {
        "answer_idx": int(answer_idx),
        "query": query[:120] + ("..." if len(query) > 120 else ""),
        "total_ms": float(metrics.get("total_ms", 0.0) or 0.0),
        "retrieval_ms": float(metrics.get("retrieval_ms", 0.0) or 0.0),
        "generation_ms": float(metrics.get("generation_ms", 0.0) or 0.0),
        "peak_rss_mb": float(metrics.get("peak_rss_mb", 0.0) or 0.0),
        "retrieved_chunks": int(metrics.get("retrieved_chunks", 0) or 0),
        "used_citations": int(metrics.get("used_citations", 0) or 0),
        "citation_coverage_ratio": float(metrics.get("citation_coverage_ratio", 0.0) or 0.0),
        "top_similarity": float(metrics.get("top_similarity", 0.0) or 0.0),
        "top_relevance_score": float(metrics.get("top_relevance_score", 0.0) or 0.0),
        "prompt_chars": int(metrics.get("prompt_chars", 0) or 0),
        "answer_chars": int(metrics.get("answer_chars", 0) or 0),
    }


def append_rag_metrics_row(metrics_history: List[Dict[str, Any]], question: str, response: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = list(metrics_history) if metrics_history else []
    next_idx = len(history) + 1
    row = metrics_row_from_response(response, answer_idx=next_idx, query=str(question or ""))
    if row is None:
        return history
    history.append(row)
    return history[-200:]


def collect_recent_rag_metrics(
    *,
    metrics_history: Optional[List[Dict[str, Any]]],
    chat_history: List[Dict[str, Any]],
    window: int = 10,
) -> List[Dict[str, Any]]:
    if isinstance(metrics_history, list) and metrics_history:
        recent_metrics = metrics_history[-max(1, int(window)) :]
        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(recent_metrics, start=1):
            if not isinstance(row, dict):
                continue
            clean = dict(row)
            clean["answer_idx"] = idx
            rows.append(clean)
        if rows:
            return rows

    recent = chat_history[-max(1, int(window)) :]
    rows_out: List[Dict[str, Any]] = []
    for idx, item in enumerate(recent, start=1):
        response = item.get("response", {})
        query = str(item.get("question", "") or "").strip()
        row = metrics_row_from_response(response, answer_idx=idx, query=query)
        if row is None:
            continue
        rows_out.append(row)
    return rows_out


def rag_perf_rollup_caption(chat_history: List[Dict[str, Any]], metrics_history: Optional[List[Dict[str, Any]]], window: int = 10) -> str:
    metrics_rows = collect_recent_rag_metrics(metrics_history=metrics_history, chat_history=chat_history, window=window)
    if not metrics_rows:
        return ""
    count = len(metrics_rows)
    avg_total = sum(row["total_ms"] for row in metrics_rows) / count
    avg_retrieval = sum(row["retrieval_ms"] for row in metrics_rows) / count
    avg_generation = sum(row["generation_ms"] for row in metrics_rows) / count
    max_rss = max(row["peak_rss_mb"] for row in metrics_rows)
    avg_relevance = sum(row["top_relevance_score"] for row in metrics_rows) / count
    avg_coverage = sum(row["citation_coverage_ratio"] for row in metrics_rows) / count
    return (
        f"Recent performance (last {count} answers): avg total {avg_total:.1f} ms | "
        f"avg retrieval {avg_retrieval:.1f} ms | avg generation {avg_generation:.1f} ms | "
        f"max peak RSS {max_rss:.1f} MB | avg relevance {avg_relevance:.3f} | "
        f"avg citation coverage {avg_coverage:.2f}"
    )


def format_rag_turn_markdown(
    response: Dict[str, Any],
    *,
    show_debug: bool,
    blur_meta_text: bool,
    show_fallback_notice: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"*Generation mode: {response.get('generation_mode', 'deterministic')}*")
    metrics = response.get("metrics", {}) or {}
    if isinstance(metrics, dict) and metrics:
        total_ms = float(metrics.get("total_ms", 0.0) or 0.0)
        retrieval_ms = float(metrics.get("retrieval_ms", 0.0) or 0.0)
        generation_ms = float(metrics.get("generation_ms", 0.0) or 0.0)
        peak_rss_mb = float(metrics.get("peak_rss_mb", 0.0) or 0.0)
        top_relevance = float(metrics.get("top_relevance_score", 0.0) or 0.0)
        citation_coverage = float(metrics.get("citation_coverage_ratio", 0.0) or 0.0)
        lines.append(
            f"*Timing: total {total_ms:.1f} ms | retrieval {retrieval_ms:.1f} ms | "
            f"generation {generation_ms:.1f} ms | peak RSS {peak_rss_mb:.1f} MB | "
            f"top relevance {top_relevance:.3f} | citation coverage {citation_coverage:.2f}*"
        )
        if show_debug:
            lines.append("\n```json\n" + json.dumps(metrics, indent=2) + "\n```\n")

    fallback_reason = str(response.get("fallback_reason", "") or "").strip()
    if fallback_reason and show_fallback_notice:
        lines.append(f"\n> Fallback used: {fallback_reason}\n")

    lines.append("\n**Answer**\n")
    lines.append(format_answer_markdown_for_display(str(response.get("answer", "") or ""), hide_meta_text=blur_meta_text))
    lines.append("\n\n**Summary**\n")
    lines.append(str(response.get("summary", "") or ""))

    follow_ups = response.get("follow_ups", []) or []
    if follow_ups:
        lines.append("\n\n**Suggested follow-ups**\n")
        for prompt in follow_ups:
            lines.append(f"- {prompt}")

    citations = response.get("citations", []) or []
    lines.append(f"\n\n**Citations ({len(citations)})**\n")
    if not citations:
        lines.append("_No citations found for this question._")
    else:
        for idx, item in enumerate(citations):
            title = str(item.get("title", "Untitled"))
            label = (
                f"{title} | {item.get('category', 'Other')} | "
                f"{item.get('learning_mode', 'unknown')} | "
                f"{item.get('source_label', 'chunk')} | "
                f"sim {float(item.get('similarity', 0.0) or 0.0):.3f}"
            )
            lines.append(f"\n<details><summary>{html.escape(label)}</summary>\n\n")
            lines.append(str(item.get("snippet", "")))
            lines.append(f"\n\n`{item.get('absolute_path', '')}`\n</details>\n")
            if show_debug:
                dbg = {
                    "citation_id": item.get("citation_id", ""),
                    "book_id": item.get("book_id", ""),
                    "absolute_path": item.get("absolute_path", ""),
                    "start_char": item.get("start_char", 0),
                    "end_char": item.get("end_char", 0),
                    "chunk_order": item.get("chunk_order", 0),
                    "chunk_len": item.get("chunk_len", 0),
                }
                lines.append("\n```json\n" + json.dumps(dbg, indent=2) + "\n```\n")

    if show_debug:
        lines.append("\n```json\n" + json.dumps(response, indent=2, default=str)[:8000] + "\n```\n")

    return "\n".join(lines)


def default_rag_session_defaults() -> Dict[str, Any]:
    """Initial slider/toggle values for Ask Books (RAG) when not using Streamlit session keys."""
    return {
        "rag-top-k-chunks": 8,
        "rag-max-citations": 6,
        "rag-min-similarity": 0.15,
        "rag-hybrid-enabled": True,
        "rag-dense-weight": 0.7,
        "rag-lexical-weight": 0.3,
        "rag-candidate-pool": 48,
        "rag-reranker-enabled": True,
        "rag-reranker-model": DEFAULT_RERANKER_MODEL,
        "rag-reranker-topn": 24,
        "rag-generation-mode": "ollama",
        "rag-ollama-base-url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "rag-ollama-model": "granite3.3:8b",
        "rag-ollama-num-ctx": 8192,
        "rag-ollama-temp": 0.2,
        "rag-ollama-top-p": 0.9,
        "rag-ollama-timeout": 180,
        "rag-llama-model-path": "",
        "rag-llama-n-ctx": 2048,
        "rag-llama-max-tokens": 420,
        "rag-llama-temp": 0.2,
        "rag-llama-top-p": 0.9,
        "rag-llama-threads": 6,
        "rag-llama-gpu-layers": 0,
        "rag-auto-profile-enabled": False,
    }
