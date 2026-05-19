#!/usr/bin/env python3
"""Streamlit dashboard for semantic search and recommendations."""

from __future__ import annotations

from datetime import date
from datetime import datetime, timezone
import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional runtime dependency
    go = None

# Ensure local project packages resolve when running from nested app folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_books.learning_mode import learning_mode_labels
from semantic_books.daily_recommend import DailyBookRecommender, DailyRecommendationWeights
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService
from semantic_books.search_service import SearchFilters, SemanticSearchService
from semantic_books.bookmap_ui_core import (
    NOTEBOOKLM_URL,
    RAG_AUTO_BALANCED_MAX_QUERY_CHARS,
    RAG_AUTO_FAST_MAX_QUERY_CHARS,
    RAG_LATENCY_TARGET_MS,
    RAG_PERFORMANCE_PROFILES,
    RAG_RETRIEVAL_PRESETS,
    RERANKER_MODEL_OPTIONS,
    BookmapPaths,
    DEFAULT_RERANKER_MODEL,
    append_rag_metrics_row,
    apply_rag_performance_profile_to_session,
    apply_rag_retrieval_preset_to_session,
    blend_results_to_surface_epubs,
    build_book_summary,
    build_cover_thumbnail as core_build_cover_thumbnail,
    build_rag_answer_payload,
    build_relationship_figure,
    call_rag_api_answer,
    card_title as _card_title,
    coerce_progress,
    collect_recent_rag_metrics,
    copy_book_to_current_read_folder as core_copy_book_to_current_read_folder,
    days_reading,
    extract_selected_graph_node_id,
    format_answer_markdown_for_display,
    get_book_format,
    get_file_dates as core_get_file_dates,
    load_currently_reading,
    make_reading_entry,
    open_notebooklm_in_browser,
    open_pdf_in_file_manager,
    rag_chunk_index_directory_options,
    rag_perf_rollup_caption,
    remove_book_copy_from_current_read_folder as core_remove_book_copy_from_current_read_folder,
    save_currently_reading,
    select_rag_auto_profile,
)

PATHS = BookmapPaths(repo_root=PROJECT_ROOT)
DEFAULT_INDEX_DIR = PATHS.semantic_index
DEFAULT_CHUNK_INDEX_DIR = PATHS.semantic_index_chunks
DEFAULT_CHUNK_INDEX_DIR_GTE_LARGE = PATHS.semantic_index_chunks_gte_large
DEFAULT_COVER_CACHE_DIR = PATHS.cover_cache
DEFAULT_READING_LIST_PATH = PATHS.reading_list
DEFAULT_DAILY_RECOMMENDATIONS_PATH = PATHS.daily_recommendations
DEFAULT_CURRENT_READ_BOOKS_DIR = PATHS.current_read_books_dir


def _rag_chunk_index_directory_options() -> List[str]:
    return rag_chunk_index_directory_options(PATHS)


@st.cache_resource
def load_service(index_dir: str) -> SemanticSearchService:
    return SemanticSearchService(Path(index_dir))


@st.cache_resource
def load_rag_service(index_dir: str) -> RagService:
    return RagService(Path(index_dir))


@st.cache_data(show_spinner=False)
def get_file_dates(file_path: str) -> Tuple[Optional[date], Optional[date]]:
    return core_get_file_dates(file_path)


@st.cache_data(show_spinner=False)
def build_cover_thumbnail(source_path: str, cache_dir: str, max_width: int = 260) -> Optional[str]:
    return core_build_cover_thumbnail(source_path, cache_dir, max_width)


def copy_book_to_current_read_folder(entry: Dict[str, Any]) -> Tuple[bool, str]:
    return core_copy_book_to_current_read_folder(entry, PATHS)


def remove_book_copy_from_current_read_folder(entry: Dict[str, Any]) -> Tuple[bool, str]:
    return core_remove_book_copy_from_current_read_folder(entry, PATHS)


def render_result_card(item: dict, cover_cache_dir: str) -> None:
    left_col, right_col = st.columns([1, 3], vertical_alignment="top")

    with left_col:
        cover_path = build_cover_thumbnail(item.get("absolute_path", ""), cover_cache_dir)
        if cover_path:
            st.image(cover_path, use_container_width=True)
        else:
            st.caption("No cover preview")

    with right_col:
        st.markdown(f"### {item.get('title', 'Untitled')}")
        st.write(
            f"Category: **{item.get('category', 'Other')}** | "
            f"Mode: **{item.get('learning_mode', 'unknown')}** | "
            f"Confidence: **{item.get('confidence', 0.0):.3f}** | "
            f"Similarity: **{item.get('similarity', 0.0):.3f}**"
        )
        keywords = item.get("matched_keywords") or []
        if keywords:
            st.caption("Matched keywords: " + ", ".join(keywords[:8]))
        st.code(item.get("absolute_path", ""), language="text")


def render_result_grid(
    items: List[dict],
    cover_cache_dir: str,
    columns_per_row: int,
    button_key_prefix: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_path: Path,
) -> None:
    if not items:
        return
    safe_cols = max(1, columns_per_row)
    for start in range(0, len(items), safe_cols):
        row_items = items[start : start + safe_cols]
        row_cols = st.columns(safe_cols)
        for col_idx, item in enumerate(row_items):
            with row_cols[col_idx]:
                with st.container(border=True):
                    cover_path = build_cover_thumbnail(item.get("absolute_path", ""), cover_cache_dir)
                    if cover_path:
                        st.image(cover_path, use_container_width=True)
                    else:
                        st.caption("No cover preview")
                    title = html.escape(_card_title(item.get("title", "Untitled")))
                    book_format = get_book_format(item)
                    meta = html.escape(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"{book_format} | "
                        f"sim {item.get('similarity', 0.0):.3f}"
                    )
                    st.markdown(f"<div class='book-card-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-card-meta'>{meta}</div>", unsafe_allow_html=True)
                    button_left, button_right = st.columns(2)
                    with button_left:
                        if st.button("View Details", key=f"{button_key_prefix}-{start}-{col_idx}-{item.get('book_id')}"):
                            ok, message = open_pdf_in_file_manager(item.get("absolute_path", ""))
                            if ok:
                                st.toast(message, icon="📂")
                            else:
                                st.warning(message)
                            st.session_state["selected_book_id"] = item.get("book_id")
                    with button_right:
                        book_id = str(item.get("book_id", ""))
                        is_reading = book_id in reading_items
                        label = "Remove Reading" if is_reading else "Mark Reading"
                        if st.button(label, key=f"{button_key_prefix}-reading-{start}-{col_idx}-{book_id}"):
                            if not book_id:
                                st.warning("Book ID missing, cannot update reading list.")
                            elif is_reading:
                                removed_entry = reading_items.pop(book_id, None)
                                if isinstance(removed_entry, dict):
                                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                                    if not removed_ok:
                                        st.warning(removed_message)
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()
                            else:
                                entry = make_reading_entry(item)
                                copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                                reading_items[book_id] = entry
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Added to currently reading.", icon="📘")
                                if not copy_ok:
                                    st.warning(copy_message)
                                st.rerun()


def render_currently_reading_page(
    cover_cache_dir: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_path: Path,
    cards_per_row: int,
) -> None:
    st.header("Currently Reading")
    if not reading_items:
        st.info("No books added yet. Use 'Mark Reading' from search or related books.")
        return

    entries_all = sorted(
        reading_items.values(),
        key=lambda item: item.get("added_at", ""),
    )
    all_formats = sorted({get_book_format(item) for item in entries_all} | {"PDF", "EPUB"})
    selected_formats = st.multiselect("Format", all_formats, default=all_formats, key="reading-formats")
    entries = [item for item in entries_all if not selected_formats or get_book_format(item) in set(selected_formats)]
    if not entries:
        st.info("No currently reading books match selected formats.")
        return
    helper_left, helper_right = st.columns([1, 3], vertical_alignment="center")
    with helper_left:
        st.markdown("**NotebookLM Upload**")
        if st.button(
            "📓 Open NotebookLM to Upload",
            key="reading-open-notebooklm",
            type="primary",
            use_container_width=True,
        ):
            ok, message = open_notebooklm_in_browser()
            if ok:
                st.toast(message, icon="📓")
            else:
                st.warning(message)
    with helper_right:
        st.markdown(
            "**NotebookLM manual upload flow**\n"
            "- Click `📓 Open NotebookLM to Upload`\n"
            "- Click `Open` on the book card to reveal the file location\n"
            "- In NotebookLM: `Add source` -> `Upload`, then pick the file from that folder\n"
            "- Limits are typically up to `200MB` or `500,000 words` per source"
        )

    safe_cols = max(1, cards_per_row)
    for row_start in range(0, len(entries), safe_cols):
        row_items = entries[row_start : row_start + safe_cols]
        row_cols = st.columns(safe_cols)
        for col_idx, item in enumerate(row_items):
            with row_cols[col_idx]:
                with st.container(border=True):
                    cover_path = build_cover_thumbnail(str(item.get("absolute_path", "")), cover_cache_dir)
                    if cover_path:
                        st.image(cover_path, use_container_width=True)

                    st.markdown(f"#### {_card_title(item.get('title', 'Untitled'))}")
                    st.caption(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"{get_book_format(item)} | "
                        f"{days_reading(str(item.get('added_at', '')))} days"
                    )
                    st.caption(f"Progress: {coerce_progress(item.get('progress_pct', 0))}%")
                    book_id = str(item.get("book_id", ""))
                    slider_value = st.slider(
                        "Progress",
                        min_value=0,
                        max_value=100,
                        value=coerce_progress(item.get("progress_pct", 0)),
                        key=f"reading-progress-slider-{row_start}-{col_idx}-{book_id}",
                    )
                    if book_id and slider_value != coerce_progress(item.get("progress_pct", 0)):
                        reading_items[book_id]["progress_pct"] = slider_value
                        save_currently_reading(reading_path, reading_items)
                        st.toast("Progress updated.", icon="📈")

                    action_left, action_right = st.columns(2)
                    with action_left:
                        if st.button("Open", key=f"reading-open-{row_start}-{col_idx}-{book_id}"):
                            ok, message = open_pdf_in_file_manager(str(item.get("absolute_path", "")))
                            if ok:
                                st.toast(message, icon="📂")
                            else:
                                st.warning(message)
                    with action_right:
                        if st.button("Remove", key=f"reading-remove-{row_start}-{col_idx}-{book_id}"):
                            if book_id:
                                removed_entry = reading_items.pop(book_id, None)
                                if isinstance(removed_entry, dict):
                                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                                    if not removed_ok:
                                        st.warning(removed_message)
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()


def render_locked_paths_sidebar() -> Tuple[str, str, Path]:
    st.sidebar.markdown("---")
    index_dir = st.sidebar.text_input(
        "Semantic index directory",
        str(DEFAULT_INDEX_DIR),
        disabled=True,
    )
    cover_cache_dir = st.sidebar.text_input(
        "Cover cache directory",
        str(DEFAULT_COVER_CACHE_DIR),
        disabled=True,
    )
    reading_list_path = Path(
        st.sidebar.text_input(
            "Reading list file",
            str(DEFAULT_READING_LIST_PATH),
            disabled=True,
        )
    )
    return index_dir, cover_cache_dir, reading_list_path


def render_library_page(service: SemanticSearchService) -> None:
    st.header("Library")
    items = service.metadata
    if not items:
        st.info("No books found in semantic metadata.")
        return

    all_categories = sorted({str(item.get("category", "Other")) for item in items})
    all_formats = sorted(
        {
            (Path(str(item.get("absolute_path", ""))).suffix.lower().lstrip(".") or "unknown").upper()
            for item in items
        }
        | {"PDF", "EPUB"}
    )
    name_filter = st.text_input("Filter by name", value="")
    st.sidebar.header("Library Filters")
    selected_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories)
    selected_formats = st.sidebar.multiselect("Format", all_formats, default=all_formats)

    today = date.today()
    created_start, created_end = st.sidebar.date_input(
        "Created date range",
        value=(date(2000, 1, 1), today),
    )
    updated_start, updated_end = st.sidebar.date_input(
        "Updated date range",
        value=(date(2000, 1, 1), today),
    )

    name_norm = name_filter.strip().lower()
    rows: List[Dict[str, Any]] = []
    for item in items:
        title = str(item.get("title", "Untitled"))
        category = str(item.get("category", "Other"))
        path = str(item.get("absolute_path", ""))
        filename = Path(path).name
        fmt = (Path(path).suffix.lower().lstrip(".") or "unknown").upper()
        if selected_categories and category not in selected_categories:
            continue
        if selected_formats and fmt not in selected_formats:
            continue
        if name_norm and all(
            name_norm not in text
            for text in (title.lower(), filename.lower(), path.lower())
        ):
            continue

        created_date, updated_date = get_file_dates(path)
        if created_date is None or updated_date is None:
            continue
        if created_date < created_start or created_date > created_end:
            continue
        if updated_date < updated_start or updated_date > updated_end:
            continue

        rows.append(
            {
                "Title": title,
                "Format": fmt,
                "Category": category,
                "Created Date": created_date.isoformat(),
                "Updated Date": updated_date.isoformat(),
                "Path": path,
            }
        )

    st.subheader(f"Books ({len(rows)})")
    if not rows:
        st.info("No books match your filters.")
        return

    df = pd.DataFrame(rows).sort_values(by="Updated Date", ascending=False, kind="stable").reset_index(drop=True)
    # Show approximately 40 rows before internal scrolling.
    row_height_px = 35
    header_height_px = 40
    target_visible_rows = 40
    table_height = header_height_px + row_height_px * min(max(1, len(df)), target_visible_rows)

    table_col, preview_col = st.columns([3, 1], vertical_alignment="top")
    selected_row_idx: Optional[int] = None
    with table_col:
        try:
            table_state = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=table_height,
                on_select="rerun",
                selection_mode="single-cell",
                key="library_table",
            )
            if isinstance(table_state, dict):
                selection = table_state.get("selection", {})
                selected_rows = selection.get("rows", [])
                if selected_rows:
                    selected_row_idx = int(selected_rows[0])
                else:
                    selected_cells = selection.get("cells", [])
                    if selected_cells:
                        first_cell = selected_cells[0]
                        if isinstance(first_cell, dict):
                            row_val = first_cell.get("row")
                            if isinstance(row_val, int):
                                selected_row_idx = row_val
                        elif isinstance(first_cell, (list, tuple)) and first_cell:
                            if isinstance(first_cell[0], int):
                                selected_row_idx = int(first_cell[0])
        except TypeError:
            # Fallback for Streamlit versions that do not support row selection in st.dataframe.
            st.dataframe(df, use_container_width=True, hide_index=True, height=table_height)

    with preview_col:
        st.subheader("Cover Preview")
        if selected_row_idx is None:
            st.caption("Select a row to preview the cover.")
        elif selected_row_idx < 0 or selected_row_idx >= len(df):
            st.caption("Invalid selection.")
        else:
            row = df.iloc[selected_row_idx].to_dict()
            cover_path = build_cover_thumbnail(str(row.get("Path", "")), str(DEFAULT_COVER_CACHE_DIR))
            if cover_path:
                st.image(cover_path, width="stretch")
            else:
                st.caption("No cover preview available.")
            st.caption(str(row.get("Title", "Untitled")))
            if st.button("Open Location", key=f"library-open-{selected_row_idx}"):
                ok, message = open_pdf_in_file_manager(str(row.get("Path", "")))
                if ok:
                    st.toast(message, icon="📂")
                else:
                    st.warning(message)


def render_daily_recommendations_page(
    service: SemanticSearchService,
    cover_cache_dir: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_list_path: Path,
    weights: DailyRecommendationWeights,
) -> None:
    st.header("Today's 18 Recommendations")
    st.subheader("Daily Weights (Read-only)")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        st.text_input("Similarity weight", f"{weights.similarity:.2f}", disabled=True)
        st.text_input("Novelty weight", f"{weights.novelty:.2f}", disabled=True)
    with w_col2:
        st.text_input("Freshness weight", f"{weights.freshness:.2f}", disabled=True)
        st.text_input("Confidence weight", f"{weights.confidence:.2f}", disabled=True)
    with w_col3:
        st.text_input("Diversity penalty", f"{weights.diversity_penalty:.2f}", disabled=True)
        st.text_input("Explore bonus", f"{weights.explore_bonus:.2f}", disabled=True)

    recommender = DailyBookRecommender(
        service=service,
        reading_list_path=reading_list_path,
        history_path=DEFAULT_DAILY_RECOMMENDATIONS_PATH,
        weights=weights,
    )

    refresh = st.button("Refresh for Today")
    today = datetime.now().astimezone().date()
    recommendations = recommender.get_or_generate_for_date(
        target_date=today,
        count=18,
        force_refresh=refresh,
    )
    all_formats = sorted({get_book_format(item) for item in recommendations} | {"PDF", "EPUB"})
    selected_formats = st.multiselect(
        "Format",
        all_formats,
        default=all_formats,
        key="daily-formats",
    )
    surface_epubs = st.checkbox(
        "Surface EPUB books in daily list",
        value=True,
        key="daily-surface-epubs",
    )

    if selected_formats:
        allowed = set(selected_formats)
        recommendations = [item for item in recommendations if get_book_format(item) in allowed]
    if surface_epubs and (not selected_formats or "EPUB" in selected_formats):
        recommendations = blend_results_to_surface_epubs(recommendations, top_k=18)

    st.caption(f"Date: {today.isoformat()} | History file: {DEFAULT_DAILY_RECOMMENDATIONS_PATH}")
    if not recommendations:
        st.info("No recommendations available right now.")
        return

    cards = recommendations[:18]
    cards_per_row = 6
    for row_start in range(0, len(cards), cards_per_row):
        row_items = cards[row_start : row_start + cards_per_row]
        row_cols = st.columns(cards_per_row)
        for col_idx, item in enumerate(row_items):
            card_idx = row_start + col_idx
            with row_cols[col_idx]:
                with st.container(border=True):
                    cover_path = build_cover_thumbnail(item.get("absolute_path", ""), cover_cache_dir)
                    if cover_path:
                        st.image(cover_path, use_container_width=True)
                    else:
                        st.caption("No cover preview")
                    st.markdown(f"#### {_card_title(item.get('title', 'Untitled'))}")
                    st.caption(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"{get_book_format(item)} | "
                        f"{item.get('daily_strategy', 'exploit')}"
                    )
                    st.caption(f"Daily score: {float(item.get('daily_score', 0.0) or 0.0):.3f}")
                    reasons = item.get("daily_reasons", {}) or {}
                    if isinstance(reasons, dict):
                        st.caption(
                            "sim "
                            f"{float(reasons.get('similarity', 0.0) or 0.0):.2f}, "
                            "fresh "
                            f"{float(reasons.get('freshness', 0.0) or 0.0):.2f}, "
                            "novel "
                            f"{float(reasons.get('novelty', 0.0) or 0.0):.2f}"
                        )

                    btn_left, btn_right = st.columns(2)
                    with btn_left:
                        if st.button("Open", key=f"daily-open-{card_idx}-{item.get('book_id')}"):
                            ok, message = open_pdf_in_file_manager(item.get("absolute_path", ""))
                            if ok:
                                st.toast(message, icon="📂")
                            else:
                                st.warning(message)
                    with btn_right:
                        book_id = str(item.get("book_id", ""))
                        is_reading = book_id in reading_items
                        label = "Remove" if is_reading else "Mark"
                        if st.button(label, key=f"daily-reading-{card_idx}-{book_id}"):
                            if not book_id:
                                st.warning("Book ID missing, cannot update reading list.")
                            elif is_reading:
                                removed_entry = reading_items.pop(book_id, None)
                                if isinstance(removed_entry, dict):
                                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                                    if not removed_ok:
                                        st.warning(removed_message)
                                save_currently_reading(reading_list_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()
                            else:
                                entry = make_reading_entry(item)
                                copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                                reading_items[book_id] = entry
                                save_currently_reading(reading_list_path, reading_items)
                                st.toast("Added to currently reading.", icon="📘")
                                if not copy_ok:
                                    st.warning(copy_message)
                                st.rerun()


def _apply_rag_retrieval_preset(preset_name: str) -> None:
    apply_rag_retrieval_preset_to_session(preset_name, st.session_state)


def _apply_rag_performance_profile(profile_name: str) -> None:
    apply_rag_performance_profile_to_session(profile_name, st.session_state)


def _render_rag_chat_response(
    turn_idx: int,
    response: Dict[str, Any],
    show_debug: bool,
    blur_meta_text: bool,
    show_fallback_notice: bool,
) -> None:
    st.caption(f"Generation mode: {response.get('generation_mode', 'deterministic')}")
    metrics = response.get("metrics", {}) or {}
    if isinstance(metrics, dict) and metrics:
        total_ms = float(metrics.get("total_ms", 0.0) or 0.0)
        retrieval_ms = float(metrics.get("retrieval_ms", 0.0) or 0.0)
        generation_ms = float(metrics.get("generation_ms", 0.0) or 0.0)
        peak_rss_mb = float(metrics.get("peak_rss_mb", 0.0) or 0.0)
        top_relevance = float(metrics.get("top_relevance_score", 0.0) or 0.0)
        citation_coverage = float(metrics.get("citation_coverage_ratio", 0.0) or 0.0)
        st.caption(
            "Timing: "
            f"total {total_ms:.1f} ms | retrieval {retrieval_ms:.1f} ms | generation {generation_ms:.1f} ms | "
            f"peak RSS {peak_rss_mb:.1f} MB | top relevance {top_relevance:.3f} | citation coverage {citation_coverage:.2f}"
        )
        with st.expander("Answer diagnostics", expanded=False):
            st.json(metrics)
    fallback_reason = str(response.get("fallback_reason", "") or "").strip()
    if fallback_reason and show_fallback_notice:
        st.info(f"Fallback used: {fallback_reason}")

    st.markdown("**Answer**")
    _render_answer_with_blur(str(response.get("answer", "") or ""), hide_meta_text=blur_meta_text)
    st.markdown("**Summary**")
    st.write(response.get("summary", ""))

    follow_ups = response.get("follow_ups", []) or []
    if follow_ups:
        st.markdown("**Suggested follow-ups**")
        for idx, prompt in enumerate(follow_ups):
            if st.button(str(prompt), key=f"rag-followup-{turn_idx}-{idx}"):
                st.session_state["rag-pending-question"] = str(prompt)
                st.rerun()

    citations = response.get("citations", []) or []
    st.markdown(f"**Citations ({len(citations)})**")
    if not citations:
        st.info("No citations found for this question.")
    for idx, item in enumerate(citations):
        title = str(item.get("title", "Untitled"))
        label = (
            f"{title} | {item.get('category', 'Other')} | "
            f"{item.get('learning_mode', 'unknown')} | "
            f"{item.get('source_label', 'chunk')} | "
            f"sim {float(item.get('similarity', 0.0) or 0.0):.3f}"
        )
        with st.expander(label, expanded=False):
            st.write(str(item.get("snippet", "")))
            if st.button(
                "Open Source Location",
                key=f"rag-open-source-{turn_idx}-{idx}-{item.get('book_id', '')}",
            ):
                ok, message = open_pdf_in_file_manager(str(item.get("absolute_path", "")))
                if ok:
                    st.toast(message, icon="📂")
                else:
                    st.warning(message)
            if show_debug:
                st.json(
                    {
                        "citation_id": item.get("citation_id", ""),
                        "book_id": item.get("book_id", ""),
                        "absolute_path": item.get("absolute_path", ""),
                        "start_char": item.get("start_char", 0),
                        "end_char": item.get("end_char", 0),
                        "chunk_order": item.get("chunk_order", 0),
                        "chunk_len": item.get("chunk_len", 0),
                    }
                )

    if show_debug:
        with st.expander("Debug response payload", expanded=False):
            st.json(response)


def _render_answer_with_blur(
    answer_text: str,
    *,
    placeholder: Optional[Any] = None,
    show_cursor: bool = False,
    hide_meta_text: bool = True,
) -> None:
    render_text = format_answer_markdown_for_display(
        str(answer_text or ""),
        hide_meta_text=hide_meta_text,
        show_cursor=show_cursor,
    )
    if placeholder is None:
        st.markdown(render_text)
    else:
        placeholder.markdown(render_text)


def _append_rag_metrics(question: str, response: Dict[str, Any]) -> None:
    st.session_state["rag-metrics-history"] = append_rag_metrics_row(
        st.session_state.get("rag-metrics-history", []),
        question,
        response,
    )


def _collect_recent_rag_metrics(chat_history: List[Dict[str, Any]], window: int = 10) -> List[Dict[str, Any]]:
    return collect_recent_rag_metrics(
        metrics_history=st.session_state.get("rag-metrics-history"),
        chat_history=chat_history,
        window=window,
    )


def _render_rag_perf_rollup(chat_history: List[Dict[str, Any]], window: int = 10) -> None:
    cap = rag_perf_rollup_caption(
        chat_history,
        st.session_state.get("rag-metrics-history"),
        window=window,
    )
    if cap:
        st.caption(cap)


def render_rag_metrics_page() -> None:
    st.header("RAG Metrics")
    st.caption("In-memory performance charts for the last 10 answers. Metrics reset when the app restarts.")

    chat_history = st.session_state.get("rag-chat-history", [])
    if not isinstance(chat_history, list):
        chat_history = []
    rows = _collect_recent_rag_metrics(chat_history=chat_history, window=10)
    if not rows:
        st.info(
            "No recent answer metrics yet. Go to Ask Books (RAG), submit questions, then return to this page."
        )
        return

    frame = pd.DataFrame(rows)
    avg_total = float(frame["total_ms"].mean())
    avg_retrieval = float(frame["retrieval_ms"].mean())
    avg_generation = float(frame["generation_ms"].mean())
    max_rss = float(frame["peak_rss_mb"].max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg total (ms)", f"{avg_total:.1f}")
    c2.metric("Avg retrieval (ms)", f"{avg_retrieval:.1f}")
    c3.metric("Avg generation (ms)", f"{avg_generation:.1f}")
    c4.metric("Max peak RSS (MB)", f"{max_rss:.1f}")

    export_cols = [
        "answer_idx",
        "query",
        "total_ms",
        "retrieval_ms",
        "generation_ms",
        "peak_rss_mb",
        "retrieved_chunks",
        "used_citations",
        "citation_coverage_ratio",
        "top_similarity",
        "top_relevance_score",
        "prompt_chars",
        "answer_chars",
    ]
    export_df = frame[export_cols].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download last 10 metrics (CSV)",
        data=csv_bytes,
        file_name="rag_last10_metrics.csv",
        mime="text/csv",
        key="rag-metrics-download-csv",
        help="Exports only in-memory metrics from the current app session.",
    )

    st.subheader("Latency Trend (Last 10 Answers)")
    if go is not None:
        fig_latency = go.Figure()
        fig_latency.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["total_ms"],
                mode="lines+markers",
                name="total_ms",
            )
        )
        fig_latency.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["retrieval_ms"],
                mode="lines+markers",
                name="retrieval_ms",
            )
        )
        fig_latency.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["generation_ms"],
                mode="lines+markers",
                name="generation_ms",
            )
        )
        fig_latency.update_layout(
            xaxis_title="Answer # (recent)",
            yaxis_title="Milliseconds",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_latency, use_container_width=True)
    else:
        st.line_chart(
            frame.set_index("answer_idx")[["total_ms", "retrieval_ms", "generation_ms"]],
            use_container_width=True,
        )

    st.subheader("Resource and Payload Signals")
    if go is not None:
        fig_resource = go.Figure()
        fig_resource.add_trace(
            go.Bar(
                x=frame["answer_idx"],
                y=frame["peak_rss_mb"],
                name="peak_rss_mb",
            )
        )
        fig_resource.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["retrieved_chunks"],
                mode="lines+markers",
                name="retrieved_chunks",
                yaxis="y2",
            )
        )
        fig_resource.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["used_citations"],
                mode="lines+markers",
                name="used_citations",
                yaxis="y2",
            )
        )
        fig_resource.update_layout(
            xaxis_title="Answer # (recent)",
            yaxis=dict(title="Peak RSS (MB)"),
            yaxis2=dict(title="Counts", overlaying="y", side="right"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_resource, use_container_width=True)
    else:
        st.bar_chart(frame.set_index("answer_idx")[["peak_rss_mb"]], use_container_width=True)
        st.line_chart(frame.set_index("answer_idx")[["retrieved_chunks", "used_citations"]], use_container_width=True)

    st.subheader("Grounding Quality Signals")
    if go is not None:
        fig_quality = go.Figure()
        fig_quality.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["top_relevance_score"],
                mode="lines+markers",
                name="top_relevance_score",
            )
        )
        fig_quality.add_trace(
            go.Scatter(
                x=frame["answer_idx"],
                y=frame["citation_coverage_ratio"],
                mode="lines+markers",
                name="citation_coverage_ratio",
            )
        )
        fig_quality.update_layout(
            xaxis_title="Answer # (recent)",
            yaxis_title="Score (0-1)",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    else:
        st.line_chart(
            frame.set_index("answer_idx")[["top_relevance_score", "citation_coverage_ratio"]],
            use_container_width=True,
        )

    with st.expander("Last 10 metrics table", expanded=False):
        st.dataframe(
            export_df,
            use_container_width=True,
            hide_index=True,
        )


def render_ask_books_rag_page(
    rag_service: RagService,
    all_categories: List[str],
    all_modes: List[str],
) -> None:
    st.header("Ask Books (RAG)")
    st.caption("Grounded answers from your PDF/EPUB folder with source citations.")

    if "rag-chat-history" not in st.session_state:
        st.session_state["rag-chat-history"] = []
    if "rag-metrics-history" not in st.session_state:
        st.session_state["rag-metrics-history"] = []
    if "rag-pending-question" not in st.session_state:
        st.session_state["rag-pending-question"] = ""
    if "rag-pinned-preset" not in st.session_state:
        st.session_state["rag-pinned-preset"] = ""

    with st.sidebar:
        st.subheader("Conversation Controls")
        preset_names = list(RAG_RETRIEVAL_PRESETS.keys())
        default_preset_name = str(st.session_state.get("rag-pinned-preset", "") or "Definition Q&A")
        if default_preset_name not in RAG_RETRIEVAL_PRESETS:
            default_preset_name = "Definition Q&A"
        preset_idx = preset_names.index(default_preset_name)
        chosen_preset = st.selectbox(
            "Retrieval preset",
            options=preset_names,
            index=preset_idx,
            key="rag-preset-choice",
        )
        controls_col1, controls_col2, controls_col3 = st.columns(3)
        with controls_col1:
            if st.button("Apply preset", key="rag-apply-preset"):
                _apply_rag_retrieval_preset(chosen_preset)
                st.rerun()
        with controls_col2:
            pinned = str(st.session_state.get("rag-pinned-preset", "") or "")
            pin_label = "Unpin preset" if pinned == chosen_preset else "Pin preset"
            if st.button(pin_label, key="rag-pin-preset"):
                if pinned == chosen_preset:
                    st.session_state["rag-pinned-preset"] = ""
                else:
                    st.session_state["rag-pinned-preset"] = chosen_preset
                st.rerun()
        with controls_col3:
            if st.button("Clear chat", key="rag-clear-chat"):
                st.session_state["rag-chat-history"] = []
                st.session_state["rag-pending-question"] = ""
                st.rerun()

        profile_names = ["Auto"] + list(RAG_PERFORMANCE_PROFILES.keys())
        selected_profile = st.selectbox(
            "Performance profile",
            options=profile_names,
            index=0,
            key="rag-performance-profile",
            help=(
                "Auto routes query complexity to Fast/Balanced/Quality. "
                f"Target interactive latency: <= {int(RAG_LATENCY_TARGET_MS)} ms."
            ),
        )
        if st.button("Apply performance profile", key="rag-apply-performance-profile"):
            if selected_profile == "Auto":
                st.session_state["rag-auto-profile-enabled"] = True
            else:
                st.session_state["rag-auto-profile-enabled"] = False
                _apply_rag_performance_profile(selected_profile)
            st.rerun()

        _render_rag_perf_rollup(st.session_state.get("rag-chat-history", []), window=10)

        top_k_chunks = st.slider("Top chunks", min_value=4, max_value=20, value=8, step=2, key="rag-top-k-chunks")
        max_citations = st.slider("Max citations", min_value=2, max_value=10, value=6, step=1, key="rag-max-citations")
        min_similarity = st.slider(
            "Min chunk similarity",
            min_value=-1.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            key="rag-min-similarity",
        )
        show_debug = st.toggle("Show retrieval debug details", value=False, key="rag-show-debug")
        blur_meta_text = st.toggle(
            "Hide model thinking/meta text",
            value=True,
            key="rag-blur-meta-text",
            help="When enabled, reasoning/meta lines such as SourcesUsed/Summary are hidden from answer text.",
        )
        show_fallback_notice = st.toggle(
            "Show fallback notices",
            value=True,
            key="rag-show-fallback-notice",
            help="Show/hide messages like: 'Fallback used: Generated answer missing valid citation markers.'",
        )
        disable_fallback = st.toggle(
            "Disable deterministic fallback (advanced)",
            value=False,
            key="rag-disable-fallback",
            help=(
                "If enabled, app will keep raw model output even when citation markers are invalid. "
                "Use only for debugging."
            ),
        )

        st.subheader("Execution Mode")
        execution_mode = st.radio(
            "Execution path",
            ["Direct (local RagService)", "API (/rag/answer)"],
            horizontal=True,
            key="rag-execution-mode",
        )
        api_answer_url = st.text_input(
            "API answer URL",
            value="http://127.0.0.1:8000/rag/answer",
            key="rag-api-answer-url",
            disabled=execution_mode != "API (/rag/answer)",
        )
        api_timeout_sec = st.slider(
            "API timeout (seconds)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            key="rag-api-timeout-sec",
            disabled=execution_mode != "API (/rag/answer)",
        )
        default_api_key = str(os.getenv("RAG_API_KEY", "") or "")
        api_key = st.text_input(
            "API key (X-API-Key)",
            value=default_api_key,
            key="rag-api-key",
            type="password",
            disabled=execution_mode != "API (/rag/answer)",
        )

        st.subheader("Advanced Retrieval")
        use_hybrid = st.toggle("Enable hybrid retrieval (dense + lexical)", value=True, key="rag-hybrid-enabled")
        dense_weight = st.slider(
            "Dense weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="rag-dense-weight",
            disabled=not use_hybrid,
        )
        lexical_weight = st.slider(
            "Lexical weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="rag-lexical-weight",
            disabled=not use_hybrid,
        )
        candidate_pool_size = st.slider(
            "Candidate pool size",
            min_value=8,
            max_value=128,
            value=48,
            step=4,
            key="rag-candidate-pool",
        )
        reranker_enabled = st.toggle("Enable reranker", value=True, key="rag-reranker-enabled")
        current_reranker_model = str(
            st.session_state.get("rag-reranker-model", DEFAULT_RERANKER_MODEL) or ""
        ).strip()
        known_rerankers = [item for item in RERANKER_MODEL_OPTIONS if item != "Custom"]
        default_reranker_choice = (
            current_reranker_model if current_reranker_model in known_rerankers else "Custom"
        )
        reranker_choice = st.selectbox(
            "Reranker model preset",
            options=RERANKER_MODEL_OPTIONS,
            index=RERANKER_MODEL_OPTIONS.index(default_reranker_choice),
            key="rag-reranker-model-preset",
            disabled=not reranker_enabled,
        )
        if reranker_choice == "Custom":
            reranker_model = st.text_input(
                "Reranker model name (CrossEncoder)",
                value=(current_reranker_model if current_reranker_model not in known_rerankers else ""),
                key="rag-reranker-model",
                disabled=not reranker_enabled,
            )
        else:
            reranker_model = reranker_choice
            st.session_state["rag-reranker-model"] = reranker_model
        reranker_top_n = st.slider(
            "Reranker top-N",
            min_value=4,
            max_value=64,
            value=24,
            step=4,
            key="rag-reranker-topn",
            disabled=not reranker_enabled,
        )

        st.subheader("Generation")
        # Default new sessions to Ollama mode for grounded generation.
        st.session_state.setdefault("rag-generation-mode", "ollama")
        generation_mode = st.radio(
            "Answer mode",
            ["deterministic", "llama.cpp", "ollama"],
            index=2,
            horizontal=True,
            key="rag-generation-mode",
        )
        llama_model_path = st.text_input(
            "llama.cpp model path (.gguf)",
            value="",
            key="rag-llama-model-path",
            disabled=generation_mode != "llama.cpp",
        )
        llama_n_ctx = st.slider(
            "llama.cpp context window",
            min_value=512,
            max_value=8192,
            value=2048,
            step=256,
            key="rag-llama-n-ctx",
            disabled=generation_mode != "llama.cpp",
        )
        llama_max_tokens = st.slider(
            "llama.cpp max output tokens",
            min_value=64,
            max_value=1024,
            value=420,
            step=32,
            key="rag-llama-max-tokens",
            disabled=generation_mode != "llama.cpp",
        )
        llama_temp = st.slider(
            "llama.cpp temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="rag-llama-temp",
            disabled=generation_mode != "llama.cpp",
        )
        llama_top_p = st.slider(
            "llama.cpp top_p",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            key="rag-llama-top-p",
            disabled=generation_mode != "llama.cpp",
        )
        llama_threads = st.slider(
            "llama.cpp threads",
            min_value=1,
            max_value=32,
            value=6,
            step=1,
            key="rag-llama-threads",
            disabled=generation_mode != "llama.cpp",
        )
        llama_gpu_layers = st.slider(
            "llama.cpp GPU layers",
            min_value=0,
            max_value=120,
            value=0,
            step=1,
            key="rag-llama-gpu-layers",
            disabled=generation_mode != "llama.cpp",
        )
        ollama_base_url = st.text_input(
            "Ollama base URL",
            value=str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            key="rag-ollama-base-url",
            disabled=generation_mode != "ollama",
        )
        ollama_model = st.text_input(
            "Ollama model tag",
            value="granite3.3:8b",
            key="rag-ollama-model",
            disabled=generation_mode != "ollama",
        )
        ollama_num_ctx = st.slider(
            "Ollama context window",
            min_value=512,
            max_value=32768,
            value=8192,
            step=256,
            key="rag-ollama-num-ctx",
            disabled=generation_mode != "ollama",
        )
        ollama_temp = st.slider(
            "Ollama temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="rag-ollama-temp",
            disabled=generation_mode != "ollama",
        )
        ollama_top_p = st.slider(
            "Ollama top_p",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            key="rag-ollama-top-p",
            disabled=generation_mode != "ollama",
        )
        ollama_timeout_sec = st.slider(
            "Ollama timeout (seconds)",
            min_value=5,
            max_value=600,
            value=180,
            step=5,
            key="rag-ollama-timeout",
            disabled=generation_mode != "ollama",
        )
        st.header("Ask Books Filters")
        selected_categories = st.multiselect("Category", all_categories, default=all_categories, key="rag-category")
        selected_modes = st.multiselect(
            "Theory vs Practical",
            all_modes,
            default=all_modes,
            format_func=lambda m: learning_mode_labels().get(m, m),
            key="rag-mode",
        )

    for turn_idx, turn in enumerate(st.session_state.get("rag-chat-history", [])):
        question = str(turn.get("question", "") or "")
        response = turn.get("response", {})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            _render_rag_chat_response(
                turn_idx=turn_idx,
                response=response,
                show_debug=show_debug,
                blur_meta_text=blur_meta_text,
                show_fallback_notice=show_fallback_notice,
            )

    pending_question = str(st.session_state.get("rag-pending-question", "") or "").strip()
    prompt_value = st.chat_input("Ask a grounded question about your books")
    query = str(prompt_value or "").strip() or pending_question
    st.session_state["rag-pending-question"] = ""

    if not query:
        return

    # Show the just-submitted prompt immediately while generation runs.
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        assistant_block = st.container()
        with assistant_block:
            stream_placeholder = st.empty()
            stream_status_placeholder = st.empty()
            stream_status_placeholder.caption("Generating answer...")

    effective_top_k_chunks = int(top_k_chunks)
    effective_max_citations = int(max_citations)
    effective_min_similarity = float(min_similarity)
    effective_candidate_pool_size = int(candidate_pool_size)
    effective_generation_mode = str(generation_mode)
    effective_ollama_model = str(ollama_model)
    effective_ollama_num_ctx = int(ollama_num_ctx)
    effective_ollama_temp = float(ollama_temp)
    effective_ollama_top_p = float(ollama_top_p)
    effective_ollama_timeout_sec = int(ollama_timeout_sec)
    effective_reranker_enabled = bool(reranker_enabled)
    effective_reranker_top_n = int(reranker_top_n)

    if bool(st.session_state.get("rag-auto-profile-enabled", False)):
        auto_profile = select_rag_auto_profile(query)
        profile = RAG_PERFORMANCE_PROFILES.get(auto_profile, {})
        effective_top_k_chunks = int(profile.get("top_k_chunks", effective_top_k_chunks))
        effective_max_citations = int(profile.get("max_citations", effective_max_citations))
        effective_min_similarity = float(profile.get("min_similarity", effective_min_similarity))
        effective_candidate_pool_size = int(profile.get("candidate_pool_size", effective_candidate_pool_size))
        effective_generation_mode = str(profile.get("generation_mode", effective_generation_mode))
        effective_ollama_model = str(profile.get("ollama_model", effective_ollama_model))
        effective_ollama_num_ctx = int(profile.get("ollama_num_ctx", effective_ollama_num_ctx))
        effective_ollama_temp = float(profile.get("ollama_temp", effective_ollama_temp))
        effective_ollama_top_p = float(profile.get("ollama_top_p", effective_ollama_top_p))
        effective_ollama_timeout_sec = int(profile.get("ollama_timeout_sec", effective_ollama_timeout_sec))
        effective_reranker_enabled = bool(profile.get("reranker_enabled", effective_reranker_enabled))
        effective_reranker_top_n = int(profile.get("reranker_top_n", effective_reranker_top_n))
        st.sidebar.caption(f"Auto profile selected: {auto_profile}")

    payload = build_rag_answer_payload(
        query=query,
        top_k_chunks=effective_top_k_chunks,
        max_citations=effective_max_citations,
        selected_categories=selected_categories,
        selected_modes=selected_modes,
        min_similarity=effective_min_similarity,
        use_hybrid=bool(use_hybrid),
        dense_weight=float(dense_weight),
        lexical_weight=float(lexical_weight),
        candidate_pool_size=effective_candidate_pool_size,
        reranker_enabled=effective_reranker_enabled,
        reranker_model=str(reranker_model),
        reranker_top_n=effective_reranker_top_n,
        generation_mode=effective_generation_mode,
        llama_model_path=str(llama_model_path),
        llama_n_ctx=int(llama_n_ctx),
        llama_max_tokens=int(llama_max_tokens),
        llama_temp=float(llama_temp),
        llama_top_p=float(llama_top_p),
        llama_threads=int(llama_threads),
        llama_gpu_layers=int(llama_gpu_layers),
        ollama_base_url=str(ollama_base_url),
        ollama_model=effective_ollama_model,
        ollama_temp=effective_ollama_temp,
        ollama_top_p=effective_ollama_top_p,
        ollama_num_ctx=effective_ollama_num_ctx,
        ollama_timeout_sec=effective_ollama_timeout_sec,
        allow_fallback=not disable_fallback,
    )

    with st.spinner("Generating grounded answer..."):
        streamed_text = ""
        try:
            if execution_mode == "API (/rag/answer)":
                response = call_rag_api_answer(
                    api_url=str(api_answer_url).strip(),
                    payload=payload,
                    timeout_sec=int(api_timeout_sec),
                    api_key=str(api_key),
                )
            else:
                filters = RagFilters(
                    categories=selected_categories or None,
                    learning_modes=selected_modes or None,
                    min_similarity=effective_min_similarity,
                )
                retrieval_config = RetrievalConfig(
                    hybrid_enabled=bool(use_hybrid),
                    dense_weight=float(dense_weight),
                    lexical_weight=float(lexical_weight),
                    candidate_pool_size=effective_candidate_pool_size,
                    final_top_k=effective_top_k_chunks,
                    reranker_enabled=effective_reranker_enabled,
                    reranker_model_name=str(reranker_model).strip() if effective_reranker_enabled else None,
                    reranker_top_n=effective_reranker_top_n,
                )
                llm_config = LlamaCppConfig(
                    enabled=effective_generation_mode == "llama.cpp",
                    model_path=str(llama_model_path).strip(),
                    n_ctx=int(llama_n_ctx),
                    max_tokens=int(llama_max_tokens),
                    temperature=float(llama_temp),
                    top_p=float(llama_top_p),
                    n_threads=int(llama_threads),
                    n_gpu_layers=int(llama_gpu_layers),
                )
                ollama_config = OllamaConfig(
                    enabled=effective_generation_mode == "ollama",
                    base_url=str(ollama_base_url).strip(),
                    model=effective_ollama_model.strip(),
                    temperature=effective_ollama_temp,
                    top_p=effective_ollama_top_p,
                    num_ctx=effective_ollama_num_ctx,
                    timeout_sec=effective_ollama_timeout_sec,
                )
                if effective_generation_mode != "ollama":
                    stream_status_placeholder.caption("Generating grounded answer...")

                def _on_token(token: str) -> None:
                    nonlocal streamed_text
                    streamed_text += str(token)
                    if effective_generation_mode == "ollama":
                        _render_answer_with_blur(
                            streamed_text,
                            placeholder=stream_placeholder,
                            show_cursor=True,
                            hide_meta_text=blur_meta_text,
                        )
                        stream_status_placeholder.caption("Generating answer...")

                response = rag_service.answer_question(
                    query=query,
                    filters=filters,
                    top_k=effective_top_k_chunks,
                    max_citations=effective_max_citations,
                    retrieval_config=retrieval_config,
                    llm_config=llm_config,
                    ollama_config=ollama_config,
                    on_token=_on_token if effective_generation_mode == "ollama" else None,
                    allow_fallback=not disable_fallback,
                )
        except Exception as exc:
            st.error(f"Could not generate answer: {exc}")
            return

    with assistant_block:
        stream_placeholder.empty()
        stream_status_placeholder.empty()
        _render_rag_chat_response(
            turn_idx=len(st.session_state.get("rag-chat-history", [])),
            response=response,
            show_debug=show_debug,
            blur_meta_text=blur_meta_text,
            show_fallback_notice=show_fallback_notice,
        )

    history = st.session_state.get("rag-chat-history", [])
    history.append({"question": query, "response": response})
    st.session_state["rag-chat-history"] = history[-30:]
    _append_rag_metrics(question=query, response=response)
    st.rerun()


def render_relationship_graph_page(
    service: SemanticSearchService,
    cover_cache_dir: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_list_path: Path,
    all_categories: List[str],
    all_modes: List[str],
) -> None:
    st.header("Relationship Graph")
    st.caption("Interactive graph of how books relate by semantic similarity.")
    if go is None:
        st.error("Plotly is not installed. Install dependencies from requirements.txt and restart.")
        return

    st.sidebar.header("Relationship Graph Filters")
    selected_categories = st.sidebar.multiselect(
        "Graph Category",
        all_categories,
        default=all_categories,
        key="graph-category",
    )
    selected_modes = st.sidebar.multiselect(
        "Graph Theory vs Practical",
        all_modes,
        default=all_modes,
        format_func=lambda m: learning_mode_labels().get(m, m),
        key="graph-mode",
    )

    graph_mode = st.radio(
        "Graph scope",
        ["Whole Library", "Focused"],
        horizontal=True,
        key="graph-scope-mode",
    )
    color_by = st.selectbox(
        "Color nodes by",
        ["Category", "Theory vs Practical"],
        index=0,
        key="graph-color-by",
    )
    min_edge_similarity = st.slider(
        "Minimum edge similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        key="graph-min-edge-sim",
    )
    neighbors_per_node = st.slider(
        "Neighbors per node",
        min_value=1,
        max_value=15,
        value=6,
        step=1,
        key="graph-neighbors-per-node",
    )
    max_nodes = st.slider(
        "Max nodes",
        min_value=30,
        max_value=600,
        value=220,
        step=10,
        key="graph-max-nodes",
    )

    filters = SearchFilters(
        categories=selected_categories or None,
        learning_modes=selected_modes or None,
        min_similarity=-1.0,
    )

    graph_payload: Dict[str, Any] = {"nodes": [], "edges": []}
    seed_ids: set[str] = set()
    if graph_mode == "Whole Library":
        if st.button("Build Whole-Library Graph", type="primary", key="graph-build-whole"):
            graph_payload = service.build_whole_relationship_graph(
                filters=filters,
                max_nodes=int(max_nodes),
                min_similarity=float(min_edge_similarity),
                neighbors_per_node=int(neighbors_per_node),
            )
            st.session_state["graph-payload"] = graph_payload
            st.session_state["graph-seed-ids"] = []
        elif "graph-payload" in st.session_state:
            graph_payload = st.session_state.get("graph-payload", {"nodes": [], "edges": []})
    else:
        query = st.text_input(
            "Focus query",
            value="deep learning theory foundations",
            key="graph-focus-query",
        )
        seed_title_filter = st.text_input(
            "Filter seed title",
            value="",
            key="graph-seed-filter",
            help="Use this to quickly narrow a seed title list.",
        )
        filtered_items = []
        for item in service.metadata:
            category = str(item.get("category", "Other"))
            mode = str(item.get("learning_mode", "unknown"))
            if selected_categories and category not in selected_categories:
                continue
            if selected_modes and mode not in selected_modes:
                continue
            title = str(item.get("title", "Untitled"))
            if seed_title_filter.strip() and seed_title_filter.strip().lower() not in title.lower():
                continue
            filtered_items.append(item)
            if len(filtered_items) >= 400:
                break

        seed_option_to_id = {"(none)": ""}
        for item in filtered_items:
            label = f"{item.get('title', 'Untitled')} | {item.get('category', 'Other')}"
            seed_option_to_id[label] = str(item.get("book_id", ""))

        selected_seed_label = st.selectbox(
            "Seed book (optional)",
            options=list(seed_option_to_id.keys()),
            index=0,
            key="graph-seed-book",
        )
        seed_book_id = seed_option_to_id.get(selected_seed_label, "")

        seed_top_k = st.slider(
            "Seed hits from query",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key="graph-seed-top-k",
        )
        neighbor_k = st.slider(
            "Neighbors around each seed",
            min_value=3,
            max_value=30,
            value=12,
            step=1,
            key="graph-neighbor-k",
        )
        if st.button("Build Focused Graph", type="primary", key="graph-build-focused"):
            graph_payload = service.build_focused_relationship_graph(
                query=query.strip() or None,
                seed_book_id=seed_book_id or None,
                filters=filters,
                seed_top_k=int(seed_top_k),
                neighbor_k=int(neighbor_k),
                max_nodes=int(max_nodes),
                min_similarity=float(min_edge_similarity),
                neighbors_per_node=int(neighbors_per_node),
            )
            st.session_state["graph-payload"] = graph_payload
            st.session_state["graph-seed-ids"] = graph_payload.get("seed_ids", []) or []
        elif "graph-payload" in st.session_state:
            graph_payload = st.session_state.get("graph-payload", {"nodes": [], "edges": []})
            st.session_state.setdefault("graph-seed-ids", [])

    nodes = graph_payload.get("nodes", []) or []
    edges = graph_payload.get("edges", []) or []
    if not nodes:
        st.info("Build a graph to view relationships.")
        return
    st.caption(f"Nodes: {len(nodes)} | Edges: {len(edges)}")

    seed_ids = set(st.session_state.get("graph-seed-ids", []) or [])
    selected_node_id = st.session_state.get("graph-selected-node-id")
    fig = build_relationship_figure(
        graph_payload=graph_payload,
        color_by=color_by,
        seed_ids=seed_ids,
        selected_node_id=selected_node_id,
    )
    if fig is None:
        st.warning("Could not build graph visualization.")
        return

    try:
        plot_state = st.plotly_chart(
            fig,
            use_container_width=True,
            key="relationship-graph-plot",
            on_select="rerun",
            selection_mode=("points",),
        )
    except TypeError:
        plot_state = st.plotly_chart(fig, use_container_width=True, key="relationship-graph-plot")

    clicked_node_id = extract_selected_graph_node_id(plot_state)
    if clicked_node_id:
        st.session_state["graph-selected-node-id"] = clicked_node_id
        selected_node_id = clicked_node_id

    node_by_id = {str(item.get("id", "")): item for item in nodes}
    selected_node = node_by_id.get(str(selected_node_id or ""))
    if not selected_node:
        st.caption("Click a node to inspect details and actions.")
        return

    st.subheader("Selected Book")
    details_col, action_col = st.columns([2, 1], vertical_alignment="top")
    with details_col:
        cover_path = build_cover_thumbnail(str(selected_node.get("absolute_path", "")), cover_cache_dir)
        if cover_path:
            st.image(cover_path, width="stretch")
        st.markdown(f"**{selected_node.get('title', 'Untitled')}**")
        st.caption(
            f"{selected_node.get('category', 'Other')} | "
            f"{selected_node.get('learning_mode', 'unknown')} | "
            f"{get_book_format(selected_node)}"
        )
    with action_col:
        if st.button("Open Source Location", key=f"graph-open-{selected_node_id}"):
            ok, message = open_pdf_in_file_manager(str(selected_node.get("absolute_path", "")))
            if ok:
                st.toast(message, icon="📂")
            else:
                st.warning(message)

        book_id = str(selected_node.get("id", ""))
        is_reading = book_id in reading_items
        label = "Remove Reading" if is_reading else "Mark Reading"
        if st.button(label, key=f"graph-reading-{selected_node_id}"):
            if not book_id:
                st.warning("Book ID missing, cannot update reading list.")
            elif is_reading:
                removed_entry = reading_items.pop(book_id, None)
                if isinstance(removed_entry, dict):
                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                    if not removed_ok:
                        st.warning(removed_message)
                save_currently_reading(reading_list_path, reading_items)
                st.toast("Removed from currently reading.", icon="📕")
                st.rerun()
            else:
                entry = make_reading_entry(
                    {
                        "book_id": book_id,
                        "title": selected_node.get("title", ""),
                        "category": selected_node.get("category", "Other"),
                        "learning_mode": selected_node.get("learning_mode", "unknown"),
                        "absolute_path": selected_node.get("absolute_path", ""),
                    }
                )
                copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                reading_items[book_id] = entry
                save_currently_reading(reading_list_path, reading_items)
                st.toast("Added to currently reading.", icon="📘")
                if not copy_ok:
                    st.warning(copy_message)
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="BookMap RAG", layout="wide")
    st.markdown(
        """
<style>
.book-card-title {
  min-height: 3.2em;
  max-height: 3.2em;
  overflow: hidden;
  font-weight: 600;
  margin: 0.35rem 0 0.2rem 0;
  line-height: 1.6em;
}
.book-card-meta {
  min-height: 2.8em;
  max-height: 2.8em;
  overflow: hidden;
  color: rgba(120, 120, 120, 1);
  font-size: 0.9rem;
  line-height: 1.4em;
  margin-bottom: 0.35rem;
}
.rag-muted-blur {
  filter: blur(2.6px);
  opacity: 0.58;
  user-select: none;
  transition: filter 120ms ease-in-out, opacity 120ms ease-in-out;
}
.rag-muted-blur:hover {
  filter: blur(0.8px);
  opacity: 0.72;
}
.rag-cursor {
  opacity: 0.8;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("BookMap RAG")

    index_dir = str(DEFAULT_INDEX_DIR)
    cover_cache_dir = str(DEFAULT_COVER_CACHE_DIR)
    reading_list_path = DEFAULT_READING_LIST_PATH
    daily_weights = DailyRecommendationWeights(
        similarity=0.55,
        freshness=0.2,
        novelty=0.15,
        confidence=0.1,
        diversity_penalty=0.1,
        explore_bonus=0.2,
    )
    view_page = st.sidebar.radio(
        "Page",
        [
            "Currently Reading",
            "Search",
            "Relationship Graph",
            "Ask Books (RAG)",
            "RAG Metrics",
            "Daily Recommendations",
            "Library",
        ],
        index=1,
    )
    cards_per_row = st.sidebar.slider("Cards per row", min_value=1, max_value=6, value=4, step=1)
    try:
        service = load_service(index_dir)
    except Exception as exc:
        st.error(
            "Unable to load semantic index. Run `index_books.py` first, then "
            f"`build_semantic_index.py`. Details: {exc}"
        )
        return

    all_categories = sorted({item.get("category", "Other") for item in service.metadata})
    all_modes = list(learning_mode_labels().keys())
    all_formats = sorted({get_book_format(item) for item in service.metadata} | {"PDF", "EPUB"})
    reading_items = load_currently_reading(reading_list_path)

    if view_page == "Currently Reading":
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        reading_items = load_currently_reading(reading_list_path)
        render_currently_reading_page(
            cover_cache_dir,
            reading_items,
            reading_list_path,
            cards_per_row=cards_per_row,
        )
        return

    if view_page == "Library":
        render_library_page(service)
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        return

    if view_page == "Daily Recommendations":
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        reading_items = load_currently_reading(reading_list_path)
        render_daily_recommendations_page(
            service=service,
            cover_cache_dir=cover_cache_dir,
            reading_items=reading_items,
            reading_list_path=reading_list_path,
            weights=daily_weights,
        )
        return

    if view_page == "Ask Books (RAG)":
        default_chunk_index_dir = (
            str(DEFAULT_CHUNK_INDEX_DIR_GTE_LARGE)
            if DEFAULT_CHUNK_INDEX_DIR_GTE_LARGE.exists()
            else str(DEFAULT_CHUNK_INDEX_DIR)
        )
        current_chunk_index_dir = str(
            st.session_state.get("rag-chunk-index-dir", default_chunk_index_dir) or ""
        ).strip()
        chunk_index_options = _rag_chunk_index_directory_options()
        known_chunk_index_dirs = [item for item in chunk_index_options if item != "Custom"]
        default_chunk_index_choice = (
            current_chunk_index_dir if current_chunk_index_dir in known_chunk_index_dirs else "Custom"
        )
        chunk_index_choice = st.sidebar.selectbox(
            "RAG chunk index preset",
            options=chunk_index_options,
            index=chunk_index_options.index(default_chunk_index_choice),
            key="rag-chunk-index-preset",
        )
        if chunk_index_choice == "Custom":
            chunk_index_dir = st.sidebar.text_input(
                "RAG chunk index directory",
                value=(
                    current_chunk_index_dir
                    if current_chunk_index_dir not in known_chunk_index_dirs
                    else default_chunk_index_dir
                ),
                key="rag-chunk-index-dir",
            )
        else:
            chunk_index_dir = str(chunk_index_choice)
            st.session_state["rag-chunk-index-dir"] = chunk_index_dir
        try:
            rag_service = load_rag_service(chunk_index_dir)
        except Exception as exc:
            st.error(
                "Unable to load RAG chunk index. Build chunk index first with "
                "`build_semantic_index.py --semantic-source ./output/semantic_chunks.jsonl "
                "--output-dir ./output/semantic_index_chunks`. "
                f"Details: {exc}"
            )
            return
        render_ask_books_rag_page(
            rag_service=rag_service,
            all_categories=all_categories,
            all_modes=all_modes,
        )
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        return

    if view_page == "RAG Metrics":
        render_rag_metrics_page()
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        return

    if view_page == "Relationship Graph":
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        reading_items = load_currently_reading(reading_list_path)
        render_relationship_graph_page(
            service=service,
            cover_cache_dir=cover_cache_dir,
            reading_items=reading_items,
            reading_list_path=reading_list_path,
            all_categories=all_categories,
            all_modes=all_modes,
        )
        return

    top_k = st.sidebar.number_input(
        "Top K results (0 = All)",
        min_value=0,
        max_value=1000000,
        value=20,
        step=5,
    )
    min_similarity = st.sidebar.slider(
        "Min similarity",
        min_value=-1.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )
    items_per_page = st.sidebar.slider("Items per page", min_value=8, max_value=60, value=20, step=4)
    selected_formats: List[str] = st.sidebar.multiselect("Format", all_formats, default=all_formats)
    surface_epubs = st.sidebar.checkbox(
        "Surface EPUB books in results",
        value=True,
        help="Ensure some EPUB books are visible near the top when available.",
    )
    st.sidebar.header("Filters")
    selected_categories: List[str] = st.sidebar.multiselect("Category", all_categories, default=all_categories)
    selected_modes: List[str] = st.sidebar.multiselect(
        "Theory vs Practical", all_modes, default=all_modes, format_func=lambda m: learning_mode_labels().get(m, m)
    )
    index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
    reading_items = load_currently_reading(reading_list_path)

    query = st.text_input(
        "Search books in natural language",
        value="give me book to learn about deep learning theory",
    )
    run_search = st.button("Search", type="primary")

    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "selected_book_id" not in st.session_state:
        st.session_state["selected_book_id"] = None
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    if run_search and query.strip():
        top_k_effective = len(service.metadata) if top_k <= 0 else top_k
        filters = SearchFilters(
            categories=selected_categories or None,
            learning_modes=selected_modes or None,
            min_similarity=float(min_similarity),
        )
        results = service.search_books(
            query=query.strip(),
            filters=filters,
            top_k=top_k_effective,
        )
        if selected_formats:
            format_set = set(selected_formats)
            results = [item for item in results if get_book_format(item) in format_set]
        if surface_epubs and (not selected_formats or "EPUB" in selected_formats):
            results = blend_results_to_surface_epubs(results, top_k=int(top_k_effective))
        else:
            results = results[: max(1, int(top_k_effective))]
        st.session_state["last_results"] = results
        st.session_state["current_page"] = 1
        st.session_state["selected_book_id"] = None

    selected_book_id = st.session_state.get("selected_book_id")
    if selected_book_id:
        book = service.get_book(selected_book_id)
        if not book:
            st.warning("Selected book no longer exists in index.")
            return

        st.header("Book Details")
        details_left, details_right = st.columns([2, 1], vertical_alignment="top")
        with details_left:
            render_result_card(book, cover_cache_dir)
        with details_right:
            st.subheader("Short Summary")
            st.write(build_book_summary(book))
            if st.button("Open PDF Location", key=f"open-location-{book.get('book_id')}"):
                ok, message = open_pdf_in_file_manager(book.get("absolute_path", ""))
                if ok:
                    st.toast(message, icon="📂")
                else:
                    st.warning(message)
            book_id = str(book.get("book_id", ""))
            is_reading = book_id in reading_items
            read_label = "Remove Reading" if is_reading else "Mark Reading"
            if st.button(read_label, key=f"details-reading-{book_id}"):
                if book_id:
                    if is_reading:
                        removed_entry = reading_items.pop(book_id, None)
                        if isinstance(removed_entry, dict):
                            removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                            if not removed_ok:
                                st.warning(removed_message)
                        save_currently_reading(reading_list_path, reading_items)
                        st.toast("Removed from currently reading.", icon="📕")
                    else:
                        entry = make_reading_entry(book)
                        copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                        reading_items[book_id] = entry
                        save_currently_reading(reading_list_path, reading_items)
                        st.toast("Added to currently reading.", icon="📘")
                        if not copy_ok:
                            st.warning(copy_message)
                    st.rerun()
            if is_reading and book_id:
                current_progress = coerce_progress(reading_items.get(book_id, {}).get("progress_pct", 0))
                detail_progress = st.slider(
                    "Reading Progress (%)",
                    min_value=0,
                    max_value=100,
                    value=current_progress,
                    key=f"details-progress-slider-{book_id}",
                )
                if detail_progress != current_progress:
                    reading_items[book_id]["progress_pct"] = detail_progress
                    save_currently_reading(reading_list_path, reading_items)
                    st.toast("Progress updated.", icon="📈")
                    st.rerun()

        st.subheader("Related Books")
        related = service.recommend_related(selected_book_id, top_k=10, same_category_boost=True)
        if not related:
            st.info("No related books found.")
        else:
            render_result_grid(
                related,
                cover_cache_dir=cover_cache_dir,
                columns_per_row=5,
                button_key_prefix=f"related-{selected_book_id}",
                reading_items=reading_items,
                reading_path=reading_list_path,
            )
        st.divider()

    results = st.session_state["last_results"]
    st.subheader(f"Search Results ({len(results)})")
    if not results:
        st.info("Run a search to see results.")
    else:
        total_pages = max(1, (len(results) + items_per_page - 1) // items_per_page)
        st.session_state["current_page"] = min(max(1, st.session_state["current_page"]), total_pages)

        pager_left, pager_mid, pager_right = st.columns([1, 2, 1])
        with pager_left:
            if st.button("Previous Page", disabled=st.session_state["current_page"] <= 1):
                st.session_state["current_page"] -= 1
                st.rerun()
        with pager_mid:
            st.write(f"Page {st.session_state['current_page']} / {total_pages}")
        with pager_right:
            if st.button("Next Page", disabled=st.session_state["current_page"] >= total_pages):
                st.session_state["current_page"] += 1
                st.rerun()

        start_idx = (st.session_state["current_page"] - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paged_results = results[start_idx:end_idx]
        render_result_grid(
            paged_results,
            cover_cache_dir=cover_cache_dir,
            columns_per_row=cards_per_row,
            button_key_prefix=f"result-page-{st.session_state['current_page']}",
            reading_items=reading_items,
            reading_path=reading_list_path,
        )


if __name__ == "__main__":
    main()

