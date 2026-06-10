#!/usr/bin/env python3
"""Gradio dashboard — parity with front-end/streamlit/dashboard.py."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_GRADIO_DIR = Path(__file__).resolve().parent
for _p in (PROJECT_ROOT, _GRADIO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rag_tab  # noqa: E402

from semantic_books.bookmap_ui_core import (  # noqa: E402
    BookmapPaths,
    DEFAULT_RERANKER_MODEL,
    RERANKER_MODEL_OPTIONS,
    RAG_RETRIEVAL_PRESETS,
    append_rag_metrics_row,
    blend_results_to_surface_epubs,
    build_book_summary,
    build_cover_thumbnail,
    build_relationship_figure,
    collect_recent_rag_metrics,
    coerce_progress,
    copy_book_to_current_read_folder,
    default_rag_session_defaults,
    get_book_format,
    get_file_dates,
    load_currently_reading,
    make_reading_entry,
    open_notebooklm_in_browser,
    open_pdf_in_file_manager,
    rag_chunk_index_directory_options,
    remove_book_copy_from_current_read_folder,
    save_currently_reading,
)
from semantic_books.daily_recommend import DailyBookRecommender, DailyRecommendationWeights  # noqa: E402
from semantic_books.learning_mode import learning_mode_labels  # noqa: E402
from semantic_books.rag_service import RagService  # noqa: E402
from semantic_books.search_service import SearchFilters, SemanticSearchService  # noqa: E402

PATHS = BookmapPaths(repo_root=PROJECT_ROOT)

BOOKMAP_CSS = """
.book-card-title { font-weight: 600; }
.book-card-meta { color: #787878; font-size: 0.9rem; }
footer {display: none !important;}
"""


@lru_cache(maxsize=4)
def _cached_search(index_dir: str) -> SemanticSearchService:
    return SemanticSearchService(Path(index_dir))


@lru_cache(maxsize=8)
def _cached_rag(index_dir: str) -> RagService:
    return RagService(Path(index_dir))


def _svc() -> Tuple[Optional[SemanticSearchService], str]:
    try:
        return _cached_search(str(PATHS.semantic_index)), ""
    except Exception as exc:
        return None, str(exc)


def _gal(items: List[dict], cache: str, lim: int = 24) -> List[Tuple[Optional[str], str]]:
    rows: List[Tuple[Optional[str], str]] = []
    for it in items[:lim]:
        cp = build_cover_thumbnail(str(it.get("absolute_path", "")), cache)
        bid = str(it.get("book_id", ""))
        title = str(it.get("title", "Untitled"))[:72]
        meta = f"{it.get('category','Other')} | sim {float(it.get('similarity',0) or 0):.3f}"
        rows.append((cp, f"{bid}||{title}\n{meta}"))
    return rows


def _bid(sel: Optional[str]) -> str:
    if not sel:
        return ""
    return str(sel.split("||", 1)[0]).strip()


def _page_chunk(items: List[dict], page: int, per: int) -> Tuple[List[dict], List[str]]:
    per = max(1, int(per))
    page = max(1, int(page))
    start = (page - 1) * per
    ch = items[start : start + per]
    lab = [f"{x.get('book_id','')}||{str(x.get('title',''))[:60]}" for x in ch]
    return ch, lab


def create_app() -> gr.Blocks:
    d0 = default_rag_session_defaults()
    init_state: Dict[str, Any] = {
        "last_results": [],
        "page": 1,
        "graph_payload": {"nodes": [], "edges": []},
        "graph_seed_ids": [],
        "graph_selected": None,
        "rag_metrics_history": [],
        "rag_chat_history": [],
        "reading_page": 1,
    }

    with gr.Blocks(title="BookMap RAG", css=BOOKMAP_CSS, theme=gr.themes.Soft()) as demo:
        st = gr.State(init_state)

        gr.Markdown("# BookMap RAG")
        banner = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Global")
                cards_per_row = gr.Slider(1, 6, value=4, step=1, label="Cards per row")
                gr.Textbox(str(PATHS.semantic_index), label="Semantic index", interactive=False)
                gr.Textbox(str(PATHS.cover_cache), label="Cover cache", interactive=False)
                gr.Textbox(str(PATHS.reading_list), label="Reading list", interactive=False)

            with gr.Column(scale=4):
                with gr.Tabs():
                    # —— Search ——
                    with gr.Tab("Search"):
                        q = gr.Textbox(label="Query", value="give me book to learn about deep learning theory")
                        top_k = gr.Number(20, label="Top K (0=all)", precision=0)
                        min_sim = gr.Slider(-1, 1, value=0.05, step=0.01, label="Min similarity")
                        ipp = gr.Slider(8, 60, value=20, step=4, label="Items per page")
                        fmt_ms = gr.Dropdown(["PDF", "EPUB"], multiselect=True, value=["PDF", "EPUB"], label="Format")
                        surf_ep = gr.Checkbox(True, label="Surface EPUBs")
                        cats = gr.Dropdown(multiselect=True, label="Category")
                        modes = gr.Dropdown(multiselect=True, label="Learning mode")
                        b_go = gr.Button("Search", variant="primary")
                        gal = gr.Gallery(columns=4, height=400, label="Results")
                        sel_dd = gr.Dropdown(label="Book on page", choices=[])
                        st_log = gr.Markdown()
                        det = gr.Markdown()
                        rel = gr.Gallery(columns=5, height=260, label="Related")
                        pg = gr.Markdown("Page 1/1")
                        b_pr = gr.Button("Prev")
                        b_nx = gr.Button("Next")
                        with gr.Row():
                            b_op = gr.Button("Open file location")
                            b_mr = gr.Button("Mark / remove reading")

                        def _init():
                            s, e = _svc()
                            banner_txt = "" if s else f"**Index error:** {e}"
                            if not s:
                                return gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), banner_txt
                            ac = sorted({str(x.get("category", "Other")) for x in s.metadata})
                            am = list(learning_mode_labels().keys())
                            return gr.update(choices=ac, value=ac), gr.update(choices=am, value=am), banner_txt

                        def _search(query, tk, ms, per, fmts, surf, csel, msel, state):
                            s, e = _svc()
                            if not s:
                                return (
                                    gr.update(value=[]),
                                    gr.update(choices=[]),
                                    "",
                                    gr.update(value=[]),
                                    e,
                                    {**state, "last_results": []},
                                )
                            tk_i = len(s.metadata) if float(tk) <= 0 else int(tk)
                            flt = SearchFilters(
                                categories=csel or None,
                                learning_modes=msel or None,
                                min_similarity=float(ms),
                            )
                            res = s.search_books(query.strip(), filters=flt, top_k=tk_i)
                            if fmts:
                                fs = set(fmts)
                                res = [r for r in res if get_book_format(r) in fs]
                            if surf and (not fmts or "EPUB" in fmts):
                                res = blend_results_to_surface_epubs(res, tk_i)
                            else:
                                res = res[: max(1, tk_i)]
                            state = {**state, "last_results": res, "page": 1}
                            return _rend(state, int(per))

                        def _rend(state: Dict[str, Any], per: int):
                            s, _ = _svc()
                            res = state.get("last_results") or []
                            page = max(1, int(state.get("page") or 1))
                            per = max(1, per)
                            tp = max(1, (len(res) + per - 1) // per) if res else 1
                            page = min(page, tp)
                            state = {**state, "page": page}
                            ch, lab = _page_chunk(res, page, per)
                            g = _gal(ch, str(PATHS.cover_cache), per)
                            one = lab[0] if lab else None
                            dtxt, rg = "", []
                            if one and s:
                                b = s.get_book(_bid(one))
                                if b:
                                    dtxt = f"### {b.get('title','')}\n\n{build_book_summary(b)}"
                                    rel_items = s.recommend_related(str(b.get("book_id")), top_k=10, same_category_boost=True)
                                    rg = _gal(rel_items, str(PATHS.cover_cache), 10)
                            return (
                                gr.update(value=g),
                                gr.update(choices=lab, value=one),
                                dtxt,
                                gr.update(value=rg),
                                f"Page {page}/{tp}",
                                state,
                            )

                        def _turn(state, per, d):
                            state = {**state, "page": max(1, int(state.get("page") or 1) + d)}
                            return _rend(state, int(per))

                        def _pick(sel, state, per):
                            s, _ = _svc()
                            if not sel or not s:
                                return "", gr.update(value=[]), state
                            b = s.get_book(_bid(sel))
                            if not b:
                                return "", gr.update(value=[]), state
                            dtxt = f"### {b.get('title','')}\n\n{build_book_summary(b)}"
                            rel_items = s.recommend_related(str(b.get("book_id")), top_k=10, same_category_boost=True)
                            return dtxt, gr.update(value=_gal(rel_items, str(PATHS.cover_cache), 10)), state

                        def _open_s(sel):
                            s, _ = _svc()
                            if not sel or not s:
                                return "Nothing to open."
                            b = s.get_book(_bid(sel))
                            if not b:
                                return "Not found."
                            return open_pdf_in_file_manager(str(b.get("absolute_path", "")))[1]

                        def _mark(sel, state):
                            s, _ = _svc()
                            if not sel or not s:
                                return state, "Select a book."
                            bid = _bid(sel)
                            b = s.get_book(bid)
                            if not b:
                                return state, "Not found."
                            reading = load_currently_reading(PATHS.reading_list)
                            if bid in reading:
                                rem = reading.pop(bid, None)
                                if isinstance(rem, dict):
                                    remove_book_copy_from_current_read_folder(rem, PATHS)
                                save_currently_reading(PATHS.reading_list, reading)
                                return state, "Removed from reading list."
                            ent = make_reading_entry(b)
                            ok, msg = copy_book_to_current_read_folder(ent, PATHS)
                            reading[bid] = ent
                            save_currently_reading(PATHS.reading_list, reading)
                            return state, msg

                        demo.load(_init, outputs=[cats, modes, banner])
                        b_go.click(_search, [q, top_k, min_sim, ipp, fmt_ms, surf_ep, cats, modes, st], [gal, sel_dd, det, rel, st_log, st])
                        sel_dd.change(_pick, [sel_dd, st, ipp], [det, rel, st])
                        b_pr.click(lambda a, b: _turn(a, b, -1), [st, ipp], [gal, sel_dd, det, rel, st_log, st])
                        b_nx.click(lambda a, b: _turn(a, b, 1), [st, ipp], [gal, sel_dd, det, rel, st_log, st])
                        b_op.click(_open_s, [sel_dd], [st_log])
                        b_mr.click(_mark, [sel_dd, st], [st, st_log])

                    # —— Currently Reading ——
                    with gr.Tab("Currently Reading"):
                        fmt_r = gr.Dropdown(["PDF", "EPUB"], multiselect=True, value=["PDF", "EPUB"], label="Formats")
                        b_nb = gr.Button("Open NotebookLM")
                        nb_m = gr.Markdown()
                        pg_r = gr.Number(1, label="Page", precision=0, minimum=1)
                        gal_r = gr.Gallery(columns=4, height=380, label="Books")
                        pick_r = gr.Dropdown(label="Select", choices=[])
                        prog = gr.Slider(0, 100, value=0, label="Progress %")
                        b_sv = gr.Button("Save progress")
                        b_or = gr.Button("Open file")
                        b_rm = gr.Button("Remove")
                        out_r = gr.Markdown()

                        def _load_r(fmt_sel, page, cpr):
                            items = load_currently_reading(PATHS.reading_list)
                            ent = sorted(items.values(), key=lambda x: x.get("added_at", ""))
                            if fmt_sel:
                                ent = [e for e in ent if get_book_format(e) in set(fmt_sel)]
                            cpr = max(1, int(cpr))
                            p = max(1, int(page))
                            tp = max(1, (len(ent) + cpr - 1) // cpr) if ent else 1
                            p = min(p, tp)
                            ch, lab = _page_chunk(
                                [{"book_id": e.get("book_id"), "title": e.get("title"), "category": e.get("category"), "learning_mode": e.get("learning_mode"), "absolute_path": e.get("absolute_path"), "similarity": 0} for e in ent],
                                p,
                                cpr,
                            )
                            g = _gal(ch, str(PATHS.cover_cache), cpr)
                            return (
                                gr.update(value=g),
                                gr.update(choices=lab, value=lab[0] if lab else None),
                                gr.update(value=p, maximum=max(1, tp)),
                                f"{len(ent)} books · page {p}/{tp}",
                            )

                        def _pick_pr(sel):
                            if not sel:
                                return gr.update(value=0)
                            items = load_currently_reading(PATHS.reading_list)
                            e = items.get(_bid(sel), {})
                            return gr.update(value=coerce_progress(e.get("progress_pct", 0)))

                        b_nb.click(lambda: open_notebooklm_in_browser()[1], outputs=[nb_m])
                        for ev in (fmt_r.change, pg_r.change, cards_per_row.change):
                            ev(_load_r, [fmt_r, pg_r, cards_per_row], [gal_r, pick_r, pg_r, out_r])
                        demo.load(_load_r, [fmt_r, pg_r, cards_per_row], [gal_r, pick_r, pg_r, out_r])
                        pick_r.change(_pick_pr, [pick_r], [prog])

                        def _sv(sel, p):
                            if not sel:
                                return "Select."
                            bid = _bid(sel)
                            items = load_currently_reading(PATHS.reading_list)
                            if bid not in items:
                                return "Missing."
                            items[bid]["progress_pct"] = int(p)
                            save_currently_reading(PATHS.reading_list, items)
                            return "Saved."

                        def _or(sel):
                            if not sel:
                                return "Select."
                            items = load_currently_reading(PATHS.reading_list)
                            e = items.get(_bid(sel), {})
                            return open_pdf_in_file_manager(str(e.get("absolute_path", "")))[1]

                        def _rm(sel, fmt_sel, page, cpr):
                            if not sel:
                                a, b, c, d = _load_r(fmt_sel, page, cpr)
                                return "Select.", a, b, c, d
                            bid = _bid(sel)
                            items = load_currently_reading(PATHS.reading_list)
                            rem = items.pop(bid, None)
                            if isinstance(rem, dict):
                                remove_book_copy_from_current_read_folder(rem, PATHS)
                            save_currently_reading(PATHS.reading_list, items)
                            a, b, c, d = _load_r(fmt_sel, page, cpr)
                            return "Removed.", a, b, c, d

                        b_sv.click(_sv, [pick_r, prog], [out_r])
                        b_or.click(_or, [pick_r], [out_r])
                        b_rm.click(_rm, [pick_r, fmt_r, pg_r, cards_per_row], [out_r, gal_r, pick_r, pg_r, out_r])

                    # —— Relationship Graph ——
                    with gr.Tab("Relationship Graph"):
                        gc = gr.Dropdown(multiselect=True, label="Category")
                        gm = gr.Dropdown(multiselect=True, label="Mode")
                        gcol = gr.Dropdown(["Category", "Theory vs Practical"], value="Category", label="Color by")
                        gme = gr.Slider(0, 1, value=0.3, label="Min edge similarity")
                        gnp = gr.Slider(1, 15, value=6, step=1, label="Neighbors per node")
                        gmx = gr.Slider(30, 600, value=220, step=10, label="Max nodes")
                        gfq = gr.Textbox("deep learning theory foundations", label="Focus query")
                        gsf = gr.Textbox("", label="Filter seed title")
                        gseed = gr.Dropdown(label="Seed book", choices=["(none)"])
                        gtk = gr.Slider(1, 5, value=2, step=1, label="Seed hits")
                        gnk = gr.Slider(3, 30, value=12, step=1, label="Neighbors / seed")
                        b_wh = gr.Button("Build whole-library graph")
                        b_fo = gr.Button("Build focused graph")
                        gplot = gr.Plot(label="Graph")
                        gnode = gr.Dropdown(label="Selected node id", choices=[])
                        gbuild = gr.Markdown()
                        gdet = gr.Markdown()
                        g_gmsg = gr.Markdown()
                        with gr.Row():
                            g_open = gr.Button("Open source")
                            g_mr = gr.Button("Mark / remove reading")

                        def _ginit():
                            s, _ = _svc()
                            if not s:
                                return gr.update(choices=[], value=[]), gr.update(choices=[], value=[])
                            ac = sorted({str(x.get("category", "Other")) for x in s.metadata})
                            am = list(learning_mode_labels().keys())
                            return gr.update(choices=ac, value=ac), gr.update(choices=am, value=am)

                        def _seed_choices(csel, msel, filt):
                            s, _ = _svc()
                            if not s:
                                return gr.update(choices=["(none)"])
                            opts = ["(none)"]
                            for it in s.metadata:
                                if csel and str(it.get("category")) not in csel:
                                    continue
                                if msel and str(it.get("learning_mode")) not in msel:
                                    continue
                                t = str(it.get("title", ""))
                                if filt.strip() and filt.strip().lower() not in t.lower():
                                    continue
                                opts.append(f"{it.get('book_id')}||{t[:80]}")
                                if len(opts) > 400:
                                    break
                            return gr.update(choices=opts, value="(none)")

                        def _bwhole(csel, msel, gcol_val, me, npn, mx, state):
                            s, e = _svc()
                            if not s:
                                return None, gr.update(), {**state, "graph_payload": {"nodes": [], "edges": []}}, e
                            flt = SearchFilters(categories=csel or None, learning_modes=msel or None, min_similarity=-1.0)
                            payload = s.build_whole_relationship_graph(
                                filters=flt,
                                max_nodes=int(mx),
                                min_similarity=float(me),
                                neighbors_per_node=int(npn),
                            )
                            state = {**state, "graph_payload": payload, "graph_seed_ids": [], "graph_selected": None}
                            fig = build_relationship_figure(payload, str(gcol_val), seed_ids=set(), selected_node_id=None)
                            ids = [str(n.get("id", "")) for n in (payload.get("nodes") or [])]
                            return fig, gr.update(choices=ids, value=None), state, f"Nodes {len(ids)}"

                        def _bfocus(csel, msel, fq, seed_sel, tk, nk, me, mx, npn, gcol_val, state):
                            s, e = _svc()
                            if not s:
                                return None, gr.update(), state, e
                            flt = SearchFilters(categories=csel or None, learning_modes=msel or None, min_similarity=-1.0)
                            seed_id = "" if not seed_sel or seed_sel == "(none)" else _bid(seed_sel)
                            payload = s.build_focused_relationship_graph(
                                query=fq.strip() or None,
                                seed_book_id=seed_id or None,
                                filters=flt,
                                seed_top_k=int(tk),
                                neighbor_k=int(nk),
                                max_nodes=int(mx),
                                min_similarity=float(me),
                                neighbors_per_node=int(npn),
                            )
                            seeds = set(str(x) for x in (payload.get("seed_ids") or []))
                            state = {**state, "graph_payload": payload, "graph_seed_ids": list(seeds), "graph_selected": None}
                            fig = build_relationship_figure(payload, str(gcol_val), seed_ids=seeds, selected_node_id=None)
                            ids = [str(n.get("id", "")) for n in (payload.get("nodes") or [])]
                            return fig, gr.update(choices=ids, value=None), state, f"Nodes {len(ids)}"

                        def _gnode_detail(nid, state):
                            if not nid:
                                return ""
                            payload = state.get("graph_payload") or {}
                            nodes = {str(n.get("id")): n for n in (payload.get("nodes") or [])}
                            node = nodes.get(str(nid))
                            if not node:
                                return ""
                            return (
                                f"### {node.get('title', 'Untitled')}\n"
                                f"{node.get('category', 'Other')} | {node.get('learning_mode', 'unknown')} | "
                                f"{get_book_format(node)}"
                            )

                        def _gopen_node(nid, state):
                            payload = state.get("graph_payload") or {}
                            nodes = {str(n.get("id")): n for n in (payload.get("nodes") or [])}
                            node = nodes.get(str(nid))
                            if not node:
                                return "No node selected."
                            return open_pdf_in_file_manager(str(node.get("absolute_path", "")))[1]

                        def _gmark_node(nid, state):
                            s, _ = _svc()
                            if not s or not nid:
                                return state, "Select a node."
                            payload = state.get("graph_payload") or {}
                            nodes = {str(n.get("id")): n for n in (payload.get("nodes") or [])}
                            node = nodes.get(str(nid))
                            if not node:
                                return state, "Unknown node."
                            bid = str(nid)
                            reading = load_currently_reading(PATHS.reading_list)
                            if bid in reading:
                                rem = reading.pop(bid, None)
                                if isinstance(rem, dict):
                                    remove_book_copy_from_current_read_folder(rem, PATHS)
                                save_currently_reading(PATHS.reading_list, reading)
                                return state, "Removed from reading list."
                            ent = make_reading_entry(
                                {
                                    "book_id": bid,
                                    "title": node.get("title", ""),
                                    "category": node.get("category", "Other"),
                                    "learning_mode": node.get("learning_mode", "unknown"),
                                    "absolute_path": node.get("absolute_path", ""),
                                }
                            )
                            ok, msg = copy_book_to_current_read_folder(ent, PATHS)
                            reading[bid] = ent
                            save_currently_reading(PATHS.reading_list, reading)
                            return state, msg if ok else msg

                        demo.load(_ginit, outputs=[gc, gm])
                        gc.change(_seed_choices, [gc, gm, gsf], [gseed])
                        gm.change(_seed_choices, [gc, gm, gsf], [gseed])
                        gsf.change(_seed_choices, [gc, gm, gsf], [gseed])
                        b_wh.click(_bwhole, [gc, gm, gcol, gme, gnp, gmx, st], [gplot, gnode, st, gbuild])
                        b_fo.click(_bfocus, [gc, gm, gfq, gseed, gtk, gnk, gme, gmx, gnp, gcol, st], [gplot, gnode, st, gbuild])
                        gnode.change(_gnode_detail, [gnode, st], [gdet])
                        g_open.click(_gopen_node, [gnode, st], [g_gmsg])
                        g_mr.click(_gmark_node, [gnode, st], [st, g_gmsg])

                    # —— Ask Books (RAG) ——
                    with gr.Tab("Ask Books (RAG)"):
                        chunk_pr = gr.Dropdown(
                            label="Chunk index",
                            choices=rag_chunk_index_directory_options(PATHS),
                            value=rag_chunk_index_directory_options(PATHS)[0],
                        )
                        chunk_cu = gr.Textbox(label="Custom chunk index path", visible=False)
                        ret_pr = gr.Dropdown(
                            label="Retrieval preset",
                            choices=["Custom"] + list(RAG_RETRIEVAL_PRESETS.keys()),
                            value="Definition Q&A",
                        )
                        perf_pr = gr.Dropdown(
                            label="Performance profile",
                            choices=["Off", "Auto", "Fast", "Balanced", "Quality"],
                            value="Off",
                        )
                        r_tkc = gr.Slider(4, 20, value=int(d0["rag-top-k-chunks"]), step=2, label="Top chunks")
                        r_mxc = gr.Slider(2, 10, value=int(d0["rag-max-citations"]), step=1, label="Max citations")
                        r_msim = gr.Slider(-1, 1, value=float(d0["rag-min-similarity"]), step=0.01, label="Min chunk similarity")
                        r_dbg = gr.Checkbox(label="Show retrieval debug", value=False)
                        r_blur = gr.Checkbox(label="Hide model thinking/meta text", value=True)
                        r_fb = gr.Checkbox(label="Show fallback notices", value=True)
                        r_nofb = gr.Checkbox(label="Disable deterministic fallback", value=False)
                        r_ex = gr.Radio(["Direct (local RagService)", "API (/rag/answer)"], value="Direct (local RagService)", label="Execution")
                        r_apiu = gr.Textbox(value="http://127.0.0.1:8000/rag/answer", label="API URL")
                        r_apit = gr.Slider(5, 120, value=30, step=5, label="API timeout (s)")
                        r_apik = gr.Textbox(value=os.getenv("RAG_API_KEY", ""), label="API key", type="password")
                        r_hyb = gr.Checkbox(value=True, label="Hybrid retrieval")
                        r_dw = gr.Slider(0, 1, value=0.7, step=0.05, label="Dense weight")
                        r_lw = gr.Slider(0, 1, value=0.3, step=0.05, label="Lexical weight")
                        r_pool = gr.Slider(8, 128, value=48, step=4, label="Candidate pool")
                        r_re = gr.Checkbox(value=True, label="Reranker")
                        r_rpre = gr.Dropdown(RERANKER_MODEL_OPTIONS, value=DEFAULT_RERANKER_MODEL, label="Reranker preset")
                        r_rcust = gr.Textbox(label="Custom reranker model", visible=False)
                        r_rtn = gr.Slider(4, 64, value=24, step=4, label="Reranker top-N")
                        r_gm = gr.Radio(["deterministic", "llama.cpp", "ollama"], value="ollama", label="Answer mode")
                        r_lpath = gr.Textbox(label="llama.cpp .gguf path")
                        r_lctx = gr.Slider(512, 8192, value=2048, step=256, label="llama ctx")
                        r_lmax = gr.Slider(64, 1024, value=420, step=32, label="llama max tokens")
                        r_ltmp = gr.Slider(0, 1, value=0.2, step=0.05, label="llama temp")
                        r_ltop = gr.Slider(0.1, 1, value=0.9, step=0.05, label="llama top_p")
                        r_lth = gr.Slider(1, 32, value=6, step=1, label="llama threads")
                        r_lgpu = gr.Slider(0, 120, value=0, step=1, label="llama GPU layers")
                        r_ob = gr.Textbox(value=d0["rag-ollama-base-url"], label="Ollama base URL")
                        r_om = gr.Textbox(value=str(d0["rag-ollama-model"]), label="Ollama model")
                        r_octx = gr.Slider(512, 32768, value=int(d0["rag-ollama-num-ctx"]), step=256, label="Ollama ctx")
                        r_ot = gr.Slider(0, 1, value=float(d0["rag-ollama-temp"]), step=0.05, label="Ollama temp")
                        r_otp = gr.Slider(0.1, 1, value=float(d0["rag-ollama-top-p"]), step=0.05, label="Ollama top_p")
                        r_oto = gr.Slider(5, 600, value=int(d0["rag-ollama-timeout"]), step=5, label="Ollama timeout")
                        r_cat = gr.Dropdown(multiselect=True, label="RAG categories")
                        r_mode = gr.Dropdown(multiselect=True, label="RAG learning modes")
                        chat = gr.Chatbot(label="Conversation", height=480, type="tuples")
                        r_q = gr.Textbox(label="Your question", placeholder="Ask a grounded question…")
                        r_go = gr.Button("Ask", variant="primary")
                        r_fu = gr.Dropdown(label="Suggested follow-up (optional)", choices=[])
                        r_fugo = gr.Button("Ask selected follow-up")

                        def _chunk_vis(preset):
                            return gr.update(visible=preset == "Custom")

                        chunk_pr.change(_chunk_vis, [chunk_pr], [chunk_cu])

                        def _rer_vis(preset):
                            return gr.update(visible=preset == "Custom")

                        r_rpre.change(_rer_vis, [r_rpre], [r_rcust])

                        def _rag_init():
                            s, _ = _svc()
                            if not s:
                                return gr.update(choices=[], value=[]), gr.update(choices=[], value=[])
                            ac = sorted({str(x.get("category", "Other")) for x in s.metadata})
                            am = list(learning_mode_labels().keys())
                            return gr.update(choices=ac, value=ac), gr.update(choices=am, value=am)

                        def _rag_ask(
                            message,
                            history,
                            state,
                            cpr,
                            ccu,
                            rpreset,
                            pperf,
                            tkc,
                            mxc,
                            msim,
                            dbg,
                            blur,
                            fb,
                            nofb,
                            ex,
                            apiu,
                            apit,
                            apik,
                            hyb,
                            dw,
                            lw,
                            pool,
                            re,
                            rpre,
                            rcust,
                            rtn,
                            gm,
                            lpath,
                            lctx,
                            lmax,
                            ltmp,
                            ltop,
                            lth,
                            lgpu,
                            ob,
                            om,
                            octx,
                            ot,
                            otp,
                            oto,
                            ccat,
                            mmode,
                        ):
                            message = (message or "").strip()
                            if not message:
                                return history or [], state, gr.update(choices=[])
                            history = list(history or [])
                            cdir = rag_tab.resolve_chunk_dir(str(cpr), str(ccu))
                            if not cdir:
                                history.append([message, "Select a valid chunk index."])
                                return history, state, gr.update()
                            try:
                                rag = _cached_rag(cdir)
                            except Exception as exc:
                                history.append([message, f"Cannot load RAG index: {exc}"])
                                return history, state, gr.update()
                            rmod = str(rpre)
                            if rpre == "Custom":
                                rmod = str(rcust).strip()
                            base = {
                                "top_k_chunks": int(tkc),
                                "max_citations": int(mxc),
                                "min_similarity": float(msim),
                                "use_hybrid": bool(hyb),
                                "dense_weight": float(dw),
                                "lexical_weight": float(lw),
                                "candidate_pool_size": int(pool),
                                "reranker_enabled": bool(re),
                                "reranker_model": rmod if re else "",
                                "reranker_top_n": int(rtn),
                                "generation_mode": str(gm),
                                "llama_model_path": str(lpath),
                                "llama_n_ctx": int(lctx),
                                "llama_max_tokens": int(lmax),
                                "llama_temp": float(ltmp),
                                "llama_top_p": float(ltop),
                                "llama_threads": int(lth),
                                "llama_gpu_layers": int(lgpu),
                                "ollama_base_url": str(ob),
                                "ollama_model": str(om),
                                "ollama_num_ctx": int(octx),
                                "ollama_temp": float(ot),
                                "ollama_top_p": float(otp),
                                "ollama_timeout_sec": int(oto),
                            }
                            rp_name = str(rpreset) if str(rpreset) != "Custom" else "Custom"
                            params = rag_tab.build_rag_params_from_ui(
                                query=message,
                                retrieval_preset_name=rp_name,
                                perf_mode=str(pperf),
                                base=base,
                            )
                            exec_api = str(ex).startswith("API")
                            try:
                                response, text = rag_tab.run_rag_turn(
                                    query=message,
                                    rag=rag,
                                    exec_api=exec_api,
                                    api_url=str(apiu).strip(),
                                    api_timeout=int(apit),
                                    api_key=str(apik),
                                    selected_categories=list(ccat or []),
                                    selected_modes=list(mmode or []),
                                    show_debug=bool(dbg),
                                    blur_meta=bool(blur),
                                    show_fallback=bool(fb),
                                    disable_fallback=bool(nofb),
                                    params=params,
                                )
                            except Exception as exc:
                                history.append([message, f"Error: {exc}"])
                                return history, state, gr.update()
                            history.append([message, text])
                            mh = state.get("rag_metrics_history") or []
                            mh = append_rag_metrics_row(mh, message, response)
                            rch = list(state.get("rag_chat_history") or [])
                            rch.append({"question": message, "response": response})
                            state = {**state, "rag_metrics_history": mh, "rag_chat_history": rch[-30:]}
                            fus = response.get("follow_ups") or []
                            fus = [str(x) for x in fus if x]
                            return history, state, gr.update(choices=fus or ["(none)"], value="(none)")

                        def _rag_fu(fu_q, history, state, *args):
                            if not fu_q or fu_q == "(none)":
                                return history or [], state, gr.update()
                            return _rag_ask(fu_q, history, state, *args)

                        demo.load(_rag_init, outputs=[r_cat, r_mode])
                        r_go.click(
                            _rag_ask,
                            [
                                r_q,
                                chat,
                                st,
                                chunk_pr,
                                chunk_cu,
                                ret_pr,
                                perf_pr,
                                r_tkc,
                                r_mxc,
                                r_msim,
                                r_dbg,
                                r_blur,
                                r_fb,
                                r_nofb,
                                r_ex,
                                r_apiu,
                                r_apit,
                                r_apik,
                                r_hyb,
                                r_dw,
                                r_lw,
                                r_pool,
                                r_re,
                                r_rpre,
                                r_rcust,
                                r_rtn,
                                r_gm,
                                r_lpath,
                                r_lctx,
                                r_lmax,
                                r_ltmp,
                                r_ltop,
                                r_lth,
                                r_lgpu,
                                r_ob,
                                r_om,
                                r_octx,
                                r_ot,
                                r_otp,
                                r_oto,
                                r_cat,
                                r_mode,
                            ],
                            [chat, st, r_fu],
                        )
                        r_fugo.click(
                            _rag_fu,
                            [
                                r_fu,
                                chat,
                                st,
                                chunk_pr,
                                chunk_cu,
                                ret_pr,
                                perf_pr,
                                r_tkc,
                                r_mxc,
                                r_msim,
                                r_dbg,
                                r_blur,
                                r_fb,
                                r_nofb,
                                r_ex,
                                r_apiu,
                                r_apit,
                                r_apik,
                                r_hyb,
                                r_dw,
                                r_lw,
                                r_pool,
                                r_re,
                                r_rpre,
                                r_rcust,
                                r_rtn,
                                r_gm,
                                r_lpath,
                                r_lctx,
                                r_lmax,
                                r_ltmp,
                                r_ltop,
                                r_lth,
                                r_lgpu,
                                r_ob,
                                r_om,
                                r_octx,
                                r_ot,
                                r_otp,
                                r_oto,
                                r_cat,
                                r_mode,
                            ],
                            [chat, st, r_fu],
                        )

                    # —— RAG Metrics ——
                    with gr.Tab("RAG Metrics"):
                        m_cap = gr.Markdown()
                        m_plot1 = gr.Plot()
                        m_plot2 = gr.Plot()
                        m_plot3 = gr.Plot()
                        m_tbl = gr.Dataframe()
                        b_ref = gr.Button("Refresh metrics")

                        def _metrics(st):
                            rows = collect_recent_rag_metrics(
                                metrics_history=st.get("rag_metrics_history"),
                                chat_history=st.get("rag_chat_history") or [],
                                window=10,
                            )
                            if not rows:
                                return (
                                    "No metrics yet. Use Ask Books first.",
                                    None,
                                    None,
                                    None,
                                    pd.DataFrame(),
                                )
                            frame = pd.DataFrame(rows)
                            cap = (
                                f"Avg total {frame['total_ms'].mean():.1f} ms | "
                                f"retrieval {frame['retrieval_ms'].mean():.1f} | "
                                f"generation {frame['generation_ms'].mean():.1f} ms"
                            )
                            if go is None:
                                return cap, None, None, None, frame
                            f1 = go.Figure()
                            f1.add_trace(go.Scatter(x=frame["answer_idx"], y=frame["total_ms"], name="total_ms", mode="lines+markers"))
                            f1.add_trace(go.Scatter(x=frame["answer_idx"], y=frame["retrieval_ms"], name="retrieval_ms", mode="lines+markers"))
                            f1.add_trace(go.Scatter(x=frame["answer_idx"], y=frame["generation_ms"], name="generation_ms", mode="lines+markers"))
                            f1.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Answer #", yaxis_title="ms")
                            f2 = go.Figure()
                            f2.add_trace(go.Bar(x=frame["answer_idx"], y=frame["peak_rss_mb"], name="peak_rss_mb"))
                            f2.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                            f3 = go.Figure()
                            f3.add_trace(
                                go.Scatter(x=frame["answer_idx"], y=frame["top_relevance_score"], name="top_relevance", mode="lines+markers")
                            )
                            f3.add_trace(
                                go.Scatter(x=frame["answer_idx"], y=frame["citation_coverage_ratio"], name="citation_cov", mode="lines+markers")
                            )
                            f3.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                            return cap, f1, f2, f3, frame

                        b_ref.click(_metrics, [st], [m_cap, m_plot1, m_plot2, m_plot3, m_tbl])

                    # —— Daily Recommendations ——
                    with gr.Tab("Daily Recommendations"):
                        d_w = gr.Markdown("Daily weights: similarity 0.55, freshness 0.2, novelty 0.15, confidence 0.1, diversity 0.1, explore 0.2")
                        b_dref = gr.Button("Refresh for today")
                        d_fmt = gr.Dropdown(["PDF", "EPUB"], multiselect=True, value=["PDF", "EPUB"], label="Format")
                        d_surf = gr.Checkbox(True, label="Surface EPUBs")
                        d_gal = gr.Gallery(columns=6, height=420, label="Today's picks")
                        d_log = gr.Markdown()
                        weights = DailyRecommendationWeights(
                            similarity=0.55,
                            freshness=0.2,
                            novelty=0.15,
                            confidence=0.1,
                            diversity_penalty=0.1,
                            explore_bonus=0.2,
                        )

                        def _daily(refresh, fmt_sel, surf, state):
                            s, e = _svc()
                            if not s:
                                return gr.update(value=[]), e, state
                            rec = DailyBookRecommender(
                                service=s,
                                reading_list_path=PATHS.reading_list,
                                history_path=PATHS.daily_recommendations,
                                weights=weights,
                            )
                            today = datetime.now().astimezone().date()
                            items = rec.get_or_generate_for_date(target_date=today, count=18, force_refresh=bool(refresh))
                            if fmt_sel:
                                items = [x for x in items if get_book_format(x) in set(fmt_sel)]
                            if surf and (not fmt_sel or "EPUB" in fmt_sel):
                                items = blend_results_to_surface_epubs(items, 18)
                            g = _gal(items[:18], str(PATHS.cover_cache), 18)
                            return gr.update(value=g), f"Date {today} · {len(items)} items", state

                        b_dref.click(lambda f, s, st: _daily(True, f, s, st), [d_fmt, d_surf, st], [d_gal, d_log, st])
                        demo.load(lambda f, s, st: _daily(False, f, s, st), [d_fmt, d_surf, st], [d_gal, d_log, st])

                    # —— Library ——
                    with gr.Tab("Library"):
                        lf = gr.Textbox(label="Filter by name")
                        lc = gr.Dropdown(multiselect=True, label="Category")
                        lfmt = gr.Dropdown(multiselect=True, label="Format")
                        lcs = gr.Textbox(value="2000-01-01", label="Created start")
                        lce = gr.Textbox(value=date.today().isoformat(), label="Created end")
                        lus = gr.Textbox(value="2000-01-01", label="Updated start")
                        lue = gr.Textbox(value=date.today().isoformat(), label="Updated end")
                        b_lf = gr.Button("Apply filters")
                        l_tbl = gr.Dataframe()
                        l_pick = gr.Dropdown(label="Row preview", choices=[])
                        l_img = gr.Image(label="Cover")
                        l_op = gr.Button("Open location")
                        l_stat = gr.Markdown()

                        def _lib_init():
                            s, _ = _svc()
                            if not s:
                                return gr.update(choices=[], value=[]), gr.update(choices=[], value=[])
                            cats = sorted({str(x.get("category", "Other")) for x in s.metadata})
                            fmts = sorted(
                                {(Path(str(x.get("absolute_path", ""))).suffix.lower().lstrip(".") or "unknown").upper() for x in s.metadata}
                                | {"PDF", "EPUB"}
                            )
                            return gr.update(choices=cats, value=cats), gr.update(choices=list(fmts), value=list(fmts))

                        def _lib_apply(name, csel, fsel, cs, ce, us, ue):
                            s, e = _svc()
                            if not s:
                                return pd.DataFrame(), gr.update(choices=[]), "No index"
                            try:
                                c0, c1 = date.fromisoformat(cs), date.fromisoformat(ce)
                                u0, u1 = date.fromisoformat(us), date.fromisoformat(ue)
                            except ValueError:
                                return pd.DataFrame(), gr.update(choices=[]), "Invalid date"
                            nn = name.strip().lower()
                            rows = []
                            for it in s.metadata:
                                title = str(it.get("title", "Untitled"))
                                cat = str(it.get("category", "Other"))
                                path = str(it.get("absolute_path", ""))
                                fn = Path(path).name
                                fmt = (Path(path).suffix.lower().lstrip(".") or "unknown").upper()
                                if csel and cat not in csel:
                                    continue
                                if fsel and fmt not in fsel:
                                    continue
                                if nn and nn not in title.lower() and nn not in path.lower() and nn not in fn.lower():
                                    continue
                                cd, ud = get_file_dates(path)
                                if cd is None or ud is None:
                                    continue
                                if cd < c0 or cd > c1 or ud < u0 or ud > u1:
                                    continue
                                rows.append(
                                    {
                                        "Title": title,
                                        "Format": fmt,
                                        "Category": cat,
                                        "Created": cd.isoformat(),
                                        "Updated": ud.isoformat(),
                                        "Path": path,
                                    }
                                )
                            df = pd.DataFrame(rows).sort_values(by="Updated", ascending=False, kind="stable").reset_index(drop=True)
                            choices = [f"{i}||{row['Title'][:60]}" for i, row in df.iterrows()]
                            return df, gr.update(choices=choices, value=choices[0] if choices else None), f"{len(df)} books"

                        def _lib_prev(sel, df):
                            if sel is None or df is None or len(df) == 0:
                                return None
                            try:
                                idx = int(str(sel).split("||", 1)[0])
                            except ValueError:
                                return None
                            if idx < 0 or idx >= len(df):
                                return None
                            path = str(df.iloc[idx]["Path"])
                            cp = build_cover_thumbnail(path, str(PATHS.cover_cache))
                            return cp

                        def _lib_open(sel, df):
                            if sel is None or df is None or len(df) == 0:
                                return "No selection."
                            try:
                                idx = int(str(sel).split("||", 1)[0])
                            except ValueError:
                                return "Bad row."
                            if idx < 0 or idx >= len(df):
                                return "Bad row."
                            path = str(df.iloc[idx]["Path"])
                            return open_pdf_in_file_manager(path)[1]

                        demo.load(_lib_init, outputs=[lc, lfmt])
                        b_lf.click(_lib_apply, [lf, lc, lfmt, lcs, lce, lus, lue], [l_tbl, l_pick, l_stat])
                        l_pick.change(_lib_prev, [l_pick, l_tbl], [l_img])
                        l_op.click(_lib_open, [l_pick, l_tbl], [l_stat])

    return demo


if __name__ == "__main__":
    create_app().queue().launch(server_name="0.0.0.0", server_port=7860)
