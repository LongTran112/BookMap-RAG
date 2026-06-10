"""RAG chat + metrics helpers for Gradio (no Gradio imports)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from semantic_books.bookmap_ui_core import (
    RAG_PERFORMANCE_PROFILES,
    RAG_RETRIEVAL_PRESETS,
    append_rag_metrics_row,
    build_rag_answer_payload,
    call_rag_api_answer,
    format_rag_turn_markdown,
    select_rag_auto_profile,
)
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService


def resolve_chunk_dir(preset: str, custom: str) -> str:
    if preset == "Custom":
        return (custom or "").strip() or ""
    return str(preset).strip()


def _apply_performance_profile(target: Dict[str, Any], profile: Dict[str, Any]) -> None:
    target["generation_mode"] = str(profile["generation_mode"])
    target["ollama_model"] = str(profile["ollama_model"])
    target["ollama_num_ctx"] = int(profile["ollama_num_ctx"])
    target["ollama_temp"] = float(profile["ollama_temp"])
    target["ollama_top_p"] = float(profile["ollama_top_p"])
    target["ollama_timeout_sec"] = int(profile["ollama_timeout_sec"])
    target["top_k_chunks"] = int(profile["top_k_chunks"])
    target["max_citations"] = int(profile["max_citations"])
    target["candidate_pool_size"] = int(profile["candidate_pool_size"])
    target["min_similarity"] = float(profile["min_similarity"])
    target["reranker_enabled"] = bool(profile.get("reranker_enabled", True))
    target["reranker_top_n"] = int(profile.get("reranker_top_n", 32))


def build_rag_params_from_ui(
    *,
    query: str,
    retrieval_preset_name: str,
    perf_mode: str,
    base: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge sliders (`base`) with retrieval preset and performance profile (Streamlit-style)."""
    p = dict(base)
    if retrieval_preset_name != "Custom" and retrieval_preset_name in RAG_RETRIEVAL_PRESETS:
        rp = RAG_RETRIEVAL_PRESETS[retrieval_preset_name]
        p["top_k_chunks"] = int(rp["top_k_chunks"])
        p["min_similarity"] = float(rp["min_similarity"])
        p["use_hybrid"] = bool(rp["use_hybrid"])
        p["dense_weight"] = float(rp["dense_weight"])
        p["lexical_weight"] = float(rp["lexical_weight"])
        p["candidate_pool_size"] = int(rp["candidate_pool_size"])
        p["reranker_enabled"] = bool(rp["reranker_enabled"])
        p["reranker_top_n"] = int(rp["reranker_top_n"])

    if perf_mode == "Auto":
        name = select_rag_auto_profile(query)
        prof = RAG_PERFORMANCE_PROFILES.get(name)
        if prof:
            _apply_performance_profile(p, prof)
    elif perf_mode != "Off" and perf_mode in RAG_PERFORMANCE_PROFILES:
        _apply_performance_profile(p, RAG_PERFORMANCE_PROFILES[perf_mode])
    return p


def run_rag_turn(
    *,
    query: str,
    rag: RagService,
    exec_api: bool,
    api_url: str,
    api_timeout: int,
    api_key: str,
    selected_categories: List[str],
    selected_modes: List[str],
    show_debug: bool,
    blur_meta: bool,
    show_fallback: bool,
    disable_fallback: bool,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    payload = build_rag_answer_payload(
        query=query,
        top_k_chunks=int(params["top_k_chunks"]),
        max_citations=int(params["max_citations"]),
        selected_categories=selected_categories,
        selected_modes=selected_modes,
        min_similarity=float(params["min_similarity"]),
        use_hybrid=bool(params["use_hybrid"]),
        dense_weight=float(params["dense_weight"]),
        lexical_weight=float(params["lexical_weight"]),
        candidate_pool_size=int(params["candidate_pool_size"]),
        reranker_enabled=bool(params["reranker_enabled"]),
        reranker_model=str(params["reranker_model"]),
        reranker_top_n=int(params["reranker_top_n"]),
        generation_mode=str(params["generation_mode"]),
        llama_model_path=str(params["llama_model_path"]),
        llama_n_ctx=int(params["llama_n_ctx"]),
        llama_max_tokens=int(params["llama_max_tokens"]),
        llama_temp=float(params["llama_temp"]),
        llama_top_p=float(params["llama_top_p"]),
        llama_threads=int(params["llama_threads"]),
        llama_gpu_layers=int(params["llama_gpu_layers"]),
        ollama_base_url=str(params["ollama_base_url"]),
        ollama_model=str(params["ollama_model"]),
        ollama_temp=float(params["ollama_temp"]),
        ollama_top_p=float(params["ollama_top_p"]),
        ollama_num_ctx=int(params["ollama_num_ctx"]),
        ollama_timeout_sec=int(params["ollama_timeout_sec"]),
        allow_fallback=not disable_fallback,
    )
    if exec_api:
        response = call_rag_api_answer(api_url, payload, timeout_sec=api_timeout, api_key=api_key)
    else:
        filters = RagFilters(
            categories=selected_categories or None,
            learning_modes=selected_modes or None,
            min_similarity=float(params["min_similarity"]),
        )
        retrieval_config = RetrievalConfig(
            hybrid_enabled=bool(params["use_hybrid"]),
            dense_weight=float(params["dense_weight"]),
            lexical_weight=float(params["lexical_weight"]),
            candidate_pool_size=int(params["candidate_pool_size"]),
            final_top_k=int(params["top_k_chunks"]),
            reranker_enabled=bool(params["reranker_enabled"]),
            reranker_model_name=str(params["reranker_model"]) if params["reranker_enabled"] else None,
            reranker_top_n=int(params["reranker_top_n"]),
        )
        llm_config = LlamaCppConfig(
            enabled=str(params["generation_mode"]) == "llama.cpp",
            model_path=str(params["llama_model_path"]).strip(),
            n_ctx=int(params["llama_n_ctx"]),
            max_tokens=int(params["llama_max_tokens"]),
            temperature=float(params["llama_temp"]),
            top_p=float(params["llama_top_p"]),
            n_threads=int(params["llama_threads"]),
            n_gpu_layers=int(params["llama_gpu_layers"]),
        )
        ollama_config = OllamaConfig(
            enabled=str(params["generation_mode"]) == "ollama",
            base_url=str(params["ollama_base_url"]).strip(),
            model=str(params["ollama_model"]).strip(),
            temperature=float(params["ollama_temp"]),
            top_p=float(params["ollama_top_p"]),
            num_ctx=int(params["ollama_num_ctx"]),
            timeout_sec=int(params["ollama_timeout_sec"]),
        )
        response = rag.answer_question(
            query=query.strip(),
            filters=filters,
            top_k=int(params["top_k_chunks"]),
            max_citations=int(params["max_citations"]),
            retrieval_config=retrieval_config,
            llm_config=llm_config,
            ollama_config=ollama_config,
            on_token=None,
            allow_fallback=not disable_fallback,
        )
    text = format_rag_turn_markdown(
        response,
        show_debug=show_debug,
        blur_meta_text=blur_meta,
        show_fallback_notice=show_fallback,
    )
    return response, text


def append_metrics(metrics_hist: List[Dict[str, Any]], question: str, response: Dict[str, Any]) -> List[Dict[str, Any]]:
    return append_rag_metrics_row(metrics_hist, question, response)
