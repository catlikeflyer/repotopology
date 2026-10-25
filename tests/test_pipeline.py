"""Unit and deterministic sanity tests for the RepoTopology pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotopology.agent import ReActAgent
from repotopology.evaluator import (
    BenchmarkEvaluator,
    create_sample_dataset,
    extract_patched_symbols,
)
from repotopology.graph_store import EMBEDDING_DIM, GraphStore
from repotopology.pruner import ContextPruner, extract_signature_and_docstring
from repotopology.server import (
    configure_server,
    get_code_neighbors,
    get_pruned_context,
    search_similar_code,
)


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    """Creates a temporary competition dataset (`tasks.jsonl`, `graphs/`, `embeddings/`)."""
    data_dir = tmp_path / "data"
    create_sample_dataset(data_dir)
    return data_dir


@pytest.fixture()
def sample_store(sample_data_dir: Path) -> GraphStore:
    """Loads the sample `GraphStore` instance."""
    return GraphStore.from_data_dir(
        instance_id="authlib__token_verify_001",
        data_dir=sample_data_dir,
    )


def test_graph_store_and_embeddings_schema(sample_store: GraphStore) -> None:
    """Verifies NetworkX MultiDiGraph schema and 256-d float32 NPZ vectors."""
    assert sample_store.graph.is_directed()
    assert sample_store.graph.is_multigraph()
    assert sample_store.has_node("auth/router.py::authenticate_request")
    assert sample_store.has_node("auth/crypto.py::verify_token_digest")
    assert sample_store.has_node("auth/primitives.py::compute_hmac_sha256")

    for vec in sample_store.embeddings.values():
        assert vec.shape == (EMBEDDING_DIM,)
        assert str(vec.dtype) == "float32"

    hits = sample_store.search_similar_nodes("authenticate_request webhook signature", top_k=2)
    assert len(hits) == 2
    assert hits[0].node_id == "auth/router.py::authenticate_request"
    assert -1.0001 <= hits[0].score <= 1.0001


def test_neighbors_and_k_hop_tiers(sample_store: GraphStore) -> None:
    """Verifies 1-hop incoming/outgoing neighbor queries and 2-hop tier resolution."""
    callees = sample_store.get_neighbors(
        "auth/router.py::authenticate_request", direction="callees"
    )
    assert len(callees) == 1
    assert callees[0].node_id == "auth/crypto.py::verify_token_digest"
    assert callees[0].edge_type == "calls"
    assert callees[0].direction == "outgoing"

    callers = sample_store.get_neighbors(
        "auth/crypto.py::verify_token_digest", direction="callers"
    )
    assert len(callers) == 1
    assert callers[0].node_id == "auth/router.py::authenticate_request"
    assert callers[0].direction == "incoming"

    tier1, tier2 = sample_store.get_k_hop_tiers("auth/router.py::authenticate_request")
    assert tier1 == ["auth/crypto.py::verify_token_digest"]
    assert tier2 == ["auth/primitives.py::compute_hmac_sha256"]


def test_ast_signature_and_docstring_extractor() -> None:
    """Verifies that function bodies are stripped while signatures, annotations, and docstrings remain."""
    raw_code = '''
def verify_token_digest(token: str, payload: bytes, leeway_seconds: int = 0) -> bool:
    """Verifies HMAC-SHA256 digest and timestamp drift for webhook tokens."""
    parts = token.split(".")
    if len(parts) != 2:
        return False
    return parts[0] == "ok"
'''
    stub = extract_signature_and_docstring(raw_code)
    assert "def verify_token_digest(token: str, payload: bytes, leeway_seconds: int" in stub
    assert ") -> bool:" in stub
    assert "Verifies HMAC-SHA256 digest and timestamp drift for webhook tokens." in stub
    assert "parts = token.split" not in stub
    assert "..." in stub


def test_three_tier_context_pruning(sample_store: GraphStore) -> None:
    """Verifies Tier 1 (full code), Tier 2 (signatures/docstrings only), and Tier 3 (symbol IDs only)."""
    pruner = ContextPruner(sample_store)
    focus_id = "auth/router.py::authenticate_request"
    result = pruner.prune(focus_id)

    # Tier 1: Full implementation of focus symbol
    assert "X-Webhook-Signature" in result.tier1_source
    assert "return verify_token_digest(token, payload)" in result.tier1_source

    # Tier 2: 1-hop neighbor has signature & docstring, but body statements are stripped
    assert "auth/crypto.py::verify_token_digest" in result.tier2_stubs
    tier2_stub = result.tier2_stubs["auth/crypto.py::verify_token_digest"]
    assert "def verify_token_digest(token: str, payload: bytes, leeway_seconds: int" in tier2_stub
    assert "Verifies HMAC-SHA256 digest" in tier2_stub
    assert "ts_raw, digest = parts[0], parts[1]" not in tier2_stub

    # Tier 3: 2-hop neighbor has qualified symbol ID only
    assert result.tier3_symbol_ids == ["auth/primitives.py::compute_hmac_sha256"]

    formatted = result.format_prompt_context()
    assert "TIER 1: FOCUS SYMBOL" in formatted
    assert "TIER 2: 1-HOP NEIGHBORS" in formatted
    assert "TIER 3: 2-HOP NEIGHBORS" in formatted
    assert "- auth/primitives.py::compute_hmac_sha256" in formatted
    # Ensure 2-hop neighbor implementation is NOT leaked into pruned context
    assert "hmac.new(secret, message, hashlib.sha256)" not in formatted


def test_fastmcp_server_tools(sample_store: GraphStore) -> None:
    """Verifies the 3 FastMCP tool functions exposed by `repotopology.server`."""
    configure_server(sample_store)

    search_out = json.loads(search_similar_code("authenticate_request", top_k=2))
    assert search_out["instance_id"] == "authlib__token_verify_001"
    assert len(search_out["results"]) == 2
    assert search_out["results"][0]["node_id"] == "auth/router.py::authenticate_request"

    neighbors_out = json.loads(
        get_code_neighbors("auth/router.py::authenticate_request", direction="callees")
    )
    assert len(neighbors_out["neighbors"]) == 1
    assert neighbors_out["neighbors"][0]["node_id"] == "auth/crypto.py::verify_token_digest"

    pruned_out = get_pruned_context("auth/router.py::authenticate_request")
    assert "TIER 1: FOCUS SYMBOL [auth/router.py::authenticate_request]" in pruned_out
    assert "TIER 2: 1-HOP NEIGHBORS" in pruned_out
    assert "TIER 3: 2-HOP NEIGHBORS" in pruned_out


class _MockGemmaClient:
    """Deterministic mock Gemma 4 E4B client for testing multi-turn ReAct execution and hallucination tracking."""

    def __init__(self) -> None:
        self.turn = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.turn += 1
        if self.turn == 1:
            return (
                "Thought: Search for entrypoint handling webhook authentication.\n"
                'Action: {"tool": "search_similar_code", "arguments": {"query": "authenticate_request", "top_k": 2}}'
            )
        if self.turn == 2:
            return (
                "Thought: Check non-existent symbol to test hallucination tracking.\n"
                'Action: {"tool": "get_code_neighbors", "arguments": {"node_id": "auth/non_existent.py::ghost"}}'
            )
        if self.turn == 3:
            return (
                "Thought: Inspect 3-tier pruned context of verify_token_digest.\n"
                'Action: {"tool": "get_pruned_context", "arguments": {"focus_node_id": "auth/crypto.py::verify_token_digest"}}'
            )
        return (
            "Thought: Localization complete.\n"
            'Action: {"tool": "finish", "arguments": {"localized_symbols": ["auth/crypto.py::verify_token_digest"], "summary": "Found bug."}}'
        )


def test_react_agent_with_mock_llm(sample_store: GraphStore) -> None:
    """Verifies multi-step ReAct loop and structural hallucination rate calculation."""
    agent = ReActAgent(store=sample_store, llm_client=_MockGemmaClient(), max_steps=5)
    traj = agent.run("Webhook authentication fails on clock skew")

    assert traj.trajectory_length == 4
    assert traj.total_tool_calls == 3
    assert traj.hallucinated_tool_calls == 1
    assert pytest.approx(traj.hallucination_rate, rel=1e-4) == 1.0 / 3.0
    assert "auth/crypto.py::verify_token_digest" in traj.localized_symbols


def test_evaluator_four_baselines(sample_data_dir: Path) -> None:
    """Verifies the 4-baseline evaluation harness and token efficiency improvements."""
    evaluator = BenchmarkEvaluator(data_dir=sample_data_dir, top_k=5)
    summary = evaluator.run_all()

    assert summary["num_tasks"] == 1
    baselines = summary["baselines"]
    assert set(baselines.keys()) == {
        "baseline_0_grep",
        "baseline_1_rag",
        "baseline_2_unpruned_graph",
        "baseline_3_repotopology",
    }

    b0 = baselines["baseline_0_grep"]
    b2 = baselines["baseline_2_unpruned_graph"]
    b3 = baselines["baseline_3_repotopology"]

    # RepoTopology (Baseline 3) hits ground truth with zero hallucinations and fewer tokens than Baseline 2 & 0
    assert b3["top_k_localization_hit_rate"] == 1.0
    assert b3["structural_hallucination_rate"] == 0.0
    assert b3["mean_total_prompt_tokens"] < b2["mean_total_prompt_tokens"]
    assert b3["mean_token_efficiency_ratio"] > b2["mean_token_efficiency_ratio"]
    assert b3["mean_token_efficiency_ratio"] > b0["mean_token_efficiency_ratio"]


def test_extract_patched_symbols() -> None:
    """Verifies unified diff parsing for ground-truth patched symbols."""
    diff = (
        "--- a/auth/crypto.py\n"
        "+++ b/auth/crypto.py\n"
        "@@ -10,4 +10,5 @@ def verify_token_digest(token: str)\n"
        "+    pass\n"
    )
    assert extract_patched_symbols(diff) == ["auth/crypto.py::verify_token_digest"]


def test_colab_utils_and_hardware_detection() -> None:
    """Verifies Colab utility functions and hardware inspection."""
    from repotopology.colab_utils import get_hardware_info, is_colab, link_drive_data

    hw = get_hardware_info()
    assert isinstance(hw, dict)
    assert "is_colab" in hw
    assert "gpu_available" in hw
    assert "device_name" in hw

    # Test link_drive_data fallback when drive directory doesn't exist
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        target = link_drive_data(
            drive_data_folder="non_existent_drive_folder",
            mount_point=tmp,
            target_data_dir=Path(tmp) / "data",
        )
        assert target.exists()

