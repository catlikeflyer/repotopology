"""Automated benchmark runner across tasks.jsonl comparing RepoTopology against 4 baselines."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from repotopology.agent import AgentTrajectory, LLMClient, ReActAgent, estimate_tokens
from repotopology.graph_store import (
    EMBEDDING_DIM,
    GraphStore,
    deterministic_text_embedding,
)

BASELINE_MODES: tuple[tuple[str, str], ...] = (
    ("baseline_0_grep", "Baseline 0: Grep / Raw File Reading"),
    ("baseline_1_rag", "Baseline 1: Dense Vector RAG Alone (search_similar_code)"),
    ("baseline_2_unpruned_graph", "Baseline 2: Unpruned Graph Agent (Full Neighbor Code)"),
    ("baseline_3_repotopology", "Baseline 3: RepoTopology (Graph Traversal + 3-Tier Pruning)"),
)


@dataclass(slots=True)
class TaskRecord:
    """Represents one benchmark task entry from `tasks.jsonl`."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    ground_truth_symbols: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        gt_syms = list(data.get("ground_truth_symbols", []))
        patch_str = str(data.get("patch", ""))
        if not gt_syms and patch_str:
            gt_syms = extract_patched_symbols(patch_str)
        return cls(
            instance_id=str(data["instance_id"]),
            repo=str(data.get("repo", "unknown/repo")),
            base_commit=str(data.get("base_commit", "HEAD")),
            problem_statement=str(data.get("problem_statement", "")),
            patch=patch_str,
            ground_truth_symbols=gt_syms,
        )


@dataclass(slots=True)
class EpisodeMetrics:
    """Per-task evaluation metrics for a single baseline mode."""

    instance_id: str
    mode: str
    localization_hit: bool
    matched_symbols: list[str]
    ground_truth_symbols: list[str]
    relevant_source_lines: int
    total_prompt_tokens: int
    token_efficiency_ratio: float
    trajectory_length: int
    structural_hallucination_rate: float


@dataclass(slots=True)
class BaselineSummary:
    """Aggregated benchmark metrics across all tasks in `tasks.jsonl` for one baseline."""

    mode: str
    label: str
    num_tasks: int
    top_k_localization_hit_rate: float
    mean_token_efficiency_ratio: float
    mean_total_prompt_tokens: float
    mean_trajectory_length: float
    structural_hallucination_rate: float
    episodes: list[EpisodeMetrics] = field(default_factory=list)


def extract_patched_symbols(patch_text: str) -> list[str]:
    """Parses git diff headers and modified `def`/`class` blocks to identify ground-truth patched symbols."""
    symbols: list[str] = []
    current_file = ""

    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue

        # Match hunk header context e.g. @@ -10,7 +10,8 @@ def verify_signature(...)
        hunk_match = re.match(r"^@@.*@@\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if hunk_match:
            sym_name = hunk_match.group(1)
            qualified = f"{current_file}::{sym_name}" if current_file else sym_name
            if qualified not in symbols:
                symbols.append(qualified)

        # Also check modified/added def or class lines
        def_match = re.match(r"^[-+\s]\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if def_match:
            sym_name = def_match.group(1)
            qualified = f"{current_file}::{sym_name}" if current_file else sym_name
            if qualified not in symbols:
                symbols.append(qualified)

    return symbols


def evaluate_trajectory(
    task: TaskRecord,
    store: GraphStore,
    trajectory: AgentTrajectory,
) -> EpisodeMetrics:
    """Computes localization hit rate, token efficiency ratio, trajectory length, and hallucination rate."""
    gt_targets = list(task.ground_truth_symbols)
    if not gt_targets and task.patch:
        gt_targets = extract_patched_symbols(task.patch)

    matched: list[str] = []
    relevant_lines = 0

    for gt in gt_targets:
        short_name = gt.split("::")[-1].split(".")[-1]
        dot_cand = gt.replace("/", ".").replace(".py::", ".").replace("::", ".")

        in_localized = any(
            gt == cand
            or cand == dot_cand
            or cand.endswith(f"::{short_name}")
            or cand.endswith(f".{short_name}")
            or cand == short_name
            for cand in trajectory.localized_symbols
        )
        in_context = (
            (gt in trajectory.final_context)
            or (dot_cand in trajectory.final_context)
            or (bool(short_name) and short_name in trajectory.final_context)
        )
        if in_localized or in_context:
            matched.append(gt)
            resolved_node_id = _resolve_node_id(store, gt, short_name)
            if resolved_node_id is not None:
                node_text = str(store.get_node_data(resolved_node_id).get("text", ""))
                relevant_lines += len([ln for ln in node_text.splitlines() if ln.strip()])

    localization_hit = len(matched) > 0 if gt_targets else False
    prompt_tokens = max(1, trajectory.total_prompt_tokens or estimate_tokens(trajectory.final_context))
    token_efficiency = round(relevant_lines / float(prompt_tokens), 6)

    return EpisodeMetrics(
        instance_id=task.instance_id,
        mode=trajectory.mode,
        localization_hit=localization_hit,
        matched_symbols=matched,
        ground_truth_symbols=gt_targets,
        relevant_source_lines=relevant_lines,
        total_prompt_tokens=prompt_tokens,
        token_efficiency_ratio=token_efficiency,
        trajectory_length=trajectory.trajectory_length,
        structural_hallucination_rate=round(trajectory.hallucination_rate, 6),
    )


def _resolve_node_id(store: GraphStore, gt_symbol: str, short_name: str) -> str | None:
    if store.has_node(gt_symbol):
        return gt_symbol

    dot_cand = gt_symbol.replace("/", ".").replace(".py::", ".").replace("::", ".")
    if store.has_node(dot_cand):
        return dot_cand

    for nid, ndata in store.graph.nodes(data=True):
        nid_str = str(nid)
        if (
            nid_str.endswith(f"::{short_name}")
            or nid_str.endswith(f".{short_name}")
            or str(ndata.get("name", "")) == short_name
        ):
            return nid_str
    return None


class BenchmarkEvaluator:
    """Runs batch evaluation across `tasks.jsonl` for all 4 baselines."""

    def __init__(
        self,
        data_dir: str | Path = "./data",
        top_k: int = 5,
        max_steps: int = 6,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self.max_steps = max_steps
        self.llm_client = llm_client

    def load_tasks(self, tasks_file: str | Path | None = None) -> list[TaskRecord]:
        """Reads all benchmark tasks from `<data_dir>/tasks.jsonl`."""
        path = Path(tasks_file) if tasks_file is not None else (self.data_dir / "tasks.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"Tasks file not found: {path}")

        tasks: list[TaskRecord] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            tasks.append(TaskRecord.from_dict(json.loads(stripped)))
        return tasks

    def run_all(
        self,
        tasks_file: str | Path | None = None,
        max_tasks: int | None = None,
    ) -> dict[str, Any]:
        """Runs all 4 baseline ablations over `tasks.jsonl` and returns a structured JSON summary."""
        tasks = self.load_tasks(tasks_file=tasks_file)
        if max_tasks is not None and max_tasks > 0:
            tasks = tasks[:max_tasks]
        stores: dict[str, GraphStore] = {
            t.instance_id: GraphStore.from_data_dir(
                t.instance_id,
                data_dir=self.data_dir,
                repo=t.repo,
                base_commit=t.base_commit,
            )
            for t in tasks
        }

        summaries: dict[str, BaselineSummary] = {}
        for mode_key, mode_label in BASELINE_MODES:
            episodes: list[EpisodeMetrics] = []
            for task in tasks:
                store = stores[task.instance_id]
                agent = ReActAgent(
                    store=store,
                    llm_client=self.llm_client,
                    max_steps=self.max_steps,
                    top_k=self.top_k,
                )
                traj = agent.run(task.problem_statement, mode=mode_key)
                metrics = evaluate_trajectory(task=task, store=store, trajectory=traj)
                episodes.append(metrics)

            n = max(1, len(episodes))
            hit_rate = sum(1 for e in episodes if e.localization_hit) / float(n)
            mean_eff = sum(e.token_efficiency_ratio for e in episodes) / float(n)
            mean_tokens = sum(e.total_prompt_tokens for e in episodes) / float(n)
            mean_len = sum(e.trajectory_length for e in episodes) / float(n)
            mean_halluc = sum(e.structural_hallucination_rate for e in episodes) / float(n)

            summaries[mode_key] = BaselineSummary(
                mode=mode_key,
                label=mode_label,
                num_tasks=len(episodes),
                top_k_localization_hit_rate=round(hit_rate, 4),
                mean_token_efficiency_ratio=round(mean_eff, 6),
                mean_total_prompt_tokens=round(mean_tokens, 2),
                mean_trajectory_length=round(mean_len, 2),
                structural_hallucination_rate=round(mean_halluc, 4),
                episodes=episodes,
            )

        return {
            "benchmark": "RepoTopology - Google Gemma 4 E4B Developer Agent Evaluation",
            "num_tasks": len(tasks),
            "top_k": self.top_k,
            "baselines": {k: asdict(v) for k, v in summaries.items()},
        }


def create_sample_dataset(data_dir: str | Path = "./data") -> Path:
    """Creates a deterministic sample competition dataset in `data_dir` for local verification."""
    base = Path(data_dir)
    graphs_dir = base / "graphs"
    emb_dir = base / "embeddings"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    instance_id = "authlib__token_verify_001"
    nodes = [
        {
            "id": "auth/router.py::authenticate_request",
            "name": "authenticate_request",
            "text": (
                "def authenticate_request(headers: dict[str, str], payload: bytes) -> bool:\n"
                '    """Entrypoint router validating incoming webhook request headers and JWT signature."""\n'
                '    token = headers.get("X-Webhook-Signature", "")\n'
                "    if not token:\n"
                "        raise ValueError('Missing X-Webhook-Signature header')\n"
                "    return verify_token_digest(token, payload)\n"
            ),
        },
        {
            "id": "auth/crypto.py::verify_token_digest",
            "name": "verify_token_digest",
            "text": (
                "def verify_token_digest(token: str, payload: bytes, leeway_seconds: int = 0) -> bool:\n"
                '    """Verifies HMAC-SHA256 digest and timestamp drift for webhook tokens."""\n'
                "    parts = token.split('.')\n"
                "    if len(parts) != 2:\n"
                "        return False\n"
                "    ts_raw, digest = parts[0], parts[1]\n"
                "    expected = compute_hmac_sha256(ts_raw.encode('utf-8') + b'.' + payload)\n"
                "    # BUG: fails when clock skew exceeds 0s because leeway_seconds is ignored\n"
                "    return digest == expected\n"
            ),
        },
        {
            "id": "auth/primitives.py::compute_hmac_sha256",
            "name": "compute_hmac_sha256",
            "text": (
                "def compute_hmac_sha256(message: bytes, secret: bytes = b'default-key') -> str:\n"
                '    """Computes hex-encoded HMAC-SHA256 digest over message bytes."""\n'
                "    import hmac\n"
                "    import hashlib\n"
                "    mac = hmac.new(secret, message, hashlib.sha256)\n"
                "    digest_hex = mac.hexdigest()\n"
                "    return digest_hex.lower()\n"
            ),
        },
        {
            "id": "billing/invoice.py::reconcile_ledger",
            "name": "reconcile_ledger",
            "text": (
                "def reconcile_ledger(account_id: str, entries: list[dict[str, float]]) -> float:\n"
                '    """Unrelated heavy module computing multi-currency ledger balances."""\n'
                "    balance = 0.0\n"
                "    for item in entries:\n"
                "        debit = float(item.get('debit', 0.0))\n"
                "        credit = float(item.get('credit', 0.0))\n"
                "        fee = float(item.get('exchange_fee', 0.0))\n"
                "        balance += (credit - debit - fee)\n"
                "    return round(balance, 4)\n"
            ),
        },
    ]

    graph = nx.MultiDiGraph()
    for n in nodes:
        graph.add_node(n["id"], name=n["name"], text=n["text"])

    graph.add_edge(
        "auth/router.py::authenticate_request",
        "auth/crypto.py::verify_token_digest",
        type="calls",
    )
    graph.add_edge(
        "auth/crypto.py::verify_token_digest",
        "auth/primitives.py::compute_hmac_sha256",
        type="calls",
    )

    try:
        graph_json = nx.node_link_data(graph, edges="links")
    except TypeError:
        graph_json = nx.node_link_data(graph)
    (graphs_dir / f"{instance_id}.json").write_text(
        json.dumps(graph_json, indent=2), encoding="utf-8"
    )

    emb_dict: dict[str, np.ndarray] = {}
    for n in nodes:
        emb_dict[n["id"]] = deterministic_text_embedding(
            f"{n['id']} {n['name']} {n['text']}", dim=EMBEDDING_DIM
        )
    np.savez(emb_dir / f"{instance_id}.npz", **emb_dict)

    task_obj = {
        "instance_id": instance_id,
        "repo": "authlib/webhook-core",
        "base_commit": "a1b2c3d4e5f6",
        "problem_statement": (
            "Webhook authentication in authenticate_request fails intermittently when clock skew "
            "occurs because downstream HMAC digest verification ignores leeway_seconds."
        ),
        "patch": (
            "--- a/auth/crypto.py\n"
            "+++ b/auth/crypto.py\n"
            "@@ -1,8 +1,9 @@ def verify_token_digest\n"
            " def verify_token_digest(token: str, payload: bytes, leeway_seconds: int = 30) -> bool:\n"
        ),
        "ground_truth_symbols": ["auth/crypto.py::verify_token_digest"],
    }
    tasks_path = base / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task_obj) + "\n", encoding="utf-8")
    return tasks_path


def main() -> None:
    """CLI entrypoint for running the RepoTopology 4-baseline benchmark."""
    parser = argparse.ArgumentParser(
        description="RepoTopology Evaluation Harness (4 Baseline Ablations)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to dataset directory containing tasks.jsonl, graphs/, embeddings/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON evaluation summary",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k localization cutoff (default: 5)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Maximum tasks to evaluate (0 for all tasks, default: 0)",
    )
    parser.add_argument(
        "--generate-sample-if-missing",
        action="store_true",
        default=True,
        help="Generate synthetic sample dataset if ./data/tasks.jsonl is not present",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tasks_file = data_dir / "tasks.jsonl"
    if not tasks_file.exists() and args.generate_sample_if_missing:
        create_sample_dataset(data_dir)

    evaluator = BenchmarkEvaluator(data_dir=data_dir, top_k=args.top_k)
    summary = evaluator.run_all(max_tasks=args.max_tasks if args.max_tasks > 0 else None)
    formatted = json.dumps(summary, indent=2)
    print(formatted)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
