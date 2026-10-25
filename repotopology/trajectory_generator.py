"""Golden trajectory synthesis for fine-tuning Gemma 4 E4B on 3-tier AST graph pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx

from repotopology.agent import DEFAULT_SYSTEM_PROMPT
from repotopology.evaluator import BenchmarkEvaluator, TaskRecord
from repotopology.graph_store import GraphStore
from repotopology.pruner import ContextPruner


def resolve_symbol_in_graph(store: GraphStore, gt_symbol: str) -> str | None:
    """Resolves a git-diff symbol (e.g. path/file.py::func) to a graph node ID."""
    if store.has_node(gt_symbol):
        return gt_symbol

    # Dot-notation conversion: fastapi/dependencies/utils.py::foo -> fastapi.dependencies.utils.foo
    dot_candidate = (
        gt_symbol.replace("/", ".").replace(".py::", ".").replace("::", ".")
    )
    if store.has_node(dot_candidate):
        return dot_candidate

    short = gt_symbol.split("::")[-1].split(".")[-1]
    # Suffix match on qualified symbol or exact name match
    for nid, ndata in store.graph.nodes(data=True):
        nid_str = str(nid)
        if nid_str.endswith(f".{short}") or str(ndata.get("name", "")) == short:
            return nid_str

    return None


def synthesize_task_trajectory(
    task: TaskRecord,
    store: GraphStore,
    pruner: ContextPruner,
) -> list[dict[str, str]] | None:
    """Synthesizes a golden multi-turn ReAct trajectory for a single task."""
    gt_targets = task.ground_truth_symbols
    if not gt_targets:
        return None

    resolved_target = None
    for cand in gt_targets:
        resolved = resolve_symbol_in_graph(store, cand)
        if resolved:
            resolved_target = resolved
            break

    if not resolved_target:
        return None

    # Step 1: Semantic entrypoint discovery
    hits = store.search_similar_nodes(query=task.problem_statement, top_k=3)
    if not hits:
        return None

    entrypoint = hits[0].node_id
    search_payload = [
        {"node_id": h.node_id, "name": h.name, "score": round(h.score, 4)} for h in hits
    ]

    problem_text = task.problem_statement.strip()[:600]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem Statement:\n{problem_text}"},
    ]

    # Turn 1: Search entrypoints
    turn1_thought = (
        f"Search 256-d AST embeddings for entrypoints relevant to the problem statement."
    )
    turn1_action = {
        "tool": "search_similar_code",
        "arguments": {"query": problem_text[:120], "top_k": 3},
    }
    messages.append({
        "role": "assistant",
        "content": f"Thought: {turn1_thought}\nAction: {json.dumps(turn1_action)}",
    })
    messages.append({
        "role": "user",
        "content": f"Observation:\n{json.dumps(search_payload, indent=2)}",
    })

    # Find path in AST graph from entrypoint to resolved_target
    undirected = store.graph.to_undirected(as_view=True)
    current_node = entrypoint
    path: list[str] = []
    if nx.has_path(undirected, entrypoint, resolved_target):
        path = nx.shortest_path(undirected, entrypoint, resolved_target)
    else:
        path = [entrypoint, resolved_target]

    # Turn 2: Traverse neighbors along path
    if len(path) > 1:
        step_target = path[1] if len(path) > 1 else resolved_target
        neighbors = store.get_neighbors(current_node, direction="both")
        # Cap to top 6 neighbors to keep token budget bounded
        neighbors_payload = [
            {
                "node_id": n.node_id,
                "name": n.name,
                "edge_type": n.edge_type,
                "direction": n.direction,
            }
            for n in neighbors[:6]
        ]
        turn2_thought = (
            f"Inspect call and import graph neighbors of '{current_node}' to locate cross-boundary logic."
        )
        turn2_action = {
            "tool": "get_code_neighbors",
            "arguments": {"node_id": current_node, "direction": "both"},
        }
        messages.append({
            "role": "assistant",
            "content": f"Thought: {turn2_thought}\nAction: {json.dumps(turn2_action)}",
        })
        messages.append({
            "role": "user",
            "content": f"Observation:\n{json.dumps(neighbors_payload, indent=2)}",
        })

    # Turn 3: 3-Tier context pruning on target
    pruned_obj = pruner.prune(resolved_target)
    # Compact bounded format to guarantee prompt fits strict token budgets
    compact_lines = [
        f"=== TIER 1: FOCUS SYMBOL [{resolved_target}] ===",
        pruned_obj.tier1_source.strip()[:800] or "# <empty>",
        "\n=== TIER 2: 1-HOP NEIGHBORS (Stubs) ===",
    ]
    for nid, stub in list(pruned_obj.tier2_stubs.items())[:4]:
        compact_lines.append(f"# {nid}\n{stub.strip()[:250]}")
    compact_lines.append("\n=== TIER 3: 2-HOP NEIGHBORS ===")
    for nid in pruned_obj.tier3_symbol_ids[:8]:
        compact_lines.append(f"- {nid}")
    pruned_ctx = "\n".join(compact_lines)[:1800]

    turn3_thought = (
        f"Extract 3-tier dynamic AST context centered on '{resolved_target}' to isolate implementation details without context saturation."
    )
    turn3_action = {
        "tool": "get_pruned_context",
        "arguments": {"focus_node_id": resolved_target},
    }
    messages.append({
        "role": "assistant",
        "content": f"Thought: {turn3_thought}\nAction: {json.dumps(turn3_action)}",
    })
    messages.append({
        "role": "user",
        "content": f"Observation:\n{pruned_ctx}",
    })

    # Turn 4: Finish with localized symbols
    turn4_thought = (
        f"Verified target logic in '{resolved_target}'. Localization complete."
    )
    turn4_action = {
        "tool": "finish",
        "arguments": {
            "localized_symbols": [resolved_target],
            "summary": f"Identified defect root cause in {resolved_target}.",
        },
    }
    messages.append({
        "role": "assistant",
        "content": f"Thought: {turn4_thought}\nAction: {json.dumps(turn4_action)}",
    })

    return messages


def split_trajectory_into_decision_turns(
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
) -> list[list[dict[str, str]]]:
    """Splits a multi-turn trajectory into single-decision samples ending on each assistant turn."""
    decision_samples: list[list[dict[str, str]]] = []
    current_prefix: list[dict[str, str]] = []

    for msg in messages:
        current_prefix.append(msg)
        if msg["role"] == "assistant":
            # Estimate token length
            total_chars = sum(len(m["content"]) for m in current_prefix)
            if total_chars // 4 <= max_tokens:
                decision_samples.append(list(current_prefix))
            else:
                # If too long, keep system prompt + latest user prompt + assistant target
                compact_sample = [current_prefix[0], current_prefix[-2], current_prefix[-1]]
                decision_samples.append(compact_sample)

    return decision_samples


def export_trajectories(
    data_dir: str | Path = "./data",
    output_path: str | Path = "./data/sft_trajectories.jsonl",
    format_type: str = "messages",
    split_turns: bool = True,
    max_tokens: int = 1024,
) -> int:
    """Exports golden multi-turn trajectories for all tasks in tasks.jsonl."""
    base_dir = Path(data_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    evaluator = BenchmarkEvaluator(data_dir=base_dir)
    tasks = evaluator.load_tasks()

    exported_count = 0
    all_records: list[dict[str, Any]] = []

    for task in tasks:
        store = GraphStore.from_data_dir(
            task.instance_id,
            data_dir=base_dir,
            repo=task.repo,
            base_commit=task.base_commit,
        )
        pruner = ContextPruner(store)
        trajectory = synthesize_task_trajectory(task, store, pruner)
        if not trajectory:
            continue

        if split_turns:
            samples = split_trajectory_into_decision_turns(trajectory, max_tokens=max_tokens)
        else:
            samples = [trajectory]

        for idx, sample in enumerate(samples):
            if format_type == "sharegpt":
                role_map = {"system": "system", "user": "human", "assistant": "gpt"}
                record = {
                    "instance_id": f"{task.instance_id}_t{idx}",
                    "conversations": [
                        {"from": role_map.get(m["role"], "human"), "value": m["content"]}
                        for m in sample
                    ],
                }
            else:
                record = {
                    "instance_id": f"{task.instance_id}_t{idx}",
                    "messages": sample,
                }
            all_records.append(record)

    with out_file.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
            exported_count += 1

    # Also automatically write split train.jsonl and valid.jsonl for MLX LM
    mlx_dir = base_dir / "mlx"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    split_idx = int(len(all_records) * 0.9)
    train_records = all_records[:split_idx]
    valid_records = all_records[split_idx:] if split_idx < len(all_records) else all_records[-10:]

    with (mlx_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")

    with (mlx_dir / "valid.jsonl").open("w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")

    print(
        f"Exported {exported_count} samples ({len(train_records)} train, {len(valid_records)} valid) to {mlx_dir}"
    )
    return exported_count


def main() -> None:
    parser = argparse.ArgumentParser(description="RepoTopology SFT Trajectory Synthesizer")
    parser.add_argument("--data-dir", type=str, default="./data", help="Competition data directory")
    parser.add_argument("--output", type=str, default="./data/sft_trajectories.jsonl", help="Output JSONL path")
    parser.add_argument("--format", choices=["messages", "sharegpt"], default="messages", help="Dataset format")
    parser.add_argument("--no-split", action="store_true", help="Do not split into per-turn samples")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max sequence length per sample")
    args = parser.parse_args()

    count = export_trajectories(
        data_dir=args.data_dir,
        output_path=args.output,
        format_type=args.format,
        split_turns=not args.no_split,
        max_tokens=args.max_tokens,
    )
    print(f"Exported {count} golden trajectory decision samples to {args.output}")


if __name__ == "__main__":
    main()
