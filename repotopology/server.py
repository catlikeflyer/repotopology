"""FastMCP server exposing RepoTopology graph traversal and 3-tier context pruning tools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover - compatibility with mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore[no-redef]

from repotopology.graph_store import DirectionType, GraphStore
from repotopology.pruner import ContextPruner

mcp = FastMCP("RepoTopology")

_ACTIVE_STORE: GraphStore | None = None
_ACTIVE_PRUNER: ContextPruner | None = None


def configure_server(store: GraphStore) -> None:
    """Binds an active `GraphStore` and `ContextPruner` to the FastMCP server runtime."""
    global _ACTIVE_STORE, _ACTIVE_PRUNER
    _ACTIVE_STORE = store
    _ACTIVE_PRUNER = ContextPruner(store)


def get_active_store() -> GraphStore:
    """Returns the currently loaded `GraphStore`, lazy-loading from environment variables if set."""
    global _ACTIVE_STORE, _ACTIVE_PRUNER
    if _ACTIVE_STORE is None:
        instance_id = os.environ.get("REPOTOPOLOGY_INSTANCE_ID", "")
        data_dir = Path(os.environ.get("REPOTOPOLOGY_DATA_DIR", "./data"))
        if instance_id:
            store = GraphStore.from_data_dir(instance_id=instance_id, data_dir=data_dir)
            configure_server(store)
        else:
            # Auto-discover first available graph in data_dir/graphs if present
            graphs_dir = data_dir / "graphs"
            if graphs_dir.exists():
                candidates = sorted(graphs_dir.glob("*.json"))
                if candidates:
                    store = GraphStore.from_data_dir(
                        instance_id=candidates[0].stem, data_dir=data_dir
                    )
                    configure_server(store)
    if _ACTIVE_STORE is None:
        raise RuntimeError(
            "No active GraphStore configured. Call `configure_server(store)` or set "
            "REPOTOPOLOGY_INSTANCE_ID and REPOTOPOLOGY_DATA_DIR."
        )
    return _ACTIVE_STORE


def get_active_pruner() -> ContextPruner:
    """Returns the active `ContextPruner` bound to the current `GraphStore`."""
    global _ACTIVE_PRUNER
    store = get_active_store()
    if _ACTIVE_PRUNER is None or _ACTIVE_PRUNER.store is not store:
        _ACTIVE_PRUNER = ContextPruner(store)
    return _ACTIVE_PRUNER


@mcp.tool()
def search_similar_code(query: str, top_k: int = 5) -> str:
    """Finds candidate entrypoints using cosine similarity against the loaded .npz 256-d vectors.

    Args:
        query: Natural language problem description, symbol name, or code query.
        top_k: Number of top candidate AST symbols to return (default: 5).

    Returns:
        Formatted JSON string listing the top-k candidate nodes (`node_id`, `name`, `score`).
    """
    store = get_active_store()
    hits = store.search_similar_nodes(query=query, top_k=top_k)
    payload = {
        "instance_id": store.instance_id,
        "query": query,
        "results": [
            {
                "node_id": h.node_id,
                "name": h.name,
                "score": round(h.score, 6),
                "kind": h.kind,
            }
            for h in hits
        ],
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_code_neighbors(node_id: str, direction: str = "callees") -> str:
    """Returns incoming/outgoing neighbors in the repository AST graph without returning full source text.

    Args:
        node_id: Qualified AST node identifier in the repository graph.
        direction: Traversal direction: "callees" ("outgoing"), "callers" ("incoming"), or "both".

    Returns:
        Formatted JSON string containing neighbor IDs, short names, edge types, and direction.
    """
    store = get_active_store()
    if not store.has_node(node_id):
        return json.dumps(
            {
                "error": f"Node '{node_id}' does not exist in graph '{store.instance_id}'.",
                "node_id": node_id,
                "neighbors": [],
            },
            indent=2,
        )

    neighbors = store.get_neighbors(
        node_id=node_id,
        direction=cast(DirectionType, direction),
    )
    payload = {
        "instance_id": store.instance_id,
        "focus_node_id": node_id,
        "direction": direction,
        "neighbors": [
            {
                "node_id": n.node_id,
                "name": n.name,
                "edge_type": n.edge_type,
                "direction": n.direction,
            }
            for n in neighbors
        ],
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_pruned_context(focus_node_id: str) -> str:
    """Executes 3-tier dynamic AST context pruning centered on `focus_node_id`.

    - Tier 1 (Focus Symbol): Full raw source code from `node["text"]`.
    - Tier 2 (1-hop Neighbors): Extracted signatures, type annotations, and docstrings only.
    - Tier 3 (2-hop Neighbors): Qualified symbol IDs only.

    Args:
        focus_node_id: Qualified AST node identifier to inspect.

    Returns:
        Formatted 3-tier context string ready for Gemma 4 E4B consumption.
    """
    store = get_active_store()
    if not store.has_node(focus_node_id):
        return f"ERROR: Node '{focus_node_id}' does not exist in graph '{store.instance_id}'."

    pruner = get_active_pruner()
    return pruner.get_pruned_context(focus_node_id)


def main() -> None:
    """Entrypoint to launch the RepoTopology FastMCP server over stdio."""
    parser = argparse.ArgumentParser(description="RepoTopology FastMCP stdio server")
    parser.add_argument(
        "--instance-id",
        type=str,
        default=os.environ.get("REPOTOPOLOGY_INSTANCE_ID", ""),
        help="Task instance ID to pre-load on server start",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("REPOTOPOLOGY_DATA_DIR", "./data"),
        help="Path to the competition dataset directory (default: ./data)",
    )
    args = parser.parse_args()

    if args.instance_id:
        os.environ["REPOTOPOLOGY_INSTANCE_ID"] = args.instance_id
    if args.data_dir:
        os.environ["REPOTOPOLOGY_DATA_DIR"] = args.data_dir

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
