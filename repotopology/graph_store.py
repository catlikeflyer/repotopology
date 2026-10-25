"""NetworkX directed multigraph and 256-d NPZ embedding store for RepoTopology."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

EMBEDDING_DIM = 256
DirectionType = Literal["callees", "callers", "outgoing", "incoming", "both"]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Represents a semantic/hybrid search hit over repository AST nodes."""

    node_id: str
    name: str
    score: float
    kind: str = "symbol"
    file_path: str = ""


@dataclass(frozen=True, slots=True)
class NeighborRecord:
    """Lightweight representation of a 1-hop neighbor edge without full source code."""

    node_id: str
    name: str
    edge_type: str
    direction: Literal["outgoing", "incoming"]


@dataclass(slots=True)
class GraphStore:
    """Manages repository AST MultiDiGraph and 256-d node embeddings."""

    instance_id: str
    graph: nx.MultiDiGraph
    embeddings: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    query_embedder: Callable[[str], NDArray[np.float32]] | None = None

    _node_ids: list[str] = field(init=False, default_factory=list, repr=False)
    _norm_matrix: NDArray[np.float32] = field(
        init=False,
        default_factory=lambda: np.empty((0, EMBEDDING_DIM), dtype=np.float32),
        repr=False,
    )
    _token_index: dict[str, list[int]] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Precomputes L2-normalized embedding matrix and lexical token index."""
        self._node_ids = []
        vectors: list[NDArray[np.float32]] = []

        for node_id, vec in self.embeddings.items():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] != EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {EMBEDDING_DIM}-d embedding for node '{node_id}', got shape {arr.shape}"
                )
            norm = float(np.linalg.norm(arr))
            normalized = arr / norm if norm > 1e-12 else arr
            self._node_ids.append(node_id)
            vectors.append(normalized)

        if vectors:
            self._norm_matrix = np.vstack(vectors).astype(np.float32)
        else:
            self._norm_matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        # Build token index mapping identifier tokens to row indices in _norm_matrix
        self._token_index.clear()
        for idx, node_id in enumerate(self._node_ids):
            node_data = self.graph.nodes.get(node_id, {})
            text_blob = f"{node_id} {node_data.get('name', '')} {node_data.get('text', '')}"
            for tok in _tokenize_identifiers(text_blob):
                self._token_index.setdefault(tok, []).append(idx)

    @classmethod
    def from_files(
        cls,
        graph_path: str | Path,
        embedding_path: str | Path | None = None,
        instance_id: str | None = None,
        query_embedder: Callable[[str], NDArray[np.float32]] | None = None,
    ) -> GraphStore:
        """Loads a NetworkX MultiDiGraph JSON and 256-d NPZ embeddings from disk."""
        graph_file = Path(graph_path)
        if not graph_file.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        resolved_id = instance_id or graph_file.stem
        raw_data = json.loads(graph_file.read_text(encoding="utf-8"))
        graph = _load_multidigraph(raw_data)

        embeddings: dict[str, NDArray[np.float32]] = {}
        if embedding_path is not None:
            emb_file = Path(embedding_path)
            if not emb_file.exists():
                raise FileNotFoundError(f"Embedding file not found: {emb_file}")
            embeddings = _load_npz_embeddings(emb_file)

        return cls(
            instance_id=resolved_id,
            graph=graph,
            embeddings=embeddings,
            query_embedder=query_embedder,
        )

    @classmethod
    def from_data_dir(
        cls,
        instance_id: str,
        data_dir: str | Path = "./data",
        repo: str | None = None,
        base_commit: str | None = None,
        query_embedder: Callable[[str], NDArray[np.float32]] | None = None,
    ) -> GraphStore:
        """Loads `<data_dir>/graphs/<instance_id>.json` (or `<repo>_<commit>.json`) and embeddings."""
        base = Path(data_dir)
        graphs_dir = base / "graphs"
        embs_dir = base / "embeddings"

        graph_path = graphs_dir / f"{instance_id}.json"
        emb_path = embs_dir / f"{instance_id}.npz"

        # Check for competition naming format: <repo_name>_<base_commit>.json
        if not graph_path.exists() and repo and base_commit:
            repo_name = repo.split("/")[-1]
            alt_graph = graphs_dir / f"{repo_name}_{base_commit}.json"
            alt_emb = embs_dir / f"{repo_name}_{base_commit}.npz"
            if alt_graph.exists():
                graph_path = alt_graph
            if alt_emb.exists():
                emb_path = alt_emb

        return cls.from_files(
            graph_path=graph_path,
            embedding_path=emb_path if emb_path.exists() else None,
            instance_id=instance_id,
            query_embedder=query_embedder,
        )

    def has_node(self, node_id: str) -> bool:
        """Returns True if `node_id` exists in the repository graph."""
        return self.graph.has_node(node_id)

    def get_node_data(self, node_id: str) -> dict[str, Any]:
        """Returns the attribute dictionary for `node_id` or raises KeyError."""
        if not self.graph.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found in graph '{self.instance_id}'.")
        data = dict(self.graph.nodes[node_id])
        data.setdefault("id", node_id)
        data.setdefault("name", node_id.split("::")[-1].split(".")[-1])
        data.setdefault("text", "")
        return data

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Produces a 256-d L2-normalized query vector for cosine similarity search."""
        if self.query_embedder is not None:
            vec = np.asarray(self.query_embedder(query), dtype=np.float32).reshape(-1)
            if vec.shape[0] != EMBEDDING_DIM:
                raise ValueError(
                    f"Custom query_embedder returned shape {vec.shape}, expected ({EMBEDDING_DIM},)"
                )
        else:
            vec = self._default_query_embedding(query)

        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 1e-12 else vec).astype(np.float32)

    def _default_query_embedding(self, query: str) -> NDArray[np.float32]:
        """Synthesizes a 256-d query vector aligned with node embeddings and hashed n-grams."""
        acc = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        tokens = _tokenize_identifiers(query)
        matched_rows = False
        if self._norm_matrix.shape[0] > 0:
            for tok in tokens:
                row_indices = self._token_index.get(tok)
                if row_indices:
                    matched_rows = True
                    # Inverse frequency weighting so specific symbol names dominate
                    idf = 1.0 / (1.0 + np.log1p(len(row_indices)))
                    acc += idf * np.sum(self._norm_matrix[row_indices], axis=0)

        if not matched_rows:
            acc = deterministic_text_embedding(query)
        else:
            # Blend a small deterministic hash component for tie-breaking stability
            acc += 0.05 * deterministic_text_embedding(query)
        return acc

    def search_similar_nodes(
        self,
        query: str | NDArray[np.float32],
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Finds top-k candidate entrypoints using cosine similarity over 256-d vectors."""
        if top_k <= 0:
            return []

        if isinstance(query, str):
            if self._norm_matrix.shape[0] == 0:
                return self._lexical_fallback_search(query, top_k=top_k)
            q_vec = self.embed_query(query)
        else:
            arr = np.asarray(query, dtype=np.float32).reshape(-1)
            if arr.shape[0] != EMBEDDING_DIM:
                raise ValueError(f"Expected {EMBEDDING_DIM}-d query vector, got {arr.shape}")
            norm = float(np.linalg.norm(arr))
            q_vec = (arr / norm if norm > 1e-12 else arr).astype(np.float32)

        if self._norm_matrix.shape[0] == 0:
            return []

        scores = self._norm_matrix @ q_vec
        k = min(top_k, len(self._node_ids))
        top_indices = np.argsort(-scores, kind="stable")[:k]

        hits: list[SearchHit] = []
        for idx in top_indices:
            nid = self._node_ids[int(idx)]
            ndata = self.graph.nodes.get(nid, {})
            hits.append(
                SearchHit(
                    node_id=nid,
                    name=str(ndata.get("name", nid)),
                    score=float(scores[int(idx)]),
                    kind=str(ndata.get("kind", "symbol")),
                    file_path=str(ndata.get("file_path", "")),
                )
            )
        return hits

    def _lexical_fallback_search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Fallback scorer if no NPZ embeddings are loaded."""
        q_tokens = set(_tokenize_identifiers(query))
        scored: list[tuple[float, str]] = []
        for nid, ndata in self.graph.nodes(data=True):
            blob = f"{nid} {ndata.get('name', '')} {ndata.get('text', '')}"
            n_tokens = set(_tokenize_identifiers(blob))
            if not q_tokens or not n_tokens:
                overlap = 0.0
            else:
                overlap = len(q_tokens & n_tokens) / float(len(q_tokens))
            scored.append((overlap, str(nid)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[SearchHit] = []
        for score, nid in scored[:top_k]:
            ndata = self.graph.nodes.get(nid, {})
            hits.append(
                SearchHit(
                    node_id=nid,
                    name=str(ndata.get("name", nid)),
                    score=score,
                    kind=str(ndata.get("kind", "symbol")),
                    file_path=str(ndata.get("file_path", "")),
                )
            )
        return hits

    def get_neighbors(
        self,
        node_id: str,
        direction: DirectionType = "callees",
    ) -> list[NeighborRecord]:
        """Returns incoming and/or outgoing 1-hop neighbors without returning full source text."""
        if not self.graph.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found in graph '{self.instance_id}'.")

        norm_dir = direction.lower().strip()
        if norm_dir not in {"callees", "callers", "outgoing", "incoming", "both"}:
            raise ValueError(
                f"Invalid direction '{direction}'. Expected 'callees', 'callers', 'outgoing', 'incoming', or 'both'."
            )

        records: list[NeighborRecord] = []
        seen: set[tuple[str, str, str]] = set()

        if norm_dir in {"callees", "outgoing", "both"}:
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                if target == node_id:
                    continue
                edge_type = str(edge_data.get("type", "calls"))
                key = (str(target), edge_type, "outgoing")
                if key not in seen:
                    seen.add(key)
                    target_name = str(self.graph.nodes[target].get("name", target))
                    records.append(
                        NeighborRecord(
                            node_id=str(target),
                            name=target_name,
                            edge_type=edge_type,
                            direction="outgoing",
                        )
                    )

        if norm_dir in {"callers", "incoming", "both"}:
            for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                if source == node_id:
                    continue
                edge_type = str(edge_data.get("type", "calls"))
                key = (str(source), edge_type, "incoming")
                if key not in seen:
                    seen.add(key)
                    source_name = str(self.graph.nodes[source].get("name", source))
                    records.append(
                        NeighborRecord(
                            node_id=str(source),
                            name=source_name,
                            edge_type=edge_type,
                            direction="incoming",
                        )
                    )

        return records

    def get_k_hop_tiers(self, focus_node_id: str) -> tuple[list[str], list[str]]:
        """Computes deterministic 1-hop and 2-hop neighbor node IDs (undirected view of MultiDiGraph)."""
        if not self.graph.has_node(focus_node_id):
            raise KeyError(f"Node '{focus_node_id}' not found in graph '{self.instance_id}'.")

        tier1_set: set[str] = set()
        for _, target in self.graph.out_edges(focus_node_id):
            if target != focus_node_id:
                tier1_set.add(str(target))
        for source, _ in self.graph.in_edges(focus_node_id):
            if source != focus_node_id:
                tier1_set.add(str(source))

        tier1_ids = sorted(tier1_set)

        tier2_set: set[str] = set()
        for n1 in tier1_ids:
            for _, target in self.graph.out_edges(n1):
                t_str = str(target)
                if t_str != focus_node_id and t_str not in tier1_set:
                    tier2_set.add(t_str)
            for source, _ in self.graph.in_edges(n1):
                s_str = str(source)
                if s_str != focus_node_id and s_str not in tier1_set:
                    tier2_set.add(s_str)

        tier2_ids = sorted(tier2_set)
        return tier1_ids, tier2_ids


def deterministic_text_embedding(text: str, dim: int = EMBEDDING_DIM) -> NDArray[np.float32]:
    """Computes a deterministic 256-d float32 unit vector from text tokens using signed feature hashing."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize_identifiers(text)
    if not tokens:
        tokens = [text.strip() or "__empty__"]

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        idx1 = int.from_bytes(digest[:4], "little") % dim
        idx2 = int.from_bytes(digest[4:8], "little") % dim
        sign1 = 1.0 if (digest[8] & 1) == 0 else -1.0
        sign2 = 1.0 if (digest[9] & 1) == 0 else -1.0
        vec[idx1] += sign1
        vec[idx2] += 0.5 * sign2

    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec /= norm
    return vec.astype(np.float32)


def _tokenize_identifiers(text: str) -> list[str]:
    """Splits source text and camelCase/snake_case symbols into normalized tokens."""
    raw_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    tokens: list[str] = []
    for word in raw_words:
        lower = word.lower()
        tokens.append(lower)
        # Split snake_case and camelCase sub-parts
        subparts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", word).lower().split("_")
        for part in subparts:
            if len(part) >= 2 and part != lower:
                tokens.append(part)
    return tokens


def _load_multidigraph(raw_data: dict[str, Any]) -> nx.MultiDiGraph:
    """Deserializes a node-link JSON object into a NetworkX MultiDiGraph."""
    data = dict(raw_data)
    data["directed"] = True
    data["multigraph"] = True
    edge_key = "edges" if "edges" in data and "links" not in data else "links"
    try:
        graph = nx.node_link_graph(data, directed=True, multigraph=True, edges=edge_key)
    except TypeError:
        # Fallback for older NetworkX versions where `link` was the keyword argument
        graph = nx.node_link_graph(data, directed=True, multigraph=True, link=edge_key)

    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    return graph


def _load_npz_embeddings(emb_file: Path) -> dict[str, NDArray[np.float32]]:
    """Loads node embeddings from an `.npz` archive keyed by `<node_id>`."""
    embeddings: dict[str, NDArray[np.float32]] = {}
    with np.load(emb_file, allow_pickle=False) as archive:
        keys = list(archive.files)
        if "node_ids" in archive and "embeddings" in archive and len(keys) == 2:
            node_ids = archive["node_ids"]
            matrix = archive["embeddings"].astype(np.float32)
            for nid, row in zip(node_ids, matrix, strict=True):
                embeddings[str(nid)] = np.asarray(row, dtype=np.float32)
        else:
            for key in keys:
                embeddings[str(key)] = np.asarray(archive[key], dtype=np.float32)
    return embeddings
