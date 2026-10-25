"""3-Tier AST dynamic context pruner for RepoTopology."""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field

from repotopology.graph_store import GraphStore


@dataclass(frozen=True, slots=True)
class PrunedContextResult:
    """Structured output of 3-tier dynamic context pruning around a focus node."""

    focus_node_id: str
    tier1_source: str
    tier2_stubs: dict[str, str] = field(default_factory=dict)
    tier3_symbol_ids: list[str] = field(default_factory=list)

    def format_prompt_context(self) -> str:
        """Renders the 3-tier context into a structured, token-efficient markdown block."""
        sections: list[str] = [
            f"=== TIER 1: FOCUS SYMBOL [{self.focus_node_id}] (Full Implementation) ===",
            self.tier1_source.strip() or "# <empty source>",
        ]

        sections.append("\n=== TIER 2: 1-HOP NEIGHBORS (Signatures & Docstrings Only) ===")
        if self.tier2_stubs:
            for node_id, stub in self.tier2_stubs.items():
                sections.append(f"# Symbol: {node_id}\n{stub.strip()}")
        else:
            sections.append("# None")

        sections.append("\n=== TIER 3: 2-HOP NEIGHBORS (Qualified Symbol IDs Only) ===")
        if self.tier3_symbol_ids:
            for sym_id in self.tier3_symbol_ids:
                sections.append(f"- {sym_id}")
        else:
            sections.append("# None")

        return "\n".join(sections)


def extract_signature_and_docstring(source_code: str) -> str:
    """Extracts Python signatures, decorators, type annotations, and docstrings while stripping bodies.

    Uses Python's `ast` module to transform function/method bodies into `...` while keeping
    docstrings and class-level type annotations. Falls back to a line-based extractor if
    the snippet is not valid standalone Python syntax.
    """
    cleaned = textwrap.dedent(source_code).strip()
    if not cleaned:
        return ""

    try:
        tree = ast.parse(cleaned)
        stubbed_tree = _StubTransformer().visit(tree)
        ast.fix_missing_locations(stubbed_tree)
        return ast.unparse(stubbed_tree).strip()
    except SyntaxError:
        return _fallback_signature_extractor(cleaned)


class _StubTransformer(ast.NodeTransformer):
    """AST transformer that strips executable statements from functions/methods, keeping docstrings."""

    def _strip_callable_body(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        new_body: list[ast.stmt] = []
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            new_body.append(node.body[0])
        new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
        node.body = new_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._strip_callable_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._strip_callable_body(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        new_body: list[ast.stmt] = []
        for idx, stmt in enumerate(node.body):
            # Keep class docstring
            if (
                idx == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                new_body.append(stmt)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visited = self.visit(stmt)
                if isinstance(visited, ast.stmt):
                    new_body.append(visited)
            elif isinstance(stmt, ast.AnnAssign):
                # Keep class attribute type annotations without noisy multi-line values
                new_body.append(
                    ast.AnnAssign(
                        target=stmt.target,
                        annotation=stmt.annotation,
                        value=None,
                        simple=stmt.simple,
                    )
                )
        if not new_body:
            new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
        node.body = new_body
        return node


def _fallback_signature_extractor(source_code: str) -> str:
    """Fallback signature & docstring extractor for partial or non-Python code snippets."""
    lines = source_code.splitlines()
    kept: list[str] = []
    in_signature = False
    in_docstring = False
    doc_delim = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if in_docstring:
            kept.append(line)
            if doc_delim and stripped.endswith(doc_delim) and len(stripped) >= len(doc_delim):
                in_docstring = False
            continue

        if stripped.startswith("@"):
            kept.append(line)
            continue

        if re.match(r"^(async\s+def|def|class|interface|fn|func|struct)\b", stripped):
            kept.append(line)
            in_signature = not (stripped.endswith(":") or stripped.endswith("{"))
            continue

        if in_signature:
            kept.append(line)
            if stripped.endswith(":") or stripped.endswith("{"):
                in_signature = False
            continue

        if stripped.startswith(('"""', "'''")):
            delim = stripped[:3]
            kept.append(line)
            if stripped.count(delim) == 1:
                in_docstring = True
                doc_delim = delim
            continue

    if not kept:
        return lines[0].strip() + " ..."
    kept.append("    ...")
    return "\n".join(kept)


class ContextPruner:
    """Executes 3-Tier AST dynamic context pruning over a `GraphStore`."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def prune(self, focus_node_id: str) -> PrunedContextResult:
        """Builds the 3-tier pruned context centered on `focus_node_id`.

        - Tier 1 (Focus Symbol): Full raw source code from `node["text"]`.
        - Tier 2 (1-hop Neighbors): Signatures, type annotations, and docstrings only.
        - Tier 3 (2-hop Neighbors): Qualified symbol IDs only.
        """
        focus_data = self.store.get_node_data(focus_node_id)
        tier1_source = str(focus_data.get("text", ""))

        tier1_neighbor_ids, tier2_neighbor_ids = self.store.get_k_hop_tiers(focus_node_id)

        tier2_stubs: dict[str, str] = {}
        for neighbor_id in tier1_neighbor_ids:
            ndata = self.store.get_node_data(neighbor_id)
            raw_text = str(ndata.get("text", ""))
            stub = extract_signature_and_docstring(raw_text)
            if not stub:
                stub = f"def {ndata.get('name', neighbor_id)}(...): ..."
            tier2_stubs[neighbor_id] = stub

        return PrunedContextResult(
            focus_node_id=focus_node_id,
            tier1_source=tier1_source,
            tier2_stubs=tier2_stubs,
            tier3_symbol_ids=tier2_neighbor_ids,
        )

    def get_pruned_context(self, focus_node_id: str) -> str:
        """Returns the formatted 3-tier pruned context string."""
        return self.prune(focus_node_id).format_prompt_context()

    def get_unpruned_context(self, focus_node_id: str) -> str:
        """Returns full unpruned code for focus node, 1-hop, and 2-hop neighbors (for Baseline 2 ablation)."""
        focus_data = self.store.get_node_data(focus_node_id)
        tier1_neighbor_ids, tier2_neighbor_ids = self.store.get_k_hop_tiers(focus_node_id)

        sections: list[str] = [
            f"=== FOCUS SYMBOL [{focus_node_id}] (Full Implementation) ===",
            str(focus_data.get("text", "")).strip(),
            "\n=== 1-HOP NEIGHBORS (Full Implementation - Unpruned) ===",
        ]
        for nid in tier1_neighbor_ids:
            ndata = self.store.get_node_data(nid)
            sections.append(f"# Symbol: {nid}\n{str(ndata.get('text', '')).strip()}")

        sections.append("\n=== 2-HOP NEIGHBORS (Full Implementation - Unpruned) ===")
        for nid in tier2_neighbor_ids:
            ndata = self.store.get_node_data(nid)
            sections.append(f"# Symbol: {nid}\n{str(ndata.get('text', '')).strip()}")

        return "\n".join(sections)
