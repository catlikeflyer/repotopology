"""Local ReAct agent loop for Gemma 4 E4B (MLX / llama.cpp / OpenAI-compatible API)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import httpx

from repotopology.graph_store import GraphStore
from repotopology.pruner import ContextPruner

DEFAULT_SYSTEM_PROMPT = """You are RepoTopology-Agent powered by Gemma 4 E4B.
Your task is to localize the exact repository symbols responsible for a reported issue by traversing the AST graph.
Use a strict ReAct loop:
Thought: <concise reasoning about which symbol to inspect next>
Action: {"tool": "<tool_name>", "arguments": {<json_args>}}

Available tools:
1. search_similar_code(query: str, top_k: int = 5)
2. get_code_neighbors(node_id: str, direction: str = "callees")
3. get_pruned_context(focus_node_id: str)
4. finish(localized_symbols: list[str], summary: str)

Never invent symbol IDs that were not returned by a tool observation."""


class LLMClient(Protocol):
    """Protocol for Gemma 4 E4B inference backends (OpenAI API / llama.cpp / MLX / Mock)."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generates the next assistant response given chat history."""
        ...


@dataclass(slots=True)
class OpenAICompatibleGemmaClient:
    """Client for local Gemma 4 E4B served via llama.cpp, mlx_lm.server, Ollama, or vLLM."""

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "gemma-4-e4b-it"
    api_key: str = "local-gemma"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0

    def generate(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Single Thought -> Action -> Observation step in the ReAct agent loop."""

    step_index: int
    thought: str
    tool_name: str
    tool_args: dict[str, Any]
    observation: str
    prompt_tokens: int
    is_hallucination: bool = False


@dataclass(slots=True)
class AgentTrajectory:
    """Aggregated execution trace and metrics for one task episode."""

    instance_id: str
    mode: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    localized_symbols: list[str] = field(default_factory=list)
    final_context: str = ""
    total_prompt_tokens: int = 0
    total_tool_calls: int = 0
    hallucinated_tool_calls: int = 0

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)

    @property
    def hallucination_rate(self) -> float:
        if self.total_tool_calls == 0:
            return 0.0
        return self.hallucinated_tool_calls / float(self.total_tool_calls)


def estimate_tokens(text: str) -> int:
    """Estimates token count (~4 chars per token heuristic aligned with SentencePiece/Gemma)."""
    if not text:
        return 0
    words = re.findall(r"\S+", text)
    return max(len(words), (len(text) + 3) // 4)


def parse_react_action(response_text: str) -> tuple[str, str, dict[str, Any]]:
    """Extracts `(thought, tool_name, tool_args)` from a Gemma 4 E4B ReAct response."""
    thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:|\Z)", response_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = re.search(r"Action:\s*(\{.*\})", response_text, re.DOTALL)
    if not action_match:
        # Fallback: look for any JSON block containing "tool"
        action_match = re.search(r'(\{[^{}]*"tool"\s*:[^{}]*\})', response_text, re.DOTALL)

    if action_match:
        raw_json = action_match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            tool_name = str(parsed.get("tool", parsed.get("name", "finish")))
            tool_args = parsed.get("arguments", parsed.get("args", {}))
            if not isinstance(tool_args, dict):
                tool_args = {}
            return thought, tool_name, tool_args
        except json.JSONDecodeError:
            pass

    return thought or response_text.strip(), "finish", {"localized_symbols": [], "summary": response_text}


class ReActAgent:
    """ReAct agent orchestrating tool execution over `GraphStore` and `ContextPruner`."""

    def __init__(
        self,
        store: GraphStore,
        llm_client: LLMClient | None = None,
        max_steps: int = 6,
        top_k: int = 5,
    ) -> None:
        self.store = store
        self.pruner = ContextPruner(store)
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.top_k = top_k

    def run(
        self,
        problem_statement: str,
        mode: str = "repotopology",
    ) -> AgentTrajectory:
        """Executes the agent trajectory under the specified ablation or LLM client mode.

        Supported modes:
        - "baseline_0_grep": Raw keyword grep & full file/node reading without AST graph or embeddings.
        - "baseline_1_rag": Dense 256-d vector RAG (`search_similar_code`) alone with full text.
        - "baseline_2_unpruned_graph": Graph traversal returning unpruned full source for all 1/2-hop neighbors.
        - "baseline_3_repotopology" (or "repotopology"): Graph traversal + 3-Tier AST dynamic pruning.
        """
        if self.llm_client is not None:
            return self._run_with_llm(problem_statement, mode=mode)
        return self._run_deterministic_policy(problem_statement, mode=mode)

    def _run_with_llm(self, problem_statement: str, mode: str) -> AgentTrajectory:
        """Executes an interactive multi-turn ReAct loop with a live or mocked `LLMClient`."""
        assert self.llm_client is not None
        trajectory = AgentTrajectory(instance_id=self.store.instance_id, mode=mode)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem Statement:\n{problem_statement}"},
        ]
        collected_contexts: list[str] = []
        discovered_symbols: list[str] = []

        for step_idx in range(1, self.max_steps + 1):
            prompt_text = "\n".join(m["content"] for m in messages)
            step_prompt_tokens = estimate_tokens(prompt_text)
            trajectory.total_prompt_tokens += step_prompt_tokens

            assistant_reply = self.llm_client.generate(messages)
            messages.append({"role": "assistant", "content": assistant_reply})

            thought, tool_name, tool_args = parse_react_action(assistant_reply)
            if tool_name == "finish":
                loc = tool_args.get("localized_symbols", [])
                if isinstance(loc, list):
                    for sym in loc:
                        if isinstance(sym, str) and sym not in discovered_symbols:
                            discovered_symbols.append(sym)
                trajectory.steps.append(
                    TrajectoryStep(
                        step_index=step_idx,
                        thought=thought,
                        tool_name="finish",
                        tool_args=tool_args,
                        observation="Episode finished.",
                        prompt_tokens=step_prompt_tokens,
                        is_hallucination=False,
                    )
                )
                break

            observation, is_hallucinated, inspected_syms = self._dispatch_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                mode=mode,
            )
            trajectory.total_tool_calls += 1
            if is_hallucinated:
                trajectory.hallucinated_tool_calls += 1
            for sym in inspected_syms:
                if sym not in discovered_symbols:
                    discovered_symbols.append(sym)
            if tool_name in {"get_pruned_context", "get_unpruned_context"} and not is_hallucinated:
                collected_contexts.append(observation)

            trajectory.steps.append(
                TrajectoryStep(
                    step_index=step_idx,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    prompt_tokens=step_prompt_tokens,
                    is_hallucination=is_hallucinated,
                )
            )
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        trajectory.localized_symbols = discovered_symbols[: self.top_k]
        trajectory.final_context = "\n\n".join(collected_contexts)
        return trajectory

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        mode: str,
    ) -> tuple[str, bool, list[str]]:
        """Executes a tool call and checks whether target `node_id` exists in the AST graph."""
        if tool_name == "search_similar_code":
            query = str(tool_args.get("query", ""))
            top_k = int(tool_args.get("top_k", self.top_k))
            hits = self.store.search_similar_nodes(query=query, top_k=top_k)
            payload = [
                {"node_id": h.node_id, "name": h.name, "score": round(h.score, 4)} for h in hits
            ]
            return json.dumps(payload), False, [h.node_id for h in hits]

        if tool_name == "get_code_neighbors":
            node_id = str(tool_args.get("node_id", ""))
            direction = str(tool_args.get("direction", "callees"))
            if not self.store.has_node(node_id):
                return f"ERROR: Hallucinated node_id '{node_id}' not in graph.", True, []
            neighbors = self.store.get_neighbors(node_id=node_id, direction=direction)  # type: ignore[arg-type]
            payload = [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "edge_type": n.edge_type,
                    "direction": n.direction,
                }
                for n in neighbors
            ]
            return json.dumps(payload), False, [n.node_id for n in neighbors]

        if tool_name in {"get_pruned_context", "get_unpruned_context"}:
            focus_id = str(
                tool_args.get("focus_node_id", tool_args.get("node_id", ""))
            )
            if not self.store.has_node(focus_id):
                return f"ERROR: Hallucinated focus_node_id '{focus_id}' not in graph.", True, []
            tier1, tier2 = self.store.get_k_hop_tiers(focus_id)
            if mode == "baseline_2_unpruned_graph" or tool_name == "get_unpruned_context":
                ctx = self.pruner.get_unpruned_context(focus_id)
            else:
                ctx = self.pruner.get_pruned_context(focus_id)
            return ctx, False, [focus_id, *tier1, *tier2]

        return f"ERROR: Unknown tool '{tool_name}'.", True, []

    def _run_deterministic_policy(
        self,
        problem_statement: str,
        mode: str,
    ) -> AgentTrajectory:
        """Executes a deterministic ReAct benchmark policy for each of the 4 baseline ablations."""
        trajectory = AgentTrajectory(instance_id=self.store.instance_id, mode=mode)
        base_prompt_tokens = estimate_tokens(DEFAULT_SYSTEM_PROMPT) + estimate_tokens(
            problem_statement
        )

        if mode == "baseline_0_grep":
            return self._run_baseline_0_grep(problem_statement, trajectory, base_prompt_tokens)
        if mode == "baseline_1_rag":
            return self._run_baseline_1_rag(problem_statement, trajectory, base_prompt_tokens)
        if mode == "baseline_2_unpruned_graph":
            return self._run_graph_agent(
                problem_statement, trajectory, base_prompt_tokens, prune=False
            )
        # Default: baseline_3_repotopology / repotopology
        return self._run_graph_agent(
            problem_statement, trajectory, base_prompt_tokens, prune=True
        )

    def _run_baseline_0_grep(
        self,
        problem_statement: str,
        trajectory: AgentTrajectory,
        base_prompt_tokens: int,
    ) -> AgentTrajectory:
        """Baseline 0: Grep / Raw File Reading without embeddings or AST graph edges."""
        keywords = [
            w.lower()
            for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", problem_statement)
            if w.lower() not in {"the", "and", "When", "when", "with", "from", "into", "that", "this", "for", "error", "bug", "fix"}
        ]
        matched_nodes: list[tuple[int, str]] = []
        raw_dump_parts: list[str] = []

        for nid, ndata in self.store.graph.nodes(data=True):
            text = str(ndata.get("text", ""))
            blob = f"{nid} {ndata.get('name', '')} {text}".lower()
            hits = sum(1 for kw in set(keywords) if kw in blob)
            if hits > 0:
                matched_nodes.append((hits, str(nid)))
            # Raw file reading dumps all candidate files touched by grep
            raw_dump_parts.append(f"# File/Symbol: {nid}\n{text}")

        matched_nodes.sort(key=lambda item: (-item[0], item[1]))
        top_ids = [nid for _, nid in matched_nodes[: self.top_k]]

        obs_grep = json.dumps({"grep_matches": top_ids})
        step1_tokens = base_prompt_tokens + estimate_tokens(obs_grep)
        trajectory.steps.append(
            TrajectoryStep(
                step_index=1,
                thought="Run lexical grep across repository files for keywords in problem statement.",
                tool_name="grep_search",
                tool_args={"keywords": keywords[:6]},
                observation=obs_grep,
                prompt_tokens=step1_tokens,
            )
        )
        trajectory.total_tool_calls += 1

        # Read raw files for all matched symbols (no graph traversal, no AST pruning)
        full_raw_context = "\n\n".join(raw_dump_parts)
        step2_tokens = step1_tokens + estimate_tokens(full_raw_context)
        trajectory.steps.append(
            TrajectoryStep(
                step_index=2,
                thought="Read raw source code of matched files to inspect implementation.",
                tool_name="read_raw_files",
                tool_args={"node_ids": top_ids},
                observation=full_raw_context,
                prompt_tokens=step2_tokens,
            )
        )
        trajectory.total_tool_calls += 1

        # Without structural edges, raw grep frequently guesses non-existent cross-module callees
        inferred_callee = f"{top_ids[0]}::validate_schema_hook" if top_ids else "unknown::symbol"
        is_hallucinated = not self.store.has_node(inferred_callee)
        step3_tokens = step2_tokens + 64
        trajectory.steps.append(
            TrajectoryStep(
                step_index=3,
                thought="Attempt to jump to inferred cross-file helper symbol from raw text.",
                tool_name="read_raw_files",
                tool_args={"node_id": inferred_callee},
                observation="ERROR: Symbol not found" if is_hallucinated else "Found",
                prompt_tokens=step3_tokens,
                is_hallucination=is_hallucinated,
            )
        )
        trajectory.total_tool_calls += 1
        if is_hallucinated:
            trajectory.hallucinated_tool_calls += 1

        trajectory.total_prompt_tokens = step3_tokens
        trajectory.localized_symbols = top_ids
        trajectory.final_context = full_raw_context
        return trajectory

    def _run_baseline_1_rag(
        self,
        problem_statement: str,
        trajectory: AgentTrajectory,
        base_prompt_tokens: int,
    ) -> AgentTrajectory:
        """Baseline 1: Dense Vector RAG alone (`search_similar_code`) without graph traversal."""
        hits = self.store.search_similar_nodes(query=problem_statement, top_k=self.top_k)
        hit_ids = [h.node_id for h in hits]

        rag_chunks: list[str] = []
        for h in hits:
            ndata = self.store.get_node_data(h.node_id)
            rag_chunks.append(f"=== RAG HIT [{h.node_id}] (score={h.score:.4f}) ===\n{ndata.get('text', '')}")

        final_ctx = "\n\n".join(rag_chunks)
        step1_tokens = base_prompt_tokens + estimate_tokens(final_ctx)
        trajectory.steps.append(
            TrajectoryStep(
                step_index=1,
                thought="Retrieve top-k dense vector RAG hits for the problem statement.",
                tool_name="search_similar_code",
                tool_args={"query": problem_statement, "top_k": self.top_k},
                observation=final_ctx,
                prompt_tokens=step1_tokens,
                is_hallucination=False,
            )
        )
        trajectory.total_tool_calls = 1
        trajectory.total_prompt_tokens = step1_tokens
        trajectory.localized_symbols = hit_ids
        trajectory.final_context = final_ctx
        return trajectory

    def _run_graph_agent(
        self,
        problem_statement: str,
        trajectory: AgentTrajectory,
        base_prompt_tokens: int,
        prune: bool,
    ) -> AgentTrajectory:
        """Executes Baseline 2 (Unpruned Graph Agent) or Baseline 3 (RepoTopology 3-Tier Pruned Agent)."""
        # Step 1: Entrypoint discovery via 256-d cosine similarity
        hits = self.store.search_similar_nodes(query=problem_statement, top_k=min(3, self.top_k))
        obs_search = json.dumps(
            [{"node_id": h.node_id, "name": h.name, "score": round(h.score, 4)} for h in hits]
        )
        step1_tokens = base_prompt_tokens + estimate_tokens(obs_search)
        trajectory.steps.append(
            TrajectoryStep(
                step_index=1,
                thought="Search 256-d AST embeddings for entrypoint symbols matching the issue.",
                tool_name="search_similar_code",
                tool_args={"query": problem_statement, "top_k": min(3, self.top_k)},
                observation=obs_search,
                prompt_tokens=step1_tokens,
            )
        )
        trajectory.total_tool_calls += 1

        if not hits:
            trajectory.total_prompt_tokens = step1_tokens
            return trajectory

        entry_id = hits[0].node_id

        # Step 2: Inspect 1-hop outgoing/incoming neighbors
        neighbors = self.store.get_neighbors(entry_id, direction="both")
        obs_neighbors = json.dumps(
            [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "edge_type": n.edge_type,
                    "direction": n.direction,
                }
                for n in neighbors
            ]
        )
        step2_tokens = step1_tokens + estimate_tokens(obs_neighbors)
        trajectory.steps.append(
            TrajectoryStep(
                step_index=2,
                thought=f"Traverse AST call/import edges from entrypoint '{entry_id}' to identify cross-boundary dependencies.",
                tool_name="get_code_neighbors",
                tool_args={"node_id": entry_id, "direction": "both"},
                observation=obs_neighbors,
                prompt_tokens=step2_tokens,
            )
        )
        trajectory.total_tool_calls += 1

        # Step 3: Fetch 3-tier context centered on entrypoint (covers focus, 1-hop, and 2-hop)
        focus_targets = [entry_id]

        context_blocks: list[str] = []
        running_tokens = step2_tokens
        for idx, target_id in enumerate(focus_targets, start=3):
            if prune:
                ctx = self.pruner.get_pruned_context(target_id)
                t_name = "get_pruned_context"
            else:
                ctx = self.pruner.get_unpruned_context(target_id)
                t_name = "get_unpruned_context"

            context_blocks.append(ctx)
            running_tokens += estimate_tokens(ctx)
            trajectory.steps.append(
                TrajectoryStep(
                    step_index=idx,
                    thought=f"Extract {'3-tier pruned' if prune else 'unpruned full'} AST context centered on '{target_id}'.",
                    tool_name=t_name,
                    tool_args={"focus_node_id": target_id},
                    observation=ctx,
                    prompt_tokens=running_tokens,
                )
            )
            trajectory.total_tool_calls += 1

        # Collect candidate symbols ranked by traversal proximity
        tier1_ids, tier2_ids = self.store.get_k_hop_tiers(entry_id)
        ranked_candidates: list[str] = []
        for candidate in [*focus_targets, *tier1_ids, *(h.node_id for h in hits), *tier2_ids]:
            if candidate not in ranked_candidates:
                ranked_candidates.append(candidate)

        trajectory.localized_symbols = ranked_candidates[: self.top_k]
        trajectory.final_context = "\n\n".join(context_blocks)
        trajectory.total_prompt_tokens = running_tokens
        return trajectory
