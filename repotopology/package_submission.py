"""Submission packager and validator for the Gemma 4 Developer Agent Kaggle Competition."""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def create_submission_bundle(
    submission_dir: str | Path = "./submission",
    adapter_source: str | Path = "./submission/adapters/main_lora",
    output_zip: str | Path = "./submission.zip",
    model_name: str = "gemma-4-e4b-it-qat-w4a16-ct",
) -> Path:
    """Prepares and validates the declarative submission folder and zips it into submission.zip."""
    base = Path(submission_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "configs").mkdir(exist_ok=True)
    (base / "prompts").mkdir(exist_ok=True)
    (base / "sub_agents").mkdir(exist_ok=True)
    (base / "adapters" / "main_lora").mkdir(parents=True, exist_ok=True)

    # 1. Root agent.yaml
    agent_yaml = f"""name: repotopology_developer_agent
model: {model_name}
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
"""
    (base / "agent.yaml").write_text(agent_yaml, encoding="utf-8")

    # 2. eval_config.yaml
    eval_config = """max_turns: 20
max_tool_calls: 35
timeout_seconds: 900
enable_sandbox_testing: true
"""
    (base / "eval_config.yaml").write_text(eval_config, encoding="utf-8")

    # 3. configs/sampling.yaml
    sampling_yaml = """temperature: 0.1
top_p: 0.95
max_output_tokens: 4096
thinking_config:
  thinking_budget: 2048
  include_thoughts: true
"""
    (base / "configs" / "sampling.yaml").write_text(sampling_yaml, encoding="utf-8")

    # 4. prompts/system.md
    system_prompt = """You are RepoTopology Developer Agent powered by Gemma 4 E4B.
Your goal is to inspect codebases with 3-tier dynamic AST pruning, localize the root cause, apply a surgical fix, and submit a verified patch.

## Workflow

### 1. Fast Semantic Entrypoint Localization
- Use `search_similar_code` to query the 256-d codebase AST embeddings using symbol names or error messages from the problem statement.
- Do NOT run full-repository grep or find sweeps; query the AST graph directly.

### 2. Cross-Boundary Graph Navigation
- Use `get_code_neighbors` to inspect incoming callers and outgoing callees of candidate symbols.
- Trace relationships between interface endpoints and internal business logic without reading full unpruned files.

### 3. Surgical Code Reading & 3-Tier Inspection
- Use `read_file` or `code_analyzer` on specific identified functions.
- Keep context lean: only inspect symbols directly connected to the reported defect.

### 4. Implement Minimal Fix
- Apply the fix to source files under `/workspace` using `edit_file` or `write_file`.
- Never modify or delete existing test files under `tests/` or `test/`.

### 5. Verification & Immediate Patch Submission
- Run ONLY targeted tests for the changed functionality using `run_command` (e.g. `pytest tests/test_target.py -k test_feature`).
- Never run bare `pytest` or `pytest .`.
- Once verified, call `submit_patch()` immediately.
"""
    (base / "prompts" / "system.md").write_text(system_prompt, encoding="utf-8")

    # 5. prompts/analyzer.md
    analyzer_prompt = """You are the RepoTopology AST Code Analyzer.
Inspect functions, call graphs, and type annotations. Return a concise 3-tier summary:
- Focus symbol defect root cause
- 1-hop dependent signature impacts
- Recommended surgical diff
"""
    (base / "prompts" / "analyzer.md").write_text(analyzer_prompt, encoding="utf-8")

    # 6. sub_agents/code_analyzer.yaml
    sub_agent_yaml = f"""name: code_analyzer
model: {model_name}
adapter: main_lora
instruction: !include prompts/analyzer.md
tools:
  - read_file
  - get_code_neighbors
  - search_similar_code
generate_content_config: !include configs/sampling.yaml
"""
    (base / "sub_agents" / "code_analyzer.yaml").write_text(sub_agent_yaml, encoding="utf-8")

    # 7. Copy adapters if available
    adapter_src = Path(adapter_source).resolve()
    adapter_dest = (base / "adapters" / "main_lora").resolve()
    if adapter_src.exists():
        for fname in ["adapter_config.json", "adapter_model.safetensors", "adapters.safetensors"]:
            src_file = adapter_src / fname
            if src_file.exists():
                target_name = "adapter_model.safetensors" if fname == "adapters.safetensors" else fname
                dst_file = adapter_dest / target_name
                if src_file != dst_file:
                    shutil.copy2(src_file, dst_file)

    # 8. Create submission.zip (only packaging final adapter_model.safetensors and config)
    zip_path = Path(output_zip)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            for file in files:
                # Skip intermediate checkpoint files
                if file.endswith("_adapters.safetensors") or file == "adapters.safetensors":
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(base)
                zf.write(full_path, arcname=str(rel_path))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {zip_path} ({size_mb:.2f} MB)")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package RepoTopology submission for Kaggle")
    parser.add_argument("--submission-dir", type=str, default="./submission")
    parser.add_argument("--adapter-source", type=str, default="./submission/adapters/main_lora")
    parser.add_argument("--output-zip", type=str, default="./submission.zip")
    parser.add_argument("--model", type=str, default="gemma-4-e4b-it-qat-w4a16-ct")
    args = parser.parse_args()

    create_submission_bundle(
        submission_dir=args.submission_dir,
        adapter_source=args.adapter_source,
        output_zip=args.output_zip,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
