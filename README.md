# VLooM

VLooM is a config-driven framework for collecting and structuring data from large vision-language models. It provides reusable building blocks for datasets, prompt templates, agents, OpenAI-compatible model clients, and concurrent batch execution.

The project is intended to be extended by downstream repositories without modifying the core package. A downstream project can define its own dataset classes, templates, and task configs, then load them through `imports` in a YAML config.

## Features

- Config-driven pipelines with `draccus` YAML configs.
- OpenAI-compatible chat completion client with retry support.
- Dry-run mode for prompt and pipeline validation without real API calls.
- Jinja2 prompt templates for task-specific instructions.
- Agent registry for built-in and downstream custom agents.
- Dataset registry for built-in and downstream custom datasets.
- Concurrent batch runner with structured JSON output.
- Robust parsing for common VLM response formats, including Markdown fenced JSON and optional `<think>...</think>` reasoning blocks.
- API key resolution from config or environment variables, with saved configs redacting keys by default.

## Installation

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

For downstream projects, install this repository as an editable dependency or add `src` to `PYTHONPATH` during development.

```bash
pip install -e /path/to/VLooM
```

## Quick Start

Run the sample config in dry-run mode:

```bash
vloom --config_path configs/sample.yaml --dry_run
```

Or invoke the module directly:

```bash
python -m vloom --config_path configs/sample.yaml --dry_run
```

Outputs are written under the configured `output_root` and experiment directory. Each task writes a JSON result file named:

```text
<save_dir>/<dataset_name>/<task_name>_results.json
```

## API Keys

VLooM uses an OpenAI-compatible chat completions API by default. You can provide credentials directly in config:

```yaml
model:
  base_url: https://api.openai.com/v1
  model_name: gpt-4o-mini
  api_key: sk-...
```

For safer workflows, leave `api_key` empty and use environment variables:

```bash
export OPENAI_API_KEY=sk-...
vloom --config_path configs/sample.yaml
```

By default VLooM checks these variables in order:

```yaml
model:
  api_key_envs:
    - OPENAI_API_KEY
    - YUNWU_API_KEY
```

You can override `api_key_envs` for another OpenAI-compatible provider. Saved run configs redact `model.api_key` so secrets are not written into output folders.

## Configuration

A typical pipeline config contains:

```yaml
output_root: outputs
exp_name: sample_run
template_dir: templates
dry_run: false
max_concurrent: 8

imports: []

model:
  base_url: https://api.openai.com/v1
  model_name: gpt-4o-mini
  api_key: ""
  generation_kwargs:
    max_tokens: 2048
    temperature: 0.2
    max_image_size: 1024

datasets:
  - type: coco
    name: sample_dataset
    # dataset-specific fields

task_definitions:
  - name: basic_qa
    agent_type: basic
    templates:
      usr: task/basic_qa.j2

run_tasks:
  - basic_qa
```

Downstream projects can register custom dataset and agent classes, then list the module path under `imports` so VLooM imports it before building the registry.

## Response Format

The built-in `basic` agent expects model responses to contain a JSON object. The parser also accepts common wrappers produced by LLM providers, for example:

````text
<think>
Brief reasoning text.
</think>

```json
{"answer": "A", "confidence": 0.87}
```
````

Parsed output is stored in `result`. Extracted reasoning is stored in `thinking` when available. Parser repair notes are stored in `repair_log`.

## Extending VLooM

Create a Python module in your downstream project and register components with VLooM registries:

```python
from dataclasses import dataclass
from vloom.config import DatasetConfig
from vloom.dataset.base import BaseDataset


@DatasetConfig.register_subclass("my_dataset")
@dataclass
class MyDatasetConfig(DatasetConfig):
    manifest: str = ""

    def dataset_class(self):
        return MyDataset


class MyDataset(BaseDataset):
    pass
```

Then load it in config:

```yaml
imports:
  - my_project.vloom_extensions
```

Keep project-specific templates, data manifests, and output paths in the downstream repository. VLooM should stay focused on generic pipeline, client, parser, agent, and dataset abstractions.

## Development Checks

Run a syntax check for the package:

```bash
python -m py_compile src/vloom/core/parser.py src/vloom/agents/basic_agent.py src/vloom/agents/base.py src/vloom/pipelines/batch_runner.py src/vloom/config.py src/vloom/cli.py
```

Run a small parser smoke check:

```bash
PYTHONPATH=src python -c 'from vloom.core.parser import parse_json_response; r=parse_json_response("<think>ok</think>```json\n{\"answer\": \"A\",}\n```"); assert r.result == {"answer": "A"}; assert r.thinking == "ok"'
```
