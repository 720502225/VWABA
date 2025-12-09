#!/usr/bin/env python3
"""
Run VWA agent on tasks from `success_tasks.json`, convert outputs to
Online-Mind2Web format and run WebJudge_Online_Mind2Web_eval.

Requirements/assumptions:
- Run from repository root (so imports of `run` work).
- A working browser / environment for VWA agent (same as running `run.py`).
- Python on PATH and same environment that can run both VWA and Online-Mind2Web scripts.

This script performs 3 stages:
1) Load task list from `success_tasks.json` and write per-task config files.
2) Invoke VWA `run.test` to execute agent on those configs and produce results under `result_dir`.
3) Convert produced results to Online-Mind2Web format and call Online-Mind2Web `src/run.py` with mode `WebJudge_Online_Mind2Web_eval`.

Usage (example):
  python scripts/run_vwa_to_online_mind2web_eval.py --tasks ./success_tasks.json --max_tasks 10 \
    --result_dir ./vwa_results --converted_dir ./Online-Mind2Web/data/example --omw_api_key YOUR_KEY --omw_model gpt-4o --score_threshold 3

"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import nest_asyncio
nest_asyncio.apply()

def build_vwa_args(result_dir: str, test_start_idx: int = 1, test_end_idx: int = 3) -> SimpleNamespace:
    # Minimal args needed by run.prepare and run.test
    ns = SimpleNamespace()
    ns.render = False
    ns.slow_mo = 0
    ns.action_set_tag = "id_accessibility_tree"
    ns.observation_type = "accessibility_tree"
    ns.current_viewport_only = True
    ns.viewport_width = 1280
    ns.viewport_height = 2048
    ns.save_trace_enabled = True
    ns.sleep_after_execution = 0.0
    ns.max_steps = 30
    ns.agent_type = "prompt"
    ns.instruction_path = "agent/prompts/jsons/p_cot_id_actree_3s.json"
    ns.parsing_failure_th = 3
    ns.repeating_action_failure_th = 5
    ns.test_config_base_dir = "configs"
    ns.eval_captioning_model_device = "cpu"
    ns.eval_captioning_model = "qwen3-vl-plus"
    ns.captioning_model = "qwen3-vl-plus"
    ns.provider = "openai"
    ns.model = "qwen-plus"
    ns.mode = "chat"
    ns.temperature = 1.0
    ns.top_p = 0.9
    ns.context_length = 0
    ns.max_tokens = 384
    ns.stop_token = None
    ns.max_retry = 1
    ns.max_obs_length = 3840
    ns.test_start_idx = test_start_idx
    ns.test_end_idx = test_end_idx
    ns.result_dir = result_dir
    ns.render_screenshot = True
    ns.save_trace_enabled = True
    return ns


def write_task_configs(tasks, configs_dir: Path, start_idx: int, end_idx: int):
    """Write per-task config files for tasks in index range [start_idx, end_idx].

    Filenames are generated sequentially starting from 0.json for the first
    selected task.
    """
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_paths = []
    total = len(tasks)

    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx < start_idx:
        end_idx = start_idx

    # clamp indices
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, total - 1)

    counter = 0
    for idx in range(start_idx, end_idx + 1):
        t = tasks[idx]
        task_id = t.get("task_id")
        intent = t.get("confirmed_task") or t.get("task") or t.get("confirmed_task")
        cfg = {
            "intent": intent,
            "task_id": task_id,
            "start_url": t.get("website"),
            # leave storage_state empty to skip auto-login
            "storage_state": None,
            # no input images by default
            "image": None,
        }
        cfg_path = configs_dir / f"{counter}.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        config_paths.append(str(cfg_path))
        counter += 1

    return config_paths


def run_vwa_agent(config_paths, vwa_args):
    # Import local run module (must be executed from repo root)
    import run_omw as vwa_run

    # Prepare result dir etc
    #vwa_run.prepare(vwa_args)

    # call test() with list
    task_dir = vwa_run.test(vwa_args, config_paths)
    return task_dir


def convert_results(result_dir: str, converted_dir: str):
    # Use the converter script we added earlier
    script = Path("Online-Mind2Web/script/convert.py")
    if not script.exists():
        raise FileNotFoundError(f"Converter script not found: {script}")
    cmd = [sys.executable, str(script), "--input_dir", str(result_dir), "--output_dir", str(converted_dir)]
    print("Running converter:", " ".join(cmd))
    subprocess.check_call(cmd)


def run_online_mind2web_eval(converted_dir: str, api_key: str, model: str = "qwen-plus", output_dir: str | None = None, score_threshold: int = 3, num_worker: int = 1, base_url: str | None = None):
    omw_run = Path("Online-Mind2Web/src/run.py")
    if not omw_run.exists():
        raise FileNotFoundError(f"Online-Mind2Web run.py not found: {omw_run}")
    if output_dir is None:
        output_dir = f"{converted_dir}/result"

    cmd = [sys.executable, str(omw_run), "--mode", "WebJudge_Online_Mind2Web_eval", "--model", model, "--trajectories_dir", str(converted_dir), "--api_key", api_key, "--output_path", str(output_dir), "--num_worker", str(num_worker), "--score_threshold", str(score_threshold)]
    if base_url:
        cmd += ["--base_url", base_url]

    print("Running Online-Mind2Web evaluation:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="success_tasks.json", help="Path to success_tasks.json")
    parser.add_argument("--test_start_idx", type=int, default=0, help="Start task index (inclusive) for processing")
    parser.add_argument("--test_end_idx", type=int, default=0, help="End task index (inclusive) for processing")
    parser.add_argument("--result_dir", type=str, default="omw_results")
    parser.add_argument("--converted_dir", type=str, default="Online-Mind2Web/data")
    parser.add_argument("--omw_api_key", type=str, help="Online-Mind2Web OpenAI API key")
    parser.add_argument("--omw_model", type=str, default="qwen-plus")
    parser.add_argument("--score_threshold", type=int, default=3)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--base_url", type=str, default=None)
    args = parser.parse_args()

    # Fallback: read API key and base_url from environment variables if not provided
    if not args.omw_api_key:
        args.omw_api_key = os.environ.get("OPENAI_API_KEY")

    if not args.base_url:
        args.base_url = os.environ.get("OPENAI_BASE_URL")

    if not args.omw_api_key:
        print("Error: Online-Mind2Web API key not provided. Set --omw_api_key or environment variable OPENAI_KEY.")
        return

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Tasks file not found: {tasks_path}")
        return

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # prepare config directory
    configs_dir = Path("config_files/Online_Mind2Web")
    if configs_dir.exists():
        # keep prior configs but we will overwrite
        pass
    else:
        configs_dir.mkdir(parents=True)

    # Use start/end indices to select tasks
    start_idx = args.test_start_idx
    end_idx = args.test_end_idx
    config_paths = write_task_configs(tasks, configs_dir, start_idx=start_idx, end_idx=end_idx)

    # build args for VWA run
    vwa_args = build_vwa_args(args.result_dir, test_start_idx=args.test_start_idx, test_end_idx=args.test_end_idx)
    # ensure result_dir exists
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    print(f"Running agent on {len(config_paths)} tasks, results -> {args.result_dir}")
    task_dir = run_vwa_agent(config_paths, vwa_args)

    #print("Converting results to Online-Mind2Web format...")
    #convert_results(args.result_dir, args.converted_dir)

    print("Running Online-Mind2Web WebJudge_Online_Mind2Web_eval...")
    run_online_mind2web_eval(task_dir, api_key=args.omw_api_key, model=args.omw_model, score_threshold=args.score_threshold, num_worker=args.num_worker, base_url=args.base_url)


if __name__ == "__main__":
    main()
