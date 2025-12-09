"""Script to run end-to-end evaluation on the benchmark.

Modified for Online-Mind2Web compatibility.
"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import List

import openai
import requests
import torch
from PIL import Image

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils

import nest_asyncio
nest_asyncio.apply()

# Use basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("logger")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    # Evaluation/Captioning models
    parser.add_argument("--eval_captioning_model_device", type=str, default="cpu")
    parser.add_argument("--eval_captioning_model", type=str, default="qwen3-vl-plus")
    parser.add_argument("--captioning_model", type=str, default="qwen3-vl-plus")

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--max_retry", type=int, default=1)
    parser.add_argument("--max_obs_length", type=int, default=3840)

    # Indexes
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1)

    # logging related
    parser.add_argument("--result_dir", type=str, default="omw_results")
    
    # Extra args passed by orchestrator but usually handled there
    parser.add_argument("--render_screenshot", action="store_true", default=True)
    
    args = parser.parse_args()
    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    # Simple parsing failure check
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]
    if len(last_k_actions) >= k:
        if all([action["action_type"] == ActionTypes.NONE for action in last_k_actions]):
            return True, f"Failed to parse actions for {k} times"

    return False, ""


def save_screenshot(page, save_path):
    """Save screenshot using playwright page object"""
    try:
        page.screenshot(path=save_path)
    except Exception as e:
        logger.error(f"Failed to take screenshot: {e}")


def test(args: argparse.Namespace, config_file_list: list[str]) -> None:
    max_steps = args.max_steps
    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    # Construct Agent
    agent = construct_agent(args) 

    # Construct Environment
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={"width": args.viewport_width, "height": args.viewport_height},
        save_trace_enabled=False, # We handle saving manually for OMW format
        sleep_after_execution=args.sleep_after_execution,
    )

    for config_file in config_file_list:
        try:
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # IMPORTANT: Expecting 'start_url' or 'website' to be present in config
                start_url = _c.get("start_url") or _c.get("website")
                image_paths = _c.get("image", None)
                
            logger.info(f"[-] Processing Task ID: {task_id}")
            logger.info(f"    Intent: {intent}")
            logger.info(f"    Start URL: {start_url}")

            # 1. Prepare Task Directory (OMW format compatibility)
            task_dir = Path(args.result_dir) / str(task_id)
            traj_dir = task_dir / "trajectory"
            if task_dir.exists():
                shutil.rmtree(task_dir)
            task_dir.mkdir(parents=True, exist_ok=True)
            traj_dir.mkdir(parents=True, exist_ok=True)

            # 2. Reset Agent & Env
            agent.reset(config_file)
            trajectory: Trajectory = []
            
            # Load images if any
            input_images = []
            if image_paths:
                if isinstance(image_paths, str): image_paths = [image_paths]
                for p in image_paths:
                    # Simple loader (omitting complex http logic for brevity)
                    input_images.append(Image.open(p))

            # Reset Env
            obs, info = env.reset(options={"config_file": config_file})
            
            # Force navigation because config file structure might differ from VWA standard
            if start_url:
                try:
                    env.page.goto(start_url)
                    time.sleep(2) # Wait for load
                    # Update observation after manual goto
                    obs = env._get_obs()
                except Exception as e:
                    logger.error(f"Failed to navigate to {start_url}: {e}")

            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)

            # Data collection for Result JSON
            action_history_str = []
            final_response = ""

            # Save Initial State Screenshot (Step 0)
            save_screenshot(env.page, str(traj_dir / "0.png"))

            step_idx = 0
            meta_data = {"action_history": ["None"]}
            
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=input_images,
                            meta_data=meta_data,
                        )
                    except ValueError as e:
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                # Get action description for history
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                action_history_str.append(action_str)
                meta_data["action_history"].append(action_str)

                # Step Environment
                if action["action_type"] == ActionTypes.STOP:
                    final_response = action.get("answer", "")
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                step_idx += 1
                # Save Screenshot for this step
                save_screenshot(env.page, str(traj_dir / f"{step_idx}.png"))

                if terminated:
                    trajectory.append(create_stop_action(""))
                    break

            # 3. Save Result JSON in OMW format
            # Convert VWA output to the format expected by Online-Mind2Web
            # Extract structured actions and thoughts (raw LLM outputs) from trajectory
            structured_actions = []
            extracted_thoughts = []
            for item in trajectory:
                # action entries are dicts with 'action_type' key
                if isinstance(item, dict) and item.get("action_type") is not None:
                    a = item
                    action_record = {
                        "action_type": a.get("action_type"),
                        "selector": a.get("selector"),
                        "value": a.get("value"),
                        "raw_prediction": a.get("raw_prediction"),
                    }
                    structured_actions.append(action_record)
                    if a.get("raw_prediction"):
                        extracted_thoughts.append(a.get("raw_prediction"))

            result_data = {
                "task_id": task_id,
                "task": intent,
                # human-readable action descriptions
                "action_history": action_history_str,
                # structured action records
                #"actions": structured_actions,
                # raw LLM responses (thoughts) extracted from actions
                "thoughts": extracted_thoughts,
                "final_result_response": final_response,
                "input_image_paths": image_paths if image_paths else [],
            }

            with open(task_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"[Task Finished] Saved results to {task_dir}")
            return task_dir

        except Exception as e:
            logger.error(f"[Unhandled Error] Task {config_file}: {e}")
            import traceback
            traceback.print_exc()

    env.close()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    # Ensure Result Dir Exists
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # Load Config List
    test_config_base_dir = args.test_config_base_dir
    test_file_list = []
    
    # If using range
    if args.test_end_idx > args.test_start_idx and os.path.exists(test_config_base_dir):
        # Determine actual file list based on directory content matching range if simple glob fails
        # But simple iteration based on orchestrator logic:
        # Orchestrator names them 0.json, 1.json etc.
        for i in range(args.test_start_idx, args.test_end_idx):
             p = os.path.join(test_config_base_dir, f"{i}.json")
             if os.path.exists(p):
                 test_file_list.append(p)
    
    # Fallback: Just grab all JSONs in config dir
    if not test_file_list and os.path.exists(test_config_base_dir):
         test_file_list = sorted(glob.glob(os.path.join(test_config_base_dir, "*.json")))

    logger.info(f"Found {len(test_file_list)} tasks to run.")
    
    # Run
    test(args, test_file_list)