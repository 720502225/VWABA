"""Multi-Agent Web Arena Runner.

This module implements a multi-agent system for web automation tasks,
using Context, Planner, Actor, and Reflector agents.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from browser_env import (
    ScriptBrowserEnv,
    Action,
    create_id_based_action,
)
from browser_env.utils import Observation
from agent import PromptAgent
from llms import lm_config, call_llm
from PIL import Image

# Import the full multi-agent coordinator
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent.prompts.prompt_constructor import PromptConstructor
from llms import lm_config


def generate_execution_summary(execution_result: Dict[str, Any]) -> str:
    """Generate a human-readable execution summary from execution result.

    Args:
        execution_result: The execution result from multi-agent coordinator

    Returns:
        Human-readable summary string
    """
    lines = []
    lines.append("Multi-Agent Web Automation Execution Summary")
    lines.append("=" * 50)
    lines.append("")

    # Task information
    if "goal" in execution_result:
        goal = execution_result["goal"]
        lines.append("Task Information:")
        lines.append(f"  Goal: {goal}")
        lines.append(f"  Completed: {'Yes' if execution_result.get('success_rate', 0) >= 0.8 else 'No'}")
        lines.append(f"  Completion: {execution_result.get('success_rate', 0) * 100:.1f}%")
        lines.append(f"  Steps Executed: {execution_result.get('total_steps', 0)}")
        if 'execution_time_formatted' in execution_result:
            lines.append(f"  Execution Time: {execution_result['execution_time_formatted']}")
        lines.append("")

    # Agent performance
    if 'reflections' in execution_result and execution_result['reflections']:
        lines.append("Agent Performance:")

        # Count successful actions
        actions = execution_result.get('actions', [])
        successful_actions = sum(1 for action in actions if action.get('action_type') != 'NONE')
        total_actions = len(actions)

        lines.append(f"  actor_agent:")
        lines.append(f"    total_intentions: {len(execution_result.get('intentions', []))}")
        lines.append(f"    successful_actions: {successful_actions}")
        lines.append(f"    failed_actions: {total_actions - successful_actions}")
        if total_actions > 0:
            fulfillment_rate = successful_actions / total_actions
            lines.append(f"    fulfillment_rate: {fulfillment_rate * 100:.1f}%")

        # Count reflection success
        reflections = execution_result['reflections']
        successful_reflections = sum(1 for reflection in reflections if reflection.get('success', False))
        helpful_reflections = sum(1 for reflection in reflections if reflection.get('helpful', False))
        stuck_reflections = sum(1 for reflection in reflections if reflection.get('stuck', False))

        lines.append(f"  reflector_agent:")
        lines.append(f"    total_reflections: {len(reflections)}")
        lines.append(f"    successful_reflections: {successful_reflections}")
        lines.append(f"    helpful_reflections: {helpful_reflections}")
        lines.append(f"    stuck_reflections: {stuck_reflections}")
        if len(reflections) > 0:
            success_rate = successful_reflections / len(reflections)
            helpful_rate = helpful_reflections / len(reflections)
            lines.append(f"    success_rate: {success_rate * 100:.1f}%")
            lines.append(f"    helpful_rate: {helpful_rate * 100:.1f}%")
            lines.append(f"    stuck_rate: {stuck_reflections * 100:.1f}%")

        lines.append("")

    # Overall assessment
    lines.append("Overall Assessment:")
    success_rate = execution_result.get('success_rate', 0)
    lines.append(f"  Success: {'Yes' if success_rate >= 0.8 else 'No'}")
    lines.append(f"  Success Rate: {success_rate * 100:.1f}%")
    lines.append(f"  Total Actions: {len(execution_result.get('actions', []))}")
    lines.append(f"  Total Steps: {execution_result.get('total_steps', 0)}")
    lines.append("")

    return "\n".join(lines)


def config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Agent Web Arena Runner")

    # Required arguments
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to JSON configuration file")

    # Commonly overridden arguments (for convenience)
    parser.add_argument("--start_url", type=str,
                       help="Override starting URL (overrides config file)")
    parser.add_argument("--intent", type=str,
                       help="Override task intent (overrides config file)")
    parser.add_argument("--max_steps", type=int,
                       help="Override maximum steps (overrides config file)")
    parser.add_argument("--result_dir", type=str,
                       help="Override result directory (overrides config file)")

    # Debugging options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (overrides config file)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show configuration without executing")

    return parser.parse_args()


def load_config_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        return {}


def merge_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge loaded config with command line arguments."""
    merged = config.copy()

    # Only process a simplified set of command line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ["config_file", "dry_run"]:
            # Handle overrides for commonly changed parameters
            if key in ["start_url", "intent", "max_steps"]:
                if "task" not in merged:
                    merged["task"] = {}
                merged["task"][key] = value
            elif key in ["result_dir", "verbose"]:
                if "output" not in merged:
                    merged["output"] = {}
                merged["output"][key] = value
            else:
                # Direct override for any other arguments
                merged[key] = value

    return merged


def test(args, config_file):
    """Run the multi-agent system."""

    # Load and merge configuration
    file_config = load_config_file(config_file)
    config = merge_config_with_args(file_config, args)

    # Handle dry run
    if args.dry_run:
        return

    # Setup result directory
    result_dir = config.get('output', {}).get('result_dir', 'results')
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created result directory: {result_dir}")

    # Add result_dir to config for coordinator
    if 'output' not in config:
        config['output'] = {}
    config['output']['result_dir'] = result_dir

    # Import the full multi-agent coordinator
    from agent.multi_agent_coordinator import MultiAgentCoordinator
    from agent import PromptAgent
    from agent.prompts.prompt_constructor import PromptConstructor
    from llms import lm_config

    # Create LM config
    try:
        # Extract model config from the config dictionary
        model_config = config.get('model', {})

        # Create LMConfig directly from the dictionary
        lm_cfg = lm_config.LMConfig(
            provider=model_config.get('provider', 'openai'),
            model=model_config.get('model', 'gpt-4'),
            mode=model_config.get('mode', 'chat')
        )

        # Add generation config if available
        if model_config:
            lm_cfg.gen_config.update({
                'temperature': model_config.get('temperature', 1.0),
                'top_p': model_config.get('top_p', 0.9),
                'max_tokens': model_config.get('max_tokens', 384),
                'context_length': model_config.get('context_length', 0),
                'stop_token': model_config.get('stop_token', None),
                'max_obs_length': model_config.get('max_obs_length', 0),
                'max_retry': model_config.get('max_retry', 3)
            })
    except (KeyError, AttributeError) as e:
        # Fallback to minimal config if required fields missing
        lm_cfg = lm_config.LMConfig(
            provider=config.get('model', {}).get('provider', 'openai'),
            model=config.get('model', {}).get('model', 'gpt-4'),
            mode=config.get('model', {}).get('mode', 'chat')
        )

    # Create browser environment (similar to run.py)
    from browser_env import ScriptBrowserEnv
    from browser_env.utils import Observation
    from typing import Optional

    # Get browser environment configuration
    browser_config = config.get('browser', {})
    # Build viewport_size from config
    viewport_size = {
        "width": browser_config.get('viewport_width', 1280),
        "height": browser_config.get('viewport_height', 720),
    }

    env = ScriptBrowserEnv(
        headless=browser_config.get('headless', False),  # Set to False for debugging
        slow_mo=browser_config.get('slow_mo', 100),
        observation_type=browser_config.get('observation_type', 'accessibility_tree'),
        current_viewport_only=browser_config.get('current_viewport_only', True),
        viewport_size=viewport_size,
        save_trace_enabled=browser_config.get('save_trace_enabled', True),
        sleep_after_execution=browser_config.get('sleep_after_execution', 0.5),
    )

    # Create prompt constructor for multi-agent system using existing framework
    from agent.prompts.prompt_constructor import DirectPromptConstructor
    from llms.tokenizers import Tokenizer

    # Get instruction path from config or use default
    instruction_path = config.get('instruction_path') or 'agent/prompts/jsons/p_cot_id_actree_3s.json'

    # Create prompt constructor using existing DirectPromptConstructor
    prompt_constructor = DirectPromptConstructor(
        instruction_path=instruction_path,
        lm_config=lm_cfg,
        tokenizer=Tokenizer(lm_cfg.provider, lm_cfg.model)
    )

    # Create base prompt agent for multi-agent coordinator
    # Use action_set_tag from configuration instead of hardcoding
    action_set_tag = config.get('observation', {}).get('action_set_tag', 'id_accessibility_tree')
    base_agent = PromptAgent(
        action_set_tag=action_set_tag,
        lm_config=lm_cfg,
        prompt_constructor=prompt_constructor,
    )

    # Create multi-agent coordinator with browser environment
    coordinator = MultiAgentCoordinator(lm_cfg, base_agent, browser_env=env, result_dir=result_dir)

    # Execute workflow with initial observation from browser
    # Use start_url from config if available
    start_url = config.get('task', {}).get('start_url')
    reset_options = {}
    if start_url:
        reset_options["start_url"] = start_url

    initial_obs, initial_info = env.reset(options=reset_options if reset_options else None)
    initial_observation = {"observation": initial_obs, "info": initial_info}

    result = coordinator.execute_task(
        user_goal=config.get('task', {}).get('intent', 'Not specified'),
        start_observation=initial_observation,
        max_steps=config.get('task', {}).get('max_steps', 3)
    )

    # Return the execution result
    return result


if __name__ == "__main__":
    args = config()
    test(args, args.config_file)