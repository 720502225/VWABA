"""Simplified Effectiveness analysis for Reflector Agent."""

from typing import Any, Dict, List

from browser_env import Action, Trajectory
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template


class EffectivenessAnalyzer:
    """Analyzes the effectiveness of actions in progressing toward the goal."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def analyze(
        self,
        trajectory: Trajectory,
        current_intention: str,
        latest_action: Action,
        context_summary: Dict[str, Any],
    ) -> str:
        """Analyze the effectiveness of the latest action.

        Args:
            trajectory: Current execution trajectory
            current_intention: The intention that was being fulfilled
            latest_action: The most recently executed action
            context_summary: Current context from Context Agent

        Returns:
            Natural language response about action effectiveness
        """
        # Extract key metrics
        previous_progress = context_summary.get("completion_status", 0.0)
        making_progress = context_summary.get("making_progress", False)
        success_rate = context_summary.get("success_rate", 0.0)

        # Get recent trajectory context
        recent_context = self._extract_recent_context(trajectory, latest_action)

        # Convert text IDs back to readable string
        text_ids = latest_action.get("text", [])
        if isinstance(text_ids, list) and text_ids:
            try:
                # Import the ID to key mapping from browser_env
                from browser_env.actions import _id2key
                action_text = ''.join(_id2key[id_num] if 0 <= id_num < len(_id2key) else '?' for id_num in text_ids)
            except (ImportError, IndexError):
                # Fallback: try to convert IDs to characters directly
                action_text = ''.join(chr(id_num) if 32 <= id_num <= 126 else '?' for id_num in text_ids)
        else:
            action_text = "N/A"

        # Build analysis prompt using template
        prompt = load_prompt_template(
            "reflector_agent",
            "effectiveness_analysis",
            user_goal="Web automation task",
            trajectory_summary=recent_context,
            current_intention=current_intention,
            latest_action=f"Type: {latest_action.get('action_type', 'UNKNOWN')}, Element: {latest_action.get('element_id', 'N/A')}, Details: {action_text}",
            context_summary=f"Previous completion: {previous_progress:.1%}, Currently making progress: {making_progress}, Success rate: {success_rate:.1%}"
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return response
        except Exception as e:
            # Fallback effectiveness analysis
            action_type = latest_action.get("action_type", "UNKNOWN")
            return f"Fallback analysis: Action '{action_type}' executed. Based on progress metrics ({previous_progress:.1%} completion, {making_progress} progress indication), this appears to be {'effective' if making_progress else 'ineffective'}."

    def _extract_recent_context(self, trajectory: Trajectory, latest_action: Action) -> str:
        """Extract relevant context from the recent trajectory."""
        context_parts = []

        # Get last 3 action-observation pairs
        recent_steps = trajectory[-6:]  # Last 3 pairs (obs-action-obs-action-obs-action-obs)

        page_count = 1
        action_count = 1

        for step in recent_steps:
            if isinstance(step, dict):
                # Check if it's an observation (StateInfo) by looking for 'observation' key
                if 'observation' in step:
                    obs_text = step.get("observation", {}).get("text", "")[:200]
                    context_parts.append(f"Page {page_count}: {obs_text}...")
                    page_count += 1
                # Check if it's an action by looking for 'action_type' key
                elif 'action_type' in step:
                    action_type = step.get("action_type", "UNKNOWN")
                    element_id = step.get("element_id", "N/A")
                    context_parts.append(f"Action {action_count}: {action_type} on {element_id}")
                    action_count += 1

        return " | ".join(context_parts)