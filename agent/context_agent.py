"""Context Agent for managing global state and progress tracking."""

from typing import Any, Dict, List, Optional

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config

from .context.summary_generator import SummaryGenerator
from .context import StateManager
from .prompts.prompt_loader import generate_llm_prompt_from_template


class ContextAgent:
    """Manages global state and context summarization.

    Responsible for maintaining task execution history and generating
    comprehensive context summaries for other agents.
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.state_manager = StateManager()
        self.summary_generator = SummaryGenerator(lm_config)

    def update_context(
        self,
        trajectory: Trajectory,
        user_goal: str,
        current_observation: Optional[Observation] = None,
        latest_action: Optional[Action] = None,
        latest_reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update context state and generate comprehensive summary.

        Args:
            trajectory: Current execution trajectory
            user_goal: Original user goal/task
            current_observation: Latest page observation
            latest_action: Most recent action taken
            latest_reflection: Most recent reflection from Reflector Agent

        Returns:
            Dictionary containing updated context information
        """
        # Update state manager with new information (avoid duplicates)
        if current_observation and (not self.state_manager.get_all_observations() or
                                   current_observation != self.state_manager.get_latest_observation()):
            self.state_manager.add_observation(current_observation)

        if latest_action and (not self.state_manager.get_all_actions() or
                             latest_action != self.state_manager.get_latest_action()):
            self.state_manager.add_action(latest_action)

        if latest_reflection and (not self.state_manager.get_all_reflections() or
                                 latest_reflection != self.state_manager.get_all_reflections()[-1]):
            self.state_manager.add_reflection(latest_reflection)

        # Get complete execution history
        history = self.state_manager.get_history()

        # Generate context summary without progress metrics
        summary, observation_summary, action_summary, reflection_summary = self.summary_generator.generate_summary(
            user_goal=user_goal,
            observations=self.state_manager.get_all_observations(),
            actions=self.state_manager.get_all_actions(),
            reflections=self.state_manager.get_all_reflections(),
        )

        # Return comprehensive context information
        return {
            "summary": summary,
            "observation_summary": observation_summary,
            "action_summary": action_summary,
            "reflection_summary": reflection_summary,
            "state_history": history,
            "latest_observation": self.state_manager.get_latest_observation(),
            "latest_action": self.state_manager.get_latest_action(),
        }

    def reset(self) -> None:
        """Reset all context state for a new task."""
        self.state_manager.clear()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current context state without updating."""
        history = self.state_manager.get_history()
        return {
            "state_history": history,
            "latest_observation": self.state_manager.get_latest_observation(),
            "latest_action": self.state_manager.get_latest_action(),
            "total_steps": history.get("total_steps", 0),
        }

    def check_task_completion(
        self, user_goal: str, completion_threshold: float = 0.95
    ) -> bool:
        """Check if the task is considered complete based on current state.

        Args:
            user_goal: Original user goal
            completion_threshold: Minimum completion percentage (default: 0.95)

        Returns:
            True if task is considered complete, False otherwise
        """
        # Simple completion check based on action count and reflections
        history = self.state_manager.get_history()
        total_steps = history.get("total_steps", 0)
        reflections = history.get("reflections", [])

        # Basic heuristic: if we have successful actions and no recent stuck patterns
        if total_steps == 0:
            return False

        # Check recent reflections for success patterns
        recent_reflections = reflections[-3:] if reflections else []
        if recent_reflections:
            successful = sum(1 for r in recent_reflections if r.get("success", False))
            stuck = sum(1 for r in recent_reflections if r.get("stuck", False))

            # Consider complete if mostly successful and no stuck patterns
            success_rate = successful / len(recent_reflections)
            return success_rate >= completion_threshold and stuck == 0

        # Default fallback: assume incomplete without reflection data
        return False