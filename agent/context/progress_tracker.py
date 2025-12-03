"""Progress tracking and evaluation for task completion assessment."""

from typing import Any, Dict, List

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config, call_llm


class ProgressTracker:
    """Tracks and evaluates task progress."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def calculate_progress(
        self,
        trajectory: Trajectory,
        user_goal: str,
        history: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate progress metrics for the current task execution.

        Args:
            trajectory: Execution trajectory containing actions and observations
            user_goal: Original user goal/task description
            history: Execution history from StateManager

        Returns:
            Dictionary containing progress metrics
        """
        observations = history.get("observations", [])
        actions = history.get("actions", [])
        reflections = history.get("reflections", [])

        # Basic progress metrics
        total_steps = len(actions)
        successful_actions = sum(1 for r in reflections if r.get("success", False))
        helpful_actions = sum(1 for r in reflections if r.get("helpful", False))
        stuck_patterns = sum(1 for r in reflections if r.get("stuck", False))

        # Calculate success rate
        success_rate = successful_actions / max(1, total_steps)
        helpful_rate = helpful_actions / max(1, total_steps)
        stuck_rate = stuck_patterns / max(1, total_steps)

        # Estimate task completion based on reflection patterns
        if reflections:
            latest_reflection = reflections[-1]
            recent_helpful = sum(1 for r in reflections[-3:] if r.get("helpful", False))
            completion_estimate = min(0.95, (recent_helpful / 3.0) * 0.8 + success_rate * 0.2)
        else:
            completion_estimate = min(0.3, total_steps * 0.1)

        # Detect if we're making progress
        if len(reflections) >= 2:
            recent_helpful = sum(1 for r in reflections[-2:] if r.get("helpful", False))
            making_progress = recent_helpful >= 1
        else:
            making_progress = total_steps > 0

        # Use LLM to estimate completion if we have enough context
        if total_steps >= 3 and observations:
            llm_estimate = self._llm_progress_estimation(user_goal, observations[-1], trajectory)
            completion_estimate = (completion_estimate + llm_estimate) / 2

        return {
            "total_steps": total_steps,
            "successful_actions": successful_actions,
            "helpful_actions": helpful_actions,
            "stuck_patterns": stuck_patterns,
            "success_rate": success_rate,
            "helpful_rate": helpful_rate,
            "stuck_rate": stuck_rate,
            "completion_percentage": completion_estimate,
            "making_progress": making_progress,
            "needs_intervention": stuck_rate > 0.5 or not making_progress,
        }

    def _llm_progress_estimation(
        self, user_goal: str, current_observation: Observation, trajectory: Trajectory
    ) -> float:
        """Use LLM to estimate task completion progress."""
        try:
            # Extract recent observations for context
            recent_observations = []
            for i, step in enumerate(trajectory[-3:]):
                if i % 2 == 0:  # Observation step
                    obs_text = step.get("observation", {}).get("text", "")[:500]
                    recent_observations.append(obs_text)

            context = "\n".join(recent_observations)
            current_state = current_observation.get("text", "")[:500]

            prompt = f"""Analyze the current progress towards completing this task.

User Goal: {user_goal}

Current Page State: {current_state}

Recent Actions and Observations:
{context}

Based on the current state and recent progress, estimate what percentage of the task has been completed (0.0 to 1.0).
Consider whether the current state shows progress toward the goal or if we're still in early stages.

Respond with only a number between 0.0 and 1.0:"""

            response = call_llm(self.lm_config, [{"role": "user", "content": prompt}])
            try:
                estimate = float(response.strip())
                return max(0.0, min(1.0, estimate))
            except ValueError:
                return 0.5  # Default fallback

        except Exception:
            return 0.5  # Default fallback on any error