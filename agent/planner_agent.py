"""Planner Agent for task decomposition and current state analysis."""

from typing import Any, Dict, List, Optional

from browser_env.utils import Observation
from llms import lm_config

from .planner.task_decomposer import TaskDecomposer
from .planner.current_state_analyzer import CurrentStateAnalyzer


class PlannerAgent:
    """Decomposes complex tasks and analyzes current state for execution planning.

    Responsible for:
    1. Initial task decomposition into manageable subtasks
    2. Current state analysis to determine progress and next atomic action
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.task_decomposer = TaskDecomposer(lm_config)
        self.state_analyzer = CurrentStateAnalyzer(lm_config)

        # Planning state
        self.subtasks: List[str] = []
        self.current_step_index: int = 0
        self.task_decomposed: bool = False

    def generate_intention(
        self,
        user_goal: str,
        context_summary: Dict[str, Any],
        current_observation: Optional[Observation] = None,
        previous_intentions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate the next execution intention based on current state.

        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation
            previous_intentions: List of intentions already completed (not used in new approach)

        Returns:
            Dictionary containing selected intention and metadata
        """

        # Step 1: Perform task decomposition only on the first step
        if not self.task_decomposed:
            print("🎯 Planner Agent: Decomposing task...")
            decomposition_result = self.task_decomposer.decompose_task(
                user_goal=user_goal,
                current_observation=current_observation or {"text": ""},
                context_summary=context_summary,
            )
            self.subtasks = decomposition_result.get("subtasks", [])
            self.task_decomposed = True

            print(f"🎯 Task decomposed into {len(self.subtasks)} subtasks:")
            for i, subtask in enumerate(self.subtasks, 1):
                print(f"   {i}. {subtask}")
        else:
            print("🎯 Planner Agent: Analyzing current state...")

        # Step 2: Analyze current state to determine current subtask and next action
        state_analysis = self.state_analyzer.analyze_current_state(
            user_goal=user_goal,
            subtasks=self.subtasks,
            current_observation=current_observation or {"text": ""},
            context_summary=context_summary,
        )

        # Extract key information from state analysis
        current_subtask = state_analysis.get("current_subtask", "")
        next_atomic_action = state_analysis.get("next_atomic_action", "")
        reasoning = state_analysis.get("reasoning", "")
        response = state_analysis.get("response", "")

        # Create the intention for the Actor Agent
        # Use the atomic action as the main intention for more precise execution
        if next_atomic_action:
            selected_intention = next_atomic_action
        elif current_subtask:
            selected_intention = current_subtask
        else:
            # Fallback intention
            if self.current_step_index < len(self.subtasks):
                selected_intention = self.subtasks[self.current_step_index]
            else:
                print(f"🎯 Planner Agent: No subtasks available, continuing with user goal: {user_goal}")
                selected_intention = f"Continue working on: {user_goal}"

        if self.current_step_index < len(self.subtasks) - 1:
            self.current_step_index += 1

        # Build planning result with comprehensive information
        planning_result = {
            "intention": selected_intention,
            "current_subtask": current_subtask,
            "next_atomic_action": next_atomic_action,
            "reasoning": reasoning,
            "all_subtasks": self.subtasks,
            "current_step_index": self.current_step_index,
            "total_subtasks": len(self.subtasks),
            "task_decomposed": self.task_decomposed,
            "state_analysis": state_analysis,
            "user_goal": user_goal,
            "response": response
        }

        return planning_result

    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get statistics about planning performance and patterns.

        Returns:
            Dictionary containing planning statistics
        """
        if not self.task_decomposed:
            return {
                "total_subtasks": 0,
                "current_step_index": 0,
                "task_decomposed": False,
                "message": "No planning history available"
            }

        return {
            "total_subtasks": len(self.subtasks),
            "current_step_index": self.current_step_index,
            "task_decomposed": self.task_decomposed,
            "completion_percentage": (self.current_step_index / len(self.subtasks)) if self.subtasks else 0.0,
            "remaining_subtasks": len(self.subtasks) - self.current_step_index,
            "all_subtasks": self.subtasks,
            "current_subtask": self.subtasks[self.current_step_index] if self.current_step_index < len(self.subtasks) else "",
            "completed_subtasks": self.subtasks[:self.current_step_index],
        }

    def should_adjust_planning_strategy(self) -> Dict[str, Any]:
        """Determine if planning strategy needs adjustment based on performance.

        Returns:
            Dictionary containing adjustment recommendations
        """
        stats = self.get_planning_statistics()

        if not stats["task_decomposed"]:
            return {"needs_adjustment": False, "reason": "No planning data available"}

        reasons = []
        needs_adjustment = False

        # Check if stuck on same subtask for too long
        if stats["current_step_index"] == 0 and stats["total_subtasks"] > 0:
            needs_adjustment = True
            reasons.append("Still on first subtask, may need different approach")

        # Check if approaching end without progress
        if stats["current_step_index"] >= stats["total_subtasks"]:
            needs_adjustment = True
            reasons.append("Exhausted all subtasks but task may not be complete")

        return {
            "needs_adjustment": needs_adjustment,
            "reasons": reasons,
            "statistics": stats,
        }

    def reset_planning_state(self) -> None:
        """Reset planning state for a new task."""
        self.subtasks.clear()
        self.current_step_index = 0
        self.task_decomposed = False

    def get_current_subtask(self) -> str:
        """Get the current subtask being worked on.

        Returns:
            Current subtask or empty string if no subtasks available
        """
        if self.current_step_index < len(self.subtasks):
            return self.subtasks[self.current_step_index]
        return ""

    def get_remaining_subtasks(self) -> List[str]:
        """Get remaining subtasks to be completed.

        Returns:
            List of remaining subtasks
        """
        return self.subtasks[self.current_step_index + 1:]

    def mark_current_subtask_completed(self) -> None:
        """Mark the current subtask as completed and move to next."""
        if self.current_step_index < len(self.subtasks) - 1:
            self.current_step_index += 1
        elif self.current_step_index == len(self.subtasks) - 1:
            self.current_step_index += 1  # Mark as beyond the end
            print("🎯 All subtasks marked as completed")