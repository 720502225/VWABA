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
            # Update current_step_index to match the LLM-determined current subtask
            # Try to find the matching subtask, handling cases where LLM adds numbering prefixes
            matched_index = self._find_subtask_index(current_subtask)
            if matched_index is not None:
                self.current_step_index = matched_index
        else:
            # Fallback intention
            if self.current_step_index < len(self.subtasks):
                selected_intention = self.subtasks[self.current_step_index]
            else:
                print(f"🎯 Planner Agent: No subtasks available, continuing with user goal: {user_goal}")
                selected_intention = f"Continue working on: {user_goal}"

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

    def reset_planning_state(self) -> None:
        """Reset planning state for a new task."""
        self.subtasks.clear()
        self.current_step_index = 0
        self.task_decomposed = False


    def _find_subtask_index(self, current_subtask: str) -> Optional[int]:
        """Find the index of a subtask, handling cases where LLM adds numbering prefixes.

        Args:
            current_subtask: The subtask text from LLM analysis (may include numbering)

        Returns:
            Index of the matching subtask, or None if not found
        """
        # First try exact match
        if current_subtask in self.subtasks:
            return self.subtasks.index(current_subtask)

        # Try removing common numbering patterns (e.g., "1. ", "2. ", "(1) ", etc.)
        import re

        # Pattern to match numbering prefixes like "1. ", "2. ", "(1) ", "1) ", etc.
        cleaned_subtask = re.sub(r'^\s*\d+\.?\s*', '', current_subtask).strip()
        cleaned_subtask = re.sub(r'^\s*\(\d+\)\s*', '', cleaned_subtask).strip()

        # Try exact match with cleaned version
        if cleaned_subtask in self.subtasks:
            return self.subtasks.index(cleaned_subtask)

        # Try partial match (first N characters) for robustness
        for i, subtask in enumerate(self.subtasks):
            # Remove numbering from stored subtask too
            cleaned_stored = re.sub(r'^\s*\d+\.?\s*', '', subtask).strip()
            cleaned_stored = re.sub(r'^\s*\(\d+\)\s*', '', cleaned_stored).strip()

            # Check if they match (case insensitive, ignore extra whitespace)
            if cleaned_subtask.lower().strip() == cleaned_stored.lower().strip():
                return i

            # Fallback: check if the cleaned subtask is contained in the stored subtask
            if len(cleaned_subtask) > 10 and cleaned_subtask.lower() in cleaned_stored.lower():
                return i

        return None