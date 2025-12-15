"""Workflow management for multi-agent coordination."""

from typing import Any, Dict, List, Optional

from browser_env import Action
from browser_env.utils import Observation


class WorkflowManager:
    """Manages the workflow and execution flow for multiple agents."""

    def __init__(self) -> None:
        self.current_step = 0
        self.max_steps = 30
        self.workflow_state = "running"
        self.execution_history: List[Dict[str, Any]] = []

    def initialize_workflow(self, max_steps: int = 30) -> Dict[str, Any]:
        """Initialize the workflow for a new task.

        Args:
            max_steps: Maximum number of execution steps

        Returns:
            Dictionary containing workflow initialization info
        """
        self.current_step = 0
        self.max_steps = max_steps
        self.workflow_state = "running"
        self.execution_history.clear()

        return {
            "max_steps": max_steps,
            "current_step": 0,
            "workflow_state": "initialized",
        }

    def should_continue_execution(
        self, context_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine if execution should continue.

        Args:
            context_summary: Current context from Context Agent (currently unused)

        Returns:
            Dictionary containing continuation decision and reasoning
        """
        # Check step limit - this is the primary stopping condition
        if self.current_step >= self.max_steps:
            return {
                "should_continue": False,
                "reason": "Maximum steps reached",
                "stop_type": "step_limit",
            }

        # Default: continue execution
        return {
            "should_continue": True,
            "reason": "Execution should continue",
            "stop_type": "none",
        }

    def record_execution_step(
        self,
        step_number: int,
        intention: str,
        action: Action,
        observation: Observation,
        reflection: Dict[str, Any],
        execution_time: Optional[float] = None,
    ) -> None:
        """Record a complete execution step.

        Args:
            step_number: Current step number
            intention: The intention that was being fulfilled
            action: The action that was executed
            observation: The result observation
            reflection: The reflection on the execution
            execution_time: Time taken for this step
        """
        step_record = {
            "step_number": step_number,
            "intention": intention,
            "action": action,
            "observation": observation,
            "reflection": reflection,
            "execution_time": execution_time,
        }

        self.execution_history.append(step_record)
        self.current_step = step_number

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current workflow execution.

        Returns:
            Dictionary containing workflow statistics
        """
        if not self.execution_history:
            return {
                "total_steps": 0,
                "workflow_state": self.workflow_state,
                "message": "No execution history available",
            }

        total_steps = len(self.execution_history)

        # Count action types
        action_types = {}
        for record in self.execution_history:
            action = record.get("action", {})
            action_type = action.get("action_type", "UNKNOWN")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        return {
            "total_steps": total_steps,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "workflow_state": self.workflow_state,
            "progress_percentage": (self.current_step / self.max_steps) * 100,
            "action_type_distribution": action_types,
        }

    def finalize_workflow(self, final_state: str, completion_reason: str) -> Dict[str, Any]:
        """Finalize the workflow execution.

        Args:
            final_state: Final state of the workflow
            completion_reason: Reason for workflow completion

        Returns:
            Dictionary containing workflow finalization info
        """
        self.workflow_state = final_state

        finalization_record = {
            "final_state": final_state,
            "completion_reason": completion_reason,
            "total_steps": self.current_step,
            "max_steps": self.max_steps,
            "statistics": self.get_workflow_statistics(),
            "execution_summary": self._generate_execution_summary(),
        }

        return finalization_record

    def _generate_execution_summary(self) -> str:
        """Generate a summary of the workflow execution."""
        if not self.execution_history:
            return "No execution steps recorded"

        stats = self.get_workflow_statistics()
        return f"Workflow completed with {stats['total_steps']} steps out of {stats['max_steps']} maximum."

    def reset_workflow(self) -> None:
        """Reset the workflow for a new task."""
        self.current_step = 0
        self.workflow_state = "ready"
        self.execution_history.clear()
