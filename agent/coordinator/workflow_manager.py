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
            "workflow_id": None,  # Would be set in actual implementation
            "max_steps": max_steps,
            "current_step": 0,
            "workflow_state": "initialized",
            "timestamp": None,  # Would be set in actual implementation
        }

    def should_continue_execution(
        self, context_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine if execution should continue.

        Args:
            context_summary: Current context from Context Agent

        Returns:
            Dictionary containing continuation decision and reasoning
        """
        # Check step limit
        if self.current_step >= self.max_steps:
            return {
                "should_continue": False,
                "reason": "Maximum steps reached",
                "stop_type": "step_limit",
            }

        # Check task completion
        completion_percentage = context_summary.get("completion_status", 0.0)
        if completion_percentage >= 0.95:
            return {
                "should_continue": False,
                "reason": f"Task completed ({completion_percentage:.1%})",
                "stop_type": "task_completed",
            }

        # Check for intervention needs
        needs_intervention = context_summary.get("needs_intervention", False)
        if needs_intervention:
            return {
                "should_continue": True,
                "reason": "Intervention needed, continuing with recovery",
                "stop_type": "none",
                "requires_intervention": True,
            }

        # Check progress
        making_progress = context_summary.get("making_progress", False)
        if not making_progress and self.current_step > 5:
            return {
                "should_continue": True,
                "reason": "Not making progress, but continuing",
                "stop_type": "none",
                "requires_monitoring": True,
            }

        # Default: continue
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
            "timestamp": None,  # Would be set in actual implementation
            "intention": intention,
            "action": action,
            "observation": observation,
            "reflection": reflection,
            "execution_time": execution_time,
            "step_success": reflection.get("success", False),
            "step_helpful": reflection.get("helpful", False),
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
        successful_steps = sum(1 for record in self.execution_history if record.get("step_success", False))
        helpful_steps = sum(1 for record in self.execution_history if record.get("step_helpful", False))

        # Calculate success rates
        success_rate = successful_steps / total_steps if total_steps > 0 else 0.0
        helpful_rate = helpful_steps / total_steps if total_steps > 0 else 0.0

        # Analyze recent performance (last 10 steps)
        recent_steps = self.execution_history[-10:]
        recent_success_rate = (
            sum(1 for record in recent_steps if record.get("step_success", False))
            / len(recent_steps)
            if recent_steps
            else 0.0
        )

        # Check for patterns in the workflow
        stuck_steps = sum(
            1
            for record in self.execution_history
            if record.get("reflection", {}).get("stuck", False)
        )

        return {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "helpful_steps": helpful_steps,
            "success_rate": success_rate,
            "helpful_rate": helpful_rate,
            "recent_success_rate": recent_success_rate,
            "stuck_steps": stuck_steps,
            "stuck_rate": stuck_steps / total_steps if total_steps > 0 else 0.0,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "workflow_state": self.workflow_state,
            "completion_percentage": (self.current_step / self.max_steps) * 100,
            "performance_trend": "improving" if recent_success_rate > success_rate else "stable",
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
            "workflow_id": None,  # Would be set in actual implementation
            "final_state": final_state,
            "completion_reason": completion_reason,
            "total_steps": self.current_step,
            "max_steps": self.max_steps,
            "statistics": self.get_workflow_statistics(),
            "timestamp": None,  # Would be set in actual implementation
            "execution_summary": self._generate_execution_summary(),
        }

        return finalization_record

    def _generate_execution_summary(self) -> str:
        """Generate a summary of the workflow execution."""
        if not self.execution_history:
            return "No execution steps recorded"

        stats = self.get_workflow_statistics()
        return (
            f"Workflow completed with {stats['success_rate']:.1%} success rate, "
            f"{stats['helpful_rate']:.1%} helpfulness rate, "
            f"and {stats['stuck_steps']} stuck steps out of {stats['total_steps']} total steps."
        )

    def reset_workflow(self) -> None:
        """Reset the workflow for a new task."""
        self.current_step = 0
        self.workflow_state = "ready"
        self.execution_history.clear()