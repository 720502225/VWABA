"""Action execution and validation for Actor Agent."""

from typing import Any, Dict, List, Optional

from browser_env import Action
from browser_env.utils import Observation
from llms import lm_config


class ActionExecutor:
    """Executes and validates actions for the Actor Agent."""

    def __init__(self, action_set_tag: str) -> None:
        self.action_set_tag = action_set_tag
        self.execution_history: List[Dict[str, Any]] = []

    def validate_action(
        self, action: Action
    ) -> Dict[str, Any]:
        """Validate an action format and content before execution.

        Args:
            action: The action to validate

        Returns:
            Dictionary containing validation results
        """
        # Record validation attempt
        validation_record = {
            "action": action,
            "timestamp": None,  # Would be set in actual implementation
        }

        try:
            # Validate action format and content
            validation_result = self._validate_action_format(action)

            # Update validation record
            validation_record.update({
                "validation_passed": validation_result["valid"],
                "validation_details": validation_result,
            })

            # Store validation history
            self.execution_history.append(validation_record)

            return {
                "valid": validation_result["valid"],
                "action": action,
                "validation_details": validation_result,
                "validation_history_length": len(self.execution_history),
            }

        except Exception as e:
            # Record failed validation
            validation_record.update({
                "validation_passed": False,
                "error": str(e),
            })
            self.execution_history.append(validation_record)

            return {
                "valid": False,
                "error": str(e),
                "action": action,
                "validation_details": {"valid": False, "error": str(e)},
                "validation_history_length": len(self.execution_history),
            }

    def _validate_action_format(self, action: Action) -> Dict[str, Any]:
        """Validate the format and content of an action.

        Args:
            action: The action to validate

        Returns:
            Dictionary containing validation results
        """
        required_fields = ["action_type"]
        validation_result = {
            "valid": True,
            "missing_fields": [],
            "invalid_fields": [],
            "warnings": [],
        }

        # Check required fields
        for field in required_fields:
            if field not in action:
                validation_result["valid"] = False
                validation_result["missing_fields"].append(field)

        # Validate action type
        if "action_type" in action:
            action_type = action["action_type"]
            valid_types = [
                "CLICK", "TYPE", "SCROLL", "KEY_PRESS", "GOTO_URL",
                "NEW_TAB", "PAGE_CLOSE", "GO_BACK", "GO_FORWARD",
                "PAGE_FOCUS", "CLEAR", "UPLOAD", "STOP", "NONE"
            ]

            if action_type not in valid_types:
                validation_result["valid"] = False
                validation_result["invalid_fields"].append(f"Invalid action_type: {action_type}")

            # Type-specific validations
            if action_type == "TYPE" and "element_id" not in action:
                validation_result["valid"] = False
                validation_result["missing_fields"].append("element_id for TYPE action")

            if action_type == "CLICK" and "element_id" not in action:
                validation_result["valid"] = False
                validation_result["missing_fields"].append("element_id for CLICK action")

            if action_type == "SCROLL" and "direction" not in action:
                validation_result["valid"] = False
                validation_result["missing_fields"].append("direction for SCROLL action")

        # Check for potential issues (warnings)
        if "element_id" in action:
            element_id = action["element_id"]
            if isinstance(element_id, str) and not element_id.strip():
                validation_result["warnings"].append("Empty element_id detected")

        return validation_result


    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about action execution performance.

        Returns:
            Dictionary containing execution statistics
        """
        if not self.execution_history:
            return {"total_executions": 0, "message": "No execution history available"}

        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record.get("execution_success", False))
        validation_passed = sum(1 for record in self.execution_history if record.get("validation_passed", False))

        # Calculate success rates
        execution_success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
        validation_success_rate = validation_passed / total_executions if total_executions > 0 else 0.0

        # Analyze action type distribution
        action_types = {}
        for record in self.execution_history:
            action = record.get("action", {})
            action_type = action.get("action_type", "UNKNOWN")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        # Get recent executions (last 10)
        recent_executions = self.execution_history[-10:]
        recent_success_rate = sum(1 for record in recent_executions if record.get("execution_success", False)) / len(recent_executions) if recent_executions else 0.0

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "validation_passed": validation_passed,
            "execution_success_rate": execution_success_rate,
            "validation_success_rate": validation_success_rate,
            "recent_success_rate": recent_success_rate,
            "action_type_distribution": action_types,
            "most_common_action_type": max(action_types.items(), key=lambda x: x[1])[0] if action_types else "NONE",
            "recent_trend": "improving" if recent_success_rate > execution_success_rate else "stable",
        }

    def reset_execution_history(self) -> None:
        """Reset execution history for a new task."""
        self.execution_history.clear()