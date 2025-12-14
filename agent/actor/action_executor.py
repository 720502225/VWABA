"""Action execution and validation for Actor Agent."""

from typing import Any, Dict, List

from browser_env import Action


class ActionExecutor:
    """Executes and validates actions for the Actor Agent."""

    def __init__(self, action_set_tag: str) -> None:
        self.action_set_tag = action_set_tag
        self.execution_history: List[Dict[str, Any]] = []

    def validate_action(self, action: Action) -> Dict[str, Any]:
        """Validate an action format and content before execution.

        Args:
            action: The action to validate

        Returns:
            Dictionary containing validation results
        """
        try:
            # Validate action format and content
            validation_result = self._validate_action_format(action)

            # Store validation history
            validation_record = {
                "action": action,
                "validation_passed": validation_result["valid"],
                "validation_details": validation_result,
            }
            self.execution_history.append(validation_record)

            return {
                "valid": validation_result["valid"],
                "action": action,
                "validation_details": validation_result,
            }

        except Exception as e:
            # Record failed validation
            validation_record = {
                "action": action,
                "validation_passed": False,
                "error": str(e),
            }
            self.execution_history.append(validation_record)

            return {
                "valid": False,
                "error": str(e),
                "action": action,
                "validation_details": {"valid": False, "error": str(e)},
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
                "PAGE_FOCUS", "CLEAR", "UPLOAD", "STOP", "NONE", "HOVER"
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

    def reset_execution_history(self) -> None:
        """Reset execution history for a new task."""
        self.execution_history.clear()
