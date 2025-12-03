"""Context manager for agent execution history and progress tracking."""

from typing import Any, Dict, List

from browser_env import Action
from browser_env.utils import Observation


class StateManager:
    """Manages execution history for context awareness."""

    def __init__(self) -> None:
        self.observations: List[Observation] = []
        self.actions: List[Action] = []
        self.reflections: List[Dict[str, Any]] = []

    def add_observation(self, observation: Observation) -> None:
        """Add a new observation to the history."""
        self.observations.append(observation)

    def add_action(self, action: Action) -> None:
        """Add a new action to the history."""
        self.actions.append(action)

    def add_reflection(self, reflection: Dict[str, Any]) -> None:
        """Add a new reflection to the history."""
        self.reflections.append(reflection)

    def get_all_observations(self) -> List[Observation]:
        """Get all observations."""
        return self.observations

    def get_all_actions(self) -> List[Action]:
        """Get all actions."""
        return self.actions

    def get_all_reflections(self) -> List[Dict[str, Any]]:
        """Get all reflections."""
        return self.reflections

    def get_latest_observation(self) -> Observation:
        """Get the most recent observation."""
        return self.observations[-1] if self.observations else None

    def get_latest_action(self) -> Action:
        """Get the most recent action."""
        return self.actions[-1] if self.actions else None

    def get_history(self) -> Dict[str, Any]:
        """Get complete execution history."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "reflections": self.reflections,
            "total_steps": len(self.actions),
            "total_observations": len(self.observations),
            "total_reflections": len(self.reflections),
        }

    def clear(self) -> None:
        """Clear all history."""
        self.observations.clear()
        self.actions.clear()
        self.reflections.clear()