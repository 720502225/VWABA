"""Communication hub for multi-agent coordination."""

from typing import Any, Dict, Optional


class CommunicationHub:
    """Manages shared context between different agents."""

    def __init__(self) -> None:
        self.agent_states: Dict[str, Any] = {}
        self.shared_context: Dict[str, Any] = {}

    def register_agent(self, agent_name: str, initial_state: Dict[str, Any]) -> None:
        """Register an agent with the communication hub.

        Args:
            agent_name: Name of the agent
            initial_state: Initial state of the agent
        """
        self.agent_states[agent_name] = {
            "state": initial_state,
        }

    def update_agent_state(self, agent_name: str, state_update: Dict[str, Any]) -> None:
        """Update the state of a registered agent.

        Args:
            agent_name: Name of the agent
            state_update: State update information
        """
        if agent_name in self.agent_states:
            self.agent_states[agent_name]["state"].update(state_update)

    def update_shared_context(self, context_key: str, context_value: Any) -> None:
        """Update shared context accessible to all agents.

        Args:
            context_key: Key for the shared context
            context_value: Value for the shared context
        """
        self.shared_context[context_key] = context_value

    def get_shared_context(self, context_key: Optional[str] = None) -> Any:
        """Get shared context information.

        Args:
            context_key: Optional specific key to retrieve

        Returns:
            Shared context value or entire context dictionary
        """
        if context_key:
            return self.shared_context.get(context_key)
        return self.shared_context.copy()

    def get_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a registered agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Current state of the agent or None if not found
        """
        return self.agent_states.get(agent_name, {}).get("state")

    def reset(self) -> None:
        """Reset the communication hub for a new task."""
        self.agent_states.clear()
        self.shared_context.clear()
