"""Communication hub for multi-agent coordination."""

from typing import Any, Dict, List, Optional

from browser_env import Action
from browser_env.utils import Observation


class CommunicationHub:
    """Manages communication between different agents."""

    def __init__(self) -> None:
        self.message_history: List[Dict[str, Any]] = []
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
            "last_updated": None,  # Would be set in actual implementation
            "message_count": 0,
        }

    def update_agent_state(self, agent_name: str, state_update: Dict[str, Any]) -> None:
        """Update the state of a registered agent.

        Args:
            agent_name: Name of the agent
            state_update: State update information
        """
        if agent_name in self.agent_states:
            self.agent_states[agent_name]["state"].update(state_update)
            self.agent_states[agent_name]["last_updated"] = None  # Would be set in actual implementation
            self.agent_states[agent_name]["message_count"] += 1

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: Any,
        priority: str = "normal",
    ) -> None:
        """Send a message from one agent to another.

        Args:
            from_agent: Name of the sending agent
            to_agent: Name of the receiving agent
            message_type: Type of message (e.g., "intention", "reflection", "context_update")
            content: Message content
            priority: Message priority ("high", "normal", "low")
        """
        message = {
            "id": len(self.message_history),  # Simple message ID
            "timestamp": None,  # Would be set in actual implementation
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "content": content,
            "priority": priority,
            "delivered": False,
        }

        self.message_history.append(message)

    def get_messages_for_agent(
        self, agent_name: str, message_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get messages intended for a specific agent.

        Args:
            agent_name: Name of the target agent
            message_type: Optional filter for specific message types

        Returns:
            List of messages for the agent
        """
        messages = [
            msg for msg in self.message_history if msg["to_agent"] == agent_name
        ]

        if message_type:
            messages = [msg for msg in messages if msg["message_type"] == message_type]

        # Sort by priority and timestamp
        priority_order = {"high": 3, "normal": 2, "low": 1}
        messages.sort(
            key=lambda x: (
                priority_order.get(x["priority"], 2),
                x.get("timestamp", 0),
            ),
            reverse=True,
        )

        return messages

    def update_shared_context(self, context_key: str, context_value: Any) -> None:
        """Update shared context accessible to all agents.

        Args:
            context_key: Key for the shared context
            context_value: Value for the shared context
        """
        self.shared_context[context_key] = {
            "value": context_value,
            "last_updated": None,  # Would be set in actual implementation
            "updated_by": "communication_hub",
        }

    def get_shared_context(self, context_key: Optional[str] = None) -> Any:
        """Get shared context information.

        Args:
            context_key: Optional specific key to retrieve

        Returns:
            Shared context value or entire context dictionary
        """
        if context_key:
            return self.shared_context.get(context_key, {}).get("value")
        return {k: v["value"] for k, v in self.shared_context.items()}

    def get_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a registered agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Current state of the agent or None if not found
        """
        return self.agent_states.get(agent_name, {}).get("state")

    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent communication.

        Returns:
            Dictionary containing communication statistics
        """
        total_messages = len(self.message_history)
        agent_message_counts = {}

        for msg in self.message_history:
            from_agent = msg["from_agent"]
            agent_message_counts[from_agent] = agent_message_counts.get(from_agent, 0) + 1

        # Calculate message types distribution
        message_types = {}
        for msg in self.message_history:
            msg_type = msg["message_type"]
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        # Calculate priority distribution
        priority_counts = {"high": 0, "normal": 0, "low": 0}
        for msg in self.message_history:
            priority = msg.get("priority", "normal")
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return {
            "total_messages": total_messages,
            "registered_agents": list(self.agent_states.keys()),
            "agent_message_counts": agent_message_counts,
            "message_type_distribution": message_types,
            "priority_distribution": priority_counts,
            "shared_context_keys": list(self.shared_context.keys()),
            "most_active_agent": max(agent_message_counts.items(), key=lambda x: x[1])[0] if agent_message_counts else None,
        }

    def mark_message_delivered(self, message_id: int) -> None:
        """Mark a message as delivered.

        Args:
            message_id: ID of the message to mark as delivered
        """
        for msg in self.message_history:
            if msg["id"] == message_id:
                msg["delivered"] = True
                break

    def clear_message_history(self, older_than_steps: Optional[int] = None) -> None:
        """Clear message history.

        Args:
            older_than_steps: Optional filter to keep recent messages
        """
        if older_than_steps is None:
            # Clear all messages
            self.message_history.clear()
        else:
            # Keep only recent messages (simple implementation)
            # In a real implementation, this would be time-based
            if len(self.message_history) > older_than_steps * 2:
                self.message_history = self.message_history[-older_than_steps :]

    def reset_communication_hub(self) -> None:
        """Reset the communication hub for a new task."""
        self.message_history.clear()
        self.agent_states.clear()
        self.shared_context.clear()