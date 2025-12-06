"""Context Agent for managing global state and progress tracking."""

from typing import Any, Dict, List, Optional

import torch

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config

from .context.summary_generator import SummaryGenerator
from .context import StateManager
from .prompts.prompt_loader import generate_llm_prompt_from_template
from .memory import MemoryBank, MemoryGenerator


class ContextAgent:
    """Manages global state and context summarization.

    Responsible for maintaining task execution history and generating
    comprehensive context summaries for other agents.
    """

    def __init__(self, lm_config: lm_config.LMConfig, memory_config: Dict[str, Any]) -> None:
        self.lm_config = lm_config
        self.state_manager = StateManager()
        self.summary_generator = SummaryGenerator(lm_config)


        # Initialize memory system
        self.enable_memory = memory_config.get("enable_memory", False)
        self.enable_memory_store = memory_config.get("enable_memory_store", False)
        self.memory_content = ""
        if self.enable_memory or self.enable_memory_store:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            self.memory_bank = MemoryBank(
                memory_dir=memory_config.get('memory_dir', 'agent_memories'),
                embedding_model=memory_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                top_k=memory_config.get('top_k', 3),
                device=device
            )
            self.memory_generator = MemoryGenerator(lm_config)
            self.window_size = memory_config.get('window_size', 3)

    def update_context(
        self,
        trajectory: Trajectory,
        user_goal: str,
        current_observation: Optional[Observation] = None,
        latest_intention: Optional[str] = None,
        latest_action: Optional[Action] = None,
        latest_reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update context state and generate comprehensive summary.

        Args:
            trajectory: Current execution trajectory
            user_goal: Original user goal/task
            current_observation: Latest page observation
            latest_action: Most recent action taken
            latest_reflection: Most recent reflection from Reflector Agent

        Returns:
            Dictionary containing updated context information
        """
        # Update state manager with new information (avoid duplicates)
        if current_observation and (not self.state_manager.get_all_observations() or
                                   current_observation != self.state_manager.get_latest_observation()):
            self.state_manager.add_observation(current_observation)

        if latest_intention and (not self.state_manager.get_all_intentions() or
                                 latest_intention != self.state_manager.get_all_intentions()[-1]):
            self.state_manager.add_intention(latest_intention)

        if latest_action and (not self.state_manager.get_all_actions() or
                             latest_action != self.state_manager.get_latest_action()):
            self.state_manager.add_action(latest_action)

        if latest_reflection and (not self.state_manager.get_all_reflections() or
                                 latest_reflection != self.state_manager.get_all_reflections()[-1]):
            self.state_manager.add_reflection(latest_reflection)

        # Get complete execution history
        history = self.state_manager.get_history()

        # Generate context summary without progress metrics
        summary, observation_summary, action_summary, reflection_summary = self.summary_generator.generate_summary(
            user_goal=user_goal,
            observations=self.state_manager.get_all_observations(),
            actions=self.state_manager.get_all_actions(),
            reflections=self.state_manager.get_all_reflections(),
        )

        # Return comprehensive context information
        return {
            "summary": summary,
            "memory_content": self.memory_content,
            "observation_summary": observation_summary,
            "action_summary": action_summary,
            "reflection_summary": reflection_summary,
            "state_history": history,
            "latest_observation": self.state_manager.get_latest_observation(),
            "latest_action": self.state_manager.get_latest_action(),
        }

    def reset(self) -> None:
        """Reset all context state for a new task."""
        self.state_manager.clear()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current context state without updating."""
        history = self.state_manager.get_history()
        return {
            "state_history": history,
            "latest_observation": self.state_manager.get_latest_observation(),
            "latest_action": self.state_manager.get_latest_action(),
            "total_steps": history.get("total_steps", 0),
        }

    def check_task_completion(
        self, user_goal: str, completion_threshold: float = 0.95
    ) -> bool:
        """Check if the task is considered complete based on current state.

        Args:
            user_goal: Original user goal
            completion_threshold: Minimum completion percentage (default: 0.95)

        Returns:
            True if task is considered complete, False otherwise
        """
        # Simple completion check based on action count and reflections
        history = self.state_manager.get_history()
        total_steps = history.get("total_steps", 0)
        reflections = history.get("reflections", [])

        # Basic heuristic: if we have successful actions and no recent stuck patterns
        if total_steps == 0:
            return False

        # Check recent reflections for success patterns
        recent_reflections = reflections[-3:] if reflections else []
        if recent_reflections:
            successful = sum(1 for r in recent_reflections if r.get("success", False))
            stuck = sum(1 for r in recent_reflections if r.get("stuck", False))

            # Consider complete if mostly successful and no stuck patterns
            success_rate = successful / len(recent_reflections)
            return success_rate >= completion_threshold and stuck == 0

        # Default fallback: assume incomplete without reflection data
        return False

    def initialize_task_memory(self, user_goal: str) -> Dict[str, Any]:
        """Initialize memory tracking for a new task.

        Args:
            user_goal: The user's goal/task description

        Returns:
            Relevant memories from past similar tasks
        """
        if not self.enable_memory:
            self.memory_content = ""
            return {"memory_content": "", "relevant_memories": []}

        # Store user goal in state manager for memory generation
        self.state_manager.set_user_goal(user_goal)

        # Retrieve relevant memories for this task
        try:
            relevant_memories = self.memory_bank.search_memories(query=user_goal)

            self.memory_content = self._get_mem_str(relevant_memories)
            return {
                "memory_content": self.memory_content,
                "relevant_memories": relevant_memories,
            }
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            self.memory_content = ""
            return {
                "memory_content": "",
                "relevant_memories": [],
            }

    def _get_mem_str(self, memories: List[Dict[str, Any]]) -> str:
        """Convert list of memories to a formatted string.

        Args:
            memories: List of memory dictionaries from memory bank

        Returns:
            Formatted string of memories
        """
        if not memories:
            return ""

        mem_str = ""
        for i, mem in enumerate(memories):
            mem_str += f"Memory {i+1}:\n"
            mem_str += f"Title: {mem['title']}\n"
            mem_str += f"Description: {mem['description']}\n"
            mem_str += f"Content: {mem['content']}\n\n"

        return mem_str.strip()

    def generate_and_store_memory(self, task_completed: bool) -> Optional[str]:
        """Generate and store memory from completed task.

        Args:
            task_completed: Whether the task was completed successfully

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.enable_memory_store:
            return None

        try:
            # Get user goal from state manager
            user_goal = self.state_manager.get_user_goal()

            # Generate memory from trajectory using state manager data
            memory_data = self.memory_generator.generate_memory_from_trajectory(
                user_goal=user_goal,
                observations=self.state_manager.get_all_observations(),
                intentions=self.state_manager.get_all_intentions(),
                actions=self.state_manager.get_all_actions(),
                reflections=self.state_manager.get_all_reflections(),
                task_completed=task_completed,
                window_size=self.window_size,
            )

            # Store memory in bank
            memory_id = self.memory_bank.add_memory(
                title=memory_data["title"],
                description=memory_data["description"],
                content=memory_data["content"],
                success_rate=memory_data.get("success_rate", 0.0),
            )

            print(f"💾 Memory stored: {memory_data['title'][:50]}... (ID: {memory_id})")
            return memory_id

        except Exception as e:
            print(f"Error generating memory: {e}")
            return None

