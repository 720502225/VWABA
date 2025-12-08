"""Task decomposition for planning Agent."""

from typing import Any, Dict, List

from browser_env.utils import Observation
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template


class TaskDecomposer:
    """Decomposes complex tasks into manageable subtasks."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def decompose_task(
        self,
        user_goal: str,
        current_observation: Observation,
        context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Decompose user goal into 3-5 manageable subtasks.

        Args:
            user_goal: Original user goal
            current_observation: Current page observation

        Returns:
            Dictionary containing task decomposition results
        """
        # Analyze current page content
        # Handle both StateInfo format and direct observation format
        if isinstance(current_observation, dict) and "observation" in current_observation:
            # StateInfo format: {"observation": obs, "info": info}
            obs_data = current_observation["observation"]
        else:
            # Direct observation format
            obs_data = current_observation

        current_page_text = obs_data.get("text", "") if isinstance(obs_data, dict) else str(obs_data)
        page_elements = self._extract_page_elements(obs_data)

        memory = context_summary.get("memory_content", "")
        # Build decomposition prompt using template
        if memory != "":
            prompt = load_prompt_template(
                "planner_agent",
                "task_decomposition_w_mem",
                memory=memory,
                user_goal=user_goal,
                current_page_text=current_page_text,
                page_elements=page_elements,
            )
        else:
            prompt = load_prompt_template(
                "planner_agent",
                "task_decomposition",
                user_goal=user_goal,
                current_page_text=current_page_text,
                page_elements=page_elements
            )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()

            # Parse the LLM response into structured format
            decomposition = self._parse_decomposition_response(response)

        except Exception as e:
            # Fallback decomposition
            decomposition = self._generate_fallback_decomposition(user_goal, str(e))

        return decomposition

    def _extract_page_elements(self, observation: Observation) -> str:
        """Extract relevant page elements for analysis."""
        # Try to get text representation with element IDs
        obs_text = observation.get("text", "")

        # Look for element IDs in the text (common pattern: [ID] description)
        import re
        elements = re.findall(r'\[\d+\][^\n]*', obs_text)

        if elements:
            # Limit to first 20 elements to avoid overwhelming context
            elements = elements[:20]
            return "Interactive Elements:\n" + "\n".join(f"• {elem}" for elem in elements)
        else:
            return "Page content detected but no specific interactive elements identified."

    def _parse_decomposition_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM task decomposition response into structured format."""
        # Simple parsing for subtasks
        decomposition = {
            "subtasks": [],
            "reasoning": response,
        }

        # Try to extract subtasks
        lines = response.split('\n')
        for line in lines:
            line = line.strip()

            # Skip empty lines and headers
            if not line or line.lower().startswith(('here are', 'subtasks:', 'steps:', 'breakdown:')):
                continue

            # Remove numbering and bullet points
            cleaned = line.lstrip('0123456789.-* ')
            cleaned = cleaned.lstrip('- ')
            cleaned = cleaned.lstrip('• ')

            # Clean up common prefixes
            for prefix in ['Subtask:', 'Step:', 'Task:']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break

            if cleaned and len(cleaned) > 10:  # Reasonable minimum length
                decomposition["subtasks"].append(cleaned)

        # If no subtasks were parsed, use the full response
        if not decomposition["subtasks"] and len(response.strip()) > 10:
            decomposition["subtasks"] = [response.strip()]

        # Limit to 3-5 subtasks
        decomposition["subtasks"] = decomposition["subtasks"][:5]

        return decomposition

    def _generate_fallback_decomposition(self, user_goal: str, error: str) -> Dict[str, Any]:
        """Generate fallback task decomposition when LLM fails."""
        # Generate generic subtasks based on common web task patterns
        subtasks = []

        # Common web task patterns
        if any(keyword in user_goal.lower() for keyword in ['search', 'find', 'look for']):
            subtasks.extend([
                "Navigate to search functionality",
                "Enter search query",
                "Review search results",
                "Select relevant option"
            ])

        elif any(keyword in user_goal.lower() for keyword in ['buy', 'purchase', 'order', 'cart']):
            subtasks.extend([
                "Locate product or service to purchase",
                "Add item to shopping cart",
                "Proceed to checkout process",
                "Complete purchase"
            ])

        elif any(keyword in user_goal.lower() for keyword in ['information', 'details', 'about']):
            subtasks.extend([
                "Look for information sections or links",
                "Navigate to relevant information pages",
                "Extract and review requested information"
            ])

        else:
            # Generic subtasks
            subtasks = [
                f"Get started with the task: {user_goal}",
                f"Make progress on: {user_goal}",
                f"Complete the task: {user_goal}"
            ]

        return {
            "subtasks": subtasks[:5],  # Limit to 5 subtasks
            "reasoning": f"Fallback decomposition due to error: {error}",
        }