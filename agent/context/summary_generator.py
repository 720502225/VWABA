"""Context summary generation for maintaining agent awareness."""

from typing import Any, Dict, List

from browser_env import Action
from browser_env.utils import Observation
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template


class SummaryGenerator:
    """Generates context summaries for agent coordination."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def generate_summary(
        self,
        user_goal: str,
        observations: List[Observation],
        actions: List[Action],
        reflections: List[Dict[str, Any]],
    ) -> str:
        """Generate a comprehensive context summary.

        Args:
            user_goal: Original user goal
            observations: List of all observations
            actions: List of all actions taken
            reflections: List of all reflections

        Returns:
            Generated context summary string
        """
        # Extract key information
        total_steps = len(actions)

        # Get recent context
        recent_observations = observations[-2:] if len(observations) >= 2 else observations
        recent_actions = actions[-3:] if len(actions) >= 3 else actions
        recent_reflections = reflections[-2:] if len(reflections) >= 2 else reflections

        # Build context string
        observation_summary = self._summarize_observations(recent_observations)
        action_summary = self._summarize_actions(recent_actions)
        reflection_summary = self._summarize_reflections(recent_reflections)

        # Generate summary using prompt template
        prompt = load_prompt_template(
            "context_agent",
            "summary_generation",
            user_goal=user_goal,
            total_steps=total_steps,
            observation_summary=observation_summary,
            action_summary=action_summary,
            reflection_summary=reflection_summary
        )

        try:
            summary = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
        except Exception as e:
            # Fallback summary
            print(f"caught exception {e}")
            print("Context Agent call_llm failed, use fallback summary")
            summary = f"""Task execution after {total_steps} steps.
Recent observations: {len(recent_observations)} pages viewed.
Recent actions: {len(recent_actions)} actions taken.
Recent reflections: {len(recent_reflections)} performance analyses."""

        return summary, observation_summary, action_summary, reflection_summary

    def _summarize_observations(self, observations: List[Observation]) -> str:
        """Summarize recent observations using LLM."""
        if not observations:
            return "No page observations available."

        # Extract text content from recent observations
        observation_texts = []
        for i, obs in enumerate(observations[-3:], 1):  # Last 3 observations
            text = obs.get("text", "")
            if text:
                # Truncate for brevity but keep meaningful content
                truncated_text = text[:800] if len(text) > 800 else text
                observation_texts.append(f"Page {i}: {truncated_text}")

        if not observation_texts:
            return "No meaningful page content."

        # Use LLM to generate intelligent summary of observations
        try:
            prompt = load_prompt_template(
                "context_agent",
                "observation_summarization",
                observations_text="\n\n".join(observation_texts)
            )

            summary = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return summary
        except Exception as e:
            # Fallback to simple concatenation if LLM fails
            print(f"Observation summarization LLM call failed: {e}")
            return " | ".join(observation_texts)

    def _summarize_actions(self, actions: List[Action]) -> str:
        """Summarize recent actions."""
        if not actions:
            return "No actions taken yet."

        action_summaries = []
        for i, action in enumerate(actions[-5:], 1):  # Last 5 actions
            action_type = action.get("action_type", "unknown")
            element_id = action.get("element_id", "N/A")

            if action_type == "TYPE":
                text = action.get("text", [])[:3]  # First 3 characters
                action_summaries.append(f"{i}. Type[{element_id}]: {''.join(text)}...")
            elif action_type == "CLICK":
                action_summaries.append(f"{i}. Click[{element_id}]")
            elif action_type == "SCROLL":
                direction = action.get("direction", "unknown")
                action_summaries.append(f"{i}. Scroll[{direction}]")
            else:
                action_summaries.append(f"{i}. {action_type}")

        return " | ".join(action_summaries) if action_summaries else "No recent actions."

    def _summarize_reflections(self, reflections: List[Dict[str, Any]]) -> str:
        """Summarize recent reflections using LLM."""
        if not reflections:
            return "No performance reflections available."

        # Extract text content from recent reflections
        reflection_texts = []
        for i, reflection in enumerate(reflections[-3:], 1):  # Last 3 reflections
            text_parts = []

            # Add reflection number and current intention
            text_parts.append(f"Reflection {reflection.get('reflection_number', i)}:")
            text_parts.append(f"Intention: {reflection.get('current_intention', 'N/A')}")

            # Add effectiveness analysis
            effectiveness = reflection.get("effectiveness_analyzer", "")
            if effectiveness:
                text_parts.append(f"Effectiveness: {effectiveness}")

            # Add triple summary
            triple_summary = reflection.get("triple_summary", "")
            if triple_summary:
                text_parts.append(f"State Transition: {triple_summary}")

            # Add pattern detection summary
            pattern_detector = reflection.get("pattern_detector", {})
            if pattern_detector.get("patterns_detected", False):
                detection_summary = pattern_detector.get("detection_summary", "")
                if detection_summary:
                    text_parts.append(f"Patterns: {detection_summary}")

            # Add latest action summary
            latest_action = reflection.get("latest_action", {})
            if latest_action:
                action_type = latest_action.get("action_type", "UNKNOWN")
                element_id = latest_action.get("element_id", "N/A")
                text_parts.append(f"Action: {action_type} on {element_id}")

            reflection_texts.append("\n".join(text_parts))

        if not reflection_texts:
            return "No meaningful reflection content available."

        # Use LLM to generate intelligent summary of reflections
        try:
            prompt = load_prompt_template(
                "context_agent",
                "reflection_summarization",
                reflections_text="\n\n".join(reflection_texts)
            )

            summary = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return summary
        except Exception as e:
            # Fallback to simple concatenation if LLM fails
            print(f"Reflection summarization LLM call failed: {e}")
            return " | ".join(reflection_texts[:2])  # Show first 2 reflections