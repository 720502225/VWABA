"""Reflector Agent for execution validation, analysis, and recovery suggestions."""

from typing import Any, Dict, List, Optional

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config, call_llm

from .reflector.effectiveness_analyzer import EffectivenessAnalyzer
from .reflector.pattern_detector import PatternDetector
from .prompts.prompt_loader import load_prompt_template


class ReflectorAgent:
    """Simplified reflector agent for execution analysis.

    Responsible for analyzing action effectiveness and detecting execution patterns
    to provide insights for better decision making.
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.effectiveness_analyzer = EffectivenessAnalyzer(lm_config)
        self.pattern_detector = PatternDetector()

        # Reflection history
        self.reflection_history: List[Dict[str, Any]] = []

    def reflect_execution(
        self,
        trajectory: Trajectory,
        intentions: List[str],
        actions: List[Action],
        current_intention: str,
        latest_action: Action,
        current_observation: Observation,
        context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reflect on the execution of the latest action.

        Args:
            trajectory: Current execution trajectory
            intentions: List of all intentions so far
            actions: List of all actions executed so far
            current_intention: The intention that was being fulfilled
            latest_action: The most recently executed action
            current_observation: The observation after action execution
            context_summary: Current context from Context Agent

        Returns:
            Dictionary containing simplified reflection results
        """
        try:
            # 1. Analyze action effectiveness (natural language only)
            effectiveness_result = self.effectiveness_analyzer.analyze(
                trajectory=trajectory,
                current_intention=current_intention,
                latest_action=latest_action,
                context_summary=context_summary,
            )

            # 2. Detect execution patterns
            pattern_result = self.pattern_detector.detect_patterns(actions, intentions)

            # 3. Generate triple summary
            triple_summary = self._generate_enhanced_triple_summary(
                trajectory, current_intention, latest_action, current_observation
            )

            # Create simplified reflection
            reflection = {
                "effectiveness_analyzer": effectiveness_result,
                "pattern_detector": pattern_result,
                "triple_summary": triple_summary,
                "current_intention": current_intention,
                "latest_action": latest_action,
                "reflection_number": len(self.reflection_history) + 1,
            }

            # Store in reflection history
            self.reflection_history.append(reflection)

            return reflection

        except Exception as e:
            # Create error reflection
            error_reflection = {
                "effectiveness_analyzer": f"Error during effectiveness analysis: {str(e)}",
                "pattern_detector": f"Error during pattern detection: {str(e)}",
                "triple_summary": f"Error during triple summary generation: {str(e)}",
                "current_intention": current_intention,
                "latest_action": latest_action,
                "reflection_number": len(self.reflection_history) + 1,
            }

            self.reflection_history.append(error_reflection)
            return error_reflection

    def _generate_enhanced_triple_summary(
        self,
        trajectory: Trajectory,
        current_intention: str,
        latest_action: Action,
        current_observation: Observation,
    ) -> str:
        """Generate an enhanced (O_{t-1}, I_t, A_t, O_t) triple summary using LLM."""

        # Get previous observation from trajectory
        obs_before = "No previous observation available"
        if len(trajectory) >= 2:
            obs_before = trajectory[-3].get("observation", {}).get("text", "")[:200]
            if len(trajectory[-3].get("observation", {}).get("text", "")) > 200:
                obs_before += "..."

        obs_after = current_observation.get("text", "")[:200]
        if len(current_observation.get("text", "")) > 200:
            obs_after += "..."

        action_type = latest_action.get("action_type", "UNKNOWN")
        element_id = latest_action.get("element_id", "N/A")

        # Correctly convert text IDs back to string
        text_ids = latest_action.get("text", [])
        if isinstance(text_ids, list) and text_ids:
            try:
                # Import the ID to key mapping from browser_env
                from browser_env.actions import _id2key
                action_text = ''.join(_id2key[id_num] if 0 <= id_num < len(_id2key) else '?' for id_num in text_ids)
            except (ImportError, IndexError):
                # Fallback: try to convert IDs to characters directly
                action_text = ''.join(chr(id_num) if 32 <= id_num <= 126 else '?' for id_num in text_ids)
        else:
            action_text = "N/A"

        # Build enhanced triple summary prompt
        prompt = load_prompt_template(
            "reflector_agent",
            "triple_summary",
            **{"t-1": obs_before},
            current_intention=current_intention,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            current_observation=obs_after
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return response
        except Exception as e:
            # Fallback simple triple summary
            success_indicator = "Success" if action_type not in ["NONE", "STOP"] else "Failed"
            return f"Intent: {current_intention[:50]}{'...' if len(current_intention) > 50 else ''} | Action: {action_type} on {element_id} | Result: {success_indicator} (Enhanced summary unavailable: {str(e)})"

    def get_recent_reflections(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent reflections.

        Args:
            count: Number of recent reflections to return

        Returns:
            List of recent reflection records
        """
        return self.reflection_history[-count:] if self.reflection_history else []

    def reset_reflection_history(self) -> None:
        """Reset reflection history for a new task."""
        self.reflection_history.clear()