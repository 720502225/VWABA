"""Modular prompt loader for multi-agent system.

This module provides a centralized way to load and manage prompt templates
for all agents in the multi-agent coordination system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from llms import lm_config, call_llm


class PromptLoader:
    """Centralized prompt loading and management system."""

    def __init__(self, prompts_file: Optional[str] = None):
        """Initialize the prompt loader.

        Args:
            prompts_file: Path to the prompts JSON configuration file
        """
        self.prompts_file = prompts_file or self._get_default_prompts_file()
        self.prompts_data = self._load_prompts()

    def _get_default_prompts_file(self) -> str:
        """Get the default prompts file path."""
        return str(Path(__file__).parent / "multi_agent_prompts_fixed.json")

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from the JSON configuration file."""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Prompts file not found: {self.prompts_file}")
            print("⚠️  Using fallback prompts")
            return self._get_fallback_prompts()
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in prompts file: {e}")
            print("⚠️  Using fallback prompts")
            return self._get_fallback_prompts()

    def _get_fallback_prompts(self) -> Dict[str, Any]:
        """Get minimal fallback prompts when file loading fails."""
        return {
            "context_agent": {
                "summary_generation": {
                    "name": "Context Summary Generation",
                    "task": "Generate a brief context summary for web automation task.",
                    "template": "Task progress: Step {current_step}/{total_steps}, {completion_pct:.0%} complete."
                }
            },
            "planner_agent": {
                "intention_generation": {
                    "name": "Intention Generation",
                    "task": "Generate next action for web automation",
                    "template": "Generate a specific web action to complete: {user_goal}"
                }
            },
            "actor_agent": {
                "action_execution": {
                    "name": "Action Execution",
                    "task": "Execute browser actions",
                    "template": "Execute browser action to: {intention}"
                }
            },
            "reflector_agent": {
                "action_validation": {
                    "name": "Action Validation",
                    "task": "Validate web actions",
                    "template": "Did this action complete the intended task?"
                }
            }
        }

    def get_prompt_template(self, agent_type: str, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt template.

        Args:
            agent_type: Type of agent (context_agent, planner_agent, etc.)
            prompt_name: Name of the prompt template

        Returns:
            Prompt template dictionary or None if not found
        """
        try:
            return self.prompts_data["agents"][agent_type][prompt_name]
        except KeyError:
            print(f"⚠️  Prompt not found: {agent_type}.{prompt_name}")
            return None

    def format_prompt(self, agent_type: str, prompt_name: str, **kwargs) -> str:
        """Format a prompt template with provided variables.

        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt template
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string
        """
        template_data = self.get_prompt_template(agent_type, prompt_name)
        if not template_data:
            return f"Prompt template not found: {agent_type}.{prompt_name}"

        template = template_data.get("template", "")
        if not template:
            return f"No template found for: {agent_type}.{prompt_name}"

        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"⚠️  Missing variable in prompt template: {e}")
            return f"Template error: missing variable {e} in {agent_type}.{prompt_name}"

    def generate_llm_prompt(
        self,
        agent_type: str,
        prompt_name: str,
        lm_config: lm_config.LMConfig,
        context_vars: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate an LLM prompt using the template system.

        Args:
            agent_type: Type of agent
            prompt_name: Name of the prompt template
            lm_config: Language model configuration
            context_vars: Variables to substitute in template

        Returns:
            Generated prompt string
        """
        template_data = self.get_prompt_template(agent_type, prompt_name)
        if not template_data:
            return f"Prompt template not found: {agent_type}.{prompt_name}"

        template = template_data.get("template", "")
        if not template:
            return f"No template found for: {agent_type}.{prompt_name}"

        # Add context variables to template if provided
        if context_vars:
            try:
                template = template.format(**context_vars)
            except KeyError as e:
                print(f"⚠️  Missing variable in prompt template: {e}")
                template = f"Template error: missing variable {e}"

        # Get LLM parameters from template
        temperature = template_data.get("temperature", 0.7)
        max_tokens = template_data.get("max_tokens", 500)

        # Generate the actual prompt using LLM
        full_prompt = f"""Task: {template_data.get('task', '')}

Context:
{template}

Requirements:
- Be specific and actionable
- Focus on completing the stated intention
- Use appropriate web automation actions
- Handle any errors or issues appropriately"""

        if lm_config.mode == "chat":
            messages = [
                {"role": "system", "content": f"You are a web automation assistant. {template_data.get('task', '')}"},
                {"role": "user", "content": full_prompt}
            ]
            return messages
        else:
            return full_prompt

    def get_all_prompts_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get all prompt templates for a specific agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Dictionary of all prompt templates for the agent
        """
        try:
            return self.prompts_data["agents"][agent_type]
        except KeyError:
            print(f"⚠️  Agent type not found in prompts: {agent_type}")
            return {}

    def reload_prompts(self) -> None:
        """Reload the prompts file."""
        self.prompts_data = self._load_prompts()
        print(f"🔄 Reloaded prompts from: {self.prompts_file}")

    def list_available_prompts(self) -> None:
        """List all available prompt templates."""
        print("📋 Available Prompt Templates:")
        print("=" * 50)

        agents = self.prompts_data.get("agents", {})
        for agent_type, agent_prompts in agents.items():
            print(f"\n🤖 {agent_type}:")
            for prompt_name, prompt_data in agent_prompts.items():
                task = prompt_data.get("task", "No task description")
                print(f"  📝 {prompt_name}: {task}")

    def validate_prompt_template(self, template_str: str, variables: List[str]) -> Dict[str, Any]:
        """Validate that a template contains all required variables.

        Args:
            template_str: Template string to validate
            variables: List of required variables

        Returns:
            Validation result with missing variables
        """
        missing_vars = []
        for var in variables:
            if f"{{{var}}}" not in template_str:
                missing_vars.append(var)

        return {
            "valid": len(missing_vars) == 0,
            "missing_variables": missing_vars,
            "template": template_str
        }

    def create_custom_prompt(
        self,
        agent_type: str,
        prompt_name: str,
        template: str,
        task_description: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Create a custom prompt template and add it to the system.

        Args:
            agent_type: Type of agent
            prompt_name: Name for the new prompt
            template: Template string with variable placeholders
            task_description: Description of what the prompt does
            temperature: LLM temperature for this prompt
            max_tokens: Maximum tokens for this prompt

        Returns:
            Created prompt template
        """
        custom_prompt = {
            "name": prompt_name,
            "task": task_description,
            "template": template,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add to prompts data
        if "agents" not in self.prompts_data:
            self.prompts_data["agents"] = {}
        if agent_type not in self.prompts_data["agents"]:
            self.prompts_data["agents"][agent_type] = {}

        self.prompts_data["agents"][agent_type][prompt_name] = custom_prompt

        print(f"✅ Created custom prompt: {agent_type}.{prompt_name}")
        return custom_prompt


# Global prompt loader instance
_global_prompt_loader = None


def get_prompt_loader(prompts_file: Optional[str] = None) -> PromptLoader:
    """Get the global prompt loader instance."""
    global _global_prompt_loader
    if _global_prompt_loader is None:
        _global_prompt_loader = PromptLoader(prompts_file)
    return _global_prompt_loader


def load_prompt_template(agent_type: str, prompt_name: str, **kwargs) -> str:
    """Quick function to load and format a prompt template.

    Args:
        agent_type: Type of agent
        prompt_name: Name of the prompt template
        **kwargs: Variables to substitute in template

    Returns:
        Formatted prompt string
    """
    loader = get_prompt_loader()
    return loader.format_prompt(agent_type, prompt_name, **kwargs)


def generate_llm_prompt_from_template(
    agent_type: str,
    prompt_name: str,
    lm_config: lm_config.LMConfig,
    context_vars: Optional[Dict[str, Any]] = None
) -> str:
    """Quick function to generate LLM prompt from template.

    Args:
        agent_type: Type of agent
        prompt_name: Name of the prompt template
        lm_config: Language model configuration
        context_vars: Variables to substitute in template

    Returns:
        Generated prompt for LLM
    """
    loader = get_prompt_loader()
    return loader.generate_llm_prompt(agent_type, prompt_name, lm_config, context_vars)