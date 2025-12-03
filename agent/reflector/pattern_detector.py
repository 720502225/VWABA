"""Pattern detection for Reflector Agent."""

from typing import Any, Dict, List

from browser_env import Action


class PatternDetector:
    """Detects patterns in action execution and intention sequences."""

    def __init__(self) -> None:
        self.detection_history: List[Dict[str, Any]] = []

    def detect_patterns(
        self, actions: List[Action], intentions: List[str]
    ) -> Dict[str, Any]:
        """Detect patterns in the execution history.

        Args:
            actions: List of executed actions
            intentions: List of intentions that were being fulfilled

        Returns:
            Dictionary containing detected patterns
        """
        if len(actions) < 3:
            return {
                "patterns_detected": False,
                "message": "Insufficient data for pattern detection",
                "total_actions": len(actions),
                "total_intentions": len(intentions),
            }

        patterns = {}

        # Detect repetitive action patterns
        patterns["repetitive_action_patterns"] = self._detect_repetitive_actions(actions)

        # Detect repetitive intention patterns
        patterns["repetitive_intention_patterns"] = self._detect_repetitive_intentions(intentions)

        # Detect action sequence patterns
        patterns["action_sequence_patterns"] = self._detect_action_sequences(actions)

        # Detect failure patterns
        patterns["failure_patterns"] = self._detect_failure_patterns(actions)

        # Detect stuck patterns (no progress)
        patterns["stuck_patterns"] = self._detect_stuck_patterns(actions, intentions)

        # Calculate overall pattern metrics
        patterns["pattern_metrics"] = self._calculate_pattern_metrics(patterns)

        # Store detection results
        detection_record = {
            "timestamp": None,  # Would be set in actual implementation
            "total_actions": len(actions),
            "total_intentions": len(intentions),
            "detected_patterns": patterns,
        }
        self.detection_history.append(detection_record)

        return {
            "patterns_detected": True,
            "detected_patterns": patterns,
            "total_actions": len(actions),
            "total_intentions": len(intentions),
            "detection_summary": self._generate_summary(patterns),
        }

    def _detect_repetitive_actions(self, actions: List[Action]) -> List[Dict[str, Any]]:
        """Detect patterns of repetitive actions."""
        repetitive_patterns = []

        # Group actions by type
        action_groups = {}
        for i, action in enumerate(actions):
            action_type = action.get("action_type", "UNKNOWN")
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append((i, action))

        # Look for consecutive repetitions
        for action_type, action_list in action_groups.items():
            if len(action_list) < 3:  # Need at least 3 to detect pattern
                continue

            # Check for consecutive repetitions
            consecutive_count = 1
            max_consecutive = 1
            for i in range(1, len(action_list)):
                if action_list[i][0] == action_list[i-1][0] + 1:  # Consecutive indices
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1

            if max_consecutive >= 3:  # 3 or more consecutive same actions
                repetitive_patterns.append({
                    "action_type": action_type,
                    "max_consecutive": max_consecutive,
                    "total_count": len(action_list),
                    "pattern_type": "consecutive_repetition",
                })

        # Check for cyclic patterns
        if len(actions) >= 6:
            last_6_actions = [a.get("action_type", "UNKNOWN") for a in actions[-6:]]
            if len(set(last_6_actions)) <= 2:  # Only 2 or fewer unique action types
                repetitive_patterns.append({
                    "action_types": list(set(last_6_actions)),
                    "sequence": last_6_actions,
                    "pattern_type": "cyclic_pattern",
                })

        return repetitive_patterns

    def _detect_repetitive_intentions(self, intentions: List[str]) -> List[Dict[str, Any]]:
        """Detect patterns of repetitive intentions."""
        repetitive_patterns = []

        if len(intentions) < 3:
            return repetitive_patterns

        # Check for duplicate intentions
        intention_counts = {}
        for i, intention in enumerate(intentions):
            # Normalize intention text for comparison
            normalized = intention.lower().strip()
            if normalized not in intention_counts:
                intention_counts[normalized] = []
            intention_counts[normalized].append((i, intention))

        # Find repeated intentions
        for normalized, occurrences in intention_counts.items():
            if len(occurrences) > 1:
                repetitive_patterns.append({
                    "intention": occurrences[0][1],  # Original text
                    "count": len(occurrences),
                    "positions": [occ[0] for occ in occurrences],
                    "pattern_type": "repeated_intention",
                })

        # Check for very similar intentions
        for i in range(len(intentions) - 2):
            current = intentions[i].lower()
            next_one = intentions[i + 1].lower()
            next_two = intentions[i + 2].lower()

            # Calculate similarity (simple word overlap)
            similarity_score = self._calculate_similarity(current, next_two)
            if similarity_score > 0.8:  # High similarity
                repetitive_patterns.append({
                    "intentions": [intentions[i], intentions[i + 2]],
                    "similarity_score": similarity_score,
                    "pattern_type": "similar_intentions",
                })

        return repetitive_patterns

    def _detect_action_sequences(self, actions: List[Action]) -> List[Dict[str, Any]]:
        """Detect common action sequences."""
        sequence_patterns = []

        if len(actions) < 4:
            return sequence_patterns

        # Look for 3-action sequences
        action_types = [a.get("action_type", "UNKNOWN") for a in actions]

        # Count 3-action sequences
        sequence_counts = {}
        for i in range(len(action_types) - 2):
            sequence = tuple(action_types[i:i+3])
            if sequence not in sequence_counts:
                sequence_counts[sequence] = 0
            sequence_counts[sequence] += 1

        # Find repeated sequences
        for sequence, count in sequence_counts.items():
            if count >= 2:  # Sequence appeared at least twice
                sequence_patterns.append({
                    "sequence": list(sequence),
                    "count": count,
                    "pattern_type": "repeated_sequence",
                })

        return sequence_patterns

    def _detect_failure_patterns(self, actions: List[Action]) -> List[Dict[str, Any]]:
        """Detect patterns related to action failures."""
        failure_patterns = []

        # Look for NONE actions (indicating failures)
        none_actions = [(i, a) for i, a in enumerate(actions) if a.get("action_type") == "NONE"]

        if len(none_actions) > 0:
            failure_patterns.append({
                "pattern_type": "none_actions",
                "count": len(none_actions),
                "positions": [pos for pos, _ in none_actions],
                "failure_rate": len(none_actions) / len(actions),
            })

        # Look for repeated failed action types
        if len(none_actions) >= 2:
            # Check if NONE actions are clustered
            for i in range(len(none_actions) - 1):
                current_pos = none_actions[i][0]
                next_pos = none_actions[i + 1][0]

                if next_pos - current_pos <= 3:  # Clustered failures
                    failure_patterns.append({
                        "pattern_type": "clustered_failures",
                        "start_position": current_pos,
                        "end_position": next_pos,
                        "span": next_pos - current_pos,
                    })

        return failure_patterns

    def _detect_stuck_patterns(self, actions: List[Action], intentions: List[str]) -> List[Dict[str, Any]]:
        """Detect patterns indicating agent is stuck."""
        stuck_patterns = []

        # Check for high repetition of simple actions
        simple_actions = ["CLICK", "SCROLL", "KEY_PRESS"]
        simple_action_count = sum(1 for a in actions if a.get("action_type") in simple_actions)

        if simple_action_count / len(actions) > 0.8:  # 80% or more are simple actions
            stuck_patterns.append({
                "pattern_type": "excessive_simple_actions",
                "simple_action_ratio": simple_action_count / len(actions),
                "threshold": 0.8,
            })

        # Check for lack of diverse action types
        unique_action_types = len(set(a.get("action_type", "UNKNOWN") for a in actions))
        if unique_action_types <= 2 and len(actions) >= 5:
            stuck_patterns.append({
                "pattern_type": "low_action_diversity",
                "unique_types": unique_action_types,
                "total_actions": len(actions),
            })

        # Check for repeating similar intentions
        if len(intentions) >= 4:
            recent_intentions = intentions[-4:]
            normalized = [i.lower().strip() for i in recent_intentions]
            unique_normalized = set(normalized)

            if len(unique_normalized) <= 2:  # Only 2 or fewer unique intentions
                stuck_patterns.append({
                    "pattern_type": "repetitive_intentions",
                    "unique_count": len(unique_normalized),
                    "total_count": len(recent_intentions),
                })

        return stuck_patterns

    def _calculate_pattern_metrics(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics about detected patterns."""
        metrics = {
            "total_pattern_types": 0,
            "high_severity_patterns": 0,
            "pattern_density": 0.0,
        }

        # Count different pattern types
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:  # Non-empty pattern list
                metrics["total_pattern_types"] += 1

        # Identify high severity patterns
        high_severity_indicators = [
            "clustered_failures", "excessive_simple_actions", "low_action_diversity"
        ]

        for pattern_list in patterns.values():
            for pattern in pattern_list:
                if pattern.get("pattern_type") in high_severity_indicators:
                    metrics["high_severity_patterns"] += 1

        # Calculate pattern density (patterns per action)
        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
        total_actions = patterns.get("pattern_metrics", {}).get("total_actions", 1)
        metrics["pattern_density"] = total_patterns / total_actions if total_actions > 0 else 0.0

        return metrics

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _generate_summary(self, patterns: Dict[str, Any]) -> str:
        """Generate a human-readable summary of detected patterns."""
        summary_parts = []

        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                if pattern_type == "repetitive_action_patterns":
                    summary_parts.append(f"Found {len(pattern_list)} repetitive action patterns")
                elif pattern_type == "failure_patterns":
                    summary_parts.append(f"Found {len(pattern_list)} failure-related patterns")
                elif pattern_type == "stuck_patterns":
                    summary_parts.append(f"Found {len(pattern_list)} patterns suggesting agent is stuck")

        return "; ".join(summary_parts) if summary_parts else "No significant patterns detected"

    def get_pattern_history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent pattern detections.

        Args:
            count: Number of recent detections to return

        Returns:
            List of recent pattern detection records
        """
        return self.detection_history[-count:] if self.detection_history else []