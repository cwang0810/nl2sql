"""Evidence integration: incorporate BIRD's external knowledge into prompts."""

from __future__ import annotations


class EvidenceIntegrator:
    """Integrate BIRD evidence (external knowledge) into generation prompts."""

    def format_evidence(self, evidence: str) -> str:
        """Format evidence string for prompt inclusion.

        Args:
            evidence: Raw evidence string from BIRD dataset.

        Returns:
            Formatted evidence string, or empty notice if no evidence.
        """
        if not evidence or not evidence.strip():
            return "No additional knowledge provided."
        return (
            f"⚠️ CRITICAL EXTERNAL KNOWLEDGE — Your SQL MUST follow these definitions EXACTLY:\n"
            f"{evidence.strip()}\n"
            f"\n"
            f"Rules for using evidence:\n"
            f"- If the evidence provides a formula (e.g., X = A / B), your SQL MUST compute it using EXACTLY that formula.\n"
            f"- If the evidence says a column or value mapping (e.g., 'A11 refers to salary'), use that EXACT column.\n"
            f"- If the evidence specifies a subtraction order (e.g., 'difference = X - Y'), use that EXACT order.\n"
            f"- If the evidence says a date format (e.g., '201309 refers to Sep 2013'), use that EXACT format.\n"
            f"- DO NOT substitute your own interpretation when the evidence provides a specific method."
        )
