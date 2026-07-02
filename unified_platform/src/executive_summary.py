"""PM-facing executive summary export.

Takes Module 5's decision output and reduces it to the five columns a
product manager actually needs — no p-values, no DCS-component breakdowns,
no "Beta-Binomial"/"HTE"/"Bonferroni"-style jargon. One row per experiment.
"""
import pandas as pd


class ExecutiveSummary:
    """Generate a PM-facing summary export from Decision Engine outputs."""

    def generate(self, decisions_dict, decisions_df=None):
        """
        Args:
            decisions_dict: dict from DecisionEngine.run_all() (first return value)
            decisions_df: unused (kept for signature compatibility with the
                audit DataFrame callers may already have on hand)

        Returns: DataFrame with columns:
          Experiment, Decision, Confidence, Primary Reason, Next Action
        """
        summary_rows = []

        for exp_id in sorted(decisions_dict.keys()):
            dec = decisions_dict[exp_id]

            if dec.get("status") == "BLOCKED":
                summary_rows.append({
                    "Experiment": exp_id,
                    "Decision": "BLOCKED",
                    "Confidence": "-",
                    "Primary Reason": "Randomization was broken before the experiment started "
                                       "(control/treatment split was uneven)",
                    "Next Action": "Fix the randomization setup and rerun before analyzing results",
                })
                continue

            decision = dec.get("decision", "UNKNOWN")
            dcs_dict = dec.get("dcs")
            dcs = dcs_dict["dcs"] if dcs_dict is not None else None
            confidence_str = f"{int(round(dcs))}" if dcs is not None else "-"

            summary_rows.append({
                "Experiment": exp_id,
                "Decision": decision,
                "Confidence": confidence_str,
                "Primary Reason": self._extract_reason(dec, decision),
                "Next Action": self._extract_action(decision),
            })

        return pd.DataFrame(summary_rows)

    def _extract_reason(self, dec, decision):
        """Primary reason in plain, PM-facing language — no jargon."""
        reasoning = dec.get("reasoning", "")
        evidence_strength = dec.get("evidence_strength", "")

        if decision == "KILL":
            if "novelty" in reasoning.lower():
                return "Looked good at first, but the benefit faded (or reversed) over time"
            if "guardrail" in reasoning.lower():
                return "Hurt a key safety metric (e.g. players quitting more often)"
            if "negative" in reasoning.lower():
                return "Made the metric worse, not better"
            return reasoning

        if decision == "SEGMENT ROLLOUT":
            # Never surface "HTE" or other acronyms here — describe the
            # finding in plain terms instead.
            return "Worked well for a specific group of players, even though the overall average effect was weak"

        if decision in ["SHIP", "EXTEND"]:
            if evidence_strength == "Strong":
                return "Consistently positive results across every check we ran"
            if evidence_strength == "Moderate":
                if "BH" in reasoning or "multiple testing" in reasoning.lower():
                    return "Positive results, but less certain once we accounted for testing six features at once"
                return "Positive results, with some uncertainty remaining"
            if evidence_strength == "Weak":
                return "A small positive signal, but not enough data yet to be confident"

        if decision == "HUMAN REVIEW":
            if "disagreement" in reasoning.lower():
                return "Two different ways of checking the data disagreed with each other"
            if "weak" in reasoning.lower():
                return "Evidence too thin to decide automatically"
            if "unknown" in reasoning.lower():
                return "Safety metrics were never checked"
            return "Needs a person to make the call, not enough to decide automatically"

        if decision == "INCONCLUSIVE":
            return "Not enough data to tell whether this worked"

        return reasoning

    def _extract_action(self, decision):
        """What should the PM actually do next?"""
        return {
            "SHIP": "Roll out to all players",
            "SEGMENT ROLLOUT": "Roll out to the specific player group it worked for; leave everyone else as-is",
            "EXTEND": "Keep the test running longer to collect more data before deciding",
            "KILL": "Do not ship; redesign the feature before testing again",
            "HUMAN REVIEW": "Bring to the team for a manual decision",
            "INCONCLUSIVE": "Run the test for longer or with more players",
            "BLOCKED": "Fix the randomization setup and rerun before analyzing results",
        }.get(decision, "Discuss with the data team")
