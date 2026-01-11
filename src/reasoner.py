"""
Reasoning Engine Module - Uses Groq LLM to analyze narrative consistency.
"""

import os
import json
import re
from typing import List, Tuple, Dict, Any
from groq import Groq


class NarrativeReasoner:
    """
    Uses Groq-hosted LLMs (LLaMA 3.1) to determine narrative consistency
    with strict structured JSON output.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "llama-3.1-8b-instant"
        # alternative (faster, cheaper):
        # model_name="llama-3.1-8b-instant"
    ):
        """
        Initialize the reasoning engine.

        Args:
            api_key: Groq API key (if None, loads from environment)
            model_name: Groq model identifier
        """
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze_consistency(
        self,
        character_name: str,
        backstory_claim: str,
        context_chunks: List[Tuple[str, float, int]]
    ) -> Dict[str, Any]:
        """
        Analyze if a backstory claim is consistent with novel context.

        Args:
            character_name: Name of the character
            backstory_claim: The claim to verify
            context_chunks: Retrieved narrative segments
                            (chunk_text, score, position)

        Returns:
            Dict with keys:
              - analysis
              - verdict (CONSISTENT / CONTRADICT / UNCLEAR)
              - confidence (0.0–1.0)
              - raw_response
        """

        context_text = self._format_context(context_chunks)
        prompt = self._build_consistency_prompt(
            character_name,
            backstory_claim,
            context_text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise narrative consistency analyst. "
                            "Follow instructions exactly. Output JSON only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2048
            )

            text = response.choices[0].message.content
            result = self._parse_structured_response(text)
            result["raw_response"] = text
            return result

        except Exception as e:
            print(f"[Groq Error] {e}")
            return {
                "analysis": f"Error during reasoning: {str(e)}",
                "verdict": "UNCLEAR",
                "confidence": 0.0,
                "raw_response": ""
            }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _format_context(
        self,
        context_chunks: List[Tuple[str, float, int]]
    ) -> str:
        """Format retrieved chunks into a readable context block."""
        formatted = []
        for i, (chunk_text, score, position) in enumerate(context_chunks, 1):
            formatted.append(
                f"--- Context Segment {i} (Relevance: {score:.3f}) ---\n{chunk_text}"
            )
        return "\n\n".join(formatted)

    def _build_consistency_prompt(
        self,
        character_name: str,
        backstory_claim: str,
        context_text: str
    ) -> str:
        """Builds a prompt enforcing strict JSON output."""
        return f"""
You are a narrative consistency expert analyzing a novel.

CHARACTER:
{character_name}

BACKSTORY CLAIM:
{backstory_claim}

RETRIEVED NOVEL CONTEXT:
{context_text}

INSTRUCTIONS:
1. Determine whether the claim is supported, contradicted, or unclear.
2. Cite specific evidence from the provided context.
3. Be conservative if evidence is weak or indirect.

You MUST respond with ONLY a valid JSON object in the following format:

{{
  "analysis": "Detailed explanation citing evidence from the context",
  "verdict": "CONSISTENT or CONTRADICT or UNCLEAR",
  "confidence": <float between 0.0 and 1.0>
}}

Do NOT include any additional text before or after the JSON.
"""

    def _parse_structured_response(
        self,
        response_text: str
    ) -> Dict[str, Any]:
        """
        Extract and validate JSON from LLM output.
        Robust to minor formatting issues.
        """
        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if match:
            try:
                result = json.loads(match.group(0))

                # Defaults
                result.setdefault("analysis", "")
                result.setdefault("verdict", "UNCLEAR")
                result.setdefault("confidence", 0.5)

                # Normalize verdict
                result["verdict"] = str(result["verdict"]).upper()
                if result["verdict"] not in {"CONSISTENT", "CONTRADICT", "UNCLEAR"}:
                    result["verdict"] = "UNCLEAR"

                # Clamp confidence
                result["confidence"] = float(result["confidence"])
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))

                return result

            except Exception as e:
                print("[JSON Parse Error]", e)

        # Fallback if parsing fails
        return {
            "analysis": response_text,
            "verdict": "UNCLEAR",
            "confidence": 0.5
        }

    # ------------------------------------------------------------------
    # OPTIONAL: BATCH MODE
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        queries: List[Tuple[str, str, List[Tuple[str, float, int]]]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple backstory claims in batch.

        Args:
            queries: List of
              (character_name, backstory_claim, context_chunks)

        Returns:
            List of analysis results
        """
        results = []
        for character_name, backstory_claim, context_chunks in queries:
            results.append(
                self.analyze_consistency(
                    character_name,
                    backstory_claim,
                    context_chunks
                )
            )
        return results
