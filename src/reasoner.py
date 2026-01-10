"""
Reasoning Engine Module - Uses LLM to analyze narrative consistency.
"""
import os
import json
import re
from typing import List, Tuple, Dict, Any
import google.generativeai as genai


class NarrativeReasoner:
    """Uses Gemini LLM to determine narrative consistency with structured output."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the reasoning engine.
        
        Args:
            api_key: Google AI API key (if None, loads from environment)
            model_name: Gemini model identifier
        """
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configure generation for structured output
        self.generation_config = {
            "temperature": 0.1,  # Low temperature for consistent reasoning
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
    
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
            context_chunks: Retrieved narrative segments (chunk_text, score, position)
            
        Returns:
            Dictionary with 'analysis', 'verdict', 'confidence', and 'raw_response'
        """
        # Prepare context from retrieved chunks
        context_text = self._format_context(context_chunks)
        
        # Build the prompt that forces JSON output
        prompt = self._build_consistency_prompt(
            character_name,
            backstory_claim,
            context_text
        )
        
        # Generate response
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Extract and parse JSON response
            result = self._parse_structured_response(response.text)
            result['raw_response'] = response.text
            
            return result
            
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {
                'analysis': f"Error: {str(e)}",
                'verdict': 'UNCLEAR',
                'confidence': 0.0,
                'raw_response': ''
            }
    
    def _format_context(self, context_chunks: List[Tuple[str, float, int]]) -> str:
        """Format retrieved chunks into context string."""
        formatted_chunks = []
        
        for i, (chunk_text, score, position) in enumerate(context_chunks, 1):
            formatted_chunks.append(
                f"--- Context Segment {i} (Relevance: {score:.3f}) ---\n{chunk_text}"
            )
        
        return "\n\n".join(formatted_chunks)
    
    def _build_consistency_prompt(
        self,
        character_name: str,
        backstory_claim: str,
        context_text: str
    ) -> str:
        """Build the prompt that enforces structured JSON output."""
        prompt = f"""You are a narrative consistency expert analyzing a novel. Your task is to determine if a character's backstory claim is consistent with the novel's narrative.

CHARACTER: {character_name}
BACKSTORY CLAIM: {backstory_claim}

RETRIEVED NOVEL CONTEXT:
{context_text}

INSTRUCTIONS:
1. Carefully analyze the provided context segments for evidence supporting or contradicting the backstory claim.
2. Consider both explicit statements and implicit narrative details.
3. Assess the strength and clarity of evidence.

You MUST respond with ONLY a valid JSON object in this exact format:
{{
    "analysis": "Detailed explanation of your reasoning, citing specific evidence from the context",
    "verdict": "CONSISTENT or CONTRADICT or UNCLEAR",
    "confidence": <float between 0.0 and 1.0>
}}

VERDICT DEFINITIONS:
- CONSISTENT: The claim is clearly supported by the narrative evidence
- CONTRADICT: The claim directly conflicts with narrative evidence
- UNCLEAR: Insufficient evidence or ambiguous information

Respond with ONLY the JSON object, no additional text before or after."""

        return prompt
    
    def _parse_structured_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate the JSON response from LLM."""
        # Try to extract JSON from response
        # Sometimes LLM may include markdown code blocks
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                
                # Validate required fields
                if 'verdict' not in result:
                    result['verdict'] = 'UNCLEAR'
                if 'analysis' not in result:
                    result['analysis'] = 'No analysis provided'
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                
                # Normalize verdict
                result['verdict'] = result['verdict'].upper()
                if result['verdict'] not in ['CONSISTENT', 'CONTRADICT', 'UNCLEAR']:
                    result['verdict'] = 'UNCLEAR'
                
                # Ensure confidence is float
                result['confidence'] = float(result['confidence'])
                result['confidence'] = max(0.0, min(1.0, result['confidence']))
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response text: {response_text}")
        
        # Fallback if parsing fails
        return {
            'analysis': response_text,
            'verdict': 'UNCLEAR',
            'confidence': 0.5
        }
    
    def batch_analyze(
        self,
        queries: List[Tuple[str, str, List[Tuple[str, float, int]]]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple backstory claims in batch.
        
        Args:
            queries: List of (character_name, backstory_claim, context_chunks) tuples
            
        Returns:
            List of analysis results
        """
        results = []
        
        for character_name, backstory_claim, context_chunks in queries:
            result = self.analyze_consistency(
                character_name,
                backstory_claim,
                context_chunks
            )
            results.append(result)
        
        return results
