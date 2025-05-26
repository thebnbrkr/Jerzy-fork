from .common import *


import asyncio
from typing import List, Dict, Any, Optional, Union

class JerzyUQLMLike:
    """
    Adapts a Jerzy LLM (with sync `.generate(prompt)`) to an async interface for UQLM.
    """
    def __init__(self, jerzy_llm):
        self.jerzy_llm = jerzy_llm

    async def generate(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Asynchronously generate a response using the underlying Jerzy LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments (passed to underlying LLM if supported)
        
        Returns:
            The generated text or structured response
        """
        try:
            # For simple implementation, just call the LLM's generate method
            return await asyncio.to_thread(self.jerzy_llm.generate, prompt)
        except Exception as e:
            print(f"Error in JerzyUQLMLike.generate: {str(e)}")
            raise

class UQLMScorer:
    """
    Utility class to score LLM responses with UQLM.
    """
    def __init__(self, jerzy_llm, scorer_type="black_box", scorers=None):
        """
        Initialize the UQLM scorer.
        
        Args:
            jerzy_llm: The Jerzy LLM to score
            scorer_type: Either "black_box" or "white_box"
            scorers: List of scorers to use (default: ["semantic_negentropy"])
        """
        try:
            from uqlm import BlackBoxUQ, WhiteBoxUQ
        except ImportError:
            raise ImportError("UQLM package not installed. Please install it with 'pip install uqlm'")
            
        self.llm_adapter = JerzyUQLMLike(jerzy_llm)
        self.scorer_type = scorer_type
        self.scorers = scorers or ["semantic_negentropy"]
        
        # Initialize the appropriate scorer
        if scorer_type == "black_box":
            self.scorer = BlackBoxUQ(llm=self.llm_adapter, scorers=self.scorers, use_best=True)
        elif scorer_type == "white_box":
            self.scorer = WhiteBoxUQ(llm=self.llm_adapter, scorers=self.scorers, use_best=True)
        else:
            raise ValueError(f"Unknown scorer type: {scorer_type}. Use 'black_box' or 'white_box'")
    
    def score_prompt(self, prompt: str, num_responses: int = 5) -> Dict[str, Any]:
        """
        Score a prompt using UQLM.
        
        Args:
            prompt: The prompt to score
            num_responses: Number of responses to generate for scoring
            
        Returns:
            Dict containing scores and responses
        """
        results = asyncio.run(self.scorer.generate_and_score(
            prompts=[prompt], 
            num_responses=num_responses
        ))
        
        # Convert to a more usable format
        return {
            'confidence': results.scores[0],
            'responses': results.responses[0],
            'details': results.to_dict(),
            'dataframe': results.to_df()
        }
    
    def score_multiple_prompts(self, prompts: List[str], num_responses: int = 5) -> Dict[str, Any]:
        """
        Score multiple prompts using UQLM.
        
        Args:
            prompts: List of prompts to score
            num_responses: Number of responses to generate for scoring
            
        Returns:
            Dict containing scores and responses for each prompt
        """
        results = asyncio.run(self.scorer.generate_and_score(
            prompts=prompts, 
            num_responses=num_responses
        ))
        
        return {
            'confidence': results.scores,
            'responses': results.responses,
            'details': results.to_dict(),
            'dataframe': results.to_df()
        }
