import numpy as np
import time

def acquisition(mu, sigma, f_best):
        """
        Kriging Believer / UCB based acquisition function.
        Score = -mu + beta * sigma
        """
        start_time = time.time()
        
        # Beta controls exploration vs exploitation
        # Higher beta -> more exploration
        beta = 2.0
        
        # Calculate UCB score for minimization
        # We want to minimize f, so we look for low mu and high sigma.
        # Score should be high for "good" candidates.
        # Good candidate: Low mu, High sigma.
        # Score = -mu + beta * sigma
        
        scores = -mu + beta * sigma
        
        # Handle non-finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        # Time guard
        if time.time() - start_time > 250:
            return scores
            
        return scores
