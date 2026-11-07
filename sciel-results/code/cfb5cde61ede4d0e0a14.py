"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track call count for adaptive beta
_call_count = 0


def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we're running long, just return a simple UCB
    # But since we can't know wall-clock easily inside this function without
    # side effects, and the task says keep it simple, we'll rely on the
    # fact that this is a vectorized numpy op on 2000 points which is fast.
    
    # Adaptive beta: start high for exploration, decay to exploitation
    # Branin has 3 minima. Early exploration is key.
    # Total evaluations = 30, first 5 are random. So 25 acquisition calls.
    # Let's decay beta from 4.0 to 1.0 over the 25 calls.
    
    # We need to know how many calls have happened.
    # Since the function is stateless in the contract but we can use a global,
    # we track it.
    
    # However, the task says "mu and sigma have shape (2000,)".
    # The evaluator draws 2000 fresh uniform candidates each step.
    
    # Let's use a schedule based on _call_count.
    # Calls 1-5: beta ~ 4.0 (heavy exploration)
    # Calls 6-15: beta ~ 2.5 (balanced)
    # Calls 16-25: beta ~ 1.5 (exploitation)
    
    # A simple linear decay might work too.
    # beta = 4.0 - (4.0 - 1.0) * (min(_call_count, 25) / 25.0)
    
    # Let's try a piecewise or smooth decay.
    # The inspiration says: "UCB with adaptive beta decaying from 4 to 1.5 over 20 acquisition calls"
    # Wait, the total evaluations are 30, 5 initial random, so 25 acquisition-driven evaluations.
    # The inspiration said "over 20 acquisition calls". Maybe they assumed 20? Or maybe it's just a parameter.
    # Let's stick to the finding: decay from 4 to 1.5.
    
    # If _call_count is 1, beta=4. If _call_count is 21 (or 25), beta=1.5.
    # Let's normalize by 25.
    
    progress = min(_call_count, 25) / 25.0
    beta = 4.0 - (4.0 - 1.5) * progress
    
    # UCB: mu + beta * sigma
    # We want to MINIMIZE, so we pick the point with the LOWEST predicted value?
    # No, the acquisition function returns a SCORE. "higher meaning evaluate next".
    # For minimization, we want to find points where the function is likely low.
    # UCB for minimization: -mu + beta * sigma ? Or mu - beta * sigma?
    # Usually UCB is for maximization: mu + beta * sigma.
    # For minimization, we can just negate the objective.
    # Let f' = -f. Then we maximize f'.
    # mu_f' = -mu_f. sigma_f' = sigma_f.
    # UCB_f' = -mu_f + beta * sigma_f.
    # So the score should be -mu + beta * sigma.
    
    # Let's verify with EI.
    # EI for minimization: E[max(0, f_best - f(x))]
    # The current code returns: (f_best - mu) * cdf(z) + sigma * pdf(z)
    # This is positive when mu < f_best. Higher is better.
    # This is correct for minimization.
    
    # So for UCB, we want to select points with low mu and high sigma (uncertainty).
    # Score = -mu + beta * sigma.
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
