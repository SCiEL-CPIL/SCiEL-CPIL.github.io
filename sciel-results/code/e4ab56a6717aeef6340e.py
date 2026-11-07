"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to track the number of evaluations to adaptively adjust beta.
# Since the function is called repeatedly in a loop, we can use a global counter
# or infer the step from the properties of mu/sigma if possible, but a global
# counter is safer if we assume the process is stateless between calls in terms of
# the function signature, but we can't persist state easily without global variables
# which might be reset or shared incorrectly. 
# However, the task says "for a loop that minimizes...". The function signature is fixed.
# We can estimate the "age" of the optimization by looking at the spread of sigma or
# simply use a fixed schedule based on the assumption that the total budget is 30.
# The problem states: "until 30 evaluations including the first 5".
# So total steps = 30.
# Steps 1-5: initial uniform.
# Steps 6-30: acquisition function is called.
# Total calls to acquisition = 25.
# We can estimate the current step count by how many times this function has been called.
# Using a global variable is the most robust way to track this across calls within a single process run.

_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Increment call count
    _call_count += 1
    
    # Total budget is 30 evaluations.
    # The first 5 are uniform random, so acquisition is called for steps 6 to 30.
    # That is 25 calls.
    # Let current_step be _call_count, ranging from 1 to 25.
    # We want beta to decay as we progress.
    
    # Early steps (low _call_count): high beta (exploration)
    # Late steps (high _call_count): low beta (exploitation)
    
    # A simple linear decay or exponential decay.
    # Let's try an exponential decay of beta from a high value to a low value.
    # beta = beta_min + (beta_max - beta_min) * exp(-decay_rate * step)
    
    # Alternatively, a simple power law or linear schedule.
    # Let's use a linear schedule from 2.0 to 0.5 over the 25 calls.
    # But wait, the strategy says "dynamically adjusted beta that decays exponentially".
    
    # Let's define the total number of acquisition calls as 25.
    total_calls = 25
    
    # Normalize step to [0, 1]
    # Step 1 -> 0, Step 25 -> 1
    progress = (_call_count - 1) / (total_calls - 1) if total_calls > 1 else 0
    
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0, 1)
    
    # Exponential decay of beta
    # beta = beta_min + (beta_max - beta_min) * exp(-k * progress)
    # Let's pick beta_max = 3.0, beta_min = 0.5, k = 3.0
    beta_max = 3.0
    beta_min = 0.5
    k = 3.0
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-k * progress)
    
    # Compute UCB score
    # Score = mu + beta * sigma
    # Note: Since we are minimizing, we want to maximize the score.
    # Lower mu is better. Higher sigma is better for exploration.
    # So maximizing (mu + beta * sigma) is correct for UCB in minimization?
    # Wait. UCB for minimization is usually: min mu - beta * sigma?
    # No, UCB is "Upper Confidence Bound". We want to find the maximum of the upper bound?
    # If we are minimizing f, we want to pick x that has a high probability of being low.
    # The "Upper Confidence Bound" for the *value* of the function is mu + beta*sigma.
    # If we pick the point with the highest mu + beta*sigma, we are exploring where the function *could* be high?
    # That's wrong.
    
    # Let's re-verify UCB for minimization.
    # We want to minimize f.
    # We estimate f(x) ~ N(mu, sigma^2).
    # We want to find x that minimizes f.
    # A common heuristic is to pick x that maximizes: -mu + beta*sigma? Or minimizes mu - beta*sigma?
    # Let's think about Expected Improvement (EI).
    # EI = E[max(f_best - f(x), 0)].
    # This encourages picking points where f(x) is likely to be less than f_best.
    #
    # UCB (Upper Confidence Bound) is typically used for maximization:
    # Maximize mu + beta * sigma.
    # For minimization, we can transform: Maximize (-mu) + beta * sigma?
    # Or Minimize (mu - beta * sigma)?
    # If we minimize (mu - beta * sigma), we are looking for points with low mean and high uncertainty.
    # Let's check the sign.
    # If mu is low (good), mu - beta*sigma is very low (good).
    # If sigma is high (uncertain), mu - beta*sigma is lower (good).
    # So minimizing (mu - beta * sigma) is equivalent to maximizing (beta * sigma - mu).
    #
    # Let's stick to the standard UCB form for minimization:
    # Score = mu - beta * sigma
    # We pick the candidate with the MINIMUM score?
    # The task says: "Return 2000 finite scores, higher meaning evaluate next".
    # So we need to return a score where HIGHER is better.
    # Therefore, we should return:
    # Score = - (mu - beta * sigma) = beta * sigma - mu
    #
    # Let's double check.
    # If mu is low (close to 0), -mu is high. Good.
    # If sigma is high, beta*sigma is high. Good.
    # So Score = beta * sigma - mu.
    
    scores = beta * sigma - mu
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
