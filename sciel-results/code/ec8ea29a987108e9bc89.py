"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Strategy:
    Instead of Expected Improvement (EI), which can be overly conservative or
    sensitive to the exact shape of the posterior, we use LCB (Lower Confidence Bound).
    LCB is a surrogate objective: LCB(x) = mu(x) - beta * sigma(x).
    
    We want to minimize the true objective, so we maximize the "expected gain" 
    relative to f_best, penalized by cost.
    
    The score is defined as:
    Score(x) = max(0, f_best - (mu(x) - beta * sigma(x))) / cost(x)
    
    This represents the potential improvement (capped at 0 if the lower bound is 
    already better than f_best, though usually we look for where the lower bound 
    suggests a value significantly lower than f_best) divided by the cost.
    
    The beta parameter is adaptive. In standard BO, beta is often set based on 
    log(n) or a fixed value. Here, we estimate the "noise" or uncertainty scale 
    from the sigma vector itself to make the exploration rate robust to the 
    current state of the GP.
    
    A key structural difference from EI:
    EI integrates the probability of improvement. LCB is a point estimate of 
    the "best possible" value given the uncertainty. LCB tends to be more 
    aggressive in exploration when beta is tuned correctly, potentially 
    escaping local optima faster than EI in low-dimensional smooth functions 
    like Hartmann 3D, especially when combined with cost-awareness.
    
    Cost-awareness:
    We divide by cost to prioritize cheap evaluations. Since cost = 1 + mean(x),
    points near the origin are cheaper. This encourages the algorithm to explore 
    the lower-cost regions first, which is efficient if the minimum is not 
    necessarily in the highest cost region, but generally helps in budget-constrained 
    settings.
    
    We also include a safety guard for time and numerical stability.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety check for finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure sigma is non-negative (it's a standard deviation)
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Adaptive Beta Calculation:
    # We want beta to scale with the uncertainty. 
    # A common heuristic is beta = sqrt(2 * log(n + 1)). 
    # However, in BO with a fixed GP, the "effective" uncertainty might vary.
    # Let's use a beta that is proportional to the mean sigma, but capped to avoid 
    # excessive exploration if the GP is very uncertain everywhere.
    # 
    # Alternative: Fixed beta might be too rigid. 
    # Let's try a beta that adapts to the scale of the problem.
    # The Hartmann 3D function range is roughly [-4, 0].
    # If sigma is large, we explore more.
    
    # Heuristic: beta = 2.0 * (1 + mean_sigma / (abs(f_best) + 1e-6))
    # This increases exploration if the model is uncertain relative to the best value found.
    mean_sigma = np.mean(sigma_safe)
    scale = np.abs(f_best) + 1e-6
    
    # We cap beta to prevent the LCB from becoming unboundedly low (exploring everywhere)
    # A beta of ~2-4 is typical for standard BO.
    # Let's set beta dynamically:
    # If mean_sigma is small, beta should be smaller to focus on exploitation.
    # If mean_sigma is large, beta should be larger to explore.
    
    # A simpler, robust approach:
    # Use a beta that ensures the LCB is a reasonable "optimistic" bound.
    # beta = 1.5 * (mean_sigma / (abs(f_best) + 1e-6)) + 1.0
    # This guarantees beta >= 1.0.
    
    beta = 1.5 * (mean_sigma / scale) + 1.0
    
    # Cap beta to avoid numerical issues or excessive exploration
    # In 3D Hartmann, beta > 5 is rarely helpful and can be detrimental.
    beta = np.clip(beta, 1.0, 5.0)
    
    # Calculate LCB: mu - beta * sigma
    # Lower LCB is better (more optimistic about finding a low value)
    lcb = mu - beta * sigma_safe
    
    # Calculate potential improvement: f_best - lcb
    # If lcb < f_best, there is potential improvement.
    # We only care about positive improvements.
    improvement = f_best - lcb
    improvement = np.maximum(improvement, 0.0)
    
    # Score: Improvement per unit cost
    scores = improvement / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # If we somehow took too long, return what we have (which is already computed)
        pass
        
    return scores
# EVOLVE-BLOCK-END
