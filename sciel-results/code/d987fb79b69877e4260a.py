"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for the deadline guard
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using UCB with dynamically increasing exploration.
    
    mu: array of shape (N,) mean predictions
    sigma: array of shape (N,) standard deviation predictions
    f_best: float best observed value so far
    
    Returns: array of shape (N,) scores, higher is better.
    """
    # Deadline guard: if we are too close to the 300s limit, return a safe fallback
    # The limit is 300s, we check against 290s to be safe.
    # However, this function is called inside a loop. The loop itself should ideally
    # check time, but we don't control the loop. 
    # We can check if the *current* call is taking too long, but we can't stop the loop.
    # The prompt says "Keep an EXPLICIT deadline guard: track elapsed time and return 
    # the best result found with a safety margin before the limit."
    # Since we return scores, we can't really "return the best result found" in a 
    # way that stops the loop from the inside easily unless we raise an exception 
    # or if the loop checks for it. 
    # But the prompt says "anything else makes the whole result invalid". 
    # So we must return valid scores.
    # If we are near the deadline, we should probably just return fast.
    
    elapsed = time.time() - _get_start_time()
    if elapsed > 290:
        # Return a simple, fast score to allow the loop to finish if it checks, 
        # or just return something valid.
        # Actually, if the loop doesn't check time, we can't stop it.
        # But the task says "Each evaluation must finish within 300 s".
        # So the *entire* process for one seed must be < 300s.
        # If we are at 290s, we have 10s left.
        # We will just proceed with a very simple calculation to be fast.
        return -mu  # Pure exploitation, very fast
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate the number of evaluations (n)
    # The variance of the posterior samples can be a proxy for the information gain
    # or simply use the magnitude of sigma relative to the range of mu.
    # A common heuristic for adaptive UCB is beta ~ sqrt(log(t)) or similar.
    # Here, we don't have t directly. 
    # However, we can estimate the "age" of the optimization by looking at 
    # how much we have reduced uncertainty or how spread out the mu is.
    
    # Alternative: Use the variance of sigma itself.
    # Early on, sigma is high and varied. Later, it drops.
    # Let's use a dynamic beta based on the spread of the predictions.
    
    # Calculate the range of mu to normalize the exploration term
    mu_range = np.ptp(mu)
    if mu_range < 1e-9:
        mu_range = 1.0
        
    # Heuristic for beta:
    # We want to explore more when uncertainty (sigma) is high relative to the 
    # improvement potential (f_best - mu).
    
    # The directive suggests: kappa scales with the square root of the number of evaluations.
    # We don't have n. But we can approximate n by looking at the number of points 
    # that have low uncertainty? No, that's hard.
    
    # Let's use a simpler structural change: 
    # The "Knowledge" says: "UCB with adaptive beta balances exploration/exploitation 
    # better than fixed EI... Measured: 1 -> 3.844"
    # Wait, 3.844 is a huge improvement? The baseline is 1.0. 
    # The score is "seed/value", where 1.0 is baseline. Higher is better.
    # So 3.844 is very good.
    
    # Let's implement UCB: score = -mu + beta * sigma
    # We need to choose beta.
    
    # Let's use the variance of the sigma values to adapt beta.
    # If sigma is very small everywhere, we are in exploitation phase -> lower beta.
    # If sigma is large, we are in exploration phase -> higher beta.
    
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Normalize sigma_std to a reasonable range
    # If sigma_std is high, we are uncertain -> explore more.
    # If sigma_std is low, we are certain -> exploit more.
    
    # Let's try a beta that starts high and decreases as we gain confidence.
    # A simple way: beta = base_beta * (1 + sigma_std / sigma_mean)
    # But sigma_mean might be close to 0.
    
    # Better: Use the ratio of the max sigma to the mean sigma.
    # Or simply use a fixed high beta initially and decay it?
    
    # The directive specifically asks for: "kappa scales with the square root of the number of evaluations"
    # Since we can't get n, let's approximate n by the number of points 
    # where sigma is below a threshold? No.
    
    # Let's look at the "inspirations" again.
    # Inspiration 3 mentions a program that scored 0.584 (worse).
    # The current program scores 1.0.
    
    # Let's try a pure UCB with a beta that is scaled by the uncertainty of the 
    # *best* candidate relative to the mean.
    
    # Actually, a very effective structural change for BO is to use 
    # **Thompson Sampling** via a single sample from the posterior?
    # But we only have mu and sigma. We can sample x ~ N(mu, sigma).
    # Score = -x.
    # This is a different design family (sampling vs deterministic acquisition).
    # It naturally balances exploration and exploitation.
    
    # Let's try Thompson Sampling.
    # To make it reproducible or at least stable, we can use a fixed seed 
    # or just rely on the randomness. 
    # The task says "Return 2000 finite scores".
    # Thompson Sampling: sample one value from N(mu, sigma) for each candidate.
    # The candidate with the lowest sampled value is the best.
    # So score = -sampled_value.
    
    # However, pure Thompson Sampling can be noisy.
    # A hybrid: Use the mean of several samples?
    # Or just use the UCB with a dynamic beta.
    
    # Let's stick to the directive's suggestion: UCB with dynamic beta.
    # "kappa scales with the square root of the number of evaluations"
    
    # How to estimate n?
    # The number of evaluations is roughly the number of times the function has been called.
    # We can track this globally if we assume the function is called once per step.
    
    global _eval_count
    try:
        _eval_count
    except NameError:
        _eval_count = 0
        
    _eval_count += 1
    
    # Estimate n. 
    # In the loop: 5 initial + (steps). 
    # Total 30 evaluations.
    # So n is between 5 and 30.
    # We can use _eval_count as a proxy for n.
    
    n = max(_eval_count, 1)
    
    # Beta = kappa * sqrt(n)
    # What is kappa?
    # We want beta to be large enough to explore.
    # Typical UCB beta is 1-5.
    # Let's set kappa such that at n=5, beta is ~2, and at n=30, beta is ~4?
    # sqrt(5) ~ 2.2, sqrt(30) ~ 5.5.
    # If kappa = 1, beta ranges from 2.2 to 5.5.
    # This seems reasonable.
    
    beta = 1.0 * np.sqrt(n)
    
    # UCB for minimization:
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    
    scores = -mu + beta * sigma_safe
    
    return scores

# EVOLVE-BLOCK-END
