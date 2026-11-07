"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard state
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Posterior Sampling of Regret (PSR).
    
    Strategy:
    Instead of computing a deterministic score like Expected Improvement (EI) or 
    Upper Confidence Bound (UCB), we sample a single realization of the posterior 
    distribution for the 2000 candidate points. We then select the point that 
    yields the maximum "regret" (improvement) in this sample.
    
    This is equivalent to Thompson Sampling for optimization. It has the property 
    of being Bayes-optimal in the limit of infinite evaluations. In practice, 
    for small budgets like 30 evaluations, it often explores more aggressively 
    than EI, which can get stuck in local optima or over-exploit regions with 
    low uncertainty but high mean.
    
    Formula:
    Let y ~ N(mu, sigma^2) be a sample from the posterior at each candidate.
    Score(x) = max(f_best - y, 0)
    
    We return these scores. The evaluator will pick the candidate with the 
    highest score, which corresponds to the candidate with the lowest sampled 
    value y (since f_best is constant).
    
    Key differences from EI:
    1. Stochasticity: Different calls to acquisition (if the GP were updated 
       between calls, which it is in the loop) would yield different rankings 
       even for the same mu/sigma if we used a different random seed. However, 
       here we are given fixed mu/sigma. The stochasticity is in the sample 
       itself. Wait, the contract says "draws 2000 fresh uniform candidates, 
       computes the posterior there, calls acquisition". The mu and sigma are 
       fixed for this call. 
       
       Actually, PSR in this context means we generate a random vector 
       eps ~ N(0, I) and set sample = mu + sigma * eps.
       Then score = max(f_best - sample, 0).
       
       Why is this better than EI?
       EI is the expectation of this score over the noise eps. 
       By sampling, we allow for "lucky" draws where a point with slightly 
       higher mean but higher variance might be selected if the random draw 
       is favorable. This injects a form of exploration that is not strictly 
       bounded by the variance term in the same way EI is.
       
       However, a single sample can be noisy. To make it more robust, we could 
       average over a few samples, but the directive asks for a structural 
       change to PSR. Let's stick to the pure PSR definition: sample once, 
       rank by regret.
       
       Note: Since we must return 2000 scores, and the evaluator picks the 
       max, we return the sampled regret.
       
       We add a small epsilon to sigma to avoid zero variance issues, though 
       the GP should provide non-zero sigma.
    """
    global _start_time
    
    # Initialize deadline if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Check deadline: if we've used > 300s, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: just return -mu to exploit the best current guess
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate a single random sample from the posterior for all candidates
    # eps ~ N(0, 1)
    eps = np.random.randn(mu.shape[0])
    
    # Sampled function values: y = mu + sigma * eps
    y_sample = mu + sigma_safe * eps
    
    # Calculate regret: how much better is this sample than f_best?
    # We want to maximize (f_best - y_sample)
    # If y_sample > f_best, regret is 0 (no improvement)
    # If y_sample < f_best, regret is positive
    
    regret = np.maximum(f_best - y_sample, 0.0)
    
    # Ensure finiteness
    regret = np.where(np.isfinite(regret), regret, 0.0)
    
    return regret
# EVOLVE-BLOCK-END
