"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track progress and enforce time limits
_state = {
    'call_count': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive exploration.
    
    mu: array of shape (2000,) - GP mean
    sigma: array of shape (2000,) - GP standard deviation
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores, higher is better
    """
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    # Deadline guard: 330s limit, leave 5s margin
    if time.time() - _state['start_time'] > 325:
        # If we are running out of time, return a safe fallback (exploitation)
        # This shouldn't happen in normal operation as EI is O(N)
        return -mu

    _state['call_count'] += 1
    step = _state['call_count']
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e9, neginf=1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e9, neginf=-1e9)

    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Avoid division by zero
    safe_sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    
    z = (f_best - mu) / safe_sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf(-10) is ~0, norm.cdf(10) is ~1
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + safe_sigma * phi
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Strategy:
    # The task runs for 30 evaluations total.
    # Steps 1-5 are initial random points.
    # Acquisition is called for steps 6-30 (25 calls).
    # 
    # Early phase (calls 1-10): Explore more. We can use a lower confidence bound (LCB) 
    # or simply add a bonus to uncertain regions.
    # However, EI naturally balances this. To improve on the seed's potential weakness,
    # we can modulate the "effective" f_best or add an exploration bonus.
    # 
    # A common robust approach is to use a mixture or simply rely on EI which is 
    # theoretically optimal for minimizing expected regret in BO.
    # 
    # Let's try a slight modification: In the very early stages (after init),
    # the GP is very uncertain. EI might be dominated by the sigma term.
    # In later stages, we want to squeeze the minimum.
    #
    # Let's implement a simple adaptive factor that scales the exploration part.
    # Actually, pure EI is often strong. The seed program (Thompson Sampling) scored 0.12.
    # Inspiration 1 (EI) scored 1.53. This suggests EI is significantly better for this task.
    # 
    # To further improve, we can use "Expected Improvement with a penalty" or 
    # just stick to standard EI but ensure it's computed correctly.
    # 
    # Let's add a small exploration bonus in the first few steps to ensure we don't 
    # get stuck in a local minimum of the surrogate model early on.
    # Exploration bonus: beta * sigma
    # beta starts high and decays.
    
    # Total 25 acquisition calls.
    # Decay beta from 2.0 to 0.5 over the 25 steps.
    # beta = 2.0 - (step / 25.0) * 1.5
    beta = 2.0 - (min(step, 25) / 25.0) * 1.5
    
    # Combine EI and Exploration Bonus
    # Score = EI + beta * sigma
    # Note: EI is in objective units, sigma is in objective units.
    # This effectively creates a Lower Confidence Bound (LCB) style score if we consider
    # that we want to minimize, so we maximize the "potential improvement".
    # LCB = mu - beta * sigma. Maximizing LCB is equivalent to minimizing mu - beta*sigma.
    # EI is a more sophisticated version of this.
    # Adding beta * sigma to EI reinforces exploration.
    
    scores = ei + beta * sigma
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
