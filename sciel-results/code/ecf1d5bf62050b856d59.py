"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an adaptive Upper Confidence Bound (UCB).
    
    Args:
        mu: np.ndarray of shape (2000,) predicted mean
        sigma: np.ndarray of shape (2000,) predicted std dev
        f_best: float, best observed value so far
        
    Returns:
        np.ndarray of shape (2000,) scores, higher is better
    """
    # The loop runs for 30 evaluations total.
    # We infer the current step from f_best relative to the known minimum (0).
    # Ackley 2D minimum is 0.
    # Early on, f_best is high (e.g., 5-10). As we converge, f_best -> 0.
    # We map this to a step count to adapt exploration.
    
    # Heuristic step estimation:
    # If f_best is close to 0, we are late. If high, we are early.
    # Let's define a progress metric.
    # A simpler robust heuristic: assume linear decrease in f_best isn't reliable.
    # Instead, let's use a fixed schedule based on a global counter if possible, 
    # but we don't have a counter. 
    # However, the prompt says "For each seed... starts from 5... until 30 evaluations".
    # Without a stateful variable, we must infer or use a static good balance.
    
    # Let's look at the inspiration: "Step 1: Switch from a static scalar acquisition... to a step-adaptive UCB".
    # Since we cannot store state between calls easily in this interface (no global state persistence guaranteed across calls in all contexts, though Python module globals persist), 
    # let's try to use a global counter.
    
    # Wait, is a global counter safe?
    # The evaluator calls acquisition multiple times.
    # If I use a global variable, it persists.
    
    # Let's implement a global step counter.
    
    if not hasattr(acquisition, '_step'):
        acquisition._step = 0
    
    # Increment step
    acquisition._step += 1
    
    # The total steps are 30.
    # We start at step 1 (first call after 5 initial points? No, the loop calls it each step).
    # The prompt says: "starts from 5 uniform random points, then at each step... until 30 evaluations including the first 5".
    # So there are 25 calls to acquisition (steps 6 through 30).
    # Step 1 in our counter corresponds to the 6th evaluation overall.
    # Step 25 corresponds to the 30th evaluation.
    
    # Let's map our internal step (1..25) to a progress ratio (0..1).
    # progress = 0 at start, 1 at end.
    total_acq_calls = 25
    current_call = acquisition._step
    if current_call > total_acq_calls:
        # Safety: if we go over, just use max exploitation
        progress = 1.0
    else:
        progress = (current_call - 1) / (total_acq_calls - 1)
    
    # Adaptive kappa for UCB:
    # Start with high exploration (kappa ~ 3.0) and decrease to low exploitation (kappa ~ 0.5).
    # Linear interpolation: kappa = 3.0 - (3.0 - 0.5) * progress
    kappa = 3.0 - 2.5 * progress
    
    # Clip sigma to avoid issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB for minimization:
    # We want to minimize f.
    # Lower mu is better.
    # Higher sigma is good for exploration.
    # Score = -mu + kappa * sigma
    # Higher score means "more promising".
    
    score = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
