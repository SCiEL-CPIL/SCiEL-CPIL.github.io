import numpy as np
import time

# Global state for tracking progress and time
class BOState:
    def __init__(self):
        self.start_time = time.time()
        self.step = 0
        self.f_best_history = []

state = BOState()

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian Optimization.
    Strategy: Pure Expected Improvement (EI) with a late-game exploitation pivot.
    Includes a deadline guard to ensure validity within the 330s limit.
    """
    state.step += 1
    
    # Deadline Guard: If we are too close to the 330s limit, return safe scores
    # to avoid timeout. In practice, this function is called 25 times, 
    # each taking <1ms, so this is a safety net.
    elapsed = time.time() - state.start_time
    if elapsed > 325.0:
        # Return a deterministic score based on mu to ensure we finish
        # and pick the best known point if we are about to time out
        return -mu 

    # Update history
    state.f_best_history.append(f_best)
    
    # Structural Change: Pure Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Late-game pivot: In the last 2 steps, switch to pure exploitation (minimize mu)
    # This ensures we don't waste the final evaluations exploring, but instead
    # refine the best found region.
    if state.step >= 24: # Steps 1..25 are acquisition calls (total 30 evals: 5 init + 25 BO)
        # Return -mu so that the point with the lowest predicted mean is chosen
        # Higher score is better for the caller, so we negate mu
        return -mu

    # Standard EI calculation
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats.norm for high-precision CDF and PDF
    # Alternatively, use numpy approximations if speed is critical, but scipy is allowed and fast
    from scipy.stats import norm
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Return raw EI scores.
    # Higher EI means higher probability of improving f_best.
    # This is materially different from the parent which normalized and blended with LCB.
    return ei
