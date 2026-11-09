"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track time and evaluation count
_start_time = None
_eval_count = 0
_MAX_EVALS = 30
_INIT_EVALS = 5

def acquisition(mu, sigma, f_best):
    global _start_time, _eval_count
    
    # Initialize time tracking if not already set
    if _start_time is None:
        _start_time = time.time()
        
    # Increment evaluation count
    _eval_count += 1
    
    elapsed = time.time() - _start_time
    
    # Safety margin: return valid scores immediately if close to limit
    # The evaluator kills after 330s, task limit is 300s.
    # If we are past 280s, just return something valid to finish quickly.
    if elapsed > 280.0:
        # Return a neutral score (zeros) to avoid crashing.
        # This effectively makes the choice arbitrary for the last step if time is out,
        # but ensures we return a valid array.
        return np.zeros_like(mu)

    # Strategy: Time-Aware Adaptive Acquisition
    # We blend LCB (exploration) and EI (exploitation).
    # LCB = mu - beta * sigma
    # EI is calculated as standard.
    # We use a weighting factor w that starts high (exploration) and drops to 0 (exploitation).
    
    # Determine the fraction of budget used.
    # Budget is 30 total evaluations. First 5 are random, so we have 25 "active" steps.
    # However, _eval_count counts all calls to acquisition.
    # Calls happen for steps 6 through 30. So active steps are 25.
    # If _eval_count is 1 (first acquisition call, i.e., step 6), we are at the start of active search.
    # If _eval_count is 25 (last acquisition call, i.e., step 30), we are at the end.
    
    active_step = _eval_count
    total_active_steps = _MAX_EVALS - _INIT_EVALS
    
    # Clamp step to [1, total_active_steps]
    if active_step < 1:
        active_step = 1
    if active_step > total_active_steps:
        active_step = total_active_steps
        
    # Progress ratio: 0.0 at start of active search, 1.0 at end.
    progress = (active_step - 1) / max(1, total_active_steps - 1)
    
    # We want high exploration early, low exploration late.
    # Let beta (exploration weight for LCB) decay from ~2.5 to ~0.5.
    # And we blend the final score: score = (1-w)*EI + w*LCB_normalized? 
    # Actually, it's better to just pick one or blend the *scores*.
    # LCB and EI have different scales. 
    # A common robust approach is to use a "Switch" or a "Mixture".
    # Let's use a simple switch: 
    # If progress < 0.5, use LCB with adaptive beta.
    # If progress >= 0.5, use EI.
    # 
    # However, a smooth blend is often better. 
    # To make them comparable, we can normalize them or just use a heuristic.
    # Let's try a different approach: 
    # Use a "Decaying LCB" where beta starts high and goes low.
    # LCB is generally better for functions with multiple local minima like Goldstein-Price.
    # EI can get stuck in local minima if the GP model is slightly off.
    # Let's stick to LCB but with a very aggressive decay of beta.
    #
    # Beta schedule:
    # beta_start = 2.5 (Strong exploration)
    # beta_end = 0.5 (Mild exploration, mostly exploitation)
    # beta = beta_start * (beta_end/beta_start)**progress
    # This is exponential decay.
    
    beta_start = 2.5
    beta_end = 0.5
    
    # Exponential decay of beta
    beta = beta_start * (beta_end / beta_start) ** progress
    
    # Calculate LCB
    sigma_safe = np.maximum(sigma, 1e-9)
    lcb = mu - beta * sigma_safe
    
    # Calculate EI (for comparison or fallback if we want to blend)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Let's use a mixture. 
    # Weight for LCB: w_lcb = 1 - progress (starts at 1, ends at 0)
    # Weight for EI: w_ei = progress (starts at 0, ends at 1)
    # But scales are different. 
    # A safer "different strategy" is to use **Probability of Improvement (PI)** 
    # or **Knowledge Gradient (KG)** approximations? No, too slow.
    #
    # Let's refine the LCB approach. The key insight from inspiration 3 (score 1.29) 
    # was switching from LCB to UCB/EI. 
    # My strategy: **Adaptive Beta LCB**.
    # Instead of a hard switch, I will use LCB with a beta that decays rapidly.
    # This ensures we explore broadly first, then focus.
    #
    # Additionally, I will add a small epsilon to prevent sigma from being too small 
    # causing numerical issues, though we already have sigma_safe.
    
    # Ensure finite scores
    lcb = np.nan_to_num(lcb, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return lcb
# EVOLVE-BLOCK-END
