"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 300s limit

def _get_elapsed_time():
    return time.time() - _start_time

def _check_deadline():
    return _get_elapsed_time() > (300.0 - _DEADLINE_MARGIN)

def _safe_div(num, den):
    return np.where(den > 1e-9, num / den, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a modified Expected Improvement (EI) that
    explicitly accounts for the cost of evaluation.
    
    Strategy:
    1. Compute standard EI for minimization.
    2. Normalize the cost to a range that allows meaningful penalty.
    3. Use a 'cost-aware' EI: EI / (1 + cost). 
       Wait, the task says "Evaluating x costs 1 + mean(x scaled to [0, 1])".
       The `cost` input is shape (2000,). It likely contains the cost of each candidate.
       
       A common approach is to divide EI by cost. However, if cost is around 1.5, 
       dividing by 1.5 is a constant scaling. 
       
       Let's look at the structure: 
       cost = 1 + mean(x_scaled). 
       If x is in [0,1]^2, mean(x_scaled) is between 0 and 1. 
       So cost is in [1, 2].
       
       If we simply do EI / cost, we are penalizing expensive points.
       But is that the right metric? 
       The goal is to minimize f_best. 
       We have a budget of 30.
       
       A more sophisticated approach: 
       We want to maximize the expected improvement per unit cost.
       score = EI / cost
       
       Let's try this. It's a simple, structural change from the UCB/EI blend.
       The current program uses a blend of EI and UCB with a decaying kappa.
       This new approach uses ONLY EI, but normalized by cost.
       
       Additionally, we can introduce a small exploration term based on sigma
       but weighted differently. 
       
       Let's stick to the core idea: Cost-Effective EI.
       score = EI / cost
       
       To ensure we don't ignore exploration entirely, if sigma is very high, 
       EI might be small if mu is far from f_best. 
       
       Let's add a small UCB term as well, but keep the primary driver as cost-aware EI.
       Actually, the directive asks for a MATERIALLY DIFFERENT strategy.
       The current one is a hybrid EI/UCB with time-decay.
       A pure Cost-Aware EI (or Cost-Effective EI) is a different family.
       
       Let's implement:
       1. Calculate EI.
       2. Calculate score = EI / cost.
       3. Add a small exploration bonus: + beta * sigma / cost.
       
       Why? Because EI can be zero if mu <= f_best. But if sigma is high, there is potential.
       The UCB term captures this.
       
       Let's use a fixed small beta, say 0.1, to encourage exploration in uncertain regions
       while still prioritizing cost-effective improvements.
    """
    if _check_deadline():
        # Fallback to simple EI/cost if running out of time
        z = _safe_div(f_best - mu, sigma)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return _safe_div(ei, cost_safe)

    # Calculate EI
    z = _safe_div(f_best - mu, sigma)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB (for minimization)
    # We want low mu, high sigma.
    ucb = -mu + 0.5 * sigma  # Fixed small exploration weight
    
    # Cost-aware score
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary term: EI per cost
    score_ei = _safe_div(ei, cost_safe)
    
    # Secondary term: UCB per cost (exploration)
    score_ucb = _safe_div(ucb, cost_safe)
    
    # Combine
    # Weight EI more heavily, but keep some exploration
    final_score = 0.9 * score_ei + 0.1 * score_ucb
    
    # Ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score
# EVOLVE-BLOCK-END
