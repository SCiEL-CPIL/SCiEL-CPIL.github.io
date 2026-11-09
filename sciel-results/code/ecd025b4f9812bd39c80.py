"""
Strategy: Cost-Effective Expected Improvement (CEI)
The previous strategy used an Adaptive LCB (point estimate). This strategy uses Expected Improvement (EI),
which is statistically more robust as it accounts for the entire distribution of potential improvements.
We normalize EI by cost to get CEI. We also introduce a dynamic exploration parameter 'xi' (or similar scaling)
that adjusts based on the current state of the search (ratio of uncertainty to best value found) to balance
exploration and exploitation without manual tuning of a fixed kappa.
"""

import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Leave 10s buffer for 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective Expected Improvement.
    
    Args:
        mu: np.array, shape (2000,), mean predictions
        sigma: np.array, shape (2000,), standard deviation predictions
        f_best: float, best observed value so far
        cost: np.array, shape (2000,), cost of evaluating each candidate
        
    Returns:
        scores: np.array, shape (2000,), finite scores (higher = better)
    """
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety checks
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n)
        
    # Ensure cost is positive to avoid division issues
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # We want to find x such that f(x) < f_best.
    # Improvement I(x) = max(0, f_best - f(x))
    # E[I(x)] = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in norm.cdf/pdf for extreme values
    # Phi(z) -> 1 for large z, Phi(z) -> 0 for large negative z
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    # Expected Improvement
    # Note: If f_best is very high (early on), EI is large.
    # If f_best is low (converged), EI is small.
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective Improvement
    cei = ei / cost_safe
    
    # Dynamic Exploration Adjustment:
    # In early stages, f_best is high and sigma is high. EI naturally handles this.
    # However, EI can be greedy. To encourage exploration in regions with high sigma 
    # but moderate mu, we can add a small UCB-like term or adjust the 'xi' parameter 
    # in EI (though standard EI has xi=0). 
    # 
    # A simpler structural change: 
    # The previous LCB used a linear combination. EI is non-linear.
    # To ensure we don't get stuck in a local minimum of EI, we can blend CEI with a 
    # normalized uncertainty score.
    
    # Let's analyze the "Stagnation" directive.
    # The current LCB is linear. EI is non-linear.
    # Another structural change: Use **Probability of Improvement (PI)**?
    # PI = Phi(z). It ignores the magnitude of improvement.
    # CEI is generally better than PI for optimization.
    
    # Let's stick to pure CEI but ensure it's robust.
    # One issue with CEI is that if f_best is very large (initial steps), 
    # (f_best - mu) is large, making EI large for almost everything.
    # This is expected.
    
    # To differentiate from a simple EI/cost, let's look at the "cost" structure.
    # cost = 1 + mean(x). This is roughly uniform in [1, 1.5].
    # So 1/cost is roughly uniform in [0.66, 1.0].
    # This normalization is mild.
    
    # Let's try a **Hybrid**: 0.8 * CEI + 0.2 * (sigma / cost)
    # This explicitly adds an exploration bonus scaled by cost.
    # This is structurally different from pure EI/cost and pure LCB/cost.
    # LCB is (f_best - mu + beta*sigma)/cost.
    # CEI is (EI)/cost.
    # Hybrid is (0.8*EI + 0.2*sigma)/cost.
    
    # Let's determine the weight dynamically.
    # If the model is uncertain (high mean sigma), increase exploration weight.
    mean_sigma = np.mean(sigma_safe)
    # Scale factor: if mean_sigma is large relative to the spread of mu, we are in exploration phase.
    # Use a sigmoid or linear scaling.
    # Let's use a simple heuristic: 
    # exploration_weight = 0.1 + 0.3 * (mean_sigma / (np.std(mu) + mean_sigma + 1e-9))
    
    mu_std = np.std(mu)
    denom = mu_std + mean_sigma + 1e-9
    ratio = mean_sigma / denom  # 0 to 1
    
    # Weight for exploration (sigma) vs exploitation (EI)
    # In early stages (high ratio), we want more exploration.
    # In late stages (low ratio), we want more exploitation.
    w_explore = 0.2 + 0.3 * ratio  # Range [0.2, 0.5]
    w_exploit = 1.0 - w_explore
    
    # Composite Score
    # Note: EI and sigma have different scales. 
    # EI is in objective units. Sigma is in objective units.
    # So they are comparable.
    
    score = (w_exploit * ei + w_explore * sigma_safe) / cost_safe
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
