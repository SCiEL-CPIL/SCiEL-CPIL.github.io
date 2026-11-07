"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking to respect the 330s wall-clock limit
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: just pick the one with lowest mu / cost
        # to avoid crashing, though the run is likely over.
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    # The current best (2.3655) used an adaptive LCB. 
    # The strongest inspiration (1.2013) used EI but with a different cost handling.
    # The directive suggests using EI * (1/cost) as a multiplicative discount.
    
    # Let's combine the robustness of the current adaptive LCB with the theoretical grounding of EI.
    # However, the directive specifically asks to try: "Model the cost as a multiplicative discount on the expected improvement... score = EI(mu, sigma, f_best) * (1.0 / cost)"
    
    # Let's implement the multiplicative discount on EI as requested, but potentially blend it with the LCB if EI is too greedy.
    # Actually, the directive says "COMBINE the strategy of the current program with the strongest ideas...".
    # Current: Adaptive LCB / cost.
    # Inspiration 2: EI / cost_factor.
    # Directive: EI * (1/cost).
    
    # Let's try the multiplicative discount on EI first, as it's a structural change from the current LCB approach.
    # EI for minimization:
    # improvement = f_best - mu
    # Z = (f_best - mu) / sigma
    # EI = improvement * Phi(Z) + sigma * phi(Z)
    
    improvement = f_best - mu
    Z = improvement / sigma_safe
    
    # Clip Z for numerical stability
    Z = np.clip(Z, -50, 50)
    
    EI = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    EI = np.maximum(EI, 0.0)
    
    # Multiplicative discount: score = EI / cost
    score = EI / cost_safe
    
    # The current program's success might be due to the exploration term (beta * sigma).
    # EI inherently includes exploration through the sigma * phi(Z) term, but it decays quickly.
    # If f_best is very low, EI becomes very small and the optimizer might get stuck.
    # The LCB approach with adaptive beta might handle the late-stage exploitation better.
    
    # Let's try a hybrid:
    # If f_best is still high (early), use LCB to explore broadly.
    # If f_best is low (late), use EI to exploit precisely.
    
    # Adaptive beta from current program:
    f_scale = 25.0
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    lcb_score = -mu + beta * sigma_safe
    
    # Blend LCB and EI based on f_best?
    # Or just use the multiplicative EI as per directive?
    # The directive says "Make a STRUCTURAL change... Strategy to try... score = EI * (1.0 / cost)"
    # So I should focus on that.
    
    # However, simple EI * (1/cost) might be too greedy.
    # Let's add a small exploration bonus to EI, similar to inspiration 3, but scaled.
    # Or, let's stick to the directive's suggestion but ensure it's robust.
    
    # One issue with EI is that when sigma is very small, EI becomes 0 even if mu < f_best.
    # This is correct behavior (no expected improvement), but it might lead to no exploration.
    # The LCB approach always has a score based on mu and sigma.
    
    # Let's try a weighted sum of normalized EI and LCB.
    # But normalization is tricky.
    
    # Let's go with the directive's suggestion: EI / cost.
    # To prevent it from being too greedy, we can add a small term proportional to sigma / cost.
    # This is essentially an LCB with a smaller beta.
    
    # Let's try: score = (EI + alpha * sigma) / cost
    # Where alpha is small.
    
    # Or, let's just use EI / cost as the primary strategy.
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
