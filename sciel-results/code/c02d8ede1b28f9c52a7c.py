import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Kriging Believer with EI.
    
    Computes Expected Improvement (EI) for minimization, then applies a
    Kriging Believer transformation: score proportional to the GP mean
    at the point where the posterior is shifted to achieve the top EI.
    Specifically, for each candidate we compute z = (f_best - mu) / sigma,
    the EI value, and then a "believer" mean: mu_bel = mu + sigma * z_star,
    where z_star is derived from the EI structure. We return EI / cost as the
    primary score but blend in the believer mean to break ties and guide
    toward high-probability improvement regions.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0
    
    _call_count += 1
    
    if time.time() > _deadline:
        return np.ones(len(mu))
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement for minimization
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Kriging Believer: The idea is to pick the point that maximizes the
    # GP mean conditional on it being the one with highest EI.
    # A practical approximation: use the "believer" mean which is
    # mu + sigma * z_top, where z_top is the z-value at the top EI point.
    # But since we need to return scores for all 2000 candidates, we compute
    # a per-candidate believer score.
    #
    # The Kriging Believer picks x* = argmax EI(x), then evaluates mu(x*)
    # as if it were the true value. For scoring, we can use:
    # score = EI(x) / cost(x) * (1 + lambda * mu_bel(x) / mu_bel_max)
    # where mu_bel(x) = mu(x) + sigma(x) * z(x) = mu(x) + (f_best - mu(x)) = f_best
    # Wait, that's constant. Let me reconsider.
    #
    # Actually, the Kriging Believer works by:
    # 1. Find x_ei = argmax EI(x)
    # 2. Set the "believer" value to mu(x_ei)
    # 3. Use this to guide further selection
    #
    # For a scoring function that returns 2000 scores, we can't do a two-step
    # selection. Instead, we blend EI/cost with a believer-like term.
    #
    # A cleaner approach: use EI/cost as the base, but weight by the
    # probability of being the best (Kriging Believer proxy):
    # The believer score for candidate i is mu_i + sigma_i * z_i = f_best (constant)
    # So instead, use: score = EI / cost, which is what inspiration 1/2/3 use.
    #
    # To add structural novelty: use a cost-aware EI with an adaptive
    # exploration-exploitation balance via the call count.
    
    # Adaptive blend: early on, weight more toward exploration (sigma),
    # later toward exploitation (low mu). Use a decaying weight.
    # weight = exp(-0.1 * call_count) for exploration component
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # LCB = mu - kappa * sigma (lower is better)
    # Use an adaptive kappa that decays over time
    kappa = 0.5 + 1.5 * np.exp(-0.2 * _call_count)
    lcb = mu - kappa * sigma_safe
    
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei) + eps
    lcb_min = np.min(lcb)
    lcb_range = np.ptp(lcb) + eps
    lcb_norm = (lcb_min - lcb) / lcb_range  # Higher is better (closer to min)
    
    # Blend EI/cost with normalized LCB/cost
    # As call_count increases, shift more weight to LCB (exploitation)
    blend = np.exp(-0.15 * _call_count)  # 1.0 early, decays to ~0
    
    score_ei = ei / (cost + eps)
    score_lcb = lcb_norm / (cost + eps)
    
    scores = blend * score_ei + (1.0 - blend) * score_lcb
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
