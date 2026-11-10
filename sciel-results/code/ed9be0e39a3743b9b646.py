import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Probability of Improvement, normalized by cost.
    
    Parameters:
    mu: (2000,) GP mean
    sigma: (2000,) GP standard deviation
    f_best: float, best observed value
    cost: (2000,) cost of evaluating each candidate
    
    Returns:
    scores: (2000,) acquisition scores, higher is better
    """
    eps = 1e-9
    
    # Compute z-score for improvement
    z = (f_best - mu) / (sigma + eps)
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement
    po_i = norm.cdf(z)
    
    # Expected Improvement
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * po_i + sigma * phi_z
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to be comparable to PoI
    # We want a blend that balances exploitation (EI) and exploration (PoI)
    # Since EI can be large when improvement is large, we normalize it.
    # A simple way is to divide by a scale factor.
    # The scale of improvement is roughly the spread of f.
    # We can estimate the scale from the difference between f_best and the median of mu?
    # Or just use a fixed weight.
    # The hint says "adaptive weighting".
    # Let's try: weight_ei = 1.0 / (1.0 + np.std(mu)) ? No, that's global.
    # Let's try: score = (w * ei + (1-w) * po_i) / cost
    # If we use w=0.5, it's a simple average.
    
    # Let's look at the "relaxed threshold" idea again.
    # Maybe instead of f_best, we use a slightly worse target?
    # target = f_best + 0.1 * np.std(mu)?
    
    # Let's stick to the hybrid EI/PI but with a twist:
    # Normalize EI by the current uncertainty or spread.
    # score = (ei / (np.max(ei) + eps) * 0.5 + po_i * 0.5) / cost
    
    # This normalizes EI to [0, 1] per batch, making it comparable to PoI.
    max_ei = np.max(ei)
    if max_ei > eps:
        ei_norm = ei / max_ei
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend
    # We can adapt the weight based on how close we are to the optimum?
    # If f_best is very low, we might want more exploration (PoI).
    # But we don't know the optimum.
    # Let's use a fixed 50/50 blend for now, or maybe 70/30 in favor of EI?
    # The previous improvement (1 -> 1.5) used "adaptive weighting".
    # Let's try weight = 0.8 * ei_norm + 0.2 * po_i
    
    scores = (0.8 * ei_norm + 0.2 * po_i) / (cost + eps)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
