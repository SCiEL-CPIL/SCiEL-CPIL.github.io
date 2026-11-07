import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB) for minimization
    # Exploration term. Kappa=2 was good. Let's try a hybrid.
    kappa = 2.0
    lcb = mu - kappa * sigma
    
    # Normalize scores to combine them effectively
    # We want higher score to be better.
    # EI is already "higher is better" (improvement magnitude).
    # LCB is "lower is better" in terms of objective, so we negate it for scoring?
    # Wait, the return value is "score", higher meaning evaluate next.
    # So we want to return a value where larger = more promising.
    
    # For EI: larger EI is better.
    # For LCB: smaller mu - kappa*sigma is better. So we should use -LCB or flip it.
    # Let's define exploration_score = -lcb = -mu + kappa*sigma.
    
    exploration_score = -lcb
    
    # Normalize both to zero mean, unit variance across the candidate set
    # This prevents one term from dominating due to scale differences
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    exp_mean = np.mean(exploration_score)
    exp_std = np.std(exploration_score) + 1e-8
    exp_norm = (exploration_score - exp_mean) / exp_std
    
    # Weighting: Favor exploration more heavily given the small budget and multimodality
    # LCB(2) alone beat EI. So let's give exploration more weight.
    alpha = 0.6 # Weight for exploration (LCB)
    beta = 1.0 - alpha # Weight for exploitation (EI)
    
    scores = alpha * exp_norm + beta * ei_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
