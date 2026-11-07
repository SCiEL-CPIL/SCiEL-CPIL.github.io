import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a continuous
    UCB-inspired exploration term weighted by promise.
    
    The score is EI (exploitation) plus a continuous exploration bonus that
    scales with uncertainty (sigma) and is weighted by how promising the
    predicted mean is relative to f_best. This creates a smooth, adaptive
    exploration signal rather than a discrete gate.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # 2. Continuous exploration term: beta * sigma * promise_weight
    # Promise weight: how much better mu is than f_best, normalized by the
    # range of mu values. This ensures exploration is focused on regions
    # that are both uncertain AND promising.
    
    # Compute the range of mu for normalization
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-6:
        mu_range = 1.0
    
    # Promise: 1 when mu is much better than f_best, 0 when mu is worse
    # Use a smooth sigmoid-like transition centered at f_best
    # promise = 1 / (1 + exp(k * (mu - f_best) / mu_range))
    # k controls the sharpness; larger k makes it more binary
    k = 3.0
    promise_arg = k * (mu - f_best) / mu_range
    # Clip to avoid overflow in exp
    promise_arg = np.clip(promise_arg, -50, 50)
    promise_weight = 1.0 / (1.0 + np.exp(promise_arg))
    
    # UCB-like exploration term
    # beta scales the exploration; we want it to be comparable to EI values
    # Use a beta that adapts to the scale of the problem
    # The typical EI scale is roughly (f_best - mu_min) when there's improvement
    ei_scale = f_best - mu_min
    if ei_scale < 1e-6:
        ei_scale = 1.0
    
    # beta: fraction of the EI scale to use as exploration weight
    # 0.3 means exploration can contribute up to 30% of the typical EI magnitude
    beta = 0.3
    
    exploration = beta * sigma_safe * promise_weight
    
    # Combine: EI + exploration
    scores = ei + exploration
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
