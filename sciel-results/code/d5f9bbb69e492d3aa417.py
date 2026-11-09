import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 280.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Structural change: Use Probability of Improvement (PI) combined with
    # a modified EI that has a sharper exploitation peak, plus an
    # uncertainty-based exploration term that decays with cost.
    
    # PI = Phi(z)
    pi = norm.cdf(z)

    # Modified EI: scale by a factor that emphasizes improvement over noise
    # Use a temperature-like parameter to sharpen
    # When mu is well below f_best, z is large positive, EI is large
    # When mu is above f_best, z is negative, EI is small
    
    # Combine PI and EI with a weighted sum
    # PI gives a smooth 0-1 signal, EI gives magnitude
    # Weight: favor PI when uncertainty is high (exploration), favor EI when low (exploitation)
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-10
    
    # Normalized uncertainty: 0 = low, 1 = high
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Blend: 0.5 * PI + 0.5 * EI_normalized
    # Normalize EI to be comparable to PI
    ei_max = np.max(ei) + 1e-10
    ei_norm = ei / ei_max
    
    # Weighted combination
    # When sigma is low (exploitative), weight EI more
    # When sigma is high (exploratory), weight PI more
    w_ei = 0.7 + 0.3 * (1 - sigma_norm)
    w_pi = 1 - w_ei
    
    combined = w_ei * ei_norm + w_pi * pi

    # Cost-aware scaling
    # Use a softer penalty: 1/sqrt(cost) instead of 1/cost
    # This allows more exploration in expensive regions while still preferring cheap ones
    cost_penalty = 1.0 / np.sqrt(cost)
    
    # Additional: small bonus for points with low predicted mean (exploitation)
    # This helps converge in flat regions of Griewank
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-10
    mu_norm = (mu - mu_min) / mu_range
    exploitation_bonus = 0.1 * (1 - mu_norm)
    
    score = (combined + exploitation_bonus) * cost_penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
