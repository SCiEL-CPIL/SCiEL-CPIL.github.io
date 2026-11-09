"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 315.0  # 15s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        return -mu.copy()
    
    # Thompson Sampling / Stochastic UCB approach
    # Instead of a fixed acquisition function, we sample an exploration
    # coefficient each time, creating stochastic exploration.
    # This is structurally different from EI/PI which are deterministic.
    
    # The scale of sigma tells us about uncertainty in the GP
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    # Sample exploration coefficient from a distribution that balances
    # exploration and exploitation. Use a half-normal or log-normal
    # to ensure positive exploration.
    # The typical sigma scale helps calibrate.
    typical_sigma = np.median(sigma_safe)
    
    # Draw kappa from a distribution centered around values that
    # balance exploration. We use a mixture: sometimes greedy,
    # sometimes more exploratory.
    # Sample from a truncated normal or use a fixed stochastic approach.
    
    # Use a random kappa that varies each call for stochastic exploration
    # kappa ~ Uniform(0, 3) * typical_sigma / (f_best + typical_sigma + 1e-9)
    # This normalizes by the scale of the problem
    
    # Actually, simpler: just use a random positive kappa
    # The key insight: Thompson sampling samples from the posterior
    # and picks the argmin. With only mu and sigma, we approximate by
    # sampling theta ~ N(0, sigma^2) for each point, but we can't do that
    # independently. Instead, we use a shared random shock.
    
    # Shared random shock approach:
    # score_i = mu_i + kappa * sigma_i
    # where kappa is a single random value drawn each call
    # This approximates sampling from the GP posterior with a common factor
    
    # Draw kappa from a distribution that provides good exploration
    # Use a scaled half-normal: kappa = |N(0,1)| * scale
    # Scale should be related to the problem's uncertainty level
    
    # The ratio of typical sigma to f_best tells us how uncertain we are
    uncertainty_ratio = typical_sigma / (abs(f_best) + typical_sigma + 1e-9)
    
    # kappa: sample from a distribution that gives good exploration
    # Use exponential for heavy-tailed exploration
    kappa = np.random.exponential(1.0) * (0.5 + uncertainty_ratio)
    
    # Ensure kappa is not too extreme
    kappa = np.clip(kappa, 0.1, 5.0)
    
    # Compute LCB scores (lower is better, but we want higher = more promising)
    # LCB for minimization: mu - kappa * sigma
    # We want to maximize this, so we return it directly
    scores = mu - kappa * sigma_safe
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
