# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Knowledge Gradient (KG) to optimize for global convergence.
    KG estimates the expected reduction in the global minimum by computing
    the probability that each candidate becomes the new best, weighted by
    the expected improvement over the current best.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Standardize for numerical stability
    # We work in the original objective space for f_best
    
    # Knowledge Gradient approximation:
    # KG(x) = E[ max(0, f_best - f(x)) - E[max(0, f_best - f(x)) | not best] ]
    # 
    # A practical approximation:
    # 1. Compute EI for each candidate: EI(x) = E[max(0, f_best - f(x))]
    # 2. Compute P(x is best) = P(f(x) < min of all other f's)
    #    Approximate by P(f(x) < f_best) since f_best is the observed minimum
    # 3. KG ≈ EI(x) * P(x is new best) + small exploration term
    #
    # More precisely, KG measures the expected improvement of the global minimum.
    # For 1D smooth functions, we can use:
    # KG(x) ≈ EI(x) * w(x) where w(x) accounts for the probability that
    # improving at x actually improves the global minimum.
    
    # Step 1: Compute EI for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Step 2: Compute the probability that candidate x would be the new best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma) = Phi(z)
    # This is already computed above
    
    # Step 3: KG approximation
    # The key insight: KG should be high when:
    # - EI is large (good expected improvement)
    # - The probability of being the new best is high
    # - But also account for the fact that we need to beat the global min
    
    # A refined KG score:
    # KG(x) = EI(x) * Phi(z) + exploration_bonus
    # The Phi(z) factor weights by the probability of actually improving the global min
    
    kg_base = ei * Phi
    
    # Add a small exploration component to avoid getting stuck
    # This helps in early stages when f_best is far from the true minimum
    # Use a UCB-like term with small weight
    beta = 0.05
    ucb_term = -mu + beta * sigma
    
    # Combine: KG dominates, UCB provides small exploration
    score = kg_base + 0.1 * ucb_term
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return EI only (fastest path)
        return ei
    
    return score
# EVOLVE-BLOCK-END
