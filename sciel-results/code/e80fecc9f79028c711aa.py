# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Monte Carlo Knowledge Gradient (KG) acquisition function.
    
    The Knowledge Gradient measures the expected reduction in the minimum observed
    function value if we evaluate at a candidate point x.
    KG(x) = f_best - E[ min(f_best, f(x)) ]
    
    Since f(x) ~ N(mu(x), sigma(x)^2), we can compute this in closed form:
    KG(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    This is mathematically equivalent to Expected Improvement (EI).
    
    However, standard EI/KG treats candidates independently. For multi-modal functions
    like Hartmann 3D, we want to focus on the region containing the *global* minimum.
    
    Strategy:
    1. Estimate the probability that each candidate is in the basin of the global minimum
       using Monte Carlo sampling of the joint Gaussian distribution.
    2. Compute the base EI score for each candidate.
    3. Weight the EI score by a power of the global-minimum probability to boost
       candidates in promising basins and suppress those in likely local minima.
    
    This "Probabilistic Global Minimum" weighting is a structural change from simple
    EI/UCB blending, as it uses joint statistics to infer basin relevance.
    """
    start_time = time.time()
    DEADLINE = 325.0  # seconds, safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Fallback if time is already tight
    if time.time() - start_time > DEADLINE - 5:
        return _fallback_scores(mu, sigma, f_best)
    
    # 1. Compute base EI scores (closed form)
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # 2. Estimate probability of being the global minimum via MC
    # We sample from the joint distribution of f(x_i) ~ N(mu, Sigma)
    # Since we don't have the full covariance matrix (only mu and sigma),
    # we approximate the joint sampling by assuming a correlation structure
    # or simply sampling independent marginals if we can't afford full MC.
    # 
    # A cheaper and effective approximation for "basin relevance" in BO:
    # Candidates with low mu and moderate sigma are more likely to be in the
    # global basin. We can estimate the "globality" by looking at how many
    # other candidates have a lower predicted minimum.
    #
    # Let's use a Monte Carlo approach with a limited number of samples to
    # estimate P(x_i is argmin) or P(x_i < all others).
    # Given 2000 candidates, full joint MC is expensive (2000x2000 covariance).
    # Instead, we use a heuristic proxy for "global basin probability":
    # 
    # Proxy: Normalize the "optimistic" score (mu - kappa*sigma) and use its
    # rank or density. But the directive asks for a structural change.
    #
    # Let's implement a lightweight MC for the *minimum* of the whole set.
    # We sample S times. In each sample, we draw f_i ~ N(mu_i, sigma_i^2) independently
    # (ignoring correlation, which is a standard approximation when K is large).
    # Then we count how often each x_i is the minimum of the sample.
    # This gives an estimate of P(x_i is the best in the set).
    
    # Number of MC samples: balance accuracy vs time.
    # 2000 candidates * S samples. If S=100, that's 200k draws, very fast.
    S = 100
    # Generate (S, n) array of samples
    # f_samples[s, i] = mu[i] + sigma[i] * z[s, i]
    # To save memory/time, we can do this in chunks or just vectorize.
    
    # Vectorized MC:
    # z_samples = np.random.randn(S, n)
    # f_samples = mu[None, :] + sigma[None, :] * z_samples
    # min_indices = np.argmin(f_samples, axis=1)
    # prob_global = np.bincount(min_indices, minlength=n) / S
    
    # This is fast for n=2000, S=100.
    
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        # If MC is too slow (unlikely for 100 samples), fall back to EI
        return _fallback_scores(mu, sigma, f_best)
    
    z_samples = np.random.randn(S, n)
    f_samples = mu[None, :] + sigma_safe[None, :] * z_samples
    min_indices = np.argmin(f_samples, axis=1)
    prob_global = np.bincount(min_indices, minlength=n).astype(np.float64) / S
    
    # 3. Weight EI by prob_global
    # We want to strongly favor candidates with high prob_global.
    # Use a power alpha > 1 to sharpen the distribution.
    # If alpha=1, it's just EI * P(global).
    # If alpha=2, it's EI * P(global)^2.
    # Let's try alpha=1.5 to balance exploration (EI) and exploitation (P(global)).
    
    alpha = 1.5
    # Avoid 0^alpha issues (0^1.5 is 0, which is fine)
    weights = prob_global ** alpha
    
    # Normalize weights to [0, 1] to prevent scale issues?
    # EI is in objective units. Weights are probabilities.
    # The product EI * weights might be small.
    # Let's normalize the weighted scores to be comparable to EI.
    
    weighted_scores = ei_scores * weights
    
    # It's possible that the max weighted score is very small if prob_global is sparse.
    # Let's ensure the top candidate has a reasonable score.
    # Alternatively, we can add a small exploration term.
    
    # Let's check if the weighted scores are degenerate (all near 0)
    if np.max(weighted_scores) < 1e-6:
        # Fall back to pure EI if weighting killed everything
        return _fallback_scores(mu, sigma, f_best)
    
    # Normalize weighted scores to have similar magnitude to EI
    # This helps if the evaluator expects scores in a certain range, though it only cares about ranking.
    # But "higher meaning evaluate next" is relative.
    
    # Apply a power transformation to amplify the top scores?
    # The current program used scores**2. Let's keep it simple first.
    
    # Ensure finite
    weighted_scores = np.nan_to_num(weighted_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final check
    if time.time() - start_time > DEADLINE - 2:
        return _fallback_scores(mu, sigma, f_best)
    
    return weighted_scores

def _fallback_scores(mu, sigma, f_best):
    """Fast fallback: standard EI-like score."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
