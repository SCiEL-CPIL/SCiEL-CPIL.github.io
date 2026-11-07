"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Multi-sample Thompson Sampling estimate of the posterior minimum.
    # We sample latent function values from the GP posterior and return
    # the negative of the per-candidate minimum across samples.
    #
    # However, sampling 100 full GP posterior samples from 2000 candidates
    # requires Cholesky of a 2000x2000 matrix per call, which is expensive.
    # Instead, we use a cheaper approximation: sample from the marginal
    # posterior at each candidate independently (which is exact for the
    # marginal, not joint). This gives us a per-candidate sample from
    # N(mu_i, sigma_i^2). We take multiple samples per candidate and
    # score by the negative of the minimum across samples.
    #
    # Actually, the key insight from the measured improvement is that
    # multi-sample Thompson Sampling reduces variance in the estimate of
    # the posterior minimum. Let's do this properly but efficiently.
    #
    # For each candidate x_i, the marginal posterior is N(mu_i, sigma_i^2).
    # We draw S samples from each marginal independently. The score for x_i
    # is -min_s f_s(x_i) where f_s ~ N(mu_i, sigma_i^2).
    #
    # Expected value of min of S i.i.d. N(mu, sigma^2) samples:
    # E[min] = mu - sigma * E[min of S i.i.d. standard normals]
    # Let c_S = E[min of S i.i.d. N(0,1)] (negative constant).
    # So E[min] = mu + sigma * c_S where c_S < 0.
    # We want to maximize the probability of finding a low value, so we
    # want to minimize E[min], i.e., minimize mu + sigma * c_S.
    # Since c_S is negative, this balances exploitation (low mu) and
    # exploration (high sigma).
    #
    # For S samples, c_S can be computed numerically or approximated.
    # For S=1, c_1 = 0 (median of single sample is the mean).
    # Actually E[min of 1 sample] = 0 for standard normal.
    # For larger S, c_S becomes more negative.
    #
    # Let's just do the direct sampling approach with a fixed number of
    # samples, using a fixed random seed for reproducibility within a call.
    # But we need to be careful about time. Let's use S=50 samples.
    
    S = 50  # number of Thompson samples
    
    # Generate S x 2000 samples from N(mu, sigma^2)
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    noise = rng.randn(S, len(mu))
    samples = mu[None, :] + sigma[None, :] * noise
    
    # Score is negative of the minimum across samples
    # Lower sample values are better (we're minimizing)
    # So we want to find candidates where the minimum sample value is lowest
    # Score = -min(samples, axis=0), higher score = better
    
    min_samples = np.min(samples, axis=0)
    scores = -min_samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
