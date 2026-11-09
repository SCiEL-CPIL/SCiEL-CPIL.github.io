"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for the evaluator session
_start_time = time.time()
# The evaluator kills after 330s. We set a safety margin.
_DEADLINE = 320.0

def _sanitize(mu, sigma):
    """Ensure mu and sigma are finite and sigma is positive."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    if not np.all(np.isfinite(mu)):
        finite_mu = mu[np.isfinite(mu)]
        fill_val = np.mean(finite_mu) if len(finite_mu) > 0 else 0.0
        mu = np.where(np.isfinite(mu), mu, fill_val)
        
    if not np.all(np.isfinite(sigma)):
        finite_sigma = sigma[np.isfinite(sigma)]
        fill_val = np.mean(finite_sigma) if len(finite_sigma) > 0 else 1e-6
        sigma = np.where(np.isfinite(sigma), sigma, fill_val)
        
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    return mu, sigma

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to standard EI to ensure valid output quickly
        mu, sigma = _sanitize(mu, sigma)
        z = (f_best - mu) / sigma
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores[~np.isfinite(scores)] = 0.0
        return scores

    mu, sigma = _sanitize(mu, sigma)
    
    # Thompson Sampling: Sample from the GP posterior
    # For each candidate i, we sample a value y_i ~ N(mu_i, sigma_i^2)
    # We want to minimize the function, so we want to pick the candidate
    # with the lowest sampled value.
    # However, the acquisition function must return scores where HIGHER is better.
    # So we return -sampled_value.
    
    # To make the sampling more robust and less prone to extreme outliers
    # dominating the choice in a single step, we can use a quantile-based approach
    # or simply sample once. Standard Thompson sampling samples once per step.
    
    # Let's sample once from the posterior for each candidate.
    # rng = np.random.default_rng()
    # samples = rng.normal(loc=mu, scale=sigma)
    
    # A single sample can be noisy. A more robust variant is to take the
    # median of multiple samples, or use a specific quantile.
    # However, the standard Thompson sampling uses a single sample.
    # Let's stick to a single sample but ensure it's reproducible within the step
    # if needed, though the task doesn't require reproducibility between calls,
    # just valid output.
    
    # Actually, since we only get mu and sigma, and we need to return 2000 scores,
    # we simulate the next observation for each candidate.
    
    # To avoid the randomness causing a single bad sample to be picked,
    # some implementations use the mean of K samples.
    # Let's use K=10 samples and take the minimum (for minimization)
    # or rather, we want the score to reflect the "best possible" outcome
    # or the expected outcome?
    
    # Standard TS: Pick argmin(sampled_values).
    # Score for candidate i = -sampled_value_i.
    
    # Let's use a single random sample.
    # Note: The evaluator draws 2000 candidates. We return scores for them.
    # The evaluator picks the one with the highest score.
    
    # To reduce variance of the selection, we can sample multiple times
    # and take the mean of the sampled values (which approximates the posterior mean)
    # or the minimum (which is more exploratory).
    # Actually, taking the minimum of K samples is equivalent to a "maximizing"
    # acquisition that is more aggressive in exploration.
    # Let's try K=5 samples and take the minimum of the samples for each candidate.
    # This creates a "robust" Thompson score.
    
    K = 5
    # Generate (K, 2000) samples
    # We need a random seed? No, just random.
    # But wait, if we use random numbers, the function is non-deterministic.
    # That is allowed.
    
    # However, generating K*2000 normal randoms is fast.
    # samples = np.random.normal(loc=mu, scale=sigma, size=(K, len(mu)))
    # min_samples = np.min(samples, axis=0)
    # scores = -min_samples
    
    # Alternatively, a simpler and very effective strategy for minimization
    # is to use the Lower Confidence Bound (LCB) but with a dynamic beta
    # or simply stick to EI which is deterministic.
    
    # Let's look at the inspiration: "deterministic, multi-candidate portfolio search".
    # The current program is a hybrid EI+LCB.
    # Let's try a pure, well-tuned EI with a "lookahead" or "optimistic" bias.
    
    # Actually, a very strong baseline for these tasks is **Expected Improvement (EI)**
    # but computed with a **robust** z-score.
    
    # Let's try a different structural change: **Knowledge Gradient (KG)** approximation
    # or **Probability of Improvement (POI)**.
    
    # POI is very exploratory. EI is balanced.
    # Let's try **UCB** with a time-varying beta.
    
    # But the directive asks for a MATERIAL difference.
    # Thompson Sampling is structurally different from EI/LCB.
    
    # Let's implement Thompson Sampling with a single sample.
    # To ensure the scores are finite and well-behaved, we clip the samples.
    
    # Sample from N(mu, sigma)
    # Using numpy's random normal
    # We need to be careful about the global state of numpy random.
    # It's better to use a local generator if possible, but standard np.random is fine.
    
    # One issue: If we just return -sample, the score is in objective units.
    # This is comparable to EI which is also in objective units.
    
    # Let's use K=10 samples and take the 10th percentile (min) to encourage exploration.
    # Or simply the mean? No, mean is just -mu, which is greedy.
    # Min of K samples is a good exploration metric.
    
    # Let's use K=20 samples and take the minimum.
    # This is equivalent to an "optimistic" acquisition.
    
    # To make it deterministic for debugging (though not required), we could seed,
    # but the task doesn't forbid randomness.
    
    # Let's use a fixed number of samples K=10.
    K = 10
    # Vectorized sampling
    # samples shape: (K, N)
    # We can generate standard normals and scale/shift
    z_samples = np.random.normal(size=(K, len(mu)))
    samples = mu[None, :] + sigma[None, :] * z_samples
    
    # For minimization, we want the candidate that has the chance of being very low.
    # The minimum of the samples is a good estimate of the "best case" scenario
    # within the posterior uncertainty.
    # Score = -min_samples
    
    min_samples = np.min(samples, axis=0)
    scores = -min_samples
    
    # Ensure finite output
    if not np.all(np.isfinite(scores)):
        finite_scores = scores[np.isfinite(scores)]
        fill_val = np.max(finite_scores) if len(finite_scores) > 0 else 0.0
        scores = np.where(np.isfinite(scores), scores, fill_val)
        
    return scores
# EVOLVE-BLOCK-END
