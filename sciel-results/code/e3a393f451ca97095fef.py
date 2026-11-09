"""
Batch-aware min-EI acquisition function.
Instead of scoring each candidate independently via standard EI, we approximate
the Expected Improvement of the minimum value across the entire candidate set.
This penalizes candidates that are close to other high-EI candidates (redundancy),
effectively spreading exploration and focusing on the most promising unique regions.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Safety handling for sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # 1. Compute standard EI for each candidate
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Approximate Min-EI
    # We want to estimate E[min_i (f(x_i) - f_best)]
    # For two points i and j, the probability that f(x_j) < f(x_i) is given by
    # P(f_j < f_i) = Phi( (mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2 - 2*rho_ij*sigma_i*sigma_j) )
    # Since we don't have the covariance matrix directly, we approximate the correlation
    # using a distance-based heuristic. However, a simpler and robust approximation
    # for "redundancy penalty" in batch EI is to assume that if another point has
    # higher EI and is "close" in prediction space, it will likely beat this one.
    
    # A computationally efficient approximation for min-EI in high dimensions
    # where we don't have the full covariance structure readily available in 
    # the function signature (only mu and sigma):
    # We can use the "Expected Minimum" approximation based on the distribution of EI values.
    # Specifically, if we assume the improvements are roughly independent or use a 
    # simple penalty for high EI neighbors, we can derive a score.
    
    # A more direct approach: 
    # The expected minimum of N correlated normals is lower than the minimum of 
    # independent ones. 
    # Let's use a penalty term that reduces the score of points that are 
    # "shadowed" by other points with significantly higher EI and similar sigma.
    
    # To do this efficiently with N=2000, O(N^2) is 4M ops, which is fast in NumPy.
    # Let's compute a pairwise "dominance" score.
    
    # For each candidate i, we estimate the probability that it is NOT the best 
    # among all candidates, based on EI values and uncertainty.
    # If candidate j has much higher EI and similar/lower sigma, it likely beats i.
    
    # Simplified Min-EI Score:
    # Score_i = EI_i * P(i is the min)
    # Approx P(i is min) by looking at how many other points have EI_j > EI_i 
    # and are "close" in the (mu, sigma) space.
    
    # Let's try a simpler, robust heuristic often used in batch acquisition:
    # "Expected Improvement of the best point" can be approximated by 
    # EI_i - lambda * (number of points with higher EI within a radius)
    
    # However, a mathematically grounded approximation for the *value* of the batch 
    # is often related to the maximum EI, but we need to select ONE point.
    # The point that maximizes the *expected minimum* of the batch is not necessarily 
    # the one with max EI if the batch is redundant.
    
    # Let's use the "Knowledge Gradient" style approximation or a simple 
    # "redundancy penalty".
    
    # Calculate pairwise distance in the (mu, sigma) space? 
    # Actually, the input `mu` and `sigma` are vectors of shape (2000,). 
    # They represent the GP mean and std at the 2000 candidate points.
    # We do NOT have the coordinates `x` to calculate physical distance.
    # We do NOT have the covariance matrix.
    
    # Without coordinates or covariance, we cannot accurately compute spatial redundancy 
    # based on physical proximity. We only have the predictive statistics.
    
    # If we only have mu and sigma, the "redundancy" must be inferred from the 
    # predictive values themselves. If two points have very similar mu and sigma, 
    # they are likely redundant in terms of the information they provide about 
    # the function value, assuming the GP is smooth.
    
    # Let's compute a penalty based on the density of points with higher EI.
    # If many points have EI > EI_i, the chance that i is the unique best is lower.
    
    # Sort by EI
    ei_sorted_idx = np.argsort(ei)[::-1]
    ei_sorted = ei[ei_sorted_idx]
    
    # For each point, how many points have higher EI?
    # This is just the rank.
    ranks = np.empty_like(ei)
    ranks[ei_sorted_idx] = np.arange(len(ei))
    
    # A simple penalty: Points with very high rank (low EI relative to others) 
    # are less likely to be the best. But this is just EI again.
    
    # Let's try a different angle: 
    # The task asks for a score for the NEXT evaluation.
    # If we treat the 2000 as a batch, and we pick the one that maximizes 
    # E[min(f(x_1), ..., f(x_2000))], this is a complex integral.
    
    # However, there is a known result: for a set of points, the point that 
    # maximizes the expected minimum is often one that balances high EI with 
    # being "far" from other high-EI points.
    
    # Since we lack coordinates, we can approximate "distance" by the difference 
    # in mu and sigma. Points with very similar (mu, sigma) are likely close 
    # in the input space if the GP is well-behaved (though not guaranteed).
    
    # Let's compute a "uniqueness" score.
    # For each point i, count how many points j have |mu_i - mu_j| < eps_mu 
    # and |sigma_i - sigma_j| < eps_sigma.
    # If the count is high, it's redundant.
    
    # But 2000x2000 comparison is fine.
    
    # Let's refine the strategy:
    # Score = EI * (1 / (1 + Redundancy))
    
    # Redundancy for point i = sum_j  exp(-((mu_i-mu_j)^2/sigma_mu^2 + (sigma_i-sigma_j)^2/sigma_sig^2))
    
    # This is expensive if done naively, but vectorized it's O(N^2).
    # 2000^2 = 4,000,000. This is very fast in NumPy (<10ms).
    
    # Parameters for the Gaussian kernel for redundancy
    # Scale of mu: f_best is around 0, mu varies. Let's use a scale relative to f_best or global mu std.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-6: mu_range = 1.0
    sigma_mu = mu_range / 5.0  # Points within 1/5 of range are considered close
    
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe)
    if sigma_range < 1e-6: sigma_range = 1.0
    sigma_sig = sigma_range / 5.0
    
    # Compute pairwise squared distances in (mu, sigma) space
    # mu_diff = mu[:, None] - mu[None, :]
    # sig_diff = sigma_safe[:, None] - sigma_safe[None, :]
    # dist_sq = (mu_diff / sigma_mu)**2 + (sig_diff / sigma_sig)**2
    # kernel = np.exp(-dist_sq)
    # redundancy = kernel.sum(axis=1) - 1.0  # Exclude self
    
    # This might be too aggressive. Let's use a simpler penalty.
    
    # Alternative: 
    # Use the "Expected Minimum" approximation via the Gumbel distribution?
    # No, let's stick to the redundancy penalty.
    
    # To save memory and time, we can subsample or use a simpler metric.
    # Actually, let's try a "Local EI" approach.
    
    # If we assume the GP is smooth, points with high EI are promising.
    # If many points have high EI, the "best" one is likely in that cluster.
    # We want to pick the point in that cluster that is most "central" or "representative"?
    # No, we want to pick the point that is most likely to be the global minimum
    # among the batch.
    
    # If I pick point i, the value I get is f(x_i).
    # The "batch" interpretation is tricky because we only evaluate ONE point.
    # The prompt says: "Treat the 2000 candidates as a single decision batch... 
    # approximating the Expected Improvement of the *minimum* value among all candidates... 
    # rather than scoring them independently."
    
    # This implies the score for candidate i should reflect its contribution to 
    # the minimum of the batch.
    # If candidate i is very likely to be the minimum of the batch, it gets a high score.
    
    # P(i is min) can be approximated by:
    # 1 / (1 + sum_j P(j < i))
    
    # P(j < i) = Phi( (mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2) )
    # (Assuming independence for the sake of approximation, which is conservative 
    # regarding redundancy - it overestimates the chance of being unique if they are correlated, 
    # but let's see).
    
    # Let's compute P(j < i) for all j.
    # This is O(N^2).
    
    # mu_diff = mu[:, None] - mu[None, :]  # (N, N)
    # sigma_sum_sq = sigma_safe[:, None]**2 + sigma_safe[None, :]**2
    # z_pair = mu_diff / np.sqrt(sigma_sum_sq + 1e-12)
    # p_j_beats_i = norm.cdf(z_pair)
    # # p_j_beats_i[i, j] is P(x_j < x_i)
    
    # # For each i, sum of probabilities that others beat it
    # # If many others beat it, it's not the min.
    # # Score_i = EI_i * (1 / (1 + sum_j P(j < i) * weight))
    
    # This is computationally intensive but doable.
    # 4M norm.cdf calls might be slow? 
    # norm.cdf is vectorized. 4M floats is ~32MB. Should be fine.
    
    # Let's implement this.
    
    # Optimization: We only need the top few EI candidates to really matter?
    # No, we need scores for all 2000.
    
    # Let's try a cheaper approximation first:
    # Sort by EI. 
    # For the top K candidates, compute the pairwise P(j < i).
    # For the rest, assume they are unlikely to be the min.
    
    # Actually, let's just do the full O(N^2) if it's fast enough.
    # 2000x2000 matrix.
    
    mu_col = mu[:, np.newaxis]
    mu_row = mu[np.newaxis, :]
    sigma_col = sigma_safe[:, np.newaxis]
    sigma_row = sigma_safe[np.newaxis, :]
    
    # z_ij = (mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2)
    # We want P(f_j < f_i) = P(f_j - f_i < 0)
    # f_j ~ N(mu_j, sigma_j^2), f_i ~ N(mu_i, sigma_i^2)
    # f_j - f_i ~ N(mu_j - mu_i, sigma_j^2 + sigma_i^2)
    # P(f_j - f_i < 0) = Phi( (0 - (mu_j - mu_i)) / sqrt(sigma_j^2 + sigma_i^2) )
    #                   = Phi( (mu_i - mu_j) / sqrt(sigma_j^2 + sigma_i^2) )
    
    # Let's compute the denominator
    denom = np.sqrt(sigma_col**2 + sigma_row**2 + 1e-12)
    
    # Numerator
    numer = mu_col - mu_row
    
    z_matrix = numer / denom
    
    # p_beat = norm.cdf(z_matrix)
    # p_beat[i, j] = P(f_j < f_i)
    
    # We want to penalize i if many j have high P(f_j < f_i).
    # Sum of p_beat[i, j] for j != i.
    
    # To avoid memory issues with 4M floats (32MB is fine), let's proceed.
    # However, norm.cdf on 4M elements might take ~50ms. Acceptable.
    
    p_beat = norm.cdf(z_matrix)
    
    # Mask diagonal (self)
    np.fill_diagonal(p_beat, 0.0)
    
    # Sum probabilities that other points beat i
    # If this sum is high, i is unlikely to be the minimum.
    # Let's use a soft penalty.
    # score = EI * exp(-sum_p_beat / alpha)
    
    # What is a good alpha?
    # If there are 10 points with P=0.9, sum=9. exp(-9) is tiny.
    # If there is 1 point with P=0.5, sum=0.5. exp(-0.5) ~ 0.6.
    
    # Let's use alpha = 5.0
    alpha = 5.0
    redundancy_penalty = np.exp(-p_beat.sum(axis=1) / alpha)
    
    scores = ei * redundancy_penalty
    
    # Safety: Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time Guard
    elapsed = time.time() - start_time
    if elapsed > 10.0:
        # Fallback to simple EI
        z_fallback = (f_best - mu) / sigma_safe
        ei_fallback = (f_best - mu) * norm.cdf(z_fallback) + sigma_safe * norm.pdf(z_fallback)
        return ei_fallback
        
    return scores
# EVOLVE-BLOCK-END
