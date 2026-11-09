"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _matern52_kernel(X, Y, length_scale=0.2):
    """
    Compute the Matern-5/2 kernel between points in X and Y.
    X and Y are assumed to be in [0, 1]^1 (scaled coordinates).
    Returns a matrix of kernel values.
    """
    # X shape: (n, 1), Y shape: (m, 1)
    # Compute pairwise squared distances
    # Using broadcasting: (n, 1, 1) - (1, m, 1) -> (n, m, 1)
    diff = X[:, :, np.newaxis] - Y[:, :, np.newaxis]
    sq_dist = np.sum(diff ** 2, axis=-1)
    
    # sqrt distance
    dist = np.sqrt(sq_dist + 1e-12)
    
    # Matern-5/2 kernel: (1 + sqrt(3)*d/l + 3*d^2/l^2) * exp(-sqrt(3)*d/l)
    scaled_dist = dist / length_scale
    sqrt3 = np.sqrt(3.0)
    
    kernel = (1.0 + sqrt3 * scaled_dist + 3.0 * scaled_dist ** 2) * np.exp(-sqrt3 * scaled_dist)
    
    return kernel


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a diversity-penalized Expected Improvement.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin well within 300s limit
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time
    if time.time() - start_time > deadline * 0.5:
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Identify top candidates for diversity penalty
    # We want to penalize points that are close to other high-EI points.
    # To be efficient, we only look at the top K candidates.
    n_candidates = len(mu)
    
    # Heuristic for K: larger in early stages (more exploration needed), smaller later.
    # But since we don't know the step count directly, we can use a fixed reasonable K.
    # 2000 candidates. Top 100 seems reasonable to capture the "front" of the distribution.
    K = min(100, n_candidates)
    
    if K < 2:
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Get indices of top K candidates by EI score
    top_k_indices = np.argsort(ei)[-K:]
    
    # We need coordinates to compute kernel distances.
    # The problem states: "length scale 0.2 on coordinates scaled to [0, 1]"
    # "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This is a critical constraint: "No coordinates are passed."
    # Therefore, we CANNOT compute the Matern kernel directly because we don't have x values!
    
    # Wait, let me re-read the contract carefully.
    # "mu and sigma have shape (2000,). Return 2000 finite scores, higher meaning evaluate next; anything else makes the whole result invalid. No coordinates are passed."
    
    # If coordinates are NOT passed, how can we compute a diversity penalty based on spatial distance?
    # We cannot. The inspiration strategy "compute the maximum pairwise correlation among the top-K candidates based on the Matern-5/2 kernel" is IMPOSSIBLE to implement directly if x is not provided.
    
    # However, maybe we can infer coordinates? No, that's not robust.
    # Maybe the "diversity" can be approximated by the spread of sigma or mu?
    # Or perhaps the "batch-aware" aspect implies we should look at the distribution of EI scores themselves?
    
    # Let's look at the inspirations again.
    # Inspiration 1: Combined LCB and EI. Score 2.9249. This is the strongest.
    # Inspiration 2: Adaptive LCB/EI. Score 1.7066.
    # Inspiration 3: UCB/EI. Score 1.5090.
    
    # The current program is pure EI (score 1.7256).
    # The best inspiration is the LCB/EI mix (2.9249).
    
    # The directive says: "Replace the point-wise Expected Improvement with a batch-aware, diversity-penalized acquisition... thereby explicitly discouraging redundant evaluations... forcing the search to explore distinct basins".
    
    # Since we don't have coordinates, we cannot do spatial diversity.
    # BUT, we can do "functional" diversity or use the GP uncertainty (sigma) as a proxy for distance?
    # Points with high sigma are "far" from observed data in the GP sense.
    # Points with low sigma are "close" to observed data.
    
    # A common trick when coordinates aren't available for diversity:
    # Use the normalized EI score itself to identify "similar" candidates? No, that's circular.
    
    # Let's reconsider the "No coordinates are passed" constraint.
    # Is it possible the evaluator passes coordinates in a global variable or something?
    # "Keep the function names and signatures of the seed program."
    # Signature: `acquisition(mu, sigma, f_best)`.
    
    # If I can't access x, I can't compute the Matern kernel between candidates.
    # Therefore, the specific strategy "compute maximum pairwise correlation... based on Matern-5/2" is not implementable as described.
    
    # I must adapt the strategy.
    # Structural change: Instead of spatial diversity, use **uncertainty-based diversity**.
    # In GP, high sigma implies the point is in an unexplored region.
    # If we pick a point with high EI, it might be in a region with moderate sigma.
    # If we pick another point with high EI, it might be in a *different* region with high sigma.
    #
    # Actually, the most effective "diversity" without coordinates is often just **LCB** (Lower Confidence Bound).
    # LCB = mu - beta * sigma.
    # This explicitly rewards high sigma (exploration) and low mu (exploitation).
    # The inspiration with score 2.9249 used exactly this: a mix of EI and LCB.
    #
    # The directive asks for a "STRUCTURAL change" and specifically mentions "diversity-penalized".
    # Since spatial diversity is blocked by the API, I will implement a **probabilistic diversity** heuristic.
    #
    # Alternative interpretation:
    # Maybe I can estimate the "distance" between candidates by looking at the change in mu?
    # No, mu is the mean prediction.
    #
    # Let's stick to the strongest empirical evidence: **Mixing EI and LCB**.
    # The inspiration 1 (2.9249) is significantly better than current (1.7256).
    # The directive says "Combine the strategy of the current program with the strongest ideas of the inspirations".
    # Current: EI.
    # Strongest Inspiration: EI + LCB mix.
    #
    # How to make it "batch-aware" or "diversity-penalized" without coordinates?
    # We can penalize candidates that have similar *predicted values* (mu) and *uncertainties* (sigma) to the top candidate.
    # This is a "feature-space" diversity penalty.
    #
    # Algorithm:
    # 1. Compute EI.
    # 2. Compute LCB.
    # 3. Identify the top candidate (argmax of combined score).
    # 4. For other candidates, calculate a "similarity" to the top candidate based on (mu, sigma).
    # 5. Subtract a penalty proportional to this similarity.
    #
    # However, this might be over-engineering and slow.
    # Let's look at the time budget. 330s total. 25 evals. ~13s per eval.
    # We have plenty of time for O(N^2) operations if N=2000? 2000^2 = 4,000,000.
    # In Python/NumPy, a 2000x2000 matrix operation is fast (<0.1s).
    #
    # Let's try to implement a diversity penalty based on the **GP posterior features**.
    # We treat (mu, sigma) as a 2D feature vector for each candidate.
    # We compute the pairwise Euclidean distance in this (mu, sigma) space.
    # Candidates that are "close" in (mu, sigma) space are considered redundant.
    #
    # Steps:
    # 1. Compute base scores (EI or LCB).
    # 2. Select top K candidates by base score.
    # 3. For the top K, compute pairwise distances in (mu, sigma) space.
    # 4. For each candidate in the top K, compute the minimum distance to any *other* candidate in the top K.
    # 5. The "diversity" of a candidate is this minimum distance.
    # 6. Final Score = Base Score - lambda * (1 / (1 + min_dist))
    #    Or better: Final Score = Base Score * (1 - penalty)
    #
    # But wait, if I penalize points close to the best point, I might penalize the best point itself?
    # Usually, we want to keep the best point, and penalize the *runners-up* that are too close to the best.
    #
    # Let's refine:
    # 1. Compute EI.
    # 2. Compute LCB.
    # 3. Combine: S = w1 * EI_norm + w2 * LCB_norm.
    # 4. Identify top K (e.g., 50) by S.
    # 5. Among these top K, compute pairwise distances in (mu, sigma) space.
    # 6. For each candidate i in top K, find the nearest neighbor j != i.
    # 7. Penalty_i = max(0, 1 - dist(i, j) / threshold).
    # 8. Adjusted_S_i = S_i * (1 - alpha * Penalty_i).
    # 9. For candidates not in top K, keep original S.
    #
    # This encourages picking distinct points from the top cluster.
    
    # Let's implement this.
    
    # First, compute EI and LCB
    lcb = mu - 2.0 * sigma_safe  # Beta = 2.0 from inspiration 1
    
    # Normalize EI and LCB to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    # For minimization, lower LCB is better. So we flip LCB for normalization.
    lcb_norm = (lcb_max - lcb) / lcb_range
    
    # Combine EI and LCB
    # We need to balance exploration (LCB) and exploitation (EI).
    # Early on, we want more exploration. Later, more exploitation.
    # Heuristic for "early" vs "late":
    # If sigma_mean is high relative to f_best, we are early.
    sigma_mean = np.mean(sigma_safe)
    f_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = np.clip(sigma_mean / f_scale, 0.0, 2.0)
    
    # Weight for LCB increases with uncertainty
    w_lcb = np.clip(0.3 + 0.4 * uncertainty_ratio, 0.2, 0.7)
    w_ei = 1.0 - w_lcb
    
    base_scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Scale back to a meaningful magnitude?
    # The contract says "higher meaning evaluate next". The absolute value doesn't matter, only the ranking.
    # So we can return the normalized scores directly, or scaled.
    # Let's stick to the normalized [0, 1] range for stability, or scale by ei_range.
    # Inspiration 1 scaled back. Let's do that.
    base_scores_scaled = base_scores * ei_range + ei_min
    
    # Now, apply diversity penalty to the top K.
    K = 50
    if K > n_candidates:
        K = n_candidates
    
    if K < 2:
        return np.where(np.isfinite(base_scores_scaled), base_scores_scaled, 0.0)
    
    # Get indices of top K
    top_k_indices = np.argsort(base_scores_scaled)[-K:]
    
    # Extract mu and sigma for these top K
    mu_top = mu[top_k_indices]
    sigma_top = sigma_safe[top_k_indices]
    
    # Normalize mu and sigma for distance calculation to avoid scale dominance
    mu_mean = np.mean(mu_top)
    mu_std = np.std(mu_top) + 1e-9
    sigma_mean_top = np.mean(sigma_top)
    sigma_std_top = np.std(sigma_top) + 1e-9
    
    mu_norm = (mu_top - mu_mean) / mu_std
    sigma_norm = (sigma_top - sigma_mean_top) / sigma_std_top
    
    # Stack into feature matrix (K, 2)
    features = np.column_stack((mu_norm, sigma_norm))
    
    # Compute pairwise Euclidean distances
    # dist[i, j] = ||features[i] - features[j]||
    # Use broadcasting
    diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # For each i, find the minimum distance to any j != i
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(dist, np.inf)
    min_dists = np.min(dist, axis=1)
    
    # Check time
    if time.time() - start_time > deadline * 0.8:
        return np.where(np.isfinite(base_scores_scaled), base_scores_scaled, 0.0)
    
    # Penalty: Points with small min_dist are penalized.
    # We want to spread out the top K.
    # Penalty = 1 / (1 + min_dist)
    # If min_dist is 0 (duplicate), penalty is 1.
    # If min_dist is large, penalty is ~0.
    
    # We need to map this penalty back to the original indices.
    # Initialize final scores with base scores
    final_scores = base_scores_scaled.copy()
    
    # Apply penalty to top K
    # We want to reduce the score of candidates that are too close to others in the top K.
    # But we should probably keep the absolute best candidate (index 0 in sorted top_k) high?
    # Actually, if we penalize the best candidate, we might pick a slightly worse but distinct point.
    # This is the goal of diversity: trade off a little quality for more coverage.
    
    # Let's apply the penalty.
    # alpha controls strength.
    alpha = 0.5
    
    for i, idx in enumerate(top_k_indices):
        if min_dists[i] < np.inf:
            penalty = 1.0 / (1.0 + min_dists[i])
            # Reduce score
            final_scores[idx] *= (1.0 - alpha * penalty)
    
    # Ensure finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
