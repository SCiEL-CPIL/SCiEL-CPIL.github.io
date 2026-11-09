"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 290.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Strategy: Stochastic local refinement.
    1. Compute EI to find the best "anchor" index.
    2. Generate 2000 candidate indices: 
       - 90% are local jitter around the anchor (Laplace-like distribution in index space).
       - 10% are uniform global (to maintain some exploration).
    3. Score these candidates with a high value (proportional to EI at anchor + small noise)
       and the rest with -inf.
    This forces the evaluator (which picks the argmax) to select from the local neighborhood,
    effectively performing a local search around the most promising region.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu

    n = len(mu)
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute EI to find the anchor
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure finite EI
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    # Find the best anchor
    anchor_idx = np.argmax(ei)
    anchor_ei = ei[anchor_idx]
    
    # If anchor_ei is -inf or very small, fall back to pure exploitation (-mu)
    if not np.isfinite(anchor_ei) or anchor_ei < 1e-6:
        # Exploit best mu
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    # 2. Generate candidate indices
    # We want 2000 indices.
    # 90% local, 10% global.
    n_local = int(0.9 * n)
    n_global = n - n_local
    
    # Local jitter: Laplace distribution around anchor_idx
    # Scale of jitter: relative to the domain size.
    # Since we don't know the exact spacing, we assume uniform grid.
    # A reasonable jitter scale is ~1% of the domain width in index space.
    # n=2000, so 1% is ~20 indices.
    jitter_scale = max(1.0, n * 0.01)
    
    # Generate local indices
    # Laplace noise
    local_noise = np.random.laplace(0, jitter_scale, size=n_local)
    local_indices = (anchor_idx + local_noise).astype(int)
    
    # Wrap around or clip? Clipping is safer to stay in bounds.
    # But wrapping might help if the function is periodic (Weierstrass isn't strictly, but domain is bounded).
    # Let's clip to [0, n-1].
    local_indices = np.clip(local_indices, 0, n - 1)
    
    # Global indices
    global_indices = np.random.randint(0, n, size=n_global)
    
    # Combine
    selected_indices = np.concatenate([local_indices, global_indices])
    
    # 3. Assign scores
    # We want the selected indices to have high scores.
    # To ensure the local ones are preferred over global ones (mostly),
    # we give them a score based on the anchor EI plus a small random perturbation
    # to break ties and allow some diversity within the local set if multiple are picked?
    # Wait, the evaluator picks ONLY ONE candidate (the argmax).
    # So we just need ONE index to have the highest score.
    # If we assign high scores to 2000 indices, the argmax will be one of them.
    # But we want it to be one of the LOCAL ones preferentially.
    
    # Strategy:
    # Assign the anchor EI value to the local indices.
    # Assign a slightly lower value (or random lower) to global indices.
    # But we need to return 2000 scores.
    
    scores = np.full(n, -np.inf)
    
    # For local indices, assign a score that is high.
    # To make the argmax likely to be a local one, we can assign the same high score to all local ones?
    # If they all have the same score, np.argmax picks the first one.
    # That would always pick the same local index (the first one generated).
    # We want to explore the local neighborhood.
    # So we should assign slightly different scores to each local index.
    # How?
    # We can use the EI value at the anchor, but add a small random noise to each local candidate's score.
    # This way, the argmax will be the local candidate with the highest noise, effectively sampling from the local set.
    
    # However, the "score" returned is just a number. The evaluator picks argmax.
    # If I assign:
    # scores[local_indices[i]] = anchor_ei + noise[i]
    # scores[global_indices[j]] = anchor_ei * 0.5 + noise_global[j]
    
    # Then argmax will likely be a local one.
    # And among local ones, it will be the one with the highest noise.
    # This effectively samples a random point from the local neighborhood.
    
    # Let's do that.
    # Noise scale: small compared to EI differences?
    # EI values can vary. Let's just use a small constant noise.
    noise_scale = 1e-3 * max(1.0, abs(anchor_ei))
    
    local_noise_scores = np.random.normal(0, noise_scale, size=n_local)
    global_noise_scores = np.random.normal(0, noise_scale, size=n_global)
    
    # Base scores
    local_base = anchor_ei
    global_base = anchor_ei * 0.5  # Lower priority for global
    
    # Assign
    # Note: indices might have duplicates. If so, the last assignment wins.
    # To avoid this, we could use unique indices, but duplicates in local jitter are unlikely if scale is small.
    # If duplicates occur, it just means one candidate is represented twice, which is fine.
    
    scores[local_indices] = local_base + local_noise_scores
    scores[global_indices] = global_base + global_noise_scores
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
