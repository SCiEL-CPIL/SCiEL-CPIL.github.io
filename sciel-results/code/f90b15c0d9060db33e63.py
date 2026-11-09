"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling with adaptive exploration weight.
    # We draw a single posterior predictive sample and blend it with an
    # uncertainty-driven exploration term. As optimization progresses
    # (f_best approaches 0 for Beale), we reduce exploration to exploit.
    
    n = len(mu)
    
    # Draw a single sample from the posterior predictive
    z = np.random.standard_normal(n)
    samples = mu + sigma * z
    
    # Adaptive exploration weight: start with strong exploration, decay as we
    # get closer to the optimum. Beale's minimum is 0, so we use f_best as
    # a proxy for progress.
    # When f_best is large (early), exploration_weight is high.
    # When f_best is small (late), exploration_weight is low.
    # Use a smooth decay: weight = 1 / (1 + f_best)
    # But f_best can be negative? No, Beale's minimum is 0 and values are >= 0.
    # Actually, Beale function minimum is 0 at (3, -0.5), and values are non-negative.
    # So f_best >= 0.
    
    # Clamp f_best to be non-negative for safety
    fb = max(f_best, 1e-8)
    
    # Exploration weight: higher when f_best is large (early in optimization)
    # Use a sigmoid-like decay. Scale: Beale values range roughly 0 to ~200+
    # at the boundaries. Let's use a scale of ~10 for the decay.
    exploration_weight = 1.0 / (1.0 + fb / 10.0)
    
    # Blend: the sample already includes exploration via sigma*z.
    # But we can add an extra uncertainty bonus to encourage exploration in
    # high-uncertainty regions, especially early on.
    # The sample is mu + sigma*z. For minimization, we want low samples.
    # We return -sample as score (higher = better).
    
    # Additional exploration: add exploration_weight * sigma to the score
    # This makes high-uncertainty points more attractive early on.
    scores = -(samples - exploration_weight * sigma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
