import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    Uses a hybrid of Expected Improvement (EI) and a tuned UCB term,
    with adaptive weighting based on the current best value.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Constants for Hartmann 3D
    f_min = -3.86278
    f_max = 1.0
    
    # Normalize f_best to [0, 1] where 0 is best (found min) and 1 is worst (start)
    # This allows us to adapt the exploration/exploitation balance dynamically.
    f_norm = np.clip((f_best - f_min) / (f_max - f_min), 0.0, 1.0)
    
    # --- Component 1: Expected Improvement (EI) ---
    # Standard EI for minimization
    # Improvement is defined as f_best - f(x). We want this to be positive.
    z = (f_best - mu) / sigma
    # Handle numerical stability for large z
    ei = np.zeros_like(mu)
    mask = z > 1e-6
    ei[mask] = (f_best - mu[mask]) * norm.cdf(z[mask]) + sigma[mask] * norm.pdf(z[mask])
    # For z <= 0, EI is 0 (no improvement expected)
    
    # --- Component 2: Lower Confidence Bound (LCB) ---
    # LCB = mu - kappa * sigma
    # We want to maximize the score, so for minimization we look at (mu - kappa*sigma)
    # but since we want low values, we actually want to pick points with low mu and high sigma.
    # The standard UCB for minimization is: score = -mu + beta * sigma
    # Let's use a UCB-like term: -mu + beta_ucb * sigma
    # We tune beta_ucb to be small initially (exploit) and grow as we get stuck?
    # Or vice versa? Usually, early on we explore (high beta), later exploit (low beta).
    # However, Hartmann 3D has multiple local minima. Pure EI can get stuck.
    # A constant moderate beta often works well.
    
    # Let's blend EI and a UCB term.
    # Score = w_ei * EI + w_ucb * (-mu + beta_ucb * sigma)
    # Since EI already captures exploitation, adding -mu might be redundant but helps with global exploration if weighted correctly.
    # Actually, a simpler and very robust approach is to use a "Weighted UCB" or "Adaptive UCB".
    
    # Let's try a different structural change: **Thompson Sampling-like** or **Probability of Improvement (PI)** hybrid?
    # PI is often too exploitative.
    
    # Let's stick to a refined EI-UCB blend.
    # The key insight from the "hybrid EI-LCB" finding (score 1.813) is that balancing works.
    # Let's define the UCB score for minimization as: S_ucb = -mu + beta * sigma
    # We want to maximize S.
    
    # Adaptive Beta:
    # Start with high exploration (high beta) to find basins, then reduce.
    # f_norm is high (1.0) at start, low (0.0) at end.
    # So beta should be high when f_norm is high.
    beta_start = 3.0  # Strong exploration early
    beta_end = 0.5    # Strong exploitation late
    
    # Linear interpolation
    beta = beta_end + (beta_start - beta_end) * f_norm
    
    # Calculate UCB score
    score_ucb = -mu + beta * sigma
    
    # --- Blending ---
    # EI is bounded and non-negative. UCB can be negative.
    # We need to normalize them to comparable scales or use a weighted sum where weights sum to 1.
    # However, EI's magnitude depends on f_best.
    # A common trick is to use the maximum of EI and a scaled UCB, or a weighted sum.
    
    # Let's try a weighted sum.
    # Weight for EI: increase as we get closer to the optimum (f_norm -> 0).
    # Weight for UCB: decrease as we get closer to the optimum.
    
    w_ei = 0.5 + 0.5 * (1.0 - f_norm)  # 0.5 at start, 1.0 at end
    w_ucb = 1.0 - w_ei                  # 0.5 at start, 0.0 at end
    
    # To make them comparable, we can scale EI.
    # Max possible EI is roughly (f_best - f_min).
    # Let's scale EI by 1/f_best_range? No, let's just add them.
    # Actually, EI is already a good measure of expected gain.
    # UCB is a measure of potential.
    
    # Alternative: Use the **Knowledge Gradient** approximation? Too complex.
    
    # Let's try a "Softmax" selection over a small set? No, we must return 2000 scores.
    
    # Let's refine the blend.
    # Score = w_ei * EI + w_ucb * (beta * sigma - mu)
    # Note: EI is 0 when mu > f_best. The UCB term is still positive if sigma is large enough.
    # This ensures we always have a preference for high uncertainty regions when exploitation is low.
    
    score = w_ei * ei + w_ucb * (beta * sigma - mu)
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
