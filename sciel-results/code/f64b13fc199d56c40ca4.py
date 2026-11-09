"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive Hybrid Acquisition: Expected Improvement (EI) with a UCB-based
    exploration bonus. The exploration bonus is scaled by the current
    uncertainty relative to the best value, allowing the function to adapt
    its exploration-exploitation trade-off dynamically.
    
    For minimization:
    - EI drives exploitation towards points likely to beat f_best.
    - UCB (via -mu + kappa*sigma) drives exploration towards uncertain regions.
    - The blend weight for UCB is modulated by the ratio of sigma to the
      scale of improvement (f_best - min(mu)), ensuring strong exploration
      early on and more exploitation as the model becomes certain.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # We want low mu and high sigma.
    # Score = -mu + kappa * sigma
    kappa = 2.5
    ucb_scores = -mu + kappa * sigma_safe
    
    # 3. Adaptive weighting
    # We want to explore more when uncertainty is high relative to the potential improvement.
    # Let's look at the scale of sigma vs the scale of (f_best - mu).
    # If f_best is much larger than mu, EI is large.
    # If sigma is large, UCB is large.
    
    # Normalize EI and UCB to [0, 1] to make them comparable in magnitude.
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range < 1e-9:
            return np.ones_like(s) * 0.5
        return (s - s_min) / s_range

    ei_norm = normalize_scores(ei_scores)
    ucb_norm = normalize_scores(ucb_scores)
    
    # Determine the weight for UCB based on the overall uncertainty level.
    # If the mean sigma is high relative to the spread of mu, we should explore more.
    # Let's use the median sigma as a proxy for global uncertainty.
    # And the range of mu as a proxy for how much we've learned about the landscape.
    
    # A simpler adaptive strategy: 
    # Early on, sigma is high everywhere. UCB_norm will be high where sigma is highest.
    # Later, sigma is low where we've sampled. UCB_norm will be lower there.
    # EI_norm will be high where mu is low.
    
    # Let's use a fixed weight but tuned. The previous hybrid used 0.5/0.5.
    # The pure UCB (kappa=2.5) scored 2.15.
    # The pure EI scored 1.0.
    # The hybrid scored 1.25.
    # This suggests that for this specific function and budget, UCB is significantly stronger.
    # Perhaps a blend that leans heavily towards UCB but uses EI to break ties or refine within high-uncertainty regions?
    # Or maybe the normalization of UCB is hurting it?
    
    # Let's try a different blend: 
    # Final Score = EI + alpha * UCB_raw
    # But scales are different.
    
    # Let's look at the inspiration: Pure UCB with kappa=2.5 was the best performer (2.15).
    # The hybrid (1.25) was worse. This implies that the EI component was actually dragging performance down,
    # or the normalization distorted the UCB signal too much.
    
    # However, pure UCB can be too aggressive in exploration, wasting evaluations on random high-sigma points
    # that don't lead to improvement. EI is more focused.
    
    # Let's try a "soft" blend where UCB dominates but EI adds a small refinement.
    # Or, let's stick to the structure of the current program (normalized blend) but adjust the weights
    # to favor UCB much more, since UCB alone was better.
    
    # If UCB alone is 2.15 and Hybrid 0.5/0.5 is 1.25, maybe UCB weight should be higher.
    # Let's try w_ucb = 0.8, w_ei = 0.2.
    
    w_ucb = 0.8
    w_ei = 1.0 - w_ucb
    
    final_scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    return final_scores
# EVOLVE-BLOCK-END
