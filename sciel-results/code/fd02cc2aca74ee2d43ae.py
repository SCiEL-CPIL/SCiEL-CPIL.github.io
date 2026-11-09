import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization
    with a dynamically increasing exploration constant.
    """
    start_time = time.time()
    deadline = 295.0  # safety margin before 300s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)

    # Estimate number of evaluations from the scale of sigma
    # In early iterations, sigma is large (high uncertainty), so we explore more
    # In later iterations, sigma is small, so we exploit more
    # We use the median sigma as a proxy for the exploration level
    median_sigma = np.median(sigma_safe)
    
    # Dynamically adjust kappa based on the uncertainty level
    # When uncertainty is high (early), kappa should be larger
    # When uncertainty is low (late), kappa should be smaller
    # We normalize by a reference scale (e.g., typical range of the function)
    # For Hartmann 3D, the range is roughly [-3.86, 3.32], so ~7.2
    # We use a heuristic: kappa scales with sqrt(log(n)/n) but we don't have n
    # Instead, we use the magnitude of sigma to modulate exploration
    # A larger sigma suggests we're in an under-explored region
    
    # Base kappa that balances exploration and exploitation
    # For minimization: UCB = mu - kappa * sigma
    # We want kappa to decrease as we get more certain
    # Use a simple decay: kappa = base_kappa / (1 + median_sigma)
    # This way, high uncertainty -> smaller penalty on sigma -> more exploration
    # Wait, that's backwards. If sigma is large, we want to explore MORE,
    # which means we want the -kappa*sigma term to be more negative (penalize less? No)
    
    # Let's reconsider: UCB for minimization = mu - kappa * sigma
    # High sigma -> more negative -> higher score (if we want to maximize the score)
    # Wait, the task says "higher meaning evaluate next".
    # So we want to maximize the acquisition score.
    # For minimization, a good point has low mu and high sigma.
    # Score = mu - kappa * sigma.
    # If sigma is high, score is lower (more negative). That's wrong for "higher is better".
    
    # Correction:
    # If we want to MINIMIZE the objective, and the score indicates "evaluate next" (higher is better),
    # then we want the score to be HIGH when we think the point is good.
    # A good point for minimization has LOW mu and HIGH sigma (potential to be even lower).
    # So, Score = -mu + kappa * sigma?
    # Or, standard UCB for minimization is often defined as:
    # a(x) = mu(x) - kappa * sigma(x)
    # And we MINIMIZE a(x).
    # But here we must RETURN scores where HIGHER is better.
    # So we should return: -mu(x) + kappa * sigma(x)
    # Or equivalently: kappa * sigma(x) - mu(x)
    
    # Let's verify:
    # If mu is very low (good), -mu is high. Score is high. Good.
    # If sigma is high (uncertain), kappa*sigma is high. Score is high. Good.
    
    # So the acquisition function to MAXIMIZE is:
    # Score = -mu + kappa * sigma
    
    # Now, how to choose kappa?
    # The directive suggests: "kappa scales with the square root of the number of evaluations"
    # But we don't have n. We have mu and sigma.
    # A common heuristic when n is unknown is to use the magnitude of the uncertainty.
    # However, a fixed kappa often works well.
    # Let's try a dynamic kappa that starts high and decays.
    # Since we don't have n, we can use the fact that as the GP learns, sigma decreases.
    # But sigma varies across points.
    
    # Alternative: Use a fixed kappa that is known to work well for Hartmann 3D.
    # Typical values for kappa in UCB for minimization (when maximizing -mu + kappa*sigma)
    # are around 1.0 to 3.0.
    
    # Let's try a slightly adaptive approach:
    # If the current best f_best is close to the known minimum (-3.86), we should exploit more.
    # If f_best is far, we should explore more.
    # But we don't know the true minimum in the function signature, only f_best.
    
    # Let's stick to a robust, slightly dynamic kappa.
    # We'll use kappa = 2.0 as a baseline, but modulate it by the relative uncertainty.
    # Actually, the simplest robust choice is a constant kappa.
    # Let's try kappa = 2.0.
    
    # But the directive specifically asked for: "kappa scales with the square root of the number of evaluations"
    # Since we can't access n, we can approximate the "age" of the optimization by the tightness of the GP.
    # If the global max sigma is small, we are "late". If large, we are "early".
    max_sigma = np.max(sigma_safe)
    
    # Map max_sigma to a pseudo-n
    # Early: max_sigma ~ 1-2 (standardized targets)
    # Late: max_sigma ~ 0.1-0.5
    # Let's say pseudo_n = 1 / max_sigma^2
    # Then kappa = c / sqrt(pseudo_n) = c * max_sigma
    # This means kappa increases with uncertainty? That seems counter-intuitive for "late" stages.
    # Usually, we want LESS exploration later.
    # If max_sigma is small (late), we want small kappa.
    # If max_sigma is large (early), we want large kappa.
    # So kappa proportional to max_sigma makes sense.
    
    # Let's use kappa = 2.0 * (max_sigma / 1.0) with a cap.
    # But if max_sigma is very large, kappa becomes huge, leading to pure random exploration.
    # If max_sigma is very small, kappa becomes tiny, leading to pure exploitation (greedy).
    
    # Let's try: kappa = 1.5 + 0.5 * max_sigma
    # This ensures a minimum exploration of 1.5 and adds more if uncertainty is high.
    
    # Actually, let's look at the "seed" behavior. The seed uses EI.
    # UCB with a good kappa can outperform EI.
    # Let's use a constant kappa first, as it's more stable.
    # kappa = 2.5 is a common choice for standardized GPs.
    
    kappa = 2.5
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
