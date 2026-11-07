# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale function.
    Uses a hybrid approach: UCB for early exploration, transitioning to
    a modified EI that accounts for the known global minimum being 0.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound
    # Beta is a temperature parameter; we use a moderate value
    beta = 1.5
    ucb = mu + beta * sigma
    
    # Since we're minimizing, we want high scores for promising points.
    # EI is already oriented for minimization (higher EI = better).
    # UCB for minimization would be: -mu + beta * sigma
    # Let's combine them.
    
    # A simple weighted sum might not be optimal. 
    # Let's try a "Knowledge Gradient" approximation or just a robust EI variant.
    
    # Actually, for the Beale function which has a deep, narrow valley,
    # pure EI can get stuck. UCB helps escape.
    # Let's use a dynamic beta that decreases as we get closer to the minimum?
    # Or simply use a UCB that is competitive.
    
    # Let's try: score = max(ei, -mu + beta * sigma) ? No, scales differ.
    
    # Let's stick to a well-tuned EI but with a noise floor to encourage exploration
    # in high-uncertainty regions even if mu is good.
    
    # Alternative: Use the "Expected Hypervolume Improvement" intuition? Too complex.
    
    # Let's try a variant that penalizes high sigma less when f_best is close to 0.
    # If f_best is very small, EI becomes very conservative.
    
    # Let's use a UCB-style acquisition for minimization:
    # Score = -mu + beta * sigma
    
    # But EI is generally superior for convergence.
    # Let's blend them based on the number of points? We don't have N here.
    
    # Let's try a different angle: The Beale function is known to have its minimum
    # at (3, 0.5) with value 0.
    # Wait, the problem says "minimum value 0".
    # f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2 + 10
    # At (3, 0.5): (1.5 - 6 + 9)(3 - 1.5 + 0.25)^2 + 10 = 4.5 * 1.5625 + 10 = 7.03 + 10 = 17.03?
    # Let me check the Beale function definition.
    # Usually: f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2 + 10
    # Min is at (3, 0.5) with value 1.
    # Wait, the prompt says "minimum value 0".
    # Perhaps the function is shifted? Or I am misremembering.
    # If the minimum is 0, then f_best will approach 0.
    
    # If f_best is close to 0, EI(z) where z = (0 - mu)/sigma.
    # If mu < 0 (impossible if min is 0 and GP is accurate, but noise might allow it),
    # EI handles it.
    
    # Let's try a UCB with a beta that adapts to the scale of f_best.
    # If f_best is large, explore more. If f_best is small, exploit.
    # But we don't know the scale of the function other than f_best.
    
    # Let's try a simple, robust UCB.
    # The Beale function is smooth but has a narrow valley.
    # UCB with beta=2.0 often works well for such landscapes.
    
    # Let's compute both and take a convex combination.
    # Normalize EI and UCB?
    
    # Let's just use UCB. It was measured to improve score in the prompt's knowledge.
    # "UCB explicitly balances exploitation and exploration... Measured: 1 -> 1.093."
    
    # So let's implement UCB.
    # For minimization, we want to minimize mu - beta * sigma? No.
    # We want to find low values.
    # High score = good candidate.
    # Good candidate = low mu (exploit) or high sigma (explore).
    # So Score = -mu + beta * sigma.
    
    beta = 2.0
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
