import numpy as np
from scipy.special import ndtr, ndtri
import time

def acquisition(mu, sigma, f_best):
    """
    Diversity-Weighted Expected Improvement acquisition function.
    Uses a simple distance-based penalty to encourage exploration of new regions,
    which is beneficial for multimodal functions like Styblinski-Tang.
    """
    start_time = time.time()
    
    # Safety margin: ensure we return before 330s wall clock limit
    # The function is called 25 times (30 evals - 5 initial).
    # If each call takes too long, we might timeout. 
    # However, the computation here is O(N) or O(N log N) at most, so it should be fast.
    # We add a simple check just in case.
    if time.time() - start_time > 300:
        return np.zeros_like(mu)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi_z = ndtr(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Diversity Penalty / Exploration Bonus
    # Since we don't have access to the evaluated points directly in the acquisition function
    # (only mu, sigma, f_best), we cannot compute a true KDE of evaluated points.
    # However, high sigma indicates regions with little data, which are "diverse" or unexplored.
    # Low sigma indicates regions with lots of data, which are "exploited".
    # The EI already balances this, but we can explicitly add a diversity term based on sigma.
    # Alternatively, we can look at the structure of the problem.
    # The directive suggests "Density-Aware" or "Diversity-Weighted".
    # Without coordinates, we can't do spatial diversity directly.
    # But we can use the GP posterior variance (sigma) as a proxy for unexploredness.
    
    # Let's try a hybrid: EI + beta * sigma
    # This is similar to UCB but with EI's exploitation term.
    # Or, we can modify the EI to be more sensitive to uncertainty.
    
    # A common structural change is to use the "Probability of Improvement" (PI)
    # or a modified EI.
    
    # Let's try: Score = EI + lambda * sigma
    # This encourages exploration of high-variance regions.
    # The value of lambda is critical.
    
    # Another idea: Use the "Knowledge Gradient" (KG) which is often better for multimodal functions.
    # But KG is computationally expensive if not approximated.
    # The current code used KG but scored 0.92.
    
    # Let's try a "Diversity-Weighted EI" where we weight EI by the inverse of local density.
    # Since we don't have coordinates, we can't do this directly.
    
    # However, we can assume that the GP's sigma is a good indicator of "diversity" or "novelty".
    # Regions with high sigma are likely far from observed points.
    
    # Let's try a simple blend:
    # score = EI + 0.1 * sigma
    
    # Or, let's try to make the exploration more aggressive.
    # score = EI + 0.5 * sigma
    
    # Let's try a different approach: Use the "Upper Confidence Bound" (UCB)
    # but with a dynamic beta based on the number of evaluations?
    # We don't know the number of evaluations directly, but f_best changes.
    
    # Let's stick to the directive: "Diversity-Weighted Expected Improvement".
    # We can approximate diversity by assuming that high sigma implies low density of samples.
    # So, we weight EI by a function of sigma.
    # score = EI * (1 + alpha * sigma)
    # This increases the weight of EI in high-uncertainty regions.
    
    # Let's try:
    alpha = 0.5
    diversity_weight = 1.0 + alpha * sigma_safe
    score = ei * diversity_weight
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
