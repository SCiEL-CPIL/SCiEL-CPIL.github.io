# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of:
    1. Expected Improvement (EI) per unit cost (baseline)
    2. Lower Confidence Bound (LCB) per unit cost
    3. Probability of Improvement (PI) as a tie-breaker/explorer
    
    The final score is a weighted blend that balances exploitation (EI) 
    with exploration (LCB/PI), all normalized by cost.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We want to maximize -LCB = -mu + beta * sigma
    # A common choice for beta is 2.0 or based on exploration-exploitation tradeoff.
    # Since we don't have n (number of observations) directly, we use a fixed beta.
    # However, to make it adaptive, we could use a heuristic. 
    # Let's use beta = 2.0 for a strong exploration term.
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Probability of Improvement (PI)
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Normalize scores to combine them meaningfully.
    # EI and LCB are in objective units. PI is in [0, 1].
    # We need to scale them to be comparable.
    
    # Heuristic scaling:
    # EI is typically small when close to optimum.
    # LCB can be large in magnitude in unexplored regions.
    # PI is a probability.
    
    # Let's create a composite score.
    # A common approach is to use EI/cost as the main driver, but add an exploration bonus.
    # Or, use a weighted sum of normalized components.
    
    # Let's try a different approach: Use the "Knowledge Gradient" idea or a modified EI.
    # Actually, a robust cost-aware metric is often:
    # Score = EI / cost + lambda * (Exploration Term) / cost
    
    # Let's define an exploration term that is high when sigma is high relative to the range of mu.
    # But we don't know the range of mu easily without more context.
    
    # Alternative: Use the "Expected Improvement" but with a penalty for high cost,
    # and add a term that encourages exploring high-uncertainty regions.
    
    # Let's stick to a simple, robust combination:
    # Score = (w1 * EI + w2 * LCB_score) / cost
    # But EI and LCB_score have different scales.
    
    # Better: Normalize EI and LCB_score by the standard deviation of the observations?
    # We don't have the observations directly, only mu and sigma.
    
    # Let's use the fact that z is a standardized variable.
    # EI = sigma * (z * Phi(z) + phi(z))
    # LCB_score = sigma * (-z + beta)
    
    # So both are proportional to sigma.
    # EI = sigma * ei_factor(z)
    # LCB_score = sigma * lcb_factor(z)
    
    # Where:
    # ei_factor(z) = z * Phi(z) + phi(z)
    # lcb_factor(z) = -z + beta
    
    # We can combine the factors and then multiply by sigma/cost.
    # This is equivalent to (w1 * EI + w2 * LCB_score) / cost.
    
    # Let's choose weights.
    # w1 = 1.0 for EI
    # w2 = 0.5 for LCB (exploration)
    
    # However, LCB_score can be negative if mu is very high.
    # EI is always non-negative.
    # We want to maximize the score.
    
    # Let's try:
    # Score = (EI + 0.5 * max(0, LCB_score)) / cost
    # This ensures we don't get penalized for high mu if EI is high.
    
    # Actually, a more standard cost-aware acquisition is:
    # Score = EI / cost
    # But this can be too greedy.
    
    # Let's use a "Cost-Adjusted Lower Confidence Bound":
    # Score = (mu - beta * sigma) / cost  (minimizing this)
    # But we want to maximize. So:
    # Score = (-mu + beta * sigma) / cost
    
    # Let's combine EI and LCB.
    # Score = (0.8 * EI + 0.2 * LCB_score) / cost
    
    # Let's compute the factors.
    ei_factor = z * norm.cdf(z) + norm.pdf(z)
    lcb_factor = -z + beta
    
    # Combine
    combined_factor = 0.8 * ei_factor + 0.2 * lcb_factor
    
    # The score is proportional to sigma * combined_factor / cost
    # But wait, EI = sigma * ei_factor, LCB_score = sigma * lcb_factor
    # So (0.8 * EI + 0.2 * LCB_score) / cost = sigma * (0.8 * ei_factor + 0.2 * lcb_factor) / cost
    
    score = sigma_safe * combined_factor / cost
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If all scores are zero, we might have a problem.
    # But sigma_safe is at least 1e-9, and cost is positive, so it should be fine.
    
    return score
# EVOLVE-BLOCK-END
