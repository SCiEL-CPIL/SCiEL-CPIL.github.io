"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0
    if time.time() - t0 > deadline - t0:
        return np.zeros_like(mu)

    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # delta = f_best - mu (potential improvement)
    delta = f_best - mu
    
    # z-score for expected improvement
    z = delta / sigma_safe
    
    # Expected Improvement (EI)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    
    # Adaptive kappa for UCB component
    # Start with moderate exploration, decrease as f_best improves relative to scale
    # Use a simple heuristic based on the magnitude of f_best
    # If f_best is close to 0 (good), reduce exploration; if far, increase
    # Scale factor: we want kappa to be around 1-3
    # Let's use a fixed but slightly lower kappa than before, combined with EI
    # Actually, let's blend EI and UCB
    
    # UCB term for minimization: f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma
    # This is delta + kappa * sigma
    kappa = 1.5  # Reduced from 2.0 to favor exploitation more
    
    ucb_term = delta + kappa * sigma_safe
    
    # Blend EI and UCB
    # EI is already in "improvement" units
    # UCB is also in "improvement" units (optimistic bound)
    # Let's try a weighted average, favoring EI slightly
    # Or just use EI as it is more principled for cost-awareness
    # The previous program used pure UCB/cost and scored 1.2193
    # The inspirations used pure EI/cost and scored 1.0
    # So UCB seems better for this specific task setup.
    # Let's stick closer to the UCB approach but maybe refine it.
    
    # Let's try a hybrid: EI + small UCB bonus
    # Or just stick with the successful UCB/cost approach from the parent.
    # The parent scored 1.2193.
    # Let's see if we can improve it.
    # The knowledge says "Adaptive kappa and weights based on mean cost better balance exploration/exploitation"
    # Let's implement adaptive kappa.
    
    # Mean cost of the candidates
    mean_cost = np.mean(cost)
    
    # Adaptive kappa: 
    # If costs are high, we should be more conservative (lower kappa) to save budget?
    # Or if we are early (low total cost implied by many low cost points?), explore more?
    # The prompt says "While total cost is below 30".
    # We don't have total cost, just the cost of the 2000 candidates.
    # Let's assume lower mean cost implies we are in a cheaper region, so we can afford more exploration?
    # Or perhaps: normalize kappa by the mean cost to make it scale-invariant?
    
    # Let's try: kappa = 2.0 * (mean_cost / np.median(cost)) ** 0.5
    # This adjusts kappa based on the spread of costs.
    
    # Actually, simpler: just use the UCB/cost approach which worked well.
    # scores = (f_best - mu + kappa * sigma) / cost
    
    # Let's try to combine the best of both:
    # Use EI as the base, but add a small UCB term to encourage exploration in high uncertainty areas.
    # Or, since UCB/cost worked best, let's refine that.
    
    # Parent: scores = (f_best - mu + 2.0 * sigma) / cost
    # Let's try kappa = 1.8 and see.
    
    # Alternative: Use the "knowledge" insight about adaptive kappa.
    # Let's compute a global scale of the function values to normalize kappa.
    # But we don't have the full history, only f_best.
    
    # Let's stick to the parent's strategy but with a slightly tuned kappa.
    # The parent used 2.0.
    # Let's try 1.5 to be more exploitative.
    
    ucb_score = delta + 1.5 * sigma_safe
    scores = ucb_score / np.maximum(cost, 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
