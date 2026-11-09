"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we are too close to the limit
    # The evaluator kills after 330s. We want to finish within 300s per eval, but the total loop is what matters.
    # Actually, the contract says "Each evaluation must finish within 300 s".
    # But the total budget is cost < 30.
    # Let's just ensure we don't hang.
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # Return a deterministic but safe score (e.g., center of domain or just zeros)
        # Returning zeros is finite and valid.
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB
    # The cost is 1 + mean(x_scaled).
    # Points with lower cost are cheaper.
    # We want to maximize the "value per cost".
    
    # Structural Change: Use a hybrid of UCB and Expected Improvement (EI)
    # EI is more robust for multimodal functions when the model is good.
    # However, with small budgets, UCB is often better.
    # Let's try a "Cost-Adjusted Expected Improvement" or a "Normalized UCB".
    
    # Let's stick to UCB but improve the beta schedule.
    # The previous adaptive beta based on f_best was simple.
    # Let's use a beta that decays with the number of evaluations (implicit in cost).
    # Total cost so far is sum(cost) roughly? No, cost is the cost of the NEXT point?
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # "f_best is the lowest observed value."
    # The input 'cost' has shape (2000,). This is the cost of evaluating each of the 2000 candidates.
    # So we should divide by the specific cost of each candidate.
    
    # Let's try a different acquisition: Probability of Improvement (PI) combined with UCB?
    # Or just a better UCB.
    
    # Let's try "Knowledge Gradient" approximation? Too complex.
    
    # Let's try "Maximizing the Probability of being the best" (PI)
    # PI = P(f(x) < f_best) = CDF((f_best - mu) / sigma)
    # This is 0 if mu > f_best + sigma, and 1 if mu < f_best - sigma.
    # It is a step function.
    
    # Let's combine UCB and PI.
    # Score = w * UCB + (1-w) * PI
    
    # Adaptive weights based on f_best?
    # If f_best is good (low), exploit (high weight on UCB with low beta or PI).
    # If f_best is bad (high), explore (high weight on UCB with high beta).
    
    # Let's use a fixed, strong exploration beta but normalized by cost.
    # The previous score 2.09 was with beta=1.0 fixed.
    # The current score 1.29 is with adaptive beta.
    # Maybe the adaptive beta was too aggressive in exploration early or too late?
    
    # Let's try a "Two-Phase" approach.
    # Phase 1: Explore. Phase 2: Exploit.
    # How to detect phase? Use f_best.
    
    # If f_best > 0.1, we are likely far from optimum. Explore.
    # If f_best < 0.1, we are close. Exploit.
    
    # Let's use a smooth transition.
    # beta = beta_min + (beta_max - beta_min) * sigmoid(k * (f_best - threshold))
    
    # Let's try:
    # beta = 1.5 + 1.0 * exp(-10 * f_best)
    # If f_best = 0.1, exp(-1) = 0.37, beta = 1.87
    # If f_best = 0.01, exp(-0.1) = 0.90, beta = 1.90
    # If f_best = 0.5, exp(-5) = 0.006, beta = 1.5
    # This keeps beta high always.
    
    # What if we use "Cost-Weighted EI"?
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Score = EI / cost
    
    # EI is better than UCB for minimization when the model is accurate.
    # Let's try EI.
    
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    # Handle numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best. We want to maximize EI.
    # If EI is negative, it's better to explore?
    # Actually, if mu > f_best, EI is negative.
    # We can clip EI to 0? No, that kills exploration of regions with high mu but high sigma.
    
    # Let's combine EI and UCB.
    # Score = max(EI, 0) + beta * sigma / cost?
    
    # Let's just try pure EI / cost.
    scores = ei / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
