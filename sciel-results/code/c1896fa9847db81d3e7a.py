"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of Improvement (QI) acquisition function.
    
    Strategy:
    1. Compute the Quantile of Improvement (QI) for minimization:
       QI(x) = E[f(x) | f(x) < f_best]
       
       For a Gaussian posterior N(mu, sigma^2), this is given by:
       QI = mu - sigma * [phi((f_best - mu) / sigma) / Phi((f_best - mu) / sigma)]
       
       Where:
       - mu, sigma are the GP mean and standard deviation.
       - f_best is the current best observed value.
       - phi is the standard normal PDF.
       - Phi is the standard normal CDF.
       
       Intuition:
       - If f_best is far above mu (high confidence improvement), the term in brackets approaches 0, and QI approaches mu (the expected value).
       - If f_best is close to mu (uncertain improvement), the term penalizes the expectation, reflecting that we only "count" the improvement if it happens.
       - Unlike PI (which only looks at probability) or EI (which looks at expected gain), QI looks at the *quality* of the expected value given improvement. This is often more robust in multimodal landscapes like Rastrigin because it avoids exploring areas where improvement is likely but the value is still poor (e.g., local minima far from the global one).

    2. Cost Awareness:
       The cost of evaluating a point is given by `cost` (1 + mean(x_scaled)).
       We want to maximize the "value" per unit cost.
       Score = QI / cost
       
       Note: Since we are minimizing, lower f(x) is better. 
       The QI formula calculates the expected value *if* it is better. 
       However, the standard QI formula is often defined for maximization or as a "regret reduction". 
       Let's stick to the definition: We want to find points where the *conditional expected value* is low.
       
       Actually, a more direct interpretation for minimization:
       We want to select x that minimizes the expected value *given* that f(x) < f_best.
       But the acquisition function should return a score where HIGHER is better.
       
       Let's define the score as the "potential improvement magnitude" weighted by probability, but normalized by the probability (i.e., the conditional mean).
       
       Wait, if QI = E[f(x) | f(x) < f_best], then a LOWER QI is better (closer to 0).
       So we should MINIMIZE QI? 
       No, the interface says "Return 2000 finite scores (higher = evaluate next)".
       
       So we need a score that is high when QI is low (good) and cost is low.
       Score = 1 / (QI + epsilon) ? Or (f_best - QI) / cost?
       
       Let's look at the standard QI acquisition function in literature.
       Usually, QI is used to minimize the expected value of the function in the region of improvement.
       If we define Score = (f_best - QI) / cost, this represents the "average improvement magnitude" per unit cost.
       This is essentially the Expected Improvement (EI) divided by the Probability of Improvement (PI)?
       EI = E[max(0, f_best - f(x))] = (f_best - mu) * Phi(d) + sigma * phi(d)
       PI = Phi(d)
       EI / PI = (f_best - mu) + sigma * (phi(d) / Phi(d)) = f_best - (mu - sigma * phi(d)/Phi(d)) = f_best - QI.
       
       So, (f_best - QI) is exactly the conditional expected improvement.
       Maximizing (f_best - QI) / cost is equivalent to maximizing the expected improvement magnitude per unit cost, conditional on improvement happening.
       
       This is a strong candidate because it filters out "low quality" improvements (small gains) and focuses on "high quality" improvements (large gains) that are likely to occur.
    """
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
         # Return a neutral score to avoid hanging or invalid computation
         return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute d = (f_best - mu) / sigma
    d = (f_best - mu) / sigma
    
    # Compute PI = Phi(d)
    PI = norm.cdf(d)
    
    # Compute phi(d)
    phi_d = norm.pdf(d)
    
    # Handle the case where PI is very small to avoid division by zero in the conditional term
    # If PI is close to 0, the conditional expectation is undefined or unstable.
    # We can mask these out or use a safe division.
    safe_PI = np.where(PI > 1e-8, PI, 1e-8)
    
    # The term phi(d) / PI is the inverse Mills ratio.
    # QI = mu - sigma * (phi_d / PI)
    # Conditional Expected Improvement (CEI) = f_best - QI = (f_best - mu) + sigma * (phi_d / PI)
    
    # To avoid numerical overflow if PI is tiny, we clip the ratio
    ratio = phi_d / safe_PI
    # If PI was actually 0, the ratio is huge. We should probably treat CEI as 0 in those cases?
    # Actually, if PI is 0, there is no improvement, so CEI should be 0.
    # Let's construct CEI carefully.
    
    # CEI = (f_best - mu) * PI + sigma * phi_d  (This is EI)
    # We want CEI_cond = EI / PI
    
    # Let's just compute CEI_cond = (f_best - mu) + sigma * (phi_d / PI)
    # But only where PI > threshold. Elsewhere, set to 0.
    
    valid_mask = PI > 1e-6
    
    cei_cond = np.zeros_like(mu)
    cei_cond[valid_mask] = (f_best - mu[valid_mask]) + sigma[valid_mask] * (phi_d[valid_mask] / PI[valid_mask])
    
    # Ensure CEI is non-negative (numerical safety)
    cei_cond = np.maximum(cei_cond, 0.0)
    
    # Cost-aware score
    # We want to maximize this score.
    # Dividing by cost penalizes expensive points.
    cost_safe = np.maximum(cost, 1e-9)
    scores = cei_cond / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
