# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    score = (mu + kappa * sigma) / cost
    
    where kappa is dynamically tuned based on the ratio of remaining budget
    to current cost to shift the exploration-exploitation balance.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Total budget is 30, but we need to estimate how much has been spent.
    # The 'cost' array contains the cost of evaluating each of the 2000 candidates.
    # However, the total spent budget isn't directly passed. We can infer it from
    # the fact that the loop runs while total cost < 30.
    # Since we don't have the exact total spent, we use a heuristic:
    # The mean cost of candidates gives us a sense of the typical evaluation cost.
    # But a better proxy for "progress" is the number of evaluations done so far,
    # which we don't have directly. 
    #
    # Alternative: Use the distribution of mu/sigma to tune kappa.
    # A common approach is to use a time-dependent or budget-dependent kappa.
    # Since we don't have explicit time/budget info, we can use the ratio of
    # the current f_best to the range of mu to estimate convergence.
    #
    # Better approach: Use a fixed but well-tuned kappa, or adapt it based on
    # the uncertainty. Let's use a dynamic kappa based on the relative spread
    # of sigma compared to the improvement potential.
    
    # Heuristic: As f_best improves (gets more negative), we should explore less.
    # We can use the ratio of the best mu to f_best as a proxy for progress.
    # If mu is close to f_best, we're in exploitation mode (lower kappa).
    # If mu is far from f_best, we're in exploration mode (higher kappa).
    
    # Compute a proxy for progress: how much improvement is left
    # Use the minimum of mu as a signal of how good the best candidate is
    mu_min = np.min(mu)
    
    # If mu_min is close to f_best, we're converging, use lower kappa
    # If mu_min is far from f_best, we're still exploring, use higher kappa
    # The range of f for hartmann3d is roughly [-3.86, 0]
    # So the total range is about 3.86
    
    # Dynamic kappa: start high (exploration), decrease as we converge
    # Use a sigmoid-like transition based on how close mu_min is to f_best
    # Let's define a threshold: if f_best - mu_min < 0.5, we're in exploitation
    # Otherwise, we're in exploration
    
    # A smoother approach:
    # kappa = kappa_min + (kappa_max - kappa_min) * exp(-(f_best - mu_min) / scale)
    # where scale controls the transition width
    
    kappa_max = 3.0  # High exploration
    kappa_min = 0.5  # Low exploration, more exploitation
    scale = 1.0
    
    # f_best is the best observed value (most negative is best)
    # mu_min is the best predicted mean
    # If f_best - mu_min is small, mu_min is close to or better than f_best
    # This means we're near the optimum, so reduce exploration
    
    gap = f_best - mu_min
    # If gap is negative, mu_min is better than f_best (improvement found)
    # If gap is positive, f_best is still better
    
    # Use a smooth transition
    kappa = kappa_min + (kappa_max - kappa_min) * np.exp(-np.abs(gap) / scale)
    
    # Compute UCB score: (mu + kappa * sigma) / cost
    # Note: We want to MINIMIZE, so lower mu is better.
    # But UCB for minimization: we want high mu + kappa*sigma? No.
    # For minimization, the acquisition should favor points with low mu and high sigma.
    # Standard UCB for minimization: score = mu - kappa * sigma (minimize this)
    # But we want to RETURN scores where HIGHER = evaluate next.
    # So we should return: -(mu - kappa * sigma) / cost = (-mu + kappa * sigma) / cost
    # Or equivalently: (kappa * sigma - mu) / cost
    
    # Wait, let me reconsider. The task says "higher = evaluate next".
    # For minimization, we want to pick points that are likely to have low f(x).
    # The expected improvement for minimization is: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # For UCB in minimization, the confidence bound is: mu - kappa * sigma
    # We want to minimize this, so the acquisition (higher is better) should be:
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    # But this doesn't account for f_best. A pure UCB might explore too much.
    # Let's use a hybrid: combine EI-like exploitation with UCB-like exploration.
    
    # Actually, the directive says to use UCB: score = (mu + kappa * sigma) / cost
    # But for MINIMIZATION, this would favor HIGH mu, which is wrong.
    # The directive likely assumes maximization. For minimization, we need to adjust.
    
    # For minimization, the correct UCB-based acquisition (higher is better) is:
    # score = (kappa * sigma - mu) / cost
    
    # But to incorporate f_best, we can use a modified form:
    # score = ((f_best - mu) + kappa * sigma) / cost
    # This combines the improvement potential (f_best - mu) with uncertainty (sigma)
    
    # Let's use this form, which is more appropriate for minimization with a known f_best
    score = ((f_best - mu) + kappa * sigma) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
