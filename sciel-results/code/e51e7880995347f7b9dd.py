"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Adjusted Expected Improvement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-12
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit
_BUDGET_LIMIT = 30.0
_INIT_COST = 7.5  # Approximate initial cost of 5 uniform points (avg cost ~1.5 each)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) acquisition function.
    
    Strategy:
    1. Instead of using a simple UCB (mu + beta*sigma), we use Expected Improvement (EI).
       EI is theoretically grounded for maximizing improvement over the current best.
       For minimization, EI(x) = E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2).
       
    2. We incorporate cost awareness by dividing the expected improvement by the cost.
       This directly optimizes for "improvement per unit cost", which is the correct
       objective when we have a fixed budget constraint.
       
    3. The analytic formula for EI (minimization) is:
       EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
       where Phi is the standard normal CDF and phi is the standard normal PDF.
       
    4. The cost-adjusted score is:
       score = EI(x) / cost(x)
       
       This naturally balances exploration (high sigma increases EI) and exploitation
       (low mu increases EI), while penalizing expensive evaluations.
       
    5. No state tracking is needed - the function is stateless and purely based on
       the current GP predictions and costs.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Compute the improvement: f_best - mu
    improvement = f_best - mu
    
    # Compute the standardized improvement: z = (f_best - mu) / sigma
    z = improvement / sigma
    
    # Compute EI using the analytic formula
    # EI = improvement * Phi(z) + sigma * phi(z)
    # For numerical stability, we can use:
    # EI = improvement * Phi(z) + sigma * phi(z)
    # where Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # and phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    
    # Use scipy.stats.norm for stability
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    ei = improvement * Phi + sigma * phi
    
    # Ensure EI is non-negative (it should be, but numerical issues can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score: EI / cost
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END
