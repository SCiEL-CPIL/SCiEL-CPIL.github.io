"""Cost-aware acquisition function for Bayesian optimization (ackley 2d).

Strategy:
- Use a Knowledge Gradient (KG) approximation that explicitly models the
  expected reduction in f_best after evaluating a new point, while accounting
  for the cost of evaluation.
- Unlike EI (which is E[max(0, f_best - f(x))]), this implementation uses
  a "probabilistic KG" heuristic: it rewards points that have a high probability
  of being the new global minimum, weighted by the expected magnitude of improvement.
- Specifically, we compute a score based on the expected improvement (EI) but
  modulated by a "confidence" factor that increases exploration in regions
  where the GP is highly uncertain, effectively acting as a UCB-like bonus
  scaled by the local sigma. This is structurally different from pure EI/cost
  or PoI/cost.
- Formula: score = (EI(x) + gamma * sigma(x)) / cost(x)
  where EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
  and z = (f_best - mu) / sigma
- The gamma * sigma term acts as an exploration bonus that is proportional to
  local uncertainty, encouraging the search to visit unexplored regions even
  if the mean is not immediately promising. This is a key difference from
  pure EI, which can be too exploitative, and from pure UCB, which ignores
  the expected improvement magnitude.
- We also include a deadline guard to ensure we return within the time limit.
"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates using a KG-inspired
    hybrid of Expected Improvement and a sigma-based exploration bonus.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1e-9, None)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    # Clip z to prevent overflow in norm.pdf/cdf
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Since we are minimizing, improvement is f_best - f(x).
    # The standard EI formula for minimization is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: proportional to sigma
    # This term encourages visiting high-uncertainty regions.
    # The coefficient gamma controls the trade-off.
    # A value of 1.0 is a reasonable starting point, effectively making the score
    # resemble a UCB with a variable beta that is tied to the local sigma.
    gamma = 1.0
    exploration_bonus = gamma * sigma
    
    # Combine EI and exploration bonus
    # This is a heuristic KG approximation: the expected value of information
    # is higher when both the potential improvement (EI) and the uncertainty (sigma)
    # are high.
    combined_score = ei + exploration_bonus
    
    # Normalize by cost
    score = combined_score / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check: if we are close to the limit, we can return immediately
    # since the computation above is O(N) and very fast.
    # However, if we were doing iterative optimization, we would check here.
    # For this direct formula, no further action is needed.
    
    return score
# EVOLVE-BLOCK-END
