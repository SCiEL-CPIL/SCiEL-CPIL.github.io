# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective Improvement (CEI)
    with a UCB exploration term that is NOT cost-normalized.
    
    Strategy:
    - Compute standard EI for each candidate.
    - Instead of dividing EI by cost (which suppresses expensive but potentially 
      high-improvement regions), we use a cost-conditional approach:
      We estimate the "effective" improvement by considering that cheaper points
      allow more evaluations within the budget. However, a simple division 
      is known to be suboptimal. 
      
    - Key Structural Change: 
      We blend a raw UCB term (for exploration) with a cost-weighted EI term.
      The UCB term is NOT divided by cost, ensuring that regions with high 
      uncertainty (high sigma) are still explored even if they are expensive, 
      because high uncertainty often implies potential for large improvements.
      The EI term is divided by a "soft" cost factor to bias towards cheaper 
      evaluations where the improvement is already high.
      
    - Specifically:
      Score = (EI / (1 + cost_penalty)) + (alpha * UCB_term)
      Where UCB_term = -mu + beta * sigma (minimization, so we want low mu, high sigma)
      But since we return "higher = evaluate next", we use:
      UCB_score = -mu + beta * sigma
      
      We need to align scales. EI is in objective units. UCB is in objective units.
      
    - Adaptive beta:
      beta increases as we have less data (lower f_best relative to range?) 
      or simply fixed moderate beta. Given the short budget, a fixed beta ~ 1.5-2.0 
      is often robust. Let's use a beta that depends on the total cost spent so far 
      if we had it, but we only have current cost vector. 
      
    - Let's look at the cost distribution. 
      cost is in [1, 2] roughly (since x in [-5,5] scaled to [0,1], mean in [0,1]).
      
    - New Approach: 
      1. Compute EI.
      2. Compute UCB = -mu + beta * sigma.
      3. Normalize both to have similar scales? Or just combine.
      4. Cost adjustment: Instead of EI/cost, use EI * exp(-lambda * (cost - min_cost)).
         This is a soft penalty.
      5. Add UCB term directly (no cost penalty) to ensure exploration.
      
    Let's try:
      score = EI * cost_weight + ucb_weight * (-mu + beta * sigma)
      
      To make them comparable, note that EI can be large if f_best is high.
      UCB can be negative if mu is very low.
      
      Let's stick to a proven variant: 
      The "inspiration 1" mentioned switching to a cost-aware approach that doesn't 
      suppress exploration. 
      
      Let's try a pure UCB with a cost-aware beta.
      If cost is high, we should be more conservative (lower beta) because we 
      "pay" more for the uncertainty. If cost is low, we can afford higher beta.
      
      beta(x) = base_beta * (min_cost / cost(x))^gamma
      
      This way, expensive points get a lower beta (less exploration bonus), 
      while cheap points get a higher beta (more exploration bonus).
      This is a structural change from dividing the *improvement* by cost to 
      modulating the *exploration coefficient* by cost.
      
      Score = -mu + beta(x) * sigma
      
      This is a Lower Confidence Bound (for minimization) where the 
      exploration width is inversely proportional to cost.
      
      Let's test this logic:
      - If a point is expensive (high cost), beta is small. We only pick it 
        if mu is very low (exploitation) or sigma is HUGE.
      - If a point is cheap (low cost), beta is large. We are more willing 
        to explore it.
      
      This seems more principled than EI/cost. EI/cost assumes the *value* 
      of improvement is divided by cost, but the *value of information* 
      (uncertainty reduction) might not scale linearly with cost in that way.
      
      Let's implement:
      base_beta = 1.5
      gamma = 1.0
      min_cost = np.min(cost)
      beta = base_beta * (min_cost / cost)
      
      score = -mu + beta * sigma
      
      To ensure finite and reasonable values, we clip.
      
      Wait, the current program scores 1.67. The inspiration 1 scored 2.51.
      The inspiration 1 description says: "Switch from ... cost-normalized LCB ... 
      to a c | uses numpy..." The description is cut off but implies a different 
      cost handling.
      
      Let's try the adaptive beta UCB.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma and cost
    sigma = np.clip(sigma, 1e-8, 1e6)
    cost = np.clip(cost, 1e-6, 100.0)
    
    # Calculate adaptive beta
    # Base beta: 1.5 is a reasonable start for UCB in BO
    base_beta = 1.5
    gamma = 1.0
    
    # To avoid division by zero or extreme values, use a reference cost
    # min_cost is the cheapest point in the current batch
    # This makes the most expensive point have beta = base_beta * (min/max)
    # and the cheapest point have beta = base_beta
    min_cost = np.min(cost)
    
    # If min_cost is very small, this could be unstable, but cost is >= 1
    beta = base_beta * (min_cost / cost) ** gamma
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.1, 10.0)
    
    # Compute UCB score for minimization: -mu + beta * sigma
    # Higher score = better
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
