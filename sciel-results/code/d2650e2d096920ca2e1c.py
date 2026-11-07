"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Leave 40s for cleanup/other overhead
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    
    Strategy: Combines Cost-Effective Expected Improvement (CEI) with UCB exploration.
    The CEI term divides EI by cost to prefer efficient gains. The UCB term provides
    exploration that decays over time. A global cost penalty further discourages
    expensive evaluations. The balance between CEI and UCB shifts towards exploitation
    (CEI) as the number of calls increases.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score: exploit current best (minimize mu)
        return -mu.copy()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-Effective EI (CEI)
    # Divide by cost to normalize improvement per unit cost
    # Cost is in [1, 2]
    cei = ei / np.maximum(cost, 1e-9)
    
    # 3. UCB-style exploration
    # beta decays with call count to shift from exploration to exploitation
    # Start with a moderate beta, decay slowly
    beta = 2.0 / (1.0 + 0.05 * _CALL_COUNT)
    ucb_score = -mu + np.sqrt(beta) * sigma
    
    # 4. Combine CEI and UCB
    # Early on, UCB helps explore; later, CEI dominates for cost-effective exploitation
    # Weight shifts from UCB to CEI as we progress
    # At call 0: w_ucb = 1.0, w_cei = 0.0 -> Pure UCB
    # At call 10: w_ucb = 1/2 = 0.5, w_cei = 0.5
    # At call 20: w_ucb = 1/3 = 0.33, w_cei = 0.67
    w_ucb = 1.0 / (1.0 + 0.1 * _CALL_COUNT)
    w_cei = 1.0 - w_ucb
    
    # Base score (higher is better)
    # We need to scale them somewhat or they might be on different scales.
    # EI is in objective units. UCB is in objective units.
    # Cost is dimensionless (or currency).
    # CEI is objective/cost.
    # Let's just sum them directly first, maybe with a small weight on UCB to keep it secondary?
    # Actually, the weight w_ucb/w_cei handles the balance.
    base_score = w_ucb * ucb_score + w_cei * cei
    
    # 5. Global Cost Penalty
    # Penalize points with cost above the mean to further steer towards cheap evaluations.
    # This is additive to the score (subtraction).
    mean_cost = np.mean(cost)
    std_cost = np.std(cost)
    
    if std_cost > 1e-12:
        cost_z = (cost - mean_cost) / std_cost
        # Penalize high cost z-scores
        # Use a linear or quadratic penalty?
        # Quadratic is softer for low cost, harsher for high.
        # Let's try a simple linear penalty on the excess cost relative to min cost.
        # Or stick to the z-score approach.
        # Penalty magnitude should be significant enough to matter but not dominate.
        # The base score magnitude varies.
        # Let's scale the penalty by the mean absolute base score to be adaptive.
        base_scale = np.mean(np.abs(base_score))
        if base_scale < 1e-12:
            base_scale = 1.0
            
        # Penalty: 0.5 * max(0, cost_z) * base_scale
        # This ensures cost > mean gets penalized, cost < mean gets 0 penalty (relative to mean).
        # This effectively shifts the optimum towards lower cost.
        penalty = 0.5 * np.maximum(cost_z, 0.0) * base_scale
        score = base_score - penalty
    else:
        score = base_score.copy()
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
