import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with noisy cost modeling.
    
    Structural change from EI/UCB blend:
    - Use Thompson Sampling: sample f(x) ~ N(mu, sigma) and score by -f_sampled / cost
    - This naturally balances exploration and exploitation without the bias of EI
      towards already-good regions
    - Cost is treated as a probabilistic resource via 1/cost scaling (BO-NC inspired)
    - Add a small exploration bonus proportional to sigma to encourage visiting
      uncertain regions, especially early on when f_best is poor
    """
    # Fast path if we're running out of time
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        # Simple Thompson sample
        f_sampled = mu + sigma * np.random.randn(len(mu))
        scores = -f_sampled / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw a sample from the posterior for each candidate
    # f(x) ~ N(mu, sigma)
    f_sampled = mu + sigma * np.random.randn(len(mu))

    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # This is the BO-NC formulation: the effective score is the negative
    # sampled objective per unit cost
    base_score = -f_sampled / cost

    # Exploration bonus: add a term that rewards high uncertainty
    # This helps escape local optima and explore the search space
    # The bonus is scaled by 1/sqrt(cost) to maintain cost-awareness
    # and by a factor that decreases as f_best improves (less exploration needed)
    f_best_clipped = np.clip(f_best, 0.0, None)
    exploration_weight = 0.3 / (1.0 + f_best_clipped)
    exploration_bonus = exploration_weight * sigma / np.sqrt(cost)

    # Combine base Thompson score with exploration bonus
    scores = base_score + exploration_bonus

    # Ensure outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
