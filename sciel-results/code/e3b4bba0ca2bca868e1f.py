import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None
_DEADLINE = 300.0  # 300s as per task constraint, with safety margin

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Return uniform random scores to finish quickly
        return np.ones_like(mu) * 0.0
    
    # Calculate Expected Improvement
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip small negative values due to numerical precision
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # cost is 1 + mean(scaled_x), so it's always >= 1
    base_score = ei / cost
    
    # Prior bias: The minimizer is in the inner 80% of [-4, 5]^4.
    # Range length is 9. Inner 80% is length 7.2.
    # The center of the original box [-4, 5] is 0.5.
    # The inner 80% region is symmetric around the center 0.5?
    # Let's check:
    # Original: [-4, 5]. Center = 0.5.
    # Inner 80%: from -4 + 0.1*9 = -3.1 to 5 - 0.1*9 = 4.1.
    # Center of inner region = (-3.1 + 4.1) / 2 = 0.5.
    # So the minimizer is uniformly distributed in a box centered at 0.5 in the original coordinates.
    
    # Coordinates are scaled to [0, 1] for the GP.
    # Original x in [-4, 5] -> scaled x in [0, 1].
    # x = -4 + 9 * x_scaled.
    # x = 0.5 -> 0.5 = -4 + 9 * x_scaled -> 4.5 = 9 * x_scaled -> x_scaled = 0.5.
    
    # So the expected location of the minimizer in scaled coordinates is 0.5.
    # Cost = 1 + mean(x_scaled).
    # Expected cost at the minimizer = 1 + 0.5 = 1.5.
    
    # We bias towards this expected cost.
    target_cost = 1.5
    
    # Gaussian weight on cost deviation
    # Lambda controls the strength of the bias.
    # A moderate lambda helps focus search without over-constraining.
    lambda_bias = 1.0
    weight = np.exp(-lambda_bias * (cost - target_cost) ** 2)
    
    # Add a small exploration bonus for high uncertainty to avoid premature convergence
    # This helps when EI is low but sigma is high, encouraging exploration of unexplored areas
    # The bonus is scaled by the inverse of cost to maintain cost-awareness
    exploration_bonus = sigma_safe / cost
    
    # Dynamic exploration budget: scale UCB weight by remaining budget ratio
    # Total cost limit is 30. Initial cost from 5 uniform points is approx 5 * 1.5 = 7.5.
    # Remaining budget = 30 - current_total_cost.
    # We don't have current_total_cost directly, but we can estimate it or use a heuristic.
    # Since cost is per-evaluation, and we don't know the total spent, we use a simpler proxy:
    # As the search progresses, f_best decreases and sigma decreases.
    # A common heuristic is to scale exploration by the inverse of the number of evaluations.
    # However, we don't have the count. 
    # Let's use the ratio of sigma to f_best as a proxy for "how much we've learned".
    # Early on, sigma is large relative to f_best (which is large), so exploration is high.
    # Later, sigma is small relative to f_best (which is small), so exploration is low.
    
    # A more direct approach based on the directive: "scales the UCB weight by the ratio of remaining budget to current cost"
    # Since we don't have the total cost, we can't compute this exactly.
    # But we can assume that the number of evaluations is roughly proportional to the total cost.
    # Let's approximate the number of evaluations by the magnitude of the data.
    # If we assume the average cost is ~1.5, then total_cost ~ 1.5 * n_evals.
    # n_evals ~ total_cost / 1.5.
    # This is circular.
    
    # Let's try a different dynamic weight based on the improvement potential.
    # If f_best is very low, we should exploit more.
    # If f_best is high, we should explore more.
    # But f_best is in objective units, and the minimum is 0.
    # So a low f_best means we are close to the optimum.
    
    # Let's use a simple decay factor based on the number of points evaluated.
    # We can estimate n_evals from the length of mu, but mu is always 2000.
    # We don't have the history.
    
    # Let's stick to a simpler dynamic adjustment:
    # Scale the exploration bonus by a factor that decreases as f_best gets closer to 0.
    # Since the minimum is 0, f_best is always >= 0.
    # Let's use 1 / (1 + f_best) as a decay factor.
    # This will be close to 1 when f_best is small (early) and close to 0 when f_best is large (late).
    # Wait, f_best is the *best* value, so it decreases over time.
    # So 1 / (1 + f_best) will *increase* over time. That's the opposite of what we want.
    # We want exploration to *decrease* over time.
    # So we want a factor that *decreases* as f_best decreases.
    # How about f_best itself? If f_best is large, explore more. If f_best is small, exploit more.
    # But f_best is in objective units, which can vary.
    
    # Let's use a fixed schedule based on the number of calls.
    # We can count the number of times acquisition is called.
    global _call_count
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    n_calls = acquisition._call_count
    
    # Decay exploration weight with number of calls
    # Start with high exploration, decay to low exploration
    # Use an exponential decay
    exploration_weight = np.exp(-0.1 * n_calls)
    
    # Combine base score and exploration bonus
    # The exploration bonus is weighted to be smaller than the base score typically
    score = (base_score + 0.1 * exploration_weight * exploration_bonus) * weight
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
