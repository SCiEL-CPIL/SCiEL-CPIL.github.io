"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for Thompson Sampling
_ts_state = {
    'last_time': 0.0,
    'sample_offset': 0.0,
    'seed': 42
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed function value
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    scores: array of shape (2000,) - higher scores are evaluated next
    """
    global _ts_state
    
    # Deadline guard: ensure we return well before 300s limit
    # The evaluator kills after 330s, so we have plenty of margin for this simple op.
    # However, we track time just in case of unexpected delays in the loop.
    current_time = time.time()
    elapsed = current_time - _ts_state['last_time']
    _ts_state['last_time'] = current_time
    
    # If we've been running for a very long time (shouldn't happen in normal operation),
    # we might want to simplify, but for now, the operation is O(N) and fast.
    
    # Update internal seed to ensure different samples each call
    # We use a simple deterministic increment to avoid relying on global np.random state
    # which might be reset or shared in unexpected ways.
    _ts_state['seed'] += 1
    seed_val = _ts_state['seed']
    
    # Create a local random generator for reproducibility within this call
    # Using a fixed seed per call index ensures we don't get the same sample twice
    # if the function is called repeatedly with the same inputs (though inputs change).
    # Actually, for Thompson Sampling, we want a *new* random sample each time.
    # Using np.random.default_rng(seed_val) is efficient and isolated.
    rng = np.random.default_rng(seed_val)
    
    # Draw a sample from the posterior predictive distribution: N(mu, sigma^2)
    # sigma is the standard deviation
    sample = mu + sigma * rng.standard_normal(size=mu.shape)
    
    # We want to minimize the function.
    # The sampled value 'sample' represents the true function value at that point.
    # Lower is better.
    # We also need to account for cost.
    # The task specifies cost = 1 + mean(x_scaled).
    # We don't have x, but we have the 'cost' array provided.
    # A common way to handle cost in TS is to subtract a penalty proportional to cost.
    # Or, we can simply rank by the sampled value, but that ignores cost.
    # The prompt says "Cost-aware acquisition".
    # The current seed program uses EI / cost.
    # In TS, a simple effective strategy is: Score = -Sampled_Value - lambda * Cost
    # What is lambda? 
    # If we look at the seed: EI / cost. This is roughly (f_best - mu) / cost.
    # This implies we prefer points with high improvement per unit cost.
    # In TS, we can approximate this by penalizing cost.
    # Let's try a simple penalty. Since f_best is around -78 and costs are around 1-2,
    # a direct subtraction of cost might be too small or too large depending on the scale.
    # However, the "cost" is defined as 1 + mean(x_scaled). 
    # x_scaled is in [0,1]. So cost is in [1, 2].
    # The function values are in [-78, ...].
    # A difference of 1 in cost is small compared to function value differences.
    # But wait, the "cost" in the problem description is the *budget* cost.
    # The acquisition function should prioritize points that are likely to be good AND cheap.
    
    # Let's look at the inspiration: "batched Thompson Sampling".
    # A standard cost-aware TS approach is to sample y ~ N(mu, sigma^2) and then
    # maximize (y - cost_penalty).
    # Alternatively, we can use the "Expected Improvement per cost" intuition by
    # normalizing the sample by cost? No, that's not standard TS.
    
    # Let's stick to a simple, robust cost penalty.
    # Score = -sample - beta * cost
    # We need to choose beta.
    # If we assume the goal is to minimize f(x) subject to total cost < 30,
    # and we have ~15 evaluations left (30/2 approx),
    # the marginal value of cost is roughly the improvement per cost.
    
    # Let's try: Score = -sample / cost
    # This favors points that have low sampled value AND low cost.
    # This is similar to the seed's EI/cost structure but stochastic.
    # However, dividing by cost can blow up if cost is small.
    # Cost is >= 1. So it's safe.
    
    # Let's try: Score = -sample - cost
    # This is a Lagrangian relaxation with lambda=1.
    # Given the scale of f (tens) vs cost (1-2), lambda=1 might be too weak.
    # The seed program effectively has a dynamic lambda that depends on f_best and mu.
    
    # Let's look at the high-scoring inspiration (1.63): "batched Thompson Sampling".
    # It likely uses a specific penalty.
    
    # Let's use a simple heuristic:
    # Score = -sample - 0.1 * cost
    # Wait, if f is -78, and we improve to -79, the delta is 1.
    # Cost delta is small.
    # Actually, the most direct translation of "cost-aware" in a black-box setting
    # where we can't easily compute the optimal Lagrange multiplier is to use
    # the sample value itself, but perhaps adjusted.
    
    # Let's try the simplest effective variant:
    # Sample y. Rank by -y.
    # But this ignores cost.
    # The seed program's score is EI/cost.
    # EI is roughly (f_best - mu).
    # So it ranks by (f_best - mu) / cost.
    # In TS, we sample y. E[y] = mu.
    # So we are ranking by (f_best - y) / cost?
    # Or just -y?
    
    # Let's try: Score = -sample - cost
    # This penalizes expensive points.
    
    # Let's refine the penalty.
    # The cost is in [1, 2].
    # The function values vary by ~10-20 near the optimum.
    # A penalty of 1.0 on cost means a point with cost 2 is penalized by 2 units of f.
    # This seems reasonable.
    
    scores = -sample - cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
