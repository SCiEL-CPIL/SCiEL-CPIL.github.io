import numpy as np
import time

# Module-level state to track elapsed time for the deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # 300s limit, keep 30s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample from the posterior distribution of the function values (mu + sigma * N(0,1)).
    2. Since we want to minimize, we look for low values. 
    3. However, we must account for cost. A high cost reduces the "effective" benefit of finding a low value 
       if we are near the budget limit, or simply because the budget is scarce.
    4. We use a dynamic penalty based on the current cost relative to the average cost or a threshold.
    5. Crucially, Thompson Sampling naturally balances exploration and exploitation without a fixed beta,
       which is often superior to static UCB in multimodal landscapes like Styblinski-Tang.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next).
    """
    # Deadline guard: If we are close to the time limit, return a deterministic, fast fallback
    # to avoid being killed. The fallback should be safe (e.g., just exploit best known or random).
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Exploit the best mean, penalized by cost. 
        # This is fast and deterministic.
        score = -mu / np.maximum(cost, 1e-6)
        return np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Ensure positive costs and sigmas
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Thompson Sampling: Sample a value from the posterior for each candidate
    # Since we need to return a score for the *same* set of 2000 candidates to pick the *next* one,
    # we can sample once per candidate.
    # Note: In a true TS loop, we'd sample a function, find its argmin, and evaluate it.
    # Here, the framework asks for scores for 2000 candidates. 
    # Standard TS for batch/sequential selection in this context:
    # We approximate the "probability of being the best" or simply use the sampled value 
    # adjusted by cost.
    
    # Draw samples from N(mu, sigma^2)
    # Use a fixed seed for the random generator within the call? 
    # No, TS relies on randomness. But since this function is called repeatedly,
    # we should use the global RNG or a local one. Using np.random is fine.
    # However, to make the "score" comparable across calls, we are effectively 
    # saying: "Given a random realization of the function, which point is best?"
    # But we return 2000 scores. The evaluator picks the max.
    # So we return: Sampled_Value - Cost_Penalty.
    
    # The sampled value is the predicted function value.
    # We want to MINIMIZE f(x).
    # So a "good" candidate has a LOW sampled value.
    # To make it "higher is better", we invert: -Sampled_Value.
    
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Cost-awareness:
    # The cost is 1 + mean(x_scaled). Range [1, 2].
    # We want to avoid high cost if it doesn't offer proportional improvement.
    # A simple way is to divide the "gain" by cost, or subtract a penalty.
    # Let's try a penalty that scales with the uncertainty or the potential improvement.
    
    # Heuristic: 
    # Score = -Sampled_Value / Cost
    # This favors low values (high -Sampled_Value) and low costs.
    
    # However, simple division can be unstable if cost varies a lot.
    # Cost is in [1, 2].
    
    # Let's refine the TS signal.
    # Pure TS on -f(x) works well.
    # Cost-aware TS: 
    # We can interpret the cost as a discount factor.
    # Effective Value = Sampled_Value * Cost? No, that would make high cost worse (larger value).
    # We want to minimize.
    # If we minimize Sampled_Value * Cost, we prefer low values AND low costs.
    # Then Score = - (Sampled_Value * Cost).
    
    # Let's try: Score = - (Sampled_Value + lambda * Cost)
    # Or Score = -Sampled_Value / Cost
    
    # Given the previous insight "UCB term rewards high-uncertainty regions",
    # TS naturally does this by sampling high values from high sigma regions.
    
    # Let's use the multiplicative penalty as it strongly discourages corners (high cost)
    # unless the GP is very confident that the value is much lower than f_best.
    
    # But wait, if f_best is very low (e.g., -78), and a corner has mu=-70, sigma=10.
    # Sample might be -60. Cost=2. Score = -(-60)*2 = 120.
    # Center point mu=-75, sigma=5. Sample=-70. Cost=1. Score = -(-70)*1 = 70.
    # The corner wins because -60 is "better" (lower) than -70? No, -70 is lower than -60.
    # Wait. Minimization.
    # -60 is WORSE than -70.
    # So Sampled_Value = -60.
    # Score = -(-60) * 2 = 120.
    # Center: Sampled_Value = -70.
    # Score = -(-70) * 1 = 70.
    # The corner (worse value) gets a HIGHER score because of the cost multiplier?
    # No, -(-60) = 60. 60*2 = 120.
    # -(-70) = 70. 70*1 = 70.
    # Yes, the corner wins. This is BAD. We want to find the minimum.
    # The corner has a higher (worse) function value. It should not be preferred just because it's expensive?
    # Actually, we want to AVOID expensive points if they are not promising.
    # If the corner is promising (low mu), we might pay the cost.
    # If the corner is NOT promising (high mu), we should definitely avoid it.
    
    # My previous logic: Score = -Sampled_Value / Cost.
    # Corner: -(-60)/2 = 30.
    # Center: -(-70)/1 = 70.
    # Center wins. This is correct.
    
    # So, Score = -Sampled_Value / Cost.
    
    score = -samples / cost
    
    # Add a small exploration bonus based on sigma to ensure we don't get stuck?
    # TS already includes exploration via the sample.
    # However, if the GP is very certain (low sigma) about a bad region, TS won't explore it.
    # This is good.
    
    # One issue: If all samples are negative (which they will be, as f_best is negative),
    # -samples are positive.
    
    # Let's add a dynamic factor. If we are far from the optimum, explore more?
    # TS is adaptive.
    
    # To improve upon the "pure" TS, we can blend with a greedy exploitation term 
    # to ensure we don't miss the immediate best improvement if the GP is confident.
    # But TS is generally robust.
    
    # Let's try a hybrid: 
    # 50% TS, 50% Greedy (LCB-like)
    # This might be more stable.
    
    # Greedy Score: -mu / cost
    # TS Score: -samples / cost
    
    # Let's stick to pure TS first, but ensure the random seed doesn't cause instability 
    # that hurts the "mean over 5 instances" metric.
    # Actually, for a single run, TS is stochastic. The evaluator runs 5 instances.
    # The randomness in TS is part of the algorithm.
    
    # One refinement: The cost is 1 + mean(x_scaled).
    # The "mean(x_scaled)" part suggests the cost is linear in the coordinates.
    # This creates a gradient in cost.
    
    # Let's return the score.
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
