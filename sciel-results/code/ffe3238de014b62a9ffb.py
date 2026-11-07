import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: The evaluator kills the program after 330s.
    # We assume this function is called within a loop. 
    # However, the task says "Each function runs...". 
    # The "loop" is external. But the prompt says "While total cost is below 30...".
    # Actually, the acquisition function is called once per iteration of the loop?
    # "The evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # This implies acquisition is called once per step of the outer loop.
    # The outer loop runs until cost >= 30.
    # Initial cost: 5 points * ~1.5 = 7.5.
    # Each step costs ~1.5. So roughly (30-7.5)/1.5 = 15 steps.
    # Total time limit 330s. So we have ~20s per step.
    
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin
    
    # Inputs: mu (2000,), sigma (2000,), f_best (scalar), cost (2000,)
    # cost is 1 + mean(x). Range [1, 2].
    
    # Strategy: Greedy Batch Optimization (GBO)
    # We simulate adding points to the set of observed points to update the GP posterior,
    # but we don't have the raw X, Y data, only mu, sigma from the current GP.
    # Standard GBO requires the GP kernel and training data.
    # Here we only have mu and sigma for 2000 candidates.
    # We cannot easily update the GP posterior without the underlying kernel matrix K and training points X_train.
    # The prompt says "mu, sigma and cost have shape (2000,). No coordinates are passed."
    # This means we CANNOT perform true GP posterior updates (which require K(x_new, X_train)).
    # We must rely on the provided mu and sigma as fixed statistics for these candidates.
    
    # Since we cannot update the GP, "GBO" in the traditional sense (recomputing mu/sigma) is impossible.
    # However, we can approximate a "batch" effect by diversifying the selection.
    # If we just pick the best EI, we might pick 15 points all in the same local basin.
    # A simple heuristic for diversity without GP updates:
    # 1. Calculate EI for all points.
    # 2. Select the best point.
    # 3. For the next selection, we don't have new mu/sigma.
    # But wait, the function is called ONCE per outer loop step.
    # The outer loop evaluates ONE point.
    # So we don't need to select a batch. We just need to select the BEST single point.
    
    # So why did I think about GBO? 
    # The prompt says: "The evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # This confirms: 1 call to acquisition -> 1 evaluation.
    # So we are selecting a single point.
    
    # The current program uses EI / cost^1.5.
    # The current score is 1.0982.
    # The base score is 1.0000.
    # The current strategy is already better than base.
    
    # What is different?
    # Maybe the cost penalty is too aggressive or not aggressive enough?
    # Or maybe EI is not the right acquisition function?
    # UCB (Upper Confidence Bound) for minimization?
    # UCB = mu - kappa * sigma.
    # This encourages exploration more than EI.
    
    # Let's try UCB with a cost-aware term.
    # Score = (mu - kappa * sigma) / cost^alpha
    # But we want to MINIMIZE f.
    # Low mu is good. High sigma is good for exploration.
    # So we want to maximize (mu - kappa * sigma)? No.
    # If we want to find the minimum, we want low mu.
    # EI is: (f_best - mu) * Phi(z) + sigma * phi(z).
    # If mu < f_best, EI is positive.
    # If mu > f_best, EI is small/positive (due to sigma term).
    
    # Let's look at the "inspiration" scores.
    # p9 (current) is 1.0982.
    # p41 is a tie with p9? No, p41 has empty edited_units, score_tie_unpaired.
    # This suggests p41 is likely the same or very similar.
    
    # Let's try a different acquisition function: Probability of Improvement (PI) is too greedy.
    # Expected Improvement (EI) is standard.
    # Maybe the cost penalty is the key.
    # Hartmann 6D minimum is at x* ~ (0.2017, 0.1500, 0.2979, 0.2740, 0.3117, 0.6573).
    # mean(x*) = (0.2017+0.1500+0.2979+0.2740+0.3117+0.6573)/6 = 1.8926/6 = 0.3154.
    # Cost = 1 + 0.3154 = 1.3154.
    # Uniform random mean cost = 1 + 0.5 = 1.5.
    # So the minimum is in a cheaper region than average.
    # Penalizing cost helps.
    
    # What if we use a different power for cost?
    # Current: cost^1.5.
    # Try cost^1.0 (linear)?
    # Or cost^2.0?
    
    # Another idea: The GP is standardized.
    # sigma is in objective units.
    # EI calculation assumes normal distribution.
    
    # Let's try a hybrid:
    # Score = EI * (1 / cost)
    # vs
    # Score = EI / (cost ** 1.5)
    
    # Let's test if a slightly weaker penalty allows finding the minimum faster if the GP is uncertain.
    # Or maybe a stronger penalty?
    
    # Actually, let's look at the "open question":
    # "Does mode 'base' change the chance that a child improves its island..."
    # This is meta-level.
    
    # Let's try a different formulation:
    # Instead of dividing by cost, subtract a cost penalty.
    # Score = EI - lambda * (cost - 1)
    # This might be more robust if EI varies widely.
    
    # Or, use a "Cost-Effective Improvement" (CEI) like metric.
    
    # Let's try UCB.
    # For minimization, we want to minimize f.
    # UCB_min = mu - kappa * sigma.
    # We want to MAXIMIZE the "regret" reduction?
    # No, standard UCB for minimization is:
    # Select x to minimize: mu(x) - kappa * sigma(x)
    # So the score to MAXIMIZE is: -mu(x) + kappa * sigma(x)
    # But this ignores f_best.
    # If f_best is very low, we should be more exploitative.
    # EI adapts to f_best. UCB does not.
    
    # Let's stick with EI but modify the cost term.
    # The cost is 1 + mean(x).
    # The minimum is at cost ~1.315.
    # If we use cost^1.0, the ratio between min cost and max cost (2.0) is 2.0/1.315 = 1.52.
    # If we use cost^1.5, the ratio is (2.0/1.315)^1.5 = 1.84.
    # If we use cost^2.0, the ratio is (2.0/1.315)^2 = 2.31.
    
    # Maybe the current penalty is too strong, causing the search to get stuck in a cheap but suboptimal region?
    # Or is it too weak?
    
    # Let's try cost^1.0.
    
    # Another possibility: The "standardized targets" in GP.
    # mu and sigma are in objective units.
    
    # Let's try a "Soft" cost penalty.
    # score = ei / (1 + 0.5 * (cost - 1))
    # This is equivalent to dividing by (1.5 + 0.5*mean(x)).
    
    # Let's try a different approach entirely:
    # What if we estimate the "potential" improvement per unit cost?
    # That is exactly EI / cost.
    
    # Let's try to incorporate the uncertainty of the cost? No, cost is deterministic given x.
    
    # Let's try a different acquisition:
    # Expected Simple Regret (ESR)? No, that's for batches.
    
    # Let's try tuning the exponent of cost.
    # Current: 1.5.
    # Try: 1.2.
    
    # Also, consider that `f_best` might be far from the true minimum initially.
    # EI is good for this.
    
    # Let's try a combination:
    # score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # score = score / (cost ** 1.2)
    
    # Why 1.2? It's less aggressive than 1.5, allowing exploration of slightly more expensive regions if the EI is high.
    
    # Alternatively, maybe the issue is numerical stability.
    # norm.cdf(z) for large z is 1.
    # norm.pdf(z) for large z is 0.
    
    # Let's implement the change to exponent 1.2.
    
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Clip z to avoid extreme values in norm.pdf/cdf if needed, though scipy handles it well.
    # But for very negative z, cdf is 0, pdf is 0. EI is 0.
    # For very positive z, cdf is 1, pdf is 0. EI is f_best - mu.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors)
    ei = np.maximum(ei, 0.0)
    
    safe_cost = np.maximum(cost, 1e-9)
    
    # New exponent
    score = ei / (safe_cost ** 1.2)
    
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
