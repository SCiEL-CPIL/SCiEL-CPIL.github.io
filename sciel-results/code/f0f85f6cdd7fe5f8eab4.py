"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 290.0 # 330s limit, 40s safety margin

def acquisition(mu, sigma, f_best, cost):
    current_time = time.time()
    if current_time > TIME_LIMIT:
        # Emergency fallback: return finite values to prevent crash/timeout penalty
        return np.zeros_like(mu)

    # 1. Reconstruct candidate coordinates from the cost signal
    # cost = 1 + mean(x_scaled), where x_scaled = (x + 2) / 4
    # mean(x_scaled) = (mean(x) + 2) / 4
    # cost - 1 = (mean(x) + 2) / 4
    # mean(x) = 4 * (cost - 1) - 2
    mean_x = 4.0 * (cost - 1.0) - 2.0
    
    # We have the mean of the 2 coordinates. 
    # The distribution of 2000 uniform points in [-2, 2]^2 is roughly uniform.
    # The cost signal is a low-resolution projection.
    # However, we don't have individual x, y. We only have the "mean x" for each candidate?
    # No, cost has shape (2000,). Each candidate i has a specific x_i, y_i.
    # cost_i = 1 + (x_i_scaled + y_i_scaled)/2
    # This means we only know the sum x_i + y_i for each candidate.
    # We cannot uniquely reconstruct x_i and y_i from cost_i alone.
    
    # WAIT. The task says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This implies we must make a decision based ONLY on mu, sigma, and cost.
    # Since we cannot know the exact x,y, we cannot calculate distance to (0,0) directly for each point.
    
    # Let's rethink. Is there a correlation between cost and proximity to (0,0)?
    # Min of cost is 1 (when x+y = -4, i.e., x=-2, y=-2).
    # Max of cost is 2 (when x+y = 4, i.e., x=2, y=2).
    # The global minimum of Rosenbrock is at (0,0).
    # At (0,0), x+y=0. cost = 1 + (0+0)/2 = 1.
    # So (0,0) has cost 1.
    # Points with cost close to 1 are either near (-2,-2) or near (0,0) or other points where x+y=-4?
    # No, x,y in [-2,2]. x+y ranges from -4 to 4.
    # cost = 1 + (x+y)/4.
    # cost = 1 => x+y = -4 => Only (-2, -2) is in the box.
    # cost = 1.5 => x+y = 0 => Line from (-2, 2) to (2, -2). Includes (0,0).
    # cost = 2 => x+y = 4 => Only (2, 2) is in the box.
    
    # So low cost favors the corner (-2, -2). High cost favors (2, 2).
    # The target (0,0) has medium cost (1.5).
    # This makes cost a poor proxy for distance to (0,0) if used alone.
    
    # However, the GP (mu, sigma) carries the spatial information.
    # The GP was trained on points. mu(x) approximates f(x).
    # If the GP has learned the landscape, mu will be low near the minimum.
    # The current strategy (EI) works well if the GP is good.
    # Why did EI get 1.0? Maybe it's good enough, or maybe it gets stuck in local minima or wastes time.
    
    # Alternative Strategy: Lower Confidence Bound (LCB)
    # For minimization, we want to find low f(x).
    # LCB = mu - kappa * sigma
    # We want to maximize the "optimism" in the presence of uncertainty.
    # Standard EI is robust but can be conservative.
    # LCB is often faster for simple functions like Rosenbrock.
    
    # Let's try a hybrid:
    # 1. Use LCB to explore.
    # 2. Penalize high cost to save budget?
    # The current EI divides by cost. This encourages low cost.
    # Low cost means x+y is negative. This biases towards the bottom-left.
    # If the minimum is at (0,0), biasing towards bottom-left is neutral-to-slightly-bad (since (0,0) is cost 1.5, bottom-left is cost 1).
    # Wait, if we bias towards low cost, we evaluate points near (-2,-2).
    # f(-2,-2) = (1 - (-2))^2 + 100*(-2 - 4)^2 = 9 + 3600 = 3609. Very high.
    # So dividing by cost might actually be HURTING performance by pulling the search towards the expensive-to-evaluate (in terms of cost metric) or rather, the "cheap" to evaluate points which are actually far from the minimum?
    # No, "cost" in the problem is the evaluation cost.
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # This is a fictional cost. It doesn't affect the function value.
    # The goal is to minimize f(x).
    # The constraint is total cost < 30.
    # If we pick cheap points (cost ~1), we can afford more evaluations.
    # If we pick expensive points (cost ~2), we afford fewer.
    # So dividing by cost (Cost-aware EI) is theoretically sound to maximize number of samples.
    
    # But maybe the GP is not calibrated well?
    # Let's try a different family: **Knowledge Gradient (KG)** approximation or **Probability of Improvement (PI)** with a twist.
    
    # Actually, let's look at the "Inspiration" that scored 0.07. It mixed EI and UCB.
    # The seed scored 1.0.
    
    # Let's try **Lower Confidence Bound (LCB)** normalized by cost.
    # LCB = mu - kappa * sigma
    # We want to select points that might have low f(x).
    # score = (mu - kappa * sigma) / cost
    # We want to MAXIMIZE this score?
    # If f(x) is small, mu is small. mu - kappa*sigma is very small (negative).
    # So we want to MINIMIZE (mu - kappa*sigma)/cost?
    # Or MAXIMIZE (cost - (mu - kappa*sigma))? No.
    
    # Standard LCB for minimization:
    # Select x that minimizes: mu(x) - kappa * sigma(x)
    # Here we return a score where HIGHER = evaluate next.
    # So we should return: - (mu - kappa * sigma) / cost
    # Or: ( -mu + kappa * sigma ) / cost
    
    # Let's test this logic.
    # If mu is low (good), -mu is high. Score is high. Good.
    # If sigma is high (uncertain), kappa*sigma is high. Score is high. Good (exploration).
    # If cost is high, score is lower. Good (cost awareness).
    
    # What value for kappa?
    # Early on, we want to explore. Kappa should be large.
    # Later, we want to exploit. Kappa should be small.
    # We can use f_best to adapt kappa.
    # f_best starts high (Rosenbrock at random points is large, e.g., 100+).
    # f_best drops to 0.
    
    # Adaptive Kappa:
    # kappa = c / (1 + f_best)
    # If f_best is 100, kappa ~ 0.
    # If f_best is 0, kappa ~ c.
    # This is backwards. We want to explore MORE when we are far from the minimum (high f_best).
    # So kappa should be LARGE when f_best is HIGH.
    # kappa = c * (1 + f_best) ? No, that explodes.
    # kappa = c * np.log(1 + f_best) ?
    
    # Let's try a simpler, robust switch.
    # Strategy: **Cost-Aware LCB with Adaptive Kappa**
    
    # 1. Calculate LCB = mu - kappa * sigma
    # 2. Score = -LCB / cost = (kappa * sigma - mu) / cost
    # 3. Maximize Score.
    
    # Adaptive Kappa:
    # If f_best > 10, we are far away. Explore aggressively. kappa = 3.0
    # If 1 < f_best < 10, moderate. kappa = 1.5
    # If f_best < 1, we are close. Exploit. kappa = 0.5
    
    # This is a discrete switch, which is robust.
    
    # Let's refine the "Different Strategy" claim.
    # Current: Expected Improvement (EI) / Cost.
    # New: Lower Confidence Bound (LCB) / Cost with Adaptive Exploration.
    # EI is a non-linear, probabilistic improvement metric. LCB is a linear, heuristic confidence interval bound.
    # LCB is often computationally faster and can escape local optima more easily if kappa is tuned, whereas EI can be "sticky" if the GP is slightly off.
    
    # Implementation:
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Determine kappa based on f_best
    # f_best is a scalar
    if f_best > 10.0:
        kappa = 3.0
    elif f_best > 1.0:
        kappa = 1.5
    else:
        kappa = 0.5
        
    # LCB = mu - kappa * sigma
    # We want to minimize LCB.
    # Score = -LCB / cost = (kappa * sigma - mu) / cost
    
    score = (kappa * sigma_safe - mu) / cost
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # The problem asks for "higher = evaluate next".
    # If mu is very low (e.g. 0), kappa*sigma is small (e.g. 0.5*0.1=0.05).
    # Score = (0.05 - 0) / 1.5 = 0.033
    # If mu is high (e.g. 100), sigma is small.
    # Score = (0.05 - 100) / 1.5 = -66
    # So low mu gives higher score. Correct.
    
    # If sigma is high (uncertain), kappa*sigma is large.
    # Score is higher. Correct.
    
    return score

# EVOLVE-BLOCK-END
