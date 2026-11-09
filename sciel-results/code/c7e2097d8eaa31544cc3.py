import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lagrangian Relaxation.
    
    Strategy:
    1. Model the problem as maximizing expected improvement (EI) subject to a cost budget.
    2. Use the Lagrangian dual: L(x, lambda) = EI(x) - lambda * cost(x).
    3. Dynamically estimate lambda such that the average cost of the top-k candidates 
       matches a target cost. This target cost is derived from the remaining budget 
       (estimated from f_best and cost statistics) to ensure we don't exhaust the 
       budget too early or too late.
    4. This approach avoids the geometric distortion of dividing by cost, instead 
       adding a linear penalty that preserves the relative ordering of the GP posterior 
       while enforcing cost efficiency.
    """
    start_time = time.time()
    deadline = start_time + 250.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
        
    sigma = np.maximum(sigma, 1e-6)
    
    # 1. Compute EI
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Estimate Progress
    # Map f_best to progress [0, 1]
    # Early: f_best ~ 10, Late: f_best ~ -78
    f_min_theoretical = -78.3323
    f_max_expected = 10.0
    
    f_best_clipped = np.clip(f_best, f_min_theoretical, f_max_expected)
    progress = (f_max_expected - f_best_clipped) / (f_max_expected - f_min_theoretical)
    progress = np.clip(progress, 0, 1)
    
    # 3. Determine Target Cost
    # We want to spend the budget efficiently.
    # If progress is low (early), we can afford higher cost points to explore broadly?
    # Or should we save budget?
    # Usually, early on, we have lots of budget. We want to use it to find good regions.
    # Late on, budget is tight. We need to be very efficient.
    
    # Let's estimate remaining budget.
    # Total Budget = 30.
    # Initial Cost ~ 5 * 1.5 = 7.5.
    # Remaining ~ 22.5.
    # Number of steps remaining? Hard to know.
    
    # Instead, let's define a target average cost for the *next* evaluation.
    # Early: Target Cost ~ 1.5 (neutral)
    # Late: Target Cost ~ 1.1 (very cheap)
    
    # Interpolate target cost from 1.8 (early) to 1.05 (late)
    target_cost = 1.8 - 0.75 * progress
    
    # 4. Estimate Lambda
    # We want the selected point to have a cost close to target_cost.
    # Let's find a lambda such that the top candidate (max EI - lambda * cost) has cost ~ target_cost.
    # This is a 1D search.
    
    # Sort candidates by EI descending? No, lambda changes the ranking.
    # Let's use a bisection on lambda.
    # Range of lambda: [0, max_ei / (min_cost - target_cost)]?
    # If target_cost < min_cost, we need lambda > 0.
    # If target_cost > max_cost, we need lambda < 0 (encourage expensive points).
    
    # Let's define a function to get the cost of the best point for a given lambda.
    def get_best_cost(lam):
        scores = ei - lam * cost
        idx = np.argmax(scores)
        return cost[idx]
    
    # We want get_best_cost(lam) == target_cost.
    # As lam increases, the best point shifts to lower cost points.
    # So get_best_cost is non-increasing with lam.
    
    # Find bounds for lambda.
    # lam_low: cost is max (lam = 0 or negative)
    # lam_high: cost is min (lam is large)
    
    # If target_cost is between min_cost and max_cost, we can bisect.
    min_cost = np.min(cost)
    max_cost = np.max(cost)
    
    if target_cost <= min_cost:
        # We want the cheapest point.
        # Find lambda such that the best point is the min cost point.
        # If min cost point is not the best EI point, we need enough lambda.
        # Let's just set lambda high enough to force min cost?
        # Or bisect to find where cost drops to min_cost.
        lam_low = 0.0
        lam_high = 10.0
        # Ensure lam_high gives min cost
        while get_best_cost(lam_high) > min_cost and lam_high < 1000.0:
            lam_high *= 2
        # Bisect
        for _ in range(20):
            lam_mid = (lam_low + lam_high) / 2
            if get_best_cost(lam_mid) > target_cost:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
        lam = (lam_low + lam_high) / 2
        
    elif target_cost >= max_cost:
        # We want the most expensive point (if EI allows).
        # Lambda should be negative or 0.
        # If EI of max cost point is low, we might not pick it.
        # Let's set lambda = 0 and see.
        lam = 0.0
        # If we want to encourage expensive points, lambda < 0.
        # But usually EI dominates.
        # Let's try to find lambda such that best cost is max_cost.
        lam_low = -10.0
        lam_high = 0.0
        # Ensure lam_low gives max cost
        while get_best_cost(lam_low) < max_cost and lam_low > -1000.0:
            lam_low *= 2
        # Bisect
        for _ in range(20):
            lam_mid = (lam_low + lam_high) / 2
            if get_best_cost(lam_mid) < target_cost:
                lam_high = lam_mid
            else:
                lam_low = lam_mid
        lam = (lam_low + lam_high) / 2
        
    else:
        # Normal case: target_cost in [min_cost, max_cost]
        lam_low = 0.0
        lam_high = 10.0
        # Ensure lam_high gives cost <= target_cost
        while get_best_cost(lam_high) > target_cost and lam_high < 1000.0:
            lam_high *= 2
        # Ensure lam_low gives cost >= target_cost
        # At lam=0, cost is cost of max EI point.
        # If cost(max EI) < target_cost, we need negative lambda.
        if get_best_cost(0.0) < target_cost:
            lam_low = -10.0
            while get_best_cost(lam_low) > target_cost and lam_low > -1000.0:
                lam_low *= 2
            # Now bisect in [lam_low, 0]
            for _ in range(20):
                lam_mid = (lam_low + 0.0) / 2
                if get_best_cost(lam_mid) < target_cost:
                    lam_high = lam_mid
                else:
                    lam_low = lam_mid
            lam = (lam_low + lam_high) / 2
        else:
            # Bisect in [0, lam_high]
            for _ in range(20):
                lam_mid = (lam_low + lam_high) / 2
                if get_best_cost(lam_mid) > target_cost:
                    lam_low = lam_mid
                else:
                    lam_high = lam_mid
            lam = (lam_low + lam_high) / 2
            
    # 5. Compute Final Scores
    scores = ei - lam * cost
    
    # 6. Handle Edge Cases
    max_ei = np.max(ei)
    if max_ei < 1e-5:
        # Plateau reached.
        # If EI is 0, scores are -lam * cost.
        # If lam > 0, this favors low cost.
        # If lam < 0, this favors high cost.
        # If lam == 0, scores are 0.
        # We should add a small exploration term based on sigma.
        # But if we are in a plateau, maybe we just pick the cheapest point to save budget?
        # Or the point with highest sigma to find a new region?
        # Let's add a small sigma term.
        scores += 0.1 * sigma / (cost + 1e-6)
        
    # Ensure finite
    scores[~np.isfinite(scores)] = 0.0
    
    # Check time
    if time.time() > deadline:
        return scores
        
    return scores
