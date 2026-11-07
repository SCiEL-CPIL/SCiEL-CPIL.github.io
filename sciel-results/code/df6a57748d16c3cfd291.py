import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-front selection.
    
    Structural Approach:
    1. Compute Expected Improvement (EI) and Uncertainty (UCB) for each candidate.
    2. Formulate a multi-objective problem: Maximize 'Potential Improvement' and Minimize 'Cost'.
    3. Identify the Pareto front of candidates (non-dominated points).
    4. Score candidates based on their membership in the Pareto front and their 
       relative efficiency (Improvement per Cost) among front members.
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a neutral score vector if we are running out of time
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds [1, 2] for this task
    cost = np.clip(cost, 1.0, 2.0)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Uncertainty-based Exploration Score (UCB-like)
    # We use a beta parameter to balance exploitation (EI) and exploration (sigma)
    # As f_best gets closer to 0 (min), we might want to explore less? 
    # Actually, for Rastrigin, the landscape is rugged.
    # Let's create a "Potential Score" that combines EI and a confidence bound.
    # Potential Score = EI + kappa * sigma
    # We choose kappa dynamically: if f_best is high, explore more. If low, exploit more.
    # However, EI already accounts for uncertainty. 
    # To make the Pareto front more robust, we use a "Weighted EI" that penalizes 
    # high sigma slightly if the expected improvement is low.
    
    # Alternative: Just use EI as the primary improvement metric, 
    # but normalize it to be comparable across instances? 
    # The task says mu/sigma are in objective units.
    
    # Let's define the "Improvement Objective" for the Pareto front.
    # We want to maximize this.
    # A simple EI is good. But to avoid getting stuck in local minima of the GP,
    # we can add a small exploration term.
    # Exploration Term = sigma * (1.0 / (1.0 + f_best)) 
    # This scales exploration down as we get closer to the global min (0).
    
    exploration_weight = 1.0 / (1.0 + np.abs(f_best))
    exploration_score = sigma * exploration_weight
    
    # Combined Improvement Metric
    # We normalize EI and Exploration to [0, 1] roughly? 
    # No, Pareto works on raw scales, but mixing scales (EI vs sigma) is tricky.
    # EI is in function units. Sigma is in function units.
    # Let's use a ratio: Improvement Efficiency = EI / (1 + sigma)
    # This rewards high EI and low uncertainty.
    
    improvement_score = ei / (1.0 + sigma)
    
    # 3. Pareto Front Selection
    # Objective 1: Maximize improvement_score
    # Objective 2: Minimize cost
    
    n = len(improvement_score)
    if n == 0:
        return np.array([])
    
    # We want to find non-dominated points.
    # Point i is dominated by j if:
    # improvement_score[j] >= improvement_score[i] AND cost[j] <= cost[i]
    # AND at least one inequality is strict.
    
    # Efficient O(N log N) Pareto front finding for 2D:
    # Sort by cost ascending.
    # Iterate, keeping track of max improvement_score seen so far.
    # If current improvement_score > max_seen_so_far, it's on the Pareto front.
    
    cost_order = np.argsort(cost)
    sorted_cost = cost[cost_order]
    sorted_improvement = improvement_score[cost_order]
    
    is_pareto = np.zeros(n, dtype=bool)
    
    # The first element (lowest cost) is always Pareto optimal
    if n > 0:
        is_pareto[cost_order[0]] = True
    
    max_improvement_so_far = sorted_improvement[0]
    
    for i in range(1, n):
        # If current improvement is strictly greater than the max improvement seen at lower costs,
        # it is non-dominated.
        if sorted_improvement[i] > max_improvement_so_far:
            is_pareto[cost_order[i]] = True
            max_improvement_so_far = sorted_improvement[i]
        else:
            # It is dominated by a previous point (lower cost, higher/equal improvement)
            pass
            
    # 4. Scoring
    # Points on the Pareto front get high scores.
    # Points off the front get low scores.
    # Among Pareto points, we rank them by their "Hypervolume contribution" or simply 
    # by their efficiency (improvement / cost) to break ties and prioritize 
    # the most efficient parts of the front.
    
    final_score = np.zeros(n, dtype=np.float64)
    
    pareto_indices = np.where(is_pareto)[0]
    non_pareto_indices = np.where(~is_pareto)[0]
    
    if len(pareto_indices) > 0:
        # Rank Pareto points by Efficiency (Improvement / Cost)
        # Higher efficiency = better score
        pareto_costs = cost[pareto_indices]
        pareto_improvements = improvement_score[pareto_indices]
        
        # Avoid division by zero (cost is >= 1.0)
        efficiency = pareto_improvements / pareto_costs
        
        # Normalize efficiency to [0, 1]
        eff_min = np.min(efficiency)
        eff_max = np.max(efficiency)
        
        if eff_max > eff_min:
            norm_eff = (efficiency - eff_min) / (eff_max - eff_min)
        else:
            norm_eff = np.ones_like(efficiency)
            
        # Assign scores in [0.8, 1.0] for Pareto front
        final_score[pareto_indices] = 0.8 + 0.2 * norm_eff
        
    if len(non_pareto_indices) > 0:
        # For non-Pareto points, we still need to return finite scores.
        # We rank them by their improvement_score, but keep them low.
        # This ensures that if the Pareto front is small, the algorithm 
        # doesn't completely ignore the rest, but strongly prefers the front.
        
        non_pareto_improvements = improvement_score[non_pareto_indices]
        
        imp_min = np.min(non_pareto_improvements)
        imp_max = np.max(non_pareto_improvements)
        
        if imp_max > imp_min:
            norm_imp = (non_pareto_improvements - imp_min) / (imp_max - imp_min)
        else:
            norm_imp = np.zeros_like(non_pareto_improvements)
            
        # Assign scores in [0.0, 0.5] for non-Pareto
        final_score[non_pareto_indices] = 0.5 * norm_imp
        
    # Final sanity check: ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_score
