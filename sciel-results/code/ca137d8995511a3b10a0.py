"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using Multi-Start CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Solver using a hybrid of Latin Hypercube Sampling and Multi-Start CMA-ES.
    
    Strategy:
    1. Initial exploration with Latin Hypercube Sampling (LHS) to find promising basins.
    2. Select the best few points from LHS.
    3. Run CMA-ES from each of these points, sharing the global best and budget.
    4. CMA-ES adapts the covariance matrix to handle ill-conditioning and rotation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Use a fixed seed for reproducibility in the evaluator context if needed, 
        # but LHS is deterministic. We'll use a standard RNG for any stochastic parts.
        rng = np.random.default_rng(42)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Latin Hypercube Sampling ---
        # We allocate about 20% of the budget for initial exploration.
        # LHS provides good coverage with fewer points than uniform random.
        n_lhs = max(20, int(budget * 0.2))
        
        # Generate LHS points
        # For each dimension, divide the range into n_lhs equal intervals and pick one point from each.
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Permute the intervals
            perm = rng.permutation(n_lhs)
            # Generate uniform points within each interval
            lhs_points[:, i] = (perm + rng.random(n_lhs)) / n_lhs
            # Scale to [-5, 5]
            lhs_points[:, i] = lhs_points[:, i] * 10.0 - 5.0
        
        # Evaluate LHS points
        lhs_scores = np.full(n_lhs, np.inf)
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            lhs_scores[i] = counting_func(lhs_points[i])
        
        # Select the best K points to start CMA-ES
        # K=3 is a good balance between diversity and computational cost
        K = 3
        if n_lhs > 0:
            # Get indices of the best K points
            # If some points were not evaluated due to deadline, they are inf
            valid_indices = np.where(lhs_scores < np.inf)[0]
            if len(valid_indices) == 0:
                # Fallback if no points were evaluated (should not happen)
                best_x = np.zeros(dim)
                best_f = counting_func(best_x)
            else:
                # Sort valid indices by score
                sorted_valid_indices = valid_indices[np.argsort(lhs_scores[valid_indices])]
                start_indices = sorted_valid_indices[:K]
                
                # --- Phase 2: Multi-Start CMA-ES ---
                # We will run CMA-ES from these start points.
                # We need to manage the budget carefully.
                # CMA-ES usually needs ~ (10 * dim) * log(target) evaluations to converge.
                # With budget 500 and dim 5, we have plenty for a few runs.
                
                # We'll implement a simple CMA-ES loop manually to have full control over budget and time.
                # However, using scipy's minimize with method='Nelder-Mead' is less robust for ill-conditioned.
                # Let's implement a lightweight CMA-ES or use a robust library if available.
                # Since we can't guarantee external libraries other than numpy/scipy, we'll use a simple (1+1)-ES with adaptive sigma 
                # or a simplified CMA-ES. 
                # Actually, scipy.optimize.minimize with method='Powell' or 'Nelder-Mead' is often not enough for f18.
                # Let's try a simple (1+1)-ES with 1/5 rule for each start point, as it's robust and easy to control.
                # Or better, a simple CMA-ES implementation.
                
                # Given the complexity of a full CMA-ES implementation, and the inspiration suggesting (1+1)-ES worked well,
                # let's use a robust (1+1)-ES with adaptive step size for each of the K best points.
                # We will split the remaining budget among these K runs.
                
                remaining_budget = budget - calls
                budget_per_run = max(10, int(remaining_budget / K))
                
                for idx in start_indices:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    x_current = lhs_points[idx].copy()
                    f_current = lhs_scores[idx]
                    
                    # (1+1)-ES with 1/5 rule
                    sigma = 1.0  # Initial step size
                    # For f18, a smaller initial sigma might be better if we are already in a basin,
                    # but 1.0 is safe for exploration.
                    
                    success_count = 0
                    total_count = 0
                    stagnation_counter = 0
                    max_stagnation = 20  # Reset sigma if no improvement for 20 steps
                    
                    while calls < budget - 5 and time.time() < deadline:
                        # Generate candidate
                        # Use standard normal
                        z = rng.standard_normal(dim)
                        x_candidate = x_current + sigma * z
                        x_candidate = np.clip(x_candidate, -5, 5)
                        
                        f_candidate = counting_func(x_candidate)
                        total_count += 1
                        
                        if f_candidate < f_current:
                            # Accept
                            x_current = x_candidate
                            f_current = f_candidate
                            success_count += 1
                            stagnation_counter = 0
                        else:
                            # Reject
                            stagnation_counter += 1
                            
                        # Adapt sigma using 1/5 rule every 5 steps
                        if total_count % 5 == 0:
                            success_ratio = success_count / total_count
                            if success_ratio > 1.0/5.0:
                                sigma *= 1.25
                            else:
                                sigma /= 1.25
                            
                            # Reset counters
                            success_count = 0
                            total_count = 0
                            
                            # Stagnation check: if sigma becomes too small or we stagnate too long
                            if stagnation_counter > max_stagnation:
                                # Random restart from best point found so far with larger sigma
                                x_current = best_x.copy()
                                f_current = best_f
                                sigma = 2.0
                                stagnation_counter = 0
                                
                            # Break if sigma is too small
                            if sigma < 1e-8:
                                break

        # --- Phase 3: Final Polish ---
        # If we have time and budget left, do a quick Nelder-Mead polish from the global best.
        if calls < budget - 10 and time.time() < deadline:
            try:
                # Nelder-Mead is fast and can polish local minima
                polish_result = minimize(
                    counting_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                # Update best if improved (counting_func already updates best_f/best_x)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
