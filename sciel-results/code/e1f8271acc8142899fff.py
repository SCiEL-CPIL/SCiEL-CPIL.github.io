"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Generate diverse candidates and pick top ones for NM starts
        n_candidates = 50
        candidates = rng.uniform(-5, 5, (n_candidates, dim))
        fvals = np.array([func(c) for c in candidates])
        
        # Select top 5 diverse candidates (simple greedy diversity selection)
        # Sort by fitness
        sorted_idx = np.argsort(fvals)
        
        # We want to start NM from the best few, but ensure they are somewhat spread out
        # For simplicity and robustness, we'll just take the top 5 distinct enough points
        # A simple heuristic: take the top 5 from the sorted list, but if they are too close, skip.
        # Given the low dim (5) and budget (500), even top 1-2 are strong.
        # Let's allocate budget among 3 restarts.
        
        # Actually, let's just pick the single best candidate and run a robust NM, 
        # or split budget into 2-3 NM runs.
        # The knowledge says "Multi-start Nelder-Mead initialized from top random samples provides better global coverage... Measured: 1 -> 1.117."
        
        # Strategy: 
        # 1. Evaluate 20 random points.
        # 2. Pick the best 3.
        # 3. Run Nelder-Mead on each, allocating remaining budget / 3 to each.
        
        # We already evaluated 50. Let's use the best 3.
        top_idx = sorted_idx[:3]
        starts = candidates[top_idx]
        
        # Remaining budget
        calls_made = n_candidates
        remaining_budget = budget - calls_made
        
        if remaining_budget <= 0:
            return candidates[np.argmin(fvals)], fvals.min()
        
        # Allocate budget to each restart
        # Simple split: remaining_budget // 3
        budget_per_restart = max(1, remaining_budget // 3)
        
        best_x = starts[0]
        best_f = func(best_x) # Re-evaluate to be sure, or use stored. Stored is fine.
        best_f = fvals[top_idx[0]]
        
        for i in range(3):
            if time.time() > deadline:
                break
                
            x0 = starts[i]
            f0 = fvals[top_idx[i]]
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run Nelder-Mead
            # Standard NM parameters
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink
            
            # Initialize simplex
            # x0 is the best point.
            # Other vertices: x0 + delta * e_i
            delta = 0.1
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            
            for j in range(dim):
                simplex[j + 1] = x0.copy()
                simplex[j + 1][j] += delta
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            simplex_f[0] = f0
            for j in range(dim):
                if time.time() > deadline:
                    break
                simplex_f[j + 1] = func(simplex[j + 1])
                calls_made += 1
            
            # If we ran out of budget or time during init, break
            if time.time() > deadline:
                break
                
            # Main NM loop
            # We have budget_per_restart calls left for this instance.
            # But we must also respect the global budget and time.
            
            # Note: simplex_f[0] was already counted in the 50 candidates.
            # simplex_f[1:] were just counted.
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Calculate centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Just reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if fr > worst, inside if fr < worst but > best
                    # Standard: Inside contraction if fr < worst
                    if fr < simplex_f[-1]:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                    else:
                        # Outside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                    
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        # Shrink all points except the best towards the best
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex_f[j] = func(simplex[j])
                            calls_made += 1
                            if calls_made >= budget:
                                break
                        
                        if calls_made >= budget:
                            break
                
                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
                    
                # Check for convergence (optional, but helps stop early if stuck)
                # If simplex volume is very small, break
                # Simple heuristic: if max_f - min_f < 1e-10
                if simplex_f[-1] - simplex_f[0] < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
