"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead with multi-start and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # 300s margin
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Track global best
        best_x = None
        best_f = float('inf')
        
        # We will use a multi-start strategy with Nelder-Mead
        # Allocate budget: 3 starts of ~100 evals, then polish the best with remaining ~200
        
        num_starts = 3
        evals_per_start = int(budget * 0.3)
        
        candidates = []
        
        for i in range(num_starts):
            if time.time() > deadline:
                break
                
            # Random start in [-5, 5]^dim
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            candidates.append((x0, f0))
            
            # Run Nelder-Mead from this start
            x_opt, f_opt = self._nelder_mead(
                func, x0, f0, 
                budget=evals_per_start, 
                rng=rng,
                deadline=deadline
            )
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Polish the best candidate with remaining budget
        if best_x is not None:
            remaining_budget = budget - (num_starts * evals_per_start)
            # Ensure we have some budget for polishing
            if remaining_budget > 10 and time.time() < deadline:
                # Restart Nelder-Mead at the best point with a smaller initial step to refine
                # Or just continue from where it left off? 
                # Since we don't store the simplex state easily across separate calls in a simple loop, 
                # let's run a fresh NM with a tighter bound around best_x.
                
                # Create a small initial simplex around best_x
                # The step size for NM initialization is usually 0.05 * (x_max - x_min) or similar.
                # Here we want to be local.
                x_start = best_x
                f_start = best_f
                
                x_opt, f_opt = self._nelder_mead(
                    func, x_start, f_start,
                    budget=remaining_budget,
                    rng=rng,
                    deadline=deadline,
                    initial_step=0.5 # Smaller initial step for polishing
                )
                
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, rng, deadline, initial_step=1.0):
        """
        Custom Nelder-Mead implementation with budget tracking and deadline guard.
        """
        dim = len(x0)
        n = dim + 1 # Number of vertices
        
        # Initialize simplex
        # x0 is the first vertex
        # Other vertices are x0 + initial_step * e_i
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        f_values = np.zeros(n)
        f_values[0] = f0
        
        evals_used = 1 # Count the initial evaluation
        
        for i in range(1, n):
            if time.time() > deadline or evals_used >= budget:
                break
            simplex[i] = x0.copy()
            simplex[i][i-1] += initial_step
            f_values[i] = func(simplex[i])
            evals_used += 1
        
        # Order vertices by function value
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        
        best_f = f_values[0]
        
        # Main loop
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Check for convergence: if the spread of function values is very small
            if np.max(f_values) - np.min(f_values) < 1e-12:
                break
                
            # Calculate centroid of all but the worst vertex
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            evals_used += 1
            
            if f_values[0] <= fr < f_values[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_values[-1] = fr
            elif fr < f_values[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                evals_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    rho = 0.5
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    rho = 0.5
                    xc = centroid - rho * (centroid - simplex[-1])
                
                fc = func(xc)
                evals_used += 1
                
                if fc < min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f_values[i] = func(simplex[i])
                        evals_used += 1
                        if evals_used >= budget or time.time() > deadline:
                            break
                    
                    if evals_used >= budget or time.time() > deadline:
                        break
            
            # Re-order
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
            
            # Stagnation check: if no improvement for many iterations, we might be stuck
            # For a simple implementation, we just rely on the budget and convergence check.
            # A more advanced check would track the best value over the last K iterations.

        return simplex[0], f_values[0]
# EVOLVE-BLOCK-END
