"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start adaptive Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, time_limit):
        """Run adaptive Nelder-Mead from x0, using at most 'budget' evaluations."""
        dim = self.dim
        start_time = time.time()
        
        # Initialize simplex
        simplex = [x0.copy()]
        f_vals = [func(x0)]
        budget_used = 1
        
        # Initial step size
        step = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            f_vals.append(func(xi))
            budget_used += 1
            if budget_used >= budget or time.time() > time_limit:
                break
        
        if budget_used >= budget or time.time() > time_limit:
            # No more budget for NM steps, just return best
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx], budget_used
        
        # NM parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5
        
        while budget_used < budget and time.time() < time_limit:
            # Sort simplex
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            budget_used += 1
            
            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                budget_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_vals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                budget_used += 1
                
                if f_c < min(f_r, f_vals[-1]):
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if budget_used < budget and time.time() < time_limit:
                            f_vals[i] = func(simplex[i])
                            budget_used += 1
                        else:
                            break
                    
                    if budget_used >= budget or time.time() > time_limit:
                        break
            
            # Check for convergence
            if np.max(f_vals) - np.min(f_vals) < 1e-10:
                break
        
        best_idx = np.argmin(f_vals)
        return simplex[best_idx], f_vals[best_idx], budget_used

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        total_budget = self.budget
        
        # Multi-start strategy:
        # 1. Coarse random sampling (20 points)
        # 2. Pick top 5
        # 3. Run Nelder-Mead on each with remaining budget divided by 5
        
        # Phase 1: Coarse sampling
        n_coarse = 20
        coarse_points = []
        coarse_fvals = []
        
        for i in range(n_coarse):
            if time.time() > time_limit:
                break
            # Random point in [-5, 5]^dim
            x = np.random.uniform(-5, 5, size=dim)
            f = func(x)
            coarse_points.append(x)
            coarse_fvals.append(f)
        
        if not coarse_points:
            return np.zeros(dim), np.inf
        
        # Phase 2: Select top 5
        coarse_fvals = np.array(coarse_fvals)
        top_indices = np.argsort(coarse_fvals)[:min(5, len(coarse_points))]
        
        # Phase 3: Nelder-Mead refinement
        remaining_budget = total_budget - len(coarse_points)
        n_refine = len(top_indices)
        budget_per_refine = max(1, remaining_budget // n_refine)
        
        best_x = coarse_points[top_indices[0]]
        best_f = coarse_fvals[top_indices[0]]
        
        for idx in top_indices:
            if time.time() > time_limit:
                break
            
            x0 = coarse_points[idx]
            f0 = coarse_fvals[idx]
            
            # Run Nelder-Mead
            x_best, f_best, used = self._nelder_mead(func, x0, budget_per_refine, time_limit)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best
        
        return best_x, best_f
# EVOLVE-BLOCK-END
