"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling + custom Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, remaining_budget, lower=-5.0, upper=5.0):
        """
        Custom Nelder-Mead that strictly respects the remaining budget.
        Returns (best_x, best_f, calls_made).
        """
        dim = self.dim
        
        # Count calls
        call_count = [0]
        
        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > remaining_budget:
                raise RuntimeError("Budget exceeded")
            return func(x)
        
        # Initialize simplex
        # Start with a small simplex around x0
        step = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = []
        for i, v in enumerate(simplex):
            fv = counted_func(v)
            fvals.append(fv)
            if call_count[0] >= remaining_budget:
                break
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Nelder-Mead iterations
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5  # shrink
        
        while call_count[0] < remaining_budget:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if len(fvals) > 1:
                if abs(fvals[-1] - fvals[0]) < 1e-10 * max(1.0, abs(fvals[0])):
                    break
            
            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, lower, upper)
            
            if call_count[0] >= remaining_budget:
                break
            f_r = counted_func(x_r)
            
            if f_r < fvals[0]:
                # Try expansion
                if call_count[0] < remaining_budget:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lower, upper)
                    f_e = counted_func(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                
                x_c = np.clip(x_c, lower, upper)
                
                if call_count[0] >= remaining_budget:
                    break
                f_c = counted_func(x_c)
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                    
                    if call_count[0] >= remaining_budget:
                        break
                    fvals[i] = counted_func(simplex[i])
            
            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()
        
        return best_x, best_f, call_count[0]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0

        # Sobol sequence for systematic sampling
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_sobol = min(50, budget)
        # Generate Sobol points in [0,1]^dim
        sobol_points = sampler.random(n_sobol)
        # Map to [-5, 5]^dim
        sobol_points = sobol_points * (upper - lower) + lower

        best_x = None
        best_f = np.inf

        for i in range(n_sobol):
            x = sobol_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_made = n_sobol
        remaining = budget - calls_made

        if remaining <= 0:
            return best_x, best_f

        # First Nelder-Mead refinement
        try:
            best_x, best_f, nm_calls = self._nelder_mead(
                func, best_x, remaining, lower, upper
            )
            calls_made += nm_calls
            remaining = budget - calls_made
        except Exception:
            pass

        # If we still have budget, do a second refinement with smaller step
        if remaining > 10:
            try:
                # Restart with a tighter simplex
                step = 0.1
                x0 = best_x.copy()
                simplex2 = [x0.copy()]
                for i in range(dim):
                    xi = x0.copy()
                    xi[i] += step
                    simplex2.append(xi)
                
                # Evaluate
                fvals2 = []
                for v in simplex2:
                    if remaining <= 0:
                        break
                    fv = func(v)
                    remaining -= 1
                    fvals2.append(fv)
                
                # Run a few NM iterations
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                for _ in range(min(20, remaining // (dim + 1))):
                    if remaining < dim + 1:
                        break
                    
                    order = np.argsort(fvals2)
                    simplex2 = [simplex2[i] for i in order]
                    fvals2 = [fvals2[i] for i in order]
                    
                    if abs(fvals2[-1] - fvals2[0]) < 1e-10 * max(1.0, abs(fvals2[0])):
                        break
                    
                    centroid = np.mean(simplex2[:-1], axis=0)
                    x_r = centroid + alpha * (centroid - simplex2[-1])
                    x_r = np.clip(x_r, lower, upper)
                    
                    if remaining <= 0:
                        break
                    f_r = func(x_r)
                    remaining -= 1
                    
                    if f_r < fvals2[0]:
                        if remaining > 0:
                            x_e = centroid + gamma * (x_r - centroid)
                            x_e = np.clip(x_e, lower, upper)
                            f_e = func(x_e)
                            remaining -= 1
                            
                            if f_e < f_r:
                                simplex2[-1] = x_e
                                fvals2[-1] = f_e
                            else:
                                simplex2[-1] = x_r
                                fvals2[-1] = f_r
                        else:
                            simplex2[-1] = x_r
                            fvals2[-1] = f_r
                    elif f_r < fvals2[-2]:
                        simplex2[-1] = x_r
                        fvals2[-1] = f_r
                    else:
                        if f_r < fvals2[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex2[-1] - centroid)
                        
                        x_c = np.clip(x_c, lower, upper)
                        
                        if remaining <= 0:
                            break
                        f_c = func(x_c)
                        remaining -= 1
                        
                        if f_c < min(f_r, fvals2[-1]):
                            simplex2[-1] = x_c
                            fvals2[-1] = f_c
                        else:
                            for i in range(1, len(simplex2)):
                                simplex2[i] = simplex2[0] + sigma * (simplex2[i] - simplex2[0])
                                simplex2[i] = np.clip(simplex2[i], lower, upper)
                            
                            if remaining <= 0:
                                break
                            fvals2[i] = func(simplex2[i])
                            remaining -= 1
                    
                    current_best_idx = np.argmin(fvals2)
                    if fvals2[current_best_idx] < best_f:
                        best_f = fvals2[current_best_idx]
                        best_x = simplex2[current_best_idx].copy()
                        
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
