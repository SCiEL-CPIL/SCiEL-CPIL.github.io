"""Latin Hypercube global search + Nelder-Mead local refinement for BBOB f4."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """LHS global coverage + Nelder-Mead basin refinement for BBOB f4 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        def remaining():
            return budget - call_count

        # ---------- Latin Hypercube Sampling for global coverage ----------
        def lhs_sample(n, dim, rng):
            """Generate n points via Latin Hypercube Sampling in [-5,5]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                # Shuffle intervals for each dimension
                perm = rng.permutation(n)
                # Each point gets a unique interval in each dimension
                points[:, d] = (perm + rng.uniform(size=n)) / n * 10.0 - 5.0
            return points

        # ---------- Nelder-Mead local refinement ----------
        def nelder_mead(x0, f0, max_calls=80):
            """Nelder-Mead with adaptive scaling for basin refinement."""
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(f0)
            
            # Initial simplex
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                if abs(xi[i]) > 1e-10:
                    xi[i] *= 1.1
                else:
                    xi[i] += 0.1
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = [f0]
            for i in range(1, dim + 1):
                if not time_ok() or remaining() <= 2:
                    break
                fi = do_call(simplex[i])
                fvals.append(fi)
            
            if len(fvals) < dim + 1:
                return x, f
            
            # Nelder-Mead iterations
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink
            
            for it in range(max_calls):
                if not time_ok() or remaining() <= 2:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = do_call(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = do_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    fc = do_call(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if not time_ok() or remaining() <= 2:
                                break
                            fvals[i] = do_call(simplex[i])
                
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                    x = simplex[min_idx].copy()
                    f = fvals[min_idx]
                
                # Check for convergence
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break
            
            return x, f

        # ---------- Phase 1: Global search with LHS ----------
        # Use ~40% of budget for global exploration
        global_budget = max(20, int(budget * 0.4))
        
        # First, evaluate origin
        f_origin = do_call(np.zeros(dim))
        
        # Generate LHS points
        n_lhs = min(global_budget - 1, 30)
        rng = np.random.default_rng(42)
        lhs_points = lhs_sample(n_lhs, dim, rng)
        
        # Evaluate LHS points, tracking best
        for pt in lhs_points:
            if not time_ok() or remaining() <= 2:
                break
            do_call(pt)
        
        # ---------- Phase 2: Local refinement with Nelder-Mead ----------
        # Take the best point and its neighborhood, run Nelder-Mead
        # We can do multiple restarts of Nelder-Mead from top candidates
        
        # Keep track of top candidates from global phase
        # Since we don't store all, we'll just use best_x and a few perturbations
        
        local_budget = remaining()
        if local_budget > 10:
            # Primary Nelder-Mead from best point
            x_best, f_best = nelder_mead(best_x, best_f, max_calls=min(local_budget, 100))
            
            # If budget remains, try a second Nelder-Mead from a perturbed best
            if remaining() > 20:
                # Perturb best in a random direction
                rng2 = np.random.default_rng(123)
                direction = rng2.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                x_pert = best_x + 0.5 * direction
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = do_call(x_pert)
                if f_pert < best_f:
                    best_f = f_pert
                    best_x = x_pert.copy()
                
                if remaining() > 20:
                    x_best2, f_best2 = nelder_mead(best_x, best_f, max_calls=min(remaining() - 2, 80))
        
        return best_x, best_f


# EVOLVE-BLOCK-END
