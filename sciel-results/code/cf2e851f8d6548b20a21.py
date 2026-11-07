"""Hybrid Nelder-Mead + (1+1)-ES optimizer with deterministic grid init for BBOB f11."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead and (1+1)-ES optimizer for the BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(start_point, max_iter, alpha=1.0, gamma=2.0, rho=0.5, sigma=1.0):
            """Nelder-Mead simplex method with budget awareness."""
            nonlocal call_count, best_f, best_x
            n = len(start_point)
            # Build initial simplex
            simplex = [start_point.copy()]
            for i in range(n):
                pt = start_point.copy()
                if start_point[i] + 0.25 >= -5.0:
                    pt[i] = min(pt[i] + 0.25, 5.0)
                else:
                    pt[i] = max(pt[i] - 0.25, -5.0)
                simplex.append(pt)
            
            # Evaluate all simplex points
            simplex_f = []
            for pt in simplex:
                f = try_eval(pt)
                if f is None:
                    return
                simplex_f.append(f)
            
            for _ in range(max_iter):
                if time.time() > deadline - 20:
                    break
                if call_count >= budget:
                    break
                
                # Sort simplex by function value
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5.0, 5.0)
                fr = try_eval(xr)
                if fr is None:
                    break
                
                # If reflection is good, replace worst
                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = try_eval(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (worst - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc < worst_f:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_shrink = try_eval(simplex[i])
                            if f_shrink is None:
                                return
                            simplex_f[i] = f_shrink
                        # Re-sort
                        idx = np.argsort(simplex_f)
                        simplex = [simplex[i] for i in idx]
                        simplex_f = [simplex_f[i] for i in idx]
                
                # Check convergence
                if len(simplex_f) > 1 and (max(simplex_f) - min(simplex_f)) < 1e-10:
                    break
            
            return simplex[0], simplex_f[0]

        def one_plus_one_es(start_point, max_iter, sigma=1.0):
            """(1+1)-ES with adaptive step size."""
            nonlocal call_count, best_f, best_x
            x = start_point.copy()
            fx = try_eval(x)
            if fx is None:
                return
            success_count = 0
            eval_count = 0
            
            for _ in range(max_iter):
                if time.time() > deadline - 20:
                    break
                if call_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = try_eval(y)
                if fy is None:
                    break
                eval_count += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    success_count += 1
                else:
                    sigma *= 0.85
                
                if eval_count % 10 == 0:
                    rate = success_count / 10.0
                    if rate > 0.5:
                        sigma *= 1.05
                    elif rate < 0.2:
                        sigma *= 0.95
                    success_count = 0
            
            return x, fx

        # Phase 1: Dense deterministic grid initialization to find good basins
        # Use a structured grid covering the [-4, 4]^5 box more thoroughly
        grid_points = []
        
        # Center
        grid_points.append(np.zeros(dim))
        
        # Axis-aligned points at multiple radii
        for i in range(dim):
            for val in [-3.5, -2.5, -1.5, -0.75, 0.75, 1.5, 2.5, 3.5]:
                pt = np.zeros(dim)
                pt[i] = val
                grid_points.append(pt)
        
        # Two-axis points to explore more of the space
        for i in range(dim):
            for j in range(i + 1, dim):
                for vi in [-2.0, 2.0]:
                    for vj in [-2.0, 2.0]:
                        pt = np.zeros(dim)
                        pt[i] = vi
                        pt[j] = vj
                        grid_points.append(pt)
        
        # Random points for broader coverage
        for _ in range(15):
            grid_points.append(rng.uniform(-4.0, 4.0, dim))
        
        # Evaluate grid points to find promising regions
        grid_results = []
        for gp in grid_points:
            if call_count >= budget or time.time() > deadline - 60:
                break
            gpx = np.clip(gp, -5.0, 5.0)
            gpf = try_eval(gpx)
            if gpf is None:
                break
            grid_results.append((gpx, gpf))
        
        # Sort by function value and pick top candidates for Nelder-Mead
        grid_results.sort(key=lambda x: x[1])
        
        # Phase 2: Apply Nelder-Mead to top grid points with more budget each
        remaining = budget - call_count
        if remaining > 0:
            # Use up to 5 of the best starting points
            n_restarts = min(5, len(grid_results))
            nm_budget_per = max(15, remaining // (n_restarts + 2))
            
            for idx in range(n_restarts):
                if call_count >= budget or time.time() > deadline - 40:
                    break
                gp, gpf = grid_results[idx]
                # Skip if too close to current best
                if best_x is not None and np.linalg.norm(gp - best_x) < 0.5:
                    continue
                nm_result = nelder_mead(gp, max_iter=nm_budget_per)
                if nm_result is not None:
                    nm_x, nm_f = nm_result
                    if nm_f < best_f:
                        best_x = nm_x
                        best_f = nm_f
            
            # Phase 3: Fine-tune with (1+1)-ES around best point
            if best_x is not None and call_count < budget:
                es_budget = min(40, budget - call_count)
                es_result = one_plus_one_es(best_x, max_iter=es_budget, sigma=0.3)
                if es_result is not None:
                    es_x, es_f = es_result
                    if es_f < best_f:
                        best_x = es_x
                        best_f = es_f
            
            # Phase 4: Final fine-tuning with very small steps
            if best_x is not None and call_count < budget:
                fine_budget = min(15, budget - call_count)
                for _ in range(fine_budget):
                    if time.time() > deadline - 10:
                        break
                    if call_count >= budget:
                        break
                    y_fine = np.clip(best_x + 0.03 * rng.standard_normal(dim), -5.0, 5.0)
                    fy_fine = try_eval(y_fine)
                    if fy_fine is None:
                        break

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
