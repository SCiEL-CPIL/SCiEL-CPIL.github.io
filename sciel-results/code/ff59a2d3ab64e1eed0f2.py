"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with Nelder-Mead polish for fast local convergence."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, max_calls=100, deadline=None):
        """Nelder-Mead simplex method with call counting."""
        dim = len(x0)
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        
        # Create other simplex vertices with small offsets
        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.1 if xi[i] < 4.9 else -0.1
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fvals.append(func(xi))
        
        total_calls = dim + 1  # Initial evaluations
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5  # shrink
        
        while total_calls < max_calls:
            if deadline is not None and time.time() >= deadline:
                break
            
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            total_calls += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                total_calls += 1
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
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    total_calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            total_calls += 1
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    total_calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            total_calls += 1
            
            # Check convergence
            if np.std(fvals) < 1e-10:
                break
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], total_calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        budget = self.budget
        dim = self.dim
        
        # Phase 1: Global exploration with (1+1)-ES portfolio
        n_global_runs = 3
        calls_per_global = 80
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        candidate_points = []
        
        # Diverse initial points
        initial_points = [
            rng.uniform(-5, 5, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-1, 1, dim),
        ]
        
        for run_idx, x0 in enumerate(initial_points):
            if time.time() >= deadline - 60:
                break
            if total_calls >= budget - 150:
                break
            
            x = x0.copy()
            fx = func(x)
            total_calls += 1
            sigma = 2.0
            
            # (1+1)-ES for global exploration
            for _ in range(calls_per_global - 1):
                if time.time() >= deadline - 60:
                    break
                if total_calls >= budget - 150:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                sigma = float(np.clip(sigma, 0.01, 10.0))
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            candidate_points.append((x.copy(), fx))
        
        # Phase 2: Nelder-Mead polish on top candidates
        # Sort candidates by fitness and polish the best few
        candidate_points.sort(key=lambda t: t[1])
        
        n_polish = min(3, len(candidate_points))
        nm_budget = (budget - total_calls) // max(1, n_polish)
        nm_budget = max(30, min(100, nm_budget))
        
        for i in range(n_polish):
            if time.time() >= deadline - 10:
                break
            if total_calls >= budget - 5:
                break
            
            x_start, f_start = candidate_points[i]
            
            # Nelder-Mead refinement
            x_nm, f_nm, nm_calls = self._nelder_mead(
                func, x_start, max_calls=nm_budget, deadline=deadline - 10
            )
            total_calls += nm_calls
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
        
        # Phase 3: Final fine-tuning with small sigma (1+1)-ES
        if best_x is not None and total_calls < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            remaining = budget - total_calls - 5
            
            for _ in range(remaining):
                if time.time() >= deadline - 2:
                    break
                if total_calls >= budget - 2:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                sigma = float(np.clip(sigma, 0.001, 0.5))
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
