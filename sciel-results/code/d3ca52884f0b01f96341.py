# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Nelder-Mead + adaptive (1+1)-ES + polishing phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s kill
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initial Sampling
        n_init = 12
        X_init = rng.uniform(-5, 5, (n_init, dim))
        F_init = np.array([safe_call(x) for x in X_init])
        
        idx_best = np.argmin(F_init)
        x = X_init[idx_best].copy()
        fx = F_init[idx_best]

        # 2. Nelder-Mead Simplex for local exploitation
        if calls_made < budget and time.time() < deadline:
            x = self._nelder_mead(x, fx, func, safe_call, dim, budget, calls_made, deadline, rng)
            fx = best_f
            x = best_x
            calls_made = calls_made  # Already tracked in safe_call

        # 3. Adaptive (1+1)-ES for further refinement
        if calls_made < budget and time.time() < deadline:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                if sigma < 1e-10:
                    sigma = 1e-10
                if sigma > 10.0:
                    sigma = 10.0

        # 4. Polishing phase: coordinate descent for precision
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step = 0.1
            for _ in range(20):  # Max polishing iterations
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        y = x.copy()
                        y[d] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-10:
                        break

        return best_x, best_f

    def _nelder_mead(self, x0, f0, func, safe_call, dim, budget, calls_made, deadline, rng):
        """Nelder-Mead simplex optimization."""
        # Create initial simplex
        simplex = [x0.copy()]
        fvals = [f0]
        
        # Add dim more points
        for i in range(dim):
            x_i = x0.copy()
            step = 0.25 if x0[i] >= 0 else -0.25
            x_i[i] += step
            x_i = np.clip(x_i, -5, 5)
            f_i = safe_call(x_i)
            simplex.append(x_i)
            fvals.append(f_i)
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_nm = 0.5  # Shrinkage
        
        max_iter = 200
        for _ in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-12:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_call(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_call(x_e)
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_call(x_c)
                
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = safe_call(simplex[i])
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
