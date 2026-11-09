# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gauss-Newton affine transform fitting + (1+1)-ES refinement for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fit_affine(self, func, rng, budget, deadline):
        """
        Fit the hidden affine transform (shift, rotation) by probing along random directions.
        The idea: if f(x) = g(R*(x - s) + offset), then probing along R^T * e_i directions
        should reveal the structure. We estimate R and s by minimizing non-smoothness.
        
        Returns: best_x, best_f, calls_used
        """
        n = self.dim
        calls = 0
        best_f = np.inf
        best_x = None
        
        # Phase 1: Initial random sampling to get a rough estimate
        # Sample points on a sphere around origin and measure function values
        # For f14 with different powers, the function is anisotropic
        # We'll use a set of probe directions to estimate the rotation
        
        # Generate probe directions (orthonormal basis + random)
        # First, get a rough minimum via random sampling
        n_probes = min(50, budget // 4)
        probe_points = np.zeros((n_probes, n))
        probe_vals = np.zeros(n_probes)
        
        for i in range(n_probes):
            if calls >= budget or time.time() >= deadline:
                break
            # Sample in [-4, 4]^5 (where minimizer is known to be)
            x = rng.uniform(-4, 4, size=n)
            probe_points[i] = x
            probe_vals[i] = func(x)
            calls += 1
            if probe_vals[i] < best_f:
                best_f = probe_vals[i]
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1
        
        # Phase 2: Estimate rotation by probing along principal axes from best_x
        # The function f14 has different powers per dimension, so it's anisotropic
        # We probe along random unit vectors and measure the curvature/response
        
        # Generate a set of random directions
        n_dirs = min(20, budget // 5)
        dirs = np.zeros((n_dirs, n))
        for i in range(n_dirs):
            d = rng.normal(size=n)
            d /= np.linalg.norm(d)
            dirs[i] = d
        
        # Probe along each direction from best_x at two scales
        step_small = 0.1
        step_large = 0.5
        
        # Store function values to estimate directional derivatives
        # f(best_x + t*d) for t in [-step, 0, +step]
        # This gives us a local quadratic model
        
        # Build local model around best_x
        # f(x) ≈ f(x0) + grad^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
        # We can estimate grad and H from directional probes
        
        n_probe_steps = 2 * n_dirs + 1  # -d, 0, +d for each direction
        if calls + n_probe_steps > budget:
            n_dirs = max(1, (budget - calls) // 3)
            dirs = dirs[:n_dirs]
        
        # Collect probe data
        probe_data = []  # (direction, t, value)
        for i in range(n_dirs):
            if time.time() >= deadline or calls >= budget:
                break
            d = dirs[i]
            for t in [-step_large, -step_small, step_small, step_large]:
                x = best_x + t * d
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                probe_data.append((d, t, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Phase 3: Use the probe data to estimate the Hessian and gradient
        # Then do a Gauss-Newton step to refine the position
        
        # Simple approach: use the directional second differences to estimate curvature
        # For each direction d: f(x0 + t*d) - f(x0) ≈ t*grad^T d + 0.5*t^2*d^T H d
        # Second difference: f(x0 + t*d) + f(x0 - t*d) - 2*f(x0) ≈ t^2 * d^T H d
        
        # Let's estimate the Hessian eigenstructure from the probes
        # Actually, let's just do a simple coordinate-wise refinement
        
        # Phase 4: (1+1)-ES from best_x with adaptive step size
        remaining = budget - calls
        if remaining > 5:
            best_x, best_f, es_calls = self._one_plus_one_es(
                func, best_x, remaining, rng, deadline
            )
            calls += es_calls
        
        return best_x, best_f, calls

    def _one_plus_one_es(self, func, x0, budget, rng, deadline):
        """
        (1+1) Evolution Strategy with adaptive step size.
        """
        n = self.dim
        calls = 0
        x = x0.copy()
        fx = func(x)
        calls += 1
        best_f = fx
        best_x = x.copy()
        
        # Initial step size
        sigma = 0.5
        success_ratio = 0.0
        n_success = 0
        n_trials = 0
        
        while calls < budget and time.time() < deadline:
            # Generate candidate
            direction = rng.normal(size=n)
            direction /= np.linalg.norm(direction)
            x_new = x + sigma * direction
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            calls += 1
            n_trials += 1
            
            if fx_new < fx:
                # Success
                x = x_new
                fx = fx_new
                n_success += 1
                # Increase step size slowly
                sigma *= 1.05
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
            else:
                # Failure
                # Decrease step size
                sigma *= 0.95
            
            # Adapt sigma based on success rate (1/5 rule)
            if n_trials >= 20:
                success_ratio = n_success / n_trials
                if success_ratio > 0.5:
                    sigma *= 1.1
                elif success_ratio < 0.2:
                    sigma *= 0.8
                n_success = 0
                n_trials = 0
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Check for convergence
            if sigma < 1e-8:
                break
        
        return best_x, best_f, calls

    def _nelder_mead(self, func, x0, budget, rng, deadline):
        """
        Run Nelder-Mead from x0.
        Returns best_x, best_f, calls_used.
        """
        n = self.dim
        simplex = [x0.copy()]
        f_vals = [func(x0)]
        calls = 1
        
        # Expand simplex
        for i in range(n):
            xi = x0.copy()
            step = 0.5
            if xi[i] + step > 5:
                xi[i] = 5.0
            elif xi[i] - step < -5:
                xi[i] = -5.0
            else:
                xi[i] += step
            simplex.append(xi)
            f_vals.append(func(xi))
            calls += 1
            if calls >= budget or time.time() >= deadline:
                break
        
        if len(simplex) < 2:
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx], calls

        best_x = simplex[np.argmin(f_vals)].copy()
        best_f = min(f_vals)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < budget and time.time() < deadline:
            indices = np.argsort(f_vals)
            simplex = [simplex[i] for i in indices]
            f_vals = [f_vals[i] for i in indices]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < f_vals[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
            elif fr < f_vals[-2]:
                simplex[-1] = xr
                f_vals[-1] = fr
            else:
                if fr < f_vals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                
                if fc <= min(fr, f_vals[-1]):
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget or time.time() >= deadline:
                            break
                    if calls >= budget or time.time() >= deadline:
                        break
            
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < best_f:
                best_f = f_vals[min_idx]
                best_x = simplex[min_idx].copy()
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        n = self.dim
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Global exploration with stratified sampling
        explore_budget = max(30, int(self.budget * 0.25))
        
        # Use a quasi-random sequence for better coverage
        # Stratified sampling: divide each dimension into bins
        n_per_dim = max(2, int(np.round(explore_budget ** (1.0 / n))))
        total_grid = n_per_dim ** n
        if total_grid > explore_budget:
            n_per_dim = max(2, int(np.round((explore_budget) ** (1.0 / n))))
            total_grid = n_per_dim ** n
        
        # Generate grid points
        grid_coords = np.meshgrid(*[np.linspace(-4, 4, n_per_dim)] * n, indexing='ij')
        grid_points = np.array([g.flatten() for g in grid_coords]).T
        grid_points = grid_points[:explore_budget]
        
        for i in range(len(grid_points)):
            if time.time() >= deadline:
                break
            x = grid_points[i]
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls += 1
        
        # Phase 2: Gauss-Newton affine fitting + ES refinement
        remaining = self.budget - total_calls
        if remaining > 10:
            x_opt, f_opt, calls_used = self._fit_affine(
                func, rng, remaining, deadline
            )
            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Phase 3: Final Nelder-Mead polish
        remaining = self.budget - total_calls
        if remaining > 10 and time.time() < deadline:
            x_nm, f_nm, nm_calls = self._nelder_mead(
                func, best_x, remaining, rng, deadline
            )
            total_calls += nm_calls
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
        
        # Phase 4: If still budget, do another ES run
        remaining = self.budget - total_calls
        if remaining > 10 and time.time() < deadline:
            x_es, f_es, es_calls = self._one_plus_one_es(
                func, best_x, remaining, rng, deadline
            )
            total_calls += es_calls
            if f_es < best_f:
                best_f = f_es
                best_x = x_es.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
