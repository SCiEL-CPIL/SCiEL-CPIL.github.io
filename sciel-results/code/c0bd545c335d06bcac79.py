import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Random Search + Coordinate Descent + Nelder-Mead
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(1234)
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Random Search - 30 samples to locate basin
        n_random = 30
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: Coordinate Descent with adaptive per-axis step sizes
        # This exploits the anisotropic nature of the sharp ridge
        x_cd = best_x.copy()
        f_cd = best_f
        
        # Initialize step sizes per axis
        step_sizes = np.ones(dim) * 0.5
        min_step = 1e-6
        max_step = 5.0
        cd_iterations = 0
        max_cd_iterations = 150
        
        while calls < budget - 20 and cd_iterations < max_cd_iterations and time.time() < deadline:
            progress_made = False
            for i in range(dim):
                if calls >= budget - 20 or time.time() > deadline:
                    break
                
                # Try positive direction
                x_plus = x_cd.copy()
                x_plus[i] += step_sizes[i]
                x_plus = np.clip(x_plus, -5.0, 5.0)
                f_plus = safe_call(x_plus)
                if f_plus is None:
                    break
                
                # Try negative direction
                x_minus = x_cd.copy()
                x_minus[i] -= step_sizes[i]
                x_minus = np.clip(x_minus, -5.0, 5.0)
                f_minus = safe_call(x_minus)
                if f_minus is None:
                    break
                
                # Choose best direction
                if f_plus < f_cd and f_plus <= f_minus:
                    x_cd = x_plus
                    f_cd = f_plus
                    step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                    progress_made = True
                elif f_minus < f_cd:
                    x_cd = x_minus
                    f_cd = f_minus
                    step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                    progress_made = True
                else:
                    # No progress, shrink step
                    step_sizes[i] = max(step_sizes[i] * 0.5, min_step)
            
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd.copy()
            
            cd_iterations += 1
            
            # If no progress across all axes, break
            if not progress_made:
                # Check if steps are too small
                if np.max(step_sizes) < min_step * 10:
                    break
                # Try a small random perturbation to escape
                if calls < budget - 30:
                    pert = rng.uniform(-0.2, 0.2, size=dim)
                    x_pert = np.clip(x_cd + pert, -5.0, 5.0)
                    f_pert = safe_call(x_pert)
                    if f_pert is not None and f_pert < f_cd:
                        x_cd = x_pert
                        f_cd = f_pert
                        if f_cd < best_f:
                            best_f = f_cd
                            best_x = x_cd.copy()
                    else:
                        break

        # Phase 3: Nelder-Mead for final refinement
        if calls < budget - 10 and time.time() < deadline:
            x_start = best_x.copy()
            
            # Adaptive initial step based on recent progress
            if cd_iterations > 0:
                step_size = max(0.1, min(1.0, np.mean(step_sizes)))
            else:
                step_size = 0.5
            
            simplex = [x_start]
            f_values = [best_f]
            
            for i in range(dim):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                xi = x_start.copy()
                xi[i] += step_size
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                f_values.append(fi)
            
            if len(simplex) >= dim + 1:
                simplex = np.array(simplex)
                f_values = np.array(f_values)
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                while calls < budget and time.time() < deadline:
                    idx = np.argsort(f_values)
                    simplex = simplex[idx]
                    f_values = f_values[idx]
                    
                    if f_values[0] < best_f:
                        best_f = f_values[0]
                        best_x = simplex[0].copy()
                    
                    spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                    spread_f = np.max(f_values) - np.min(f_values)
                    
                    if np.max(spread_x) < 1e-8 or spread_f < 1e-14:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = safe_call(xr)
                    if fr is None:
                        break
                    
                    if fr < f_values[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = safe_call(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            f_values[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_values[-1] = fr
                    elif fr < f_values[-2]:
                        simplex[-1] = xr
                        f_values[-1] = fr
                    else:
                        if fr < f_values[-1]:
                            xc = centroid + rho * (xr - centroid)
                        else:
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc <= min(fr, f_values[-1]):
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            for i in range(1, len(simplex)):
                                if calls >= budget or time.time() > deadline:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fi = safe_call(simplex[i])
                                if fi is None:
                                    break
                                f_values[i] = fi
                            if fi is None:
                                break

        return best_x, best_f
