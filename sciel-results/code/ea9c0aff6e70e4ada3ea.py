import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using a Basin Hopping strategy with adaptive step size.
    Combines global exploration via perturbations and local exploitation
    using L-BFGS-B and Nelder-Mead.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 20s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def coordinate_descent(x, max_iter=200):
            """Perform coordinate descent with adaptive step size."""
            if calls[0] >= budget or time.time() > deadline:
                return x, wrapped(x) if calls[0] < budget else np.inf
            
            x = np.array(x, dtype=float)
            f = wrapped(x)
            if calls[0] >= budget or time.time() > deadline:
                return x, f
            
            step = 1.0
            for _ in range(max_iter):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(dim):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[i] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = wrapped(x_trial)
                    
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        continue
                    elif calls[0] >= budget:
                        break
                    
                    # Try negative direction
                    x_trial = x.copy()
                    x_trial[i] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = wrapped(x_trial)
                    
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                    elif calls[0] >= budget:
                        break
                
                if improved:
                    step *= 0.95  # Slightly shrink step when improving
                else:
                    step *= 0.5   # Shrink step when stuck
                    if step < 1e-6:
                        break
            
            return x, f

        rng = np.random.default_rng(42)

        # 1. Generate diverse initial starts
        starts = [np.zeros(dim)]
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))
        for i in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        
        for s in unique_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                update_best(s, f)
            except Exception:
                pass

        if best_x is None:
             best_x = rng.uniform(-5, 5, dim)
             best_f = wrapped(best_x)

        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.5
        stagnation_count = 0
        max_stagnation = 5

        # 2. Basin Hopping Loop with more aggressive local optimization
        while calls[0] < budget and time.time() < deadline:
            # Perturb current solution
            direction = rng.standard_normal(dim)
            perturbed_x = np.clip(current_x + step_size * direction, -5, 5)
            
            # Local Optimization
            new_x, new_f = perturbed_x, np.inf
            
            # Try L-BFGS-B first (fast, good for smooth)
            try:
                res = minimize(
                    wrapped,
                    perturbed_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'gtol': 1e-5}
                )
                new_x, new_f = res.x, res.fun
            except Exception:
                pass

            # If L-BFGS-B didn't improve significantly or failed, try Nelder-Mead
            if (new_f >= current_f or new_f == np.inf) and calls[0] < budget:
                try:
                    res_nm = minimize(
                        wrapped,
                        perturbed_x,
                        method='Nelder-Mead',
                        options={'maxiter': 80, 'xatol': 1e-7, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res_nm.fun < new_f:
                        new_x, new_f = res_nm.x, res_nm.fun
                except Exception:
                    pass

            # Additional Coordinate Descent pass for fine-tuning
            if calls[0] < budget and time.time() < deadline:
                try:
                    cd_x, cd_f = coordinate_descent(new_x)
                    if cd_f < new_f:
                        new_x, new_f = cd_x, cd_f
                except Exception:
                    pass

            # Accept if better
            if new_f < current_f:
                current_x = new_x.copy()
                current_f = new_f
                update_best(current_x, current_f)
                step_size *= 0.85  # Shrink less aggressively
                stagnation_count = 0
            else:
                step_size *= 1.25  # Expand less aggressively
                if step_size > 6.0:
                    step_size = 6.0
                stagnation_count += 1
                
                # If stuck for too long, do a random jump
                if stagnation_count >= max_stagnation:
                    jump_size = rng.uniform(2.0, 4.0)
                    jump_dir = rng.standard_normal(dim)
                    jump_x = np.clip(current_x + jump_size * jump_dir, -5, 5)
                    jump_f = wrapped(jump_x)
                    
                    # Try to improve from the jump
                    if calls[0] < budget:
                        try:
                            res_jump = minimize(
                                wrapped,
                                jump_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * dim,
                                options={'maxiter': 30, 'maxls': 15, 'gtol': 1e-4}
                            )
                            if res_jump.fun < jump_f:
                                jump_x, jump_f = res_jump.x, res_jump.fun
                        except Exception:
                            pass
                    
                    if jump_f < current_f:
                        current_x = jump_x.copy()
                        current_f = jump_f
                        update_best(current_x, current_f)
                        step_size = 1.5
                        stagnation_count = 0
                    else:
                        stagnation_count = 0  # Reset after attempting jump

        # 3. Final Polish with tighter tolerances
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 30, 'gtol': 1e-8}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass
            
            if calls[0] < budget and time.time() < deadline:
                try:
                    res_nm_final = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                    )
                    update_best(res_nm_final.x, res_nm_final.fun)
                except Exception:
                    pass
            
            # Final coordinate descent polish
            if calls[0] < budget and time.time() < deadline:
                try:
                    cd_x, cd_f = coordinate_descent(best_x, max_iter=100)
                    if cd_f < best_f:
                        update_best(cd_x, cd_f)
                except Exception:
                    pass

        return best_x, best_f
