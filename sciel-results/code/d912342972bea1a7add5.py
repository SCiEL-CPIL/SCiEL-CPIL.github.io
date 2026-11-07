"""Hybrid Nelder-Mead + (1+1)-ES optimizer for BBOB f11 Discus.
Uses Nelder-Mead to adapt to anisotropy and (1+1)-ES for final polishing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex followed by (1+1)-ES for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
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
            # Ensure x is a 1D numpy array of correct shape
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                return None
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def alive():
            return call_count < budget and time.time() < deadline

        # --- Phase 0: Initialization ---
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = try_eval(x0)
        if f0 is None:
            return best_x, best_f

        # Probe axes to help orient the simplex
        # This helps NM find the scale of the function in each dimension
        probe_delta = 1.0
        candidates = [x0]
        for i in range(dim):
            for sign in [-1, 1]:
                x_probe = x0.copy()
                x_probe[i] += sign * probe_delta
                candidates.append(x_probe)
        
        # Add a few random points to avoid getting stuck if origin is bad
        for _ in range(4):
            x_rand = rng.uniform(-4, 4, dim)
            candidates.append(x_rand)

        # Evaluate all candidates to find the best starting point for NM
        for c in candidates:
            if not alive():
                break
            fc = try_eval(c)
            if fc is not None and fc < best_f:
                # best_f and best_x are updated in try_eval
                pass

        if best_x is None:
            return np.zeros(dim), float('inf')

        x_curr = best_x.copy()
        f_curr = best_f

        # --- Phase 1: Nelder-Mead Simplex Method ---
        # Initialize simplex around x_curr
        # Standard NM initialization: x_curr + delta * e_i
        # We use a slightly larger delta if we are far from origin, smaller if close
        dist_from_origin = np.linalg.norm(x_curr)
        delta = 0.5 if dist_from_origin < 2.0 else 1.0
        
        simplex = [x_curr.copy()]
        fvals = [f_curr]
        
        for i in range(dim):
            x_i = x_curr.copy()
            x_i[i] += delta
            x_i = np.clip(x_i, -5.0, 5.0)
            f_i = try_eval(x_i)
            if f_i is None:
                break
            simplex.append(x_i)
            fvals.append(f_i)
        
        if len(simplex) < dim + 1:
            # If we ran out of budget during initialization, fallback to (1+1)-ES
            pass
        else:
            # Run NM
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Max iterations for NM: leave some budget for (1+1)-ES
            # NM uses ~1-2 calls per iteration. 
            # Budget remaining: budget - call_count
            # Let's allocate 70% of remaining budget to NM
            max_iters = int((budget - call_count) * 0.7)
            
            for _ in range(max_iters):
                if not alive():
                    break
                
                # Sort vertices by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    x_curr = best_x.copy()
                    f_curr = best_f

                # Convergence check
                if abs(fvals[0] - fvals[-1]) < 1e-12 * max(1.0, abs(fvals[0])):
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = try_eval(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Try Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = try_eval(xe)
                    if fe is None:
                        # Budget ran out during expansion, accept reflection if better
                        if fr < fvals[-1]:
                            simplex[-1] = xr
                            fvals[-1] = fr
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept Reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside Contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside Contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = try_eval(xc)
                    if fc is None:
                        break
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_i = try_eval(simplex[i])
                            if f_i is None:
                                break
                            fvals[i] = f_i
                        if f_i is None:
                            break

        # --- Phase 2: (1+1)-ES Polish ---
        # Use remaining budget to fine-tune
        # Start with a small sigma, adapt it
        x_curr = best_x.copy()
        f_curr = best_f
        sigma = 0.1  # Small initial step for polishing
        
        # Success rate adaptation
        target_success = 0.2
        success_count = 0
        total_trials = 0
        window_size = 10
        
        while alive():
            # Generate mutant
            z = rng.standard_normal(dim)
            x_trial = x_curr + sigma * z
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = try_eval(x_trial)
            if f_trial is None:
                break
            
            total_trials += 1
            
            # Accept if better
            if f_trial < f_curr:
                success_count += 1
                x_curr = x_trial.copy()
                f_curr = f_trial
            
            # Adapt step size
            if total_trials % window_size == 0:
                recent_success = success_count / window_size
                if recent_success > target_success:
                    sigma *= 1.5
                elif recent_success < target_success:
                    sigma *= 0.7
                sigma = min(sigma, 5.0)
                sigma = max(sigma, 1e-5)
                success_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
