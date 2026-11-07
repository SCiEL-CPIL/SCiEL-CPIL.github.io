"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid adaptive-step (1+1)-ES + Nelder-Mead for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Per-dimension adaptive step sizes (key for anisotropic f14)
        sigma = np.ones(dim) * 1.5
        # Track per-dimension success to adapt sigma independently
        dim_success = np.ones(dim) * 0.3
        calls_made = 0

        def check_deadline():
            return time.time() > deadline or calls_made >= budget

        # Phase 1: Diversified initialization
        n_init = min(8, budget)
        candidates = [np.zeros(dim)]
        # Sample corners along each axis
        for i in range(dim):
            for v in [-3.0, 3.0]:
                c = np.zeros(dim)
                c[i] = v
                candidates.append(c)
        # Random points
        while len(candidates) < n_init:
            candidates.append(rng.uniform(-4.0, 4.0, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            if check_deadline():
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if check_deadline():
            return best_x, best_f

        # Phase 2: Adaptive per-dimension (1+1)-ES
        x = best_x.copy()
        fx = best_f
        no_improve = 0
        nm_cooldown = 0  # cooldown after NM before running it again

        while not check_deadline():
            # Try Nelder-Mead every 50 evals if we've made some progress
            if nm_cooldown <= 0 and calls_made > 30 and (calls_made % 50 == 0 or no_improve > 20):
                try:
                    def wrapped_func(v):
                        nonlocal calls_made
                        if time.time() > deadline or calls_made >= budget:
                            return 1e10
                        v = np.clip(v, -5, 5)
                        calls_made += 1
                        return func(v)

                    # NM with limited iterations to not blow budget
                    max_nm_calls = min(80, budget - calls_made - 5)
                    if max_nm_calls > 10:
                        res = minimize(wrapped_func, x.copy(), method='Nelder-Mead',
                                       options={'maxiter': max(10, max_nm_calls // 2),
                                                'xatol': 1e-4, 'fatol': 1e-8,
                                                'adaptive': True})
                        if check_deadline():
                            break
                        if res.fun < fx:
                            x = np.clip(res.x, -5, 5)
                            fx = res.fun
                            no_improve = 0
                        nm_cooldown = 30  # wait 30 evals before next NM
                except Exception:
                    pass

            if nm_cooldown > 0:
                nm_cooldown -= 1

            if check_deadline():
                break

            # Adaptive per-dimension step: scale sigma by success rate
            for d in range(dim):
                # Adjust sigma based on per-dimension success
                if dim_success[d] > 0.5:
                    sigma[d] *= 1.15
                elif dim_success[d] < 0.2:
                    sigma[d] *= 0.85
                sigma[d] = np.clip(sigma[d], 1e-5, 8.0)

            # Generate candidate: perturb each dimension with its own sigma
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: update per-dimension success rates
                for d in range(dim):
                    moved = abs(y[d] - x[d]) > sigma[d] * 0.1
                    if moved:
                        dim_success[d] = 0.7 * dim_success[d] + 0.3 * 1.0
                    else:
                        dim_success[d] = 0.7 * dim_success[d] + 0.3 * 0.5
                x = y
                fx = fy
                no_improve = 0
            else:
                for d in range(dim):
                    dim_success[d] = 0.7 * dim_success[d] + 0.3 * 0.0
                no_improve += 1

            # If stuck for a while, try a larger random jump
            if no_improve > 40:
                jump = x + np.clip(sigma * 3.0, 0.5, 5.0) * rng.standard_normal(dim)
                jump = np.clip(jump, -5, 5)
                if check_deadline():
                    break
                fj = func(jump)
                calls_made += 1
                if fj < fx:
                    x = jump
                    fx = fj
                    no_improve = 0
                    # Reset sigma to moderate values
                    sigma[:] = 1.0
                    dim_success[:] = 0.3
                else:
                    no_improve = 0  # reset counter to avoid repeated jumps

        return x, fx
# EVOLVE-BLOCK-END
