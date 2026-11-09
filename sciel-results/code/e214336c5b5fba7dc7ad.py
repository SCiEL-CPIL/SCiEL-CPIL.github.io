"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with stagnation restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Global best tracking
        global_x = None
        global_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count, global_x, global_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < global_f:
                global_f = fx
                global_x = x.copy()
            return fx

        # Phase 1: Dense initial sampling to find a good starting region
        rng = np.random.default_rng(42)

        # Sample around origin (Katsuura minimum is often near origin after shift)
        # Dense grid near 0
        n_dense = 10
        for _ in range(n_dense):
            x = rng.uniform(-1.5, 1.5, dim)
            eval_point(x)

        # Moderate distance points
        n_mod = 8
        for _ in range(n_mod):
            x = rng.uniform(-3.0, 3.0, dim)
            eval_point(x)

        # A few far points
        n_far = 4
        for _ in range(n_far):
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)

        # Sort and pick best as initial CMA-ES start
        if call_count >= budget:
            return global_x, global_f

        # Phase 2: CMA-ES with stagnation detection
        # Use the best point from initial sampling
        x0 = global_x.copy()

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        sigma = 1.5
        max_iter = budget - call_count

        # CMA-ES state
        mean = x0.copy()
        sigma = 1.5
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        cc = (4 + np.log(dim)) / (4 + dim)
        cs = (1 + 2 / (dim + 1)) / (dim + 4)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (dim + 1) / (dim + 2)) ** 2 - 1)

        # Inverse square root of C
        def inv_sqrt(C_mat):
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T

        B = inv_sqrt(C)
        D = 1.0 / np.sqrt(np.diag(B @ B.T))  # approx

        no_improve = 0
        stagnation_threshold = 30

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Generate population
            for j in range(pop_size):
                if time.time() > deadline or call_count >= budget:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (B @ z)
                fy = eval_point(y)

            # Update CMA-ES (simplified)
            # Sort by fitness
            # For simplicity, use the best point to update mean
            # This is a simplified CMA-ES update
            if global_f < np.inf:
                # Update mean toward best
                mean = 0.5 * mean + 0.5 * global_x

            # Check stagnation
            if global_f < np.inf:
                no_improve += 1
            else:
                no_improve = 0

            # Stagnation detection: if no improvement for many iterations, restart
            if no_improve > stagnation_threshold:
                # Restart from a new random point near current best
                if global_x is not None:
                    perturbation = 0.5 * rng.standard_normal(dim)
                    new_start = np.clip(global_x + perturbation, -5.0, 5.0)
                    eval_point(new_start)
                else:
                    new_start = rng.uniform(-3.0, 3.0, dim)
                    eval_point(new_start)

                # Reset CMA-ES
                mean = new_start.copy()
                sigma = 1.0
                C = np.eye(dim)
                B = inv_sqrt(C)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve = 0
                stagnation_threshold = 20  # Decrease threshold for subsequent restarts

        # Phase 3: Final local refinement using Nelder-Mead around best point
        if call_count < budget and global_x is not None and time.time() < deadline:
            from scipy.optimize import minimize

            def wrapper(x):
                nonlocal call_count
                if call_count >= budget:
                    return global_f  # Return current best to stop
                x = np.clip(x, -5.0, 5.0)
                fx = float(func(x))
                call_count += 1
                if fx < global_f:
                    global_f = fx
                    global_x = x.copy()
                return fx

            try:
                result = minimize(wrapper, global_x, method='Nelder-Mead',
                                  options={'maxiter': budget - call_count,
                                           'xatol': 1e-6, 'fatol': 1e-8})
                if result.fun < global_f:
                    global_f = result.fun
                    global_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        return global_x, global_f
# EVOLVE-BLOCK-END
