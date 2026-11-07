"""Optimizer for BBOB-style f12 (Bent cigar) with quadratic surrogate + adaptive step."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: quadratic surrogate for large steps + fine-tuning ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        # Evaluation counter
        calls = [0]

        def eval(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Phase 1: Global exploration with random restarts + local quadratic model ---
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Initial diverse sampling
        n_init = min(20, budget)
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 2: Iterative local optimization with quadratic surrogate ---
        sigma = 1.0
        stagnation_count = 0
        max_stagnation = 15

        while calls[0] < budget and time.time() < deadline:
            # Build local quadratic model around best_x
            # Use 2*dim+1 points: center + +/- sigma*e_i
            points = [best_x]
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = sigma
                points.append(np.clip(best_x + e, -5, 5))
                points.append(np.clip(best_x - e, -5, 5))

            # Evaluate if we haven't already
            vals = []
            for p in points:
                if calls[0] >= budget or time.time() > deadline:
                    break
                vals.append(eval(p))
            if len(vals) < 2 * dim + 1:
                break

            # Fit quadratic model: f(x) ≈ c + g^T (x - x0) + 0.5 (x-x0)^T H (x-x0)
            # Using coordinate-wise quadratic fit for stability
            x0 = best_x
            c = vals[0]
            g = np.zeros(dim)
            h_diag = np.zeros(dim)

            for i in range(dim):
                fp = vals[1 + 2 * i]
                fm = vals[2 + 2 * i]
                # f(x0 + sigma*e_i) = c + sigma*g_i + 0.5*sigma^2*h_ii
                # f(x0 - sigma*e_i) = c - sigma*g_i + 0.5*sigma^2*h_ii
                # Adding: fp + fm = 2c + sigma^2*h_ii  => h_ii = (fp + fm - 2c) / sigma^2
                # Subtracting: fp - fm = 2*sigma*g_i  => g_i = (fp - fm) / (2*sigma)
                h_diag[i] = (fp + fm - 2 * c) / (sigma ** 2)
                g[i] = (fp - fm) / (2 * sigma)

            # Compute Newton step: solve H * d = -g  =>  d = -g / h_diag (element-wise)
            # Clip step to avoid divergence
            step = np.zeros(dim)
            for i in range(dim):
                if abs(h_diag[i]) > 1e-12:
                    step[i] = -g[i] / h_diag[i]
                else:
                    step[i] = -np.sign(g[i]) * sigma if g[i] != 0 else 0.0

            # Limit step size
            step_norm = np.linalg.norm(step)
            if step_norm > 3 * sigma:
                step = step * (3 * sigma / step_norm)

            x_new = np.clip(x0 + step, -5, 5)
            f_new = eval(x_new)

            if f_new < best_f - 1e-12:
                best_f = f_new
                best_x = x_new.copy()
                stagnation_count = 0
                sigma = max(sigma * 0.5, 0.01)
            else:
                # Try a smaller step in the same direction
                if step_norm > 1e-10:
                    x_half = np.clip(x0 + 0.5 * step, -5, 5)
                    f_half = eval(x_half)
                    if f_half < best_f - 1e-12:
                        best_f = f_half
                        best_x = x_half.copy()
                        stagnation_count = 0
                        sigma = max(sigma * 0.3, 0.005)
                    else:
                        stagnation_count += 1
                        sigma *= 0.7
                else:
                    stagnation_count += 1
                    sigma *= 0.7

            # If stagnating too long, do a random jump
            if stagnation_count >= max_stagnation:
                jump = rng.uniform(-2, 2, dim)
                x_jump = np.clip(best_x + jump, -5, 5)
                f_jump = eval(x_jump)
                if f_jump < best_f:
                    best_f = f_jump
                    best_x = x_jump.copy()
                    sigma = 1.0
                    stagnation_count = 0
                else:
                    # Shrink and try again
                    sigma *= 0.5
                    stagnation_count = 0

            # If we're very close to convergence, switch to fine ES
            if sigma < 0.01:
                break

        # --- Phase 3: Fine-tuning with (1+1)-ES on remaining budget ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            sigma_es = 0.05
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_es *= 1.3
                else:
                    sigma_es *= 0.85
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
