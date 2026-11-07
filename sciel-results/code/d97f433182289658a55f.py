"""Rewritten optimizer: Trust-region quadratic surrogate with ridge-aware step for Sharp Ridge."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using a sequential trust-region method with a quadratic surrogate model.
    Explicitly identifies the ridge's flat direction and constrains steps to achieve
    rapid convergence across the sharp ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = clip(x)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Initial sampling to locate basin ----------
        n_starts = 15
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: Trust-region quadratic optimization ----------
        # We maintain a quadratic model m(x) = f0 + g^T (x - x0) + 0.5 (x-x0)^T H (x-x0)
        # and a trust region radius. We minimize m within the trust region,
        # then update the model based on the actual function value.

        def trust_region_optimize(start_x, max_evals=200):
            """Run trust-region quadratic optimization from start_x."""
            nonlocal evaluated, best_x, best_f

            x = start_x.copy()
            f = eval_point(x)

            # Initialize gradient and Hessian estimates
            # Use finite differences to build initial model
            h = 1e-3
            g = np.zeros(dim)
            H = np.zeros((dim, dim))

            for i in range(dim):
                if not elapsed_ok() or evaluated >= budget:
                    return x, f
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = clip(x_plus)
                f_plus = eval_point(x_plus)

                x_minus = x.copy()
                x_minus[i] -= h
                x_minus = clip(x_minus)
                f_minus = eval_point(x_minus)

                g[i] = (f_plus - f_minus) / (2 * h)
                H[i, i] = (f_plus - 2 * f + f_minus) / (h ** 2)

                for j in range(i + 1, dim):
                    if not elapsed_ok() or evaluated >= budget:
                        return x, f
                    x_pp = x.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pp = clip(x_pp)
                    f_pp = eval_point(x_pp)

                    x_pm = x.copy()
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_pm = clip(x_pm)
                    f_pm = eval_point(x_pm)

                    x_mp = x.copy()
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mp = clip(x_mp)
                    f_mp = eval_point(x_mp)

                    x_mm = x.copy()
                    x_mm[i] -= h
                    x_mm[j] -= h
                    x_mm = clip(x_mm)
                    f_mm = eval_point(x_mm)

                    H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h ** 2)
                    H[j, i] = H[i, j]

            # Ensure H is symmetric positive semi-definite for stability
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.maximum(eigvals, 1e-8)
            H = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Trust region radius
            radius = 1.0
            min_radius = 1e-6
            max_radius = 5.0

            # Iterate
            for _ in range(100):
                if not elapsed_ok() or evaluated >= budget:
                    break

                # Solve the trust-region subproblem: minimize m(x) s.t. ||x - x0|| <= radius
                # Use a simple projected gradient step in the trust region
                # First, find the unconstrained minimizer of the quadratic model
                try:
                    x_uncon = x - np.linalg.solve(H + 1e-10 * np.eye(dim), g)
                except np.linalg.LinAlgError:
                    x_uncon = x - 0.1 * g

                # Project to trust region
                dist = np.linalg.norm(x_uncon - x)
                if dist > radius:
                    x_trial = x + (x_uncon - x) * (radius / dist)
                else:
                    x_trial = x_uncon

                x_trial = clip(x_trial)

                # Evaluate
                f_trial = eval_point(x_trial)

                # Check improvement
                if f_trial < f:
                    # Success: move to trial point, update model
                    x = x_trial
                    f = f_trial

                    # Update gradient and Hessian via finite differences at new point
                    h2 = 1e-3
                    g_new = np.zeros(dim)
                    H_new = np.zeros((dim, dim))

                    for i in range(dim):
                        if not elapsed_ok() or evaluated >= budget:
                            break
                        x_plus = x.copy()
                        x_plus[i] += h2
                        x_plus = clip(x_plus)
                        f_plus = eval_point(x_plus)

                        x_minus = x.copy()
                        x_minus[i] -= h2
                        x_minus = clip(x_minus)
                        f_minus = eval_point(x_minus)

                        g_new[i] = (f_plus - f_minus) / (2 * h2)
                        H_new[i, i] = (f_plus - 2 * f + f_minus) / (h2 ** 2)

                        for j in range(i + 1, dim):
                            if not elapsed_ok() or evaluated >= budget:
                                break
                            x_pp = x.copy()
                            x_pp[i] += h2
                            x_pp[j] += h2
                            x_pp = clip(x_pp)
                            f_pp = eval_point(x_pp)

                            x_pm = x.copy()
                            x_pm[i] += h2
                            x_pm[j] -= h2
                            x_pm = clip(x_pm)
                            f_pm = eval_point(x_pm)

                            x_mp = x.copy()
                            x_mp[i] -= h2
                            x_mp[j] += h2
                            x_mp = clip(x_mp)
                            f_mp = eval_point(x_mp)

                            x_mm = x.copy()
                            x_mm[i] -= h2
                            x_mm[j] -= h2
                            x_mm = clip(x_mm)
                            f_mm = eval_point(x_mm)

                            H_new[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h2 ** 2)
                            H_new[j, i] = H_new[i, j]

                    # Blend old and new Hessian for stability
                    H = 0.5 * H + 0.5 * H_new
                    g = g_new

                    # Increase trust region
                    radius = min(radius * 1.5, max_radius)
                else:
                    # Failure: shrink trust region
                    radius = max(radius * 0.5, min_radius)
                    if radius <= min_radius:
                        break

                    # Try a smaller step in the steepest descent direction
                    if np.linalg.norm(g) > 1e-10:
                        x_trial = x - (radius / np.linalg.norm(g)) * g
                        x_trial = clip(x_trial)
                        f_trial = eval_point(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            # Update gradient
                            h2 = 1e-3
                            g_new = np.zeros(dim)
                            for i in range(dim):
                                if not elapsed_ok() or evaluated >= budget:
                                    break
                                x_plus = x.copy()
                                x_plus[i] += h2
                                x_plus = clip(x_plus)
                                f_plus = eval_point(x_plus)
                                x_minus = x.copy()
                                x_minus[i] -= h2
                                x_minus = clip(x_minus)
                                f_minus = eval_point(x_minus)
                                g_new[i] = (f_plus - f_minus) / (2 * h2)
                            g = g_new

            return x, f

        # Run trust-region optimization from the best initial point
        x_opt, f_opt_val = trust_region_optimize(best_x.copy())

        # ---------- Phase 3: Ridge-aware refinement ----------
        # Identify the flat direction (smallest eigenvalue of Hessian) and do
        # a line search along it to refine the solution.
        if evaluated < budget and elapsed_ok() and best_x is not None:
            # Estimate Hessian at current best point
            h = 1e-3
            H_est = np.zeros((dim, dim))
            x_curr = best_x.copy()
            f_curr = best_f

            for i in range(dim):
                if not elapsed_ok() or evaluated >= budget:
                    break
                x_plus = x_curr.copy()
                x_plus[i] += h
                x_plus = clip(x_plus)
                f_plus = eval_point(x_plus)

                x_minus = x_curr.copy()
                x_minus[i] -= h
                x_minus = clip(x_minus)
                f_minus = eval_point(x_minus)

                H_est[i, i] = (f_plus - 2 * f_curr + f_minus) / (h ** 2)

                for j in range(i + 1, dim):
                    if not elapsed_ok() or evaluated >= budget:
                        break
                    x_pp = x_curr.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pp = clip(x_pp)
                    f_pp = eval_point(x_pp)

                    x_pm = x_curr.copy()
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_pm = clip(x_pm)
                    f_pm = eval_point(x_pm)

                    x_mp = x_curr.copy()
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mp = clip(x_mp)
                    f_mp = eval_point(x_mp)

                    x_mm = x_curr.copy()
                    x_mm[i] -= h
                    x_mm[j] -= h
                    x_mm = clip(x_mm)
                    f_mm = eval_point(x_mm)

                    H_est[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h ** 2)
                    H_est[j, i] = H_est[i, j]

            # Find the flat direction (smallest eigenvalue)
            eigvals, eigvecs = np.linalg.eigh(H_est)
            flat_dir = eigvecs[:, 0]  # Smallest eigenvalue direction

            # Line search along flat direction
            for step_size in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                if not elapsed_ok() or evaluated >= budget:
                    break
                for sign in [1.0, -1.0]:
                    x_trial = best_x + sign * step_size * flat_dir
                    x_trial = clip(x_trial)
                    f_trial = eval_point(x_trial)
                    if f_trial < best_f:
                        best_x = x_trial
                        best_f = f_trial

        # ---------- Phase 4: Final coordinate descent polish ----------
        if evaluated < budget and elapsed_ok() and best_x is not None:
            x_current = best_x.copy()
            f_current = best_f

            for pass_num in range(4):
                step = 0.1 * (0.5 ** pass_num)
                for coord in range(dim):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    for direction in [1.0, -1.0]:
                        x_trial = x_current.copy()
                        x_trial[coord] += direction * step
                        x_trial = clip(x_trial)
                        f_trial = eval_point(x_trial)
                        if f_trial < f_current:
                            x_current = x_trial
                            f_current = f_trial
                    if evaluated >= budget or not elapsed_ok():
                        break

            if f_current < best_f:
                best_f = f_current
                best_x = x_current.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
