"""Derivative-free SQP optimizer with quadratic model for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free Sequential Quadratic Programming optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        num_calls = 0
        best_x = None
        best_f = np.inf

        def check_time():
            return time.time() > deadline

        def evaluate(x):
            nonlocal num_calls
            x = np.clip(x, -5, 5)
            f = func(x)
            num_calls += 1
            return f

        # Phase 1: Global Initialization with Sobol-like spread
        n_init = max(15, budget // 4)
        candidates = []

        # Use a stratified sampling approach for better coverage
        for _ in range(n_init):
            if num_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            candidates.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if not candidates:
            return best_x, best_f

        # Sort by fitness and pick best
        candidates.sort(key=lambda t: t[1])
        x = candidates[0][0].copy()
        f = candidates[0][1]

        # Phase 2: DF-SQP with quadratic model
        # We'll maintain a local quadratic model: q(z) = f0 + g^T (z-x) + 0.5 (z-x)^T B (z-x)
        # where B is a symmetric positive definite approximation of the Hessian

        # Initialize gradient via central differences
        def compute_gradient(center):
            g = np.zeros(dim)
            h = 1e-4
            for i in range(dim):
                xp = center.copy()
                xm = center.copy()
                xp[i] += h
                xm[i] -= h
                fp = evaluate(np.clip(xp, -5, 5))
                fm = evaluate(np.clip(xm, -5, 5))
                g[i] = (fp - fm) / (2 * h)
            return g

        # Initial Hessian approximation: identity scaled
        B = np.eye(dim) * 1.0
        g = compute_gradient(x) if num_calls + 2 * dim < budget and not check_time() else np.zeros(dim)

        # Trust region radius
        delta = 0.5
        max_delta = 5.0
        min_delta = 1e-6

        # Model points for updating Hessian (store recent (x, f) pairs)
        model_points = [(x.copy(), f)]

        while num_calls < budget and not check_time():
            # Check if we have enough budget for gradient computation
            if num_calls + 2 * dim + 1 >= budget or check_time():
                break

            # Recompute gradient if it's stale or we moved significantly
            # For now, recompute every iteration for accuracy
            g_new = compute_gradient(x)
            if not np.all(np.isfinite(g_new)):
                break
            g = g_new

            # Check convergence
            if np.linalg.norm(g) < 1e-10:
                break

            # Solve trust region subproblem: min g^T p + 0.5 p^T B p s.t. ||p|| <= delta
            # Use conjugate gradient with trust region constraint
            p = self._solve_trust_region(g, B, delta)

            if p is None or np.linalg.norm(p) < 1e-12:
                # Try pure gradient direction
                g_norm = np.linalg.norm(g)
                if g_norm < 1e-12:
                    break
                p = -g * (delta / g_norm)

            # Evaluate trial point
            if num_calls + 1 >= budget or check_time():
                break

            x_trial = x + p
            x_trial = np.clip(x_trial, -5, 5)
            f_trial = evaluate(x_trial)

            # Update model points
            model_points.append((x_trial.copy(), f_trial))
            if len(model_points) > 20:
                model_points.pop(0)

            # Compute actual and predicted reduction
            pred = g @ p + 0.5 * (p @ B @ p)
            act = f - f_trial

            if abs(pred) > 1e-12:
                rho = act / pred
            else:
                rho = 1.0 if act > 0 else 0.0

            if rho > 0.25:
                # Accept step
                x = x_trial
                f = f_trial
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

                # Update Hessian approximation using BFGS-like formula
                # We need gradient at new point
                if num_calls + 2 * dim < budget and not check_time():
                    g_trial = compute_gradient(x)
                    s = x - model_points[-2][0] if len(model_points) >= 2 else p
                    y = g_trial - g
                    ys = y @ s
                    if ys > 1e-10:
                        Bs = B @ s
                        sBs = s @ Bs
                        if sBs > 1e-10:
                            B = B + np.outer(y, y) / ys - np.outer(Bs, Bs) / sBs
                            B = (B + B.T) / 2
                            # Ensure positive definiteness
                            eigvals = np.linalg.eigvalsh(B)
                            if np.min(eigvals) < 1e-6:
                                B += (1e-5 - np.min(eigvals)) * np.eye(dim)
                    g = g_trial

                # Adjust trust region
                if rho > 0.75:
                    delta = min(2.0 * delta, max_delta)
                else:
                    delta = min(1.5 * delta, max_delta)
            else:
                # Reject step
                delta *= 0.5
                if delta < min_delta:
                    break

        # Final polish: try a few more random perturbations around best
        if num_calls < budget and not check_time():
            for _ in range(min(10, budget - num_calls)):
                if num_calls >= budget or check_time():
                    break
                x_pert = best_x + rng.normal(0, 0.1, dim)
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = evaluate(x_pert)
                if f_pert < best_f:
                    best_f = f_pert
                    best_x = x_pert.copy()

        return best_x, best_f

    def _solve_trust_region(self, g, B, delta):
        """Solve min g^T p + 0.5 p^T B p s.t. ||p|| <= delta using CG."""
        dim = len(g)
        x = np.zeros(dim)
        r = -g  # residual: -g - B x
        p = r.copy()
        rr = r @ r
        if rr < 1e-20:
            return np.zeros(dim)

        for _ in range(dim * 3):
            Bp = B @ p
            pBp = p @ Bp
            if pBp < 1e-15:
                break
            alpha = rr / pBp

            # Check if step exceeds trust region
            x_new = x + alpha * p
            if np.linalg.norm(x_new) > delta:
                # Find alpha that stays within trust region
                # ||x + alpha p||^2 = delta^2
                # alpha^2 (p^T p) + 2 alpha (x^T p) + (x^T x - delta^2) = 0
                a = p @ p
                b = 2 * (x @ p)
                c = x @ x - delta * delta
                if a < 1e-15:
                    break
                disc = b * b - 4 * a * c
                if disc < 0:
                    break
                alpha = (-b + np.sqrt(disc)) / (2 * a)
                if alpha <= 0:
                    alpha = (-b - np.sqrt(disc)) / (2 * a)
                if alpha <= 0:
                    break
                x = x + alpha * p
                return x

            x = x_new
            r = r - alpha * Bp
            rr_new = r @ r
            if rr_new < 1e-20:
                break
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new

        # Clip to trust region if needed
        x_norm = np.linalg.norm(x)
        if x_norm > delta:
            x = x * (delta / x_norm)
        return x
# EVOLVE-BLOCK-END
