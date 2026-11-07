"""Derivative-Free Trust-Region (DFTR) optimizer for BBOB f18 (Schaffers F7).

Uses a local quadratic surrogate model built from directional derivatives to
navigate the ill-conditioned landscape. This structural change from population-based
CMA-ES to model-based local search exploits high information density of derivative
approximations for faster convergence in early iterations (AOCC benefit).
"""
import time
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region with quadratic surrogate and wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 295.0  # evaluator kills at 330 s; keep safe margin
        calls = [0]

        def time_ok():
            return (time.time() - t0) < WALL

        def safe(x):
            """One recorded, exception-safe, budget/time-guarded evaluation."""
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        n = self.dim
        rng = np.random.default_rng(0)

        # ---- DFTR parameters
        # Trust-region radius
        delta = 1.5
        delta_min = 1e-4
        delta_max = 5.0

        # Initial point: center of box (minimizer is in [-4,4]^5)
        x0 = np.zeros(n)
        f0 = safe(x0)
        if f0 is None:
            return x0, 1e18
        best_x = x0.copy()
        best_f = f0

        # Quadratic model: f(x) ≈ f0 + g^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
        # We maintain:
        #   g: gradient estimate at x0
        #   H: Hessian estimate (symmetric)
        #   delta: trust region radius

        def compute_gradient_hessian(xc, fc, H_current, g_current):
            """
            Estimate gradient and update Hessian using central differences
            along coordinate axes, with finite difference step based on trust region.
            Returns (g, H, used_calls).
            """
            # Use a step size proportional to trust region for better conditioning
            eps = max(delta * 0.1, 1e-4)

            # Estimate gradient via central differences
            g = np.zeros(n)
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                xp = xc + eps * e
                xm = xc - eps * e
                xp = np.clip(xp, -5.0, 5.0)
                xm = np.clip(xm, -5.0, 5.0)

                fp = safe(xp)
                fm = safe(xm)
                if fp is None or fm is None:
                    # Fall back to forward difference
                    fp = safe(xp)
                    if fp is None:
                        return g, H_current, True
                    g[i] = (fp - fc) / eps
                else:
                    g[i] = (fp - fm) / (2 * eps)

            # Update Hessian using second-order differences
            # H[i,j] ≈ (f(x+e_i*e_j*h) - f(x+e_i*h) - f(x+e_j*h) + f(x)) / h^2 for i!=j
            # For diagonal: H[i,i] ≈ (f(x+e_i*h) - 2*f(x) + f(x-e_i*h)) / h^2
            h = max(delta * 0.05, 1e-3)
            H = H_current.copy()

            # Update diagonal
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                xp = xc + h * e
                xm = xc - h * e
                xp = np.clip(xp, -5.0, 5.0)
                xm = np.clip(xm, -5.0, 5.0)

                fp = safe(xp)
                fm = safe(xm)
                if fp is not None and fm is not None:
                    H[i, i] = (fp - 2 * fc + fm) / (h * h)
                else:
                    # Keep previous estimate
                    pass

            # Update off-diagonals (sparse: only update a few per iteration to save budget)
            # We'll update off-diagonals periodically
            return g, H, False

        # Initial gradient and Hessian
        g = np.zeros(n)
        H = np.eye(n) * 1e-3  # small regularization

        # Main DFTR loop
        while calls[0] < self.budget and time_ok():
            # Check time budget
            if not time_ok():
                break

            # Compute/update gradient and Hessian at current best
            # To save budget, we update gradient every iteration but Hessian less frequently
            if calls[0] < self.budget - 2 * n:
                g, H, broke = compute_gradient_hessian(best_x, best_f, H, g)
                if broke:
                    break

            # Solve the trust-region subproblem:
            # minimize 0.5 d^T H d + g^T d  s.t. ||d|| <= delta
            #
            # Use a simple approach: solve H d = -g if within trust region,
            # otherwise project onto trust region boundary.
            try:
                # Check if H is well-conditioned
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) > 1e-10:
                    # Positive definite: unconstrained minimum
                    d = np.linalg.solve(H, -g)
                    if np.linalg.norm(d) > delta:
                        # Project onto trust region boundary
                        d = delta * d / np.linalg.norm(d)
                else:
                    # Indefinite or singular: use steepest descent direction
                    d = -g / max(np.linalg.norm(g), 1e-10)
                    if np.linalg.norm(d) > delta:
                        d = delta * d / np.linalg.norm(d)
            except np.linalg.LinAlgError:
                d = -g / max(np.linalg.norm(g), 1e-10)
                if np.linalg.norm(d) > delta:
                    d = delta * d / np.linalg.norm(d)

            # Ensure step is not too small
            if np.linalg.norm(d) < delta_min:
                # Use random perturbation
                d = delta * rng.standard_normal(n)
                d = delta * d / np.linalg.norm(d)

            # Propose new point
            x_new = best_x + d
            x_new = np.clip(x_new, -5.0, 5.0)

            # Evaluate
            f_new = safe(x_new)
            if f_new is None:
                break

            # Compute actual vs predicted reduction
            actual_reduction = best_f - f_new
            predicted_reduction = -g @ d - 0.5 * (d @ H @ d)

            # Update trust region
            if actual_reduction > 0:
                # Accept step
                ratio = actual_reduction / max(abs(predicted_reduction), 1e-10)
                if ratio > 0.75:
                    delta = min(delta * 2.0, delta_max)
                elif ratio < 0.25:
                    delta = max(delta * 0.5, delta_min)
                # else: keep delta

                best_x = x_new
                best_f = f_new

                # Update model at new point (recompute gradient/Hessian)
                # But to save budget, we'll only do full recompute every few steps
                # For now, just shift the model
            else:
                # Reject step, shrink trust region
                delta = max(delta * 0.5, delta_min)

                # If trust region is too small, try a random restart in a wider region
                if delta < delta_min * 2:
                    # Local restart: perturb best_x
                    perturbation = 0.5 * rng.standard_normal(n)
                    x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                    f_try = safe(x_try)
                    if f_try is not None and f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        delta = 1.0
                    else:
                        # Keep trying with slightly larger delta
                        delta = 0.5

            # Periodically recompute gradient/Hessian for accuracy
            # (This is already done at the top of the loop)

        return best_x, float(best_f)
