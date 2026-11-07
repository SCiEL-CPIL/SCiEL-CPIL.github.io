"""Derivative-free trust-region optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time


class Optimizer:
    """DFTR with local quadratic surrogates built from directional derivatives."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Start near center; f18 optimum is shifted but within [-4,4]^5
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Initial trust region radius
        delta = 2.0
        min_delta = 1e-4
        max_delta = 8.0

        # Build a local quadratic model from a set of probe points
        def build_model(x_center, fx_center, delta, rng, func, calls_ref):
            """Build quadratic model around x_center using 2n+1 points."""
            nonlocal calls
            # Use coordinate directions scaled by delta
            H = np.zeros((n, n))
            g = np.zeros(n)
            f0 = fx_center

            for i in range(n):
                ei = np.zeros(n)
                ei[i] = 1.0
                xp = x_center + delta * ei
                xm = x_center - delta * ei
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                f_p = func(xp)
                calls += 1
                f_m = func(xm)
                calls += 1
                # Central difference for gradient
                g[i] = (f_p - f_m) / (2 * delta)
                # Second derivative approximation
                H[i, i] = (f_p - 2 * f0 + f_m) / (delta ** 2)
                # Cross terms: use one-sided differences in cross directions
                for j in range(i + 1, n):
                    ej = np.zeros(n)
                    ej[j] = 1.0
                    # f(x + delta*e_i + delta*e_j)
                    xpp = x_center + delta * (ei + ej)
                    xpp = np.clip(xpp, -5, 5)
                    f_pp = func(xpp)
                    calls += 1
                    # f(x + delta*e_i - delta*e_j)
                    xpm = x_center + delta * (ei - ej)
                    xpm = np.clip(xpm, -5, 5)
                    f_pm = func(xpm)
                    calls += 1
                    # f(x - delta*e_i + delta*e_j)
                    xmp = x_center - delta * ei + delta * ej
                    xmp = np.clip(xmp, -5, 5)
                    f_mp = func(xmp)
                    calls += 1
                    # f(x - delta*e_i - delta*e_j)
                    xmm = x_center - delta * (ei + ej)
                    xmm = np.clip(xmm, -5, 5)
                    f_mm = func(xmm)
                    calls += 1
                    # Mixed second derivative
                    H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * delta ** 2)
                    H[j, i] = H[i, j]

            return g, H

        def solve_trust_region(g, H, delta, x_center):
            """Solve min 0.5*x^T H x + g^T x s.t. ||x|| <= delta in local coords."""
            n = len(g)
            # Regularize H for numerical stability
            H_reg = H + 1e-8 * np.eye(n)
            # Try unconstrained solution first
            try:
                dx_uncon = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                dx_uncon = -g / max(np.linalg.norm(g), 1e-12)

            if np.linalg.norm(dx_uncon) <= delta:
                return dx_uncon
            else:
                # Bisection on lambda for trust region constraint
                lo, hi = 0.0, delta / max(np.linalg.norm(g), 1e-12)
                for _ in range(50):
                    mid = (lo + hi) / 2
                    A = H_reg + mid * np.eye(n)
                    try:
                        dx = np.linalg.solve(A, -g)
                    except np.linalg.LinAlgError:
                        dx = -g / (mid + 1e-12)
                    if np.linalg.norm(dx) > delta:
                        lo = mid
                    else:
                        hi = mid
                dx = np.linalg.solve(H_reg + hi * np.eye(n), -g)
                if np.linalg.norm(dx) > delta:
                    dx = dx * delta / np.linalg.norm(dx)
                return dx

        # Main loop
        while calls < budget and time.time() < deadline:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Build local model
            g, H = build_model(x, fx, delta, None, func, calls)

            # Solve trust region subproblem
            dx = solve_trust_region(g, H, delta, x)

            # Clip to box
            x_trial = np.clip(x + dx, -5, 5)
            fx_trial = func(x_trial)
            calls += 1

            # Acceptance criterion
            actual = fx - fx_trial
            predicted = -g @ dx - 0.5 * dx @ H @ dx
            rho = actual / max(abs(predicted), 1e-12) if predicted != 0 else 1.0

            if rho > 0.25:
                x, fx = x_trial, fx_trial
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # Expand trust region
                delta = min(max(delta * 1.5, min_delta), max_delta)
            else:
                # Shrink trust region
                delta = max(delta * 0.5, min_delta)
                if delta <= min_delta * 1.1:
                    # Stuck; do a random restart within the box
                    x = np.random.default_rng(calls).uniform(-4, 4, n)
                    fx = func(x)
                    calls += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    delta = 1.0

        return best_x, best_fx
