"""Optimizer for BBOB-style f13 (Sharp ridge) using BOBYQA-style trust-region."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Trust-region optimizer with quadratic model fitting for sharp ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_eval(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize with a point in the center
        x0 = np.zeros(n)
        f0 = safe_eval(x0)
        best_f = f0
        best_x = x0.copy()

        # Trust-region radius
        rho = 2.0
        min_rho = 1e-4
        max_rho = 5.0

        # Build initial point set for quadratic model
        # Use a sphere of 2n points around current center
        def build_points(center, radius):
            """Generate 2n+1 points: center + 2n along axes."""
            pts = [center.copy()]
            for i in range(n):
                d_plus = center.copy()
                d_minus = center.copy()
                d_plus[i] += radius
                d_minus[i] -= radius
                # Clip to box
                d_plus = np.clip(d_plus, -5.0, 5.0)
                d_minus = np.clip(d_minus, -5.0, 5.0)
                pts.append(d_plus)
                pts.append(d_minus)
            return pts

        # Quadratic model: f(x) ≈ f0 + g^T (x - c) + (x-c)^T H (x-c)
        # We fit using points around center c
        def fit_quadratic(center, points, values):
            """
            Fit quadratic model f(x) ≈ a + g^T(x-c) + (x-c)^T H (x-c)
            where H is symmetric. We solve for g and H entries.
            Number of params: 1 + n + n*(n+1)/2
            """
            m = len(points)
            # Design matrix
            num_params = 1 + n + n * (n + 1) // 2
            A = np.zeros((m, num_params))
            for i, p in enumerate(points):
                d = p - center
                A[i, 0] = 1.0
                for j in range(n):
                    A[i, 1 + j] = d[j]
                # Hessian terms: for each pair (j,k) with j<=k
                idx = 1 + n
                for j in range(n):
                    for k in range(j, n):
                        if j == k:
                            A[i, idx] = d[j] * d[k]
                        else:
                            A[i, idx] = d[j] * d[k] + d[k] * d[j]  # = 2*d[j]*d[k]
                        idx += 1

            vals = np.array(values)
            # Least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
            except np.linalg.LinAlgError:
                return None

            g = coeffs[1:1 + n].copy()
            # Reconstruct H from coefficients
            H = np.zeros((n, n))
            idx = 1 + n
            for j in range(n):
                for k in range(j, n):
                    if j == k:
                        H[j, j] = coeffs[idx]
                    else:
                        # coeff = 2*H[j,k]
                        val = coeffs[idx] / 2.0
                        H[j, k] = val
                        H[k, j] = val
                    idx += 1
            return g, H

        def minimize_quadratic(g, H, center, radius):
            """
            Minimize q(d) = g^T d + d^T H d  s.t. ||d|| <= radius
            Use projected gradient or simple Newton with line search.
            """
            d = np.zeros(n)
            # Try Newton step
            try:
                H_reg = H + 1e-6 * np.eye(n)
                d_newton = np.linalg.solve(H_reg, -g)
                if np.linalg.norm(d_newton) > radius:
                    d_newton = d_newton / np.linalg.norm(d_newton) * radius
                d = d_newton
            except np.linalg.LinAlgError:
                # Fallback: projected gradient
                for _ in range(50):
                    grad = 2.0 * H @ d + g
                    lr = radius / (np.linalg.norm(grad) + 1e-8)
                    d = d - lr * grad
                    if np.linalg.norm(d) > radius:
                        d = d / np.linalg.norm(d) * radius
                    if np.linalg.norm(grad) < 1e-8:
                        break

            # Ensure within radius
            if np.linalg.norm(d) > radius:
                d = d / np.linalg.norm(d) * radius
            return center + d

        # Main loop
        iterations = 0
        while eval_count < budget and time.time() < deadline:
            iterations += 1

            # Build local point set
            points = build_points(best_x, rho)
            # Evaluate points we haven't seen (or re-evaluate for model accuracy)
            # To save budget, only evaluate new points
            values = []
            for p in points:
                # Check if we've evaluated this point before (approximate)
                # For simplicity, just evaluate all
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                vals = safe_eval(p)
                values.append(vals)

            if len(values) < 2 * n + 1:
                break

            # Fit quadratic
            model = fit_quadratic(best_x, points, values)
            if model is None:
                # Shrink and try again
                rho *= 0.5
                if rho < min_rho:
                    break
                continue

            g, H = model

            # Minimize the model within trust region
            x_try = minimize_quadratic(g, H, best_x, rho)
            x_try = np.clip(x_try, -5.0, 5.0)

            # Evaluate the trial point
            if eval_count >= budget or time.time() > deadline:
                break
            f_try = safe_eval(x_try)

            # Compute predicted improvement
            d = x_try - best_x
            f_pred = f0 + g @ d + d @ H @ d
            actual_improve = f0 - f_try
            predicted_improve = f0 - f_pred

            if predicted_improve > 1e-10:
                rho_ratio = actual_improve / predicted_improve
            else:
                rho_ratio = 0.0

            # Update trust region
            if rho_ratio > 0.7:
                rho = min(rho * 2.0, max_rho)
            elif rho_ratio > 0.1:
                rho = rho * 1.5
            else:
                rho = max(rho * 0.5, min_rho)

            # If the trial point is better, move there
            if f_try < best_f:
                # Update center to new best
                best_x = x_try.copy()
                best_f = f_try
                f0 = f_try
            else:
                # Stay at current center, but maybe adjust
                f0 = best_f

            # Check for convergence
            if rho < min_rho * 1.1:
                # Try a random restart in a larger region
                if eval_count < budget - 10 and time.time() < deadline:
                    x_restart = np.clip(best_x + 3.0 * np.random.randn(n), -5.0, 5.0)
                    f_restart = safe_eval(x_restart)
                    if f_restart < best_f:
                        best_x = x_restart.copy()
                        best_f = f_restart
                        f0 = f_restart
                        rho = 2.0
                    else:
                        break
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
