import numpy as np
import time

class Optimizer:
    """
    Optimizer using a derivative-free trust-region method with quadratic model fitting
    for BBOB f13 (Sharp Ridge). This approach constructs local surrogate models to
    capture the anisotropic curvature of the sharp ridge, allowing large steps along
    the valley floor while constraining movement across the narrow width.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Trust-region optimization with quadratic model
        def trust_region(x_start, f_start, max_iter=100):
            nonlocal calls, best_f, best_x

            # Initialize trust region radius
            delta = 0.5
            delta_min = 1e-4
            delta_max = 5.0

            # Collect points for model building
            points = [x_start.copy()]
            f_values = [f_start]

            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Build quadratic model from available points
                # Use points within current trust region
                mask = [np.linalg.norm(points[i] - x_start) <= delta * 1.5 for i in range(len(points))]
                if sum(mask) < dim + 2:
                    # Need more points, add some
                    n_needed = dim + 2 - sum(mask)
                    for _ in range(n_needed):
                        if calls >= budget or time.time() > deadline:
                            break
                        # Generate point within trust region
                        direction = rng.normal(0, 1, size=dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        step = rng.uniform(0, delta, size=1)
                        x_new = x_start + step * direction
                        f_new = safe_call(x_new)
                        if f_new is None:
                            break
                        points.append(x_new)
                        f_values.append(f_new)
                        mask.append(np.linalg.norm(x_new - x_start) <= delta * 1.5)

                if sum(mask) < dim + 2:
                    break

                # Extract points for model
                idx = [i for i, m in enumerate(mask) if m]
                X = np.array([points[i] for i in idx])
                Y = np.array([f_values[i] for i in idx])

                # Fit quadratic model: f(x) ≈ f0 + g^T (x - x0) + (x - x0)^T H (x - x0)
                # Use least squares to fit
                n_points = len(X)
                if n_points < dim + 2:
                    break

                # Center and scale
                x0 = x_start
                dx = X - x0
                dy = Y - Y[0]

                # Build design matrix for quadratic model
                # Columns: dx_i, dx_i*dx_j (i<=j)
                n_features = dim + dim * (dim + 1) // 2
                A = np.zeros((n_points, n_features))
                for i in range(n_points):
                    A[i, :dim] = dx[i]
                    j = dim
                    for a in range(dim):
                        for b in range(a, dim):
                            A[i, j] = dx[i, a] * dx[i, b]
                            j += 1

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, dy, rcond=None)
                except Exception:
                    break

                # Extract gradient and Hessian
                g = coeffs[:dim]
                H = np.zeros((dim, dim))
                j = dim
                for a in range(dim):
                    for b in range(a, dim):
                        H[a, b] = coeffs[j]
                        H[b, a] = coeffs[j]
                        j += 1

                # Make Hessian symmetric and add regularization
                H = (H + H.T) / 2.0
                H += 1e-4 * np.eye(dim)

                # Find step that minimizes quadratic model within trust region
                # Solve: min 0.5 * s^T H s + g^T s  s.t. ||s|| <= delta
                # Use projected gradient or Newton step with trust region constraint

                # Try Newton step
                try:
                    s_newton = np.linalg.solve(H, -g)
                except Exception:
                    s_newton = -g / (np.linalg.norm(g) + 1e-10)

                # Check if Newton step is within trust region
                norm_s = np.linalg.norm(s_newton)
                if norm_s > delta:
                    # Scale to trust region boundary
                    s_newton = s_newton * (delta / norm_s)
                    # Check if this is better than Cauchy point
                    cauchy_step = -g / (np.linalg.norm(g) + 1e-10)
                    cauchy_step = cauchy_step * min(1.0, delta / (np.linalg.norm(cauchy_step) + 1e-10))

                    # Compare model values
                    f_newton_model = 0.5 * s_newton @ H @ s_newton + g @ s_newton
                    f_cauchy_model = 0.5 * cauchy_step @ H @ cauchy_step + g @ cauchy_step

                    if f_cauchy_model < f_newton_model:
                        s_newton = cauchy_step

                # Evaluate actual function at proposed step
                x_trial = x_start + s_newton
                f_trial = safe_call(x_trial)
                if f_trial is None:
                    break

                # Update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()

                # Compute actual vs predicted reduction
                predicted_reduction = f_start - (f_start + g @ s_newton + 0.5 * s_newton @ H @ s_newton)
                actual_reduction = f_start - f_trial

                # Update trust region
                if actual_reduction > 0:
                    rho = actual_reduction / max(abs(predicted_reduction), 1e-10)
                    if rho > 0.75:
                        delta = min(2 * delta, delta_max)
                        x_start = x_trial.copy()
                        f_start = f_trial
                        # Add point to model
                        points.append(x_trial.copy())
                        f_values.append(f_trial)
                    elif rho > 0.25:
                        # Keep delta, accept step
                        x_start = x_trial.copy()
                        f_start = f_trial
                        points.append(x_trial.copy())
                        f_values.append(f_trial)
                    else:
                        delta = 0.5 * delta
                else:
                    delta = 0.5 * delta

                # Check convergence
                if delta < delta_min:
                    break

            return x_start, f_start

        # Run trust-region optimization with restarts
        x_start = best_x.copy()
        f_start = best_f

        n_restarts = 8
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if r == 0:
                x_start, f_start = trust_region(x_start, f_start, max_iter=80)
            else:
                # New random start, biased towards current best
                x_new = best_x + rng.normal(0, 1.0, size=dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = safe_call(x_new)
                if f_new is None:
                    break
                x_start, f_start = trust_region(x_new, f_new, max_iter=60)

        return best_x, best_f
