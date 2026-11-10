import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for Bent Cigar (f12) using a trust-region surrogate model
    approach combined with Sobol global exploration.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_x = None
        best_fx = float('inf')

        def guarded(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            x = np.asarray(x, dtype=np.float64)
            calls_used += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol global scan to find a good starting point
        n_scan = min(40, budget // 5)
        if n_scan < 10:
            n_scan = 10

        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            u = sampler.random(n_scan)
        except Exception:
            rng_scan = np.random.default_rng(42)
            u = rng_scan.random((n_scan, dim))

        pts = -5.0 + 10.0 * u

        for i in range(n_scan):
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                guarded(pts[i])
            except Exception:
                break

        if best_x is None:
            try:
                guarded(np.zeros(dim))
            except Exception:
                return np.zeros(dim), float('inf')

        # Phase 2: Trust-region surrogate model optimization
        # We maintain a set of recent points and fit a quadratic model
        x = best_x.copy()
        fx = best_fx

        # Collect initial points for the surrogate model
        # Use local perturbations around the best point found
        rng = np.random.default_rng(12345)
        sigma_init = 2.0

        # Store points and values for the surrogate
        X_points = [x.copy()]
        F_values = [fx]

        # Add initial local points to build the model
        n_local = min(2 * dim + 1, 30)
        for _ in range(n_local):
            if time.time() >= deadline or calls_used >= budget:
                break
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma_init * direction, -5, 5)
            try:
                fy = guarded(y)
            except Exception:
                break
            X_points.append(y.copy())
            F_values.append(fy)
            if fy < fx:
                fx = fy
                x = y.copy()
                # Re-center if we found something better
                X_points[0] = x.copy()
                F_values[0] = fx

        # Trust-region optimization loop
        trust_radius = 1.0
        min_radius = 1e-4
        max_radius = 5.0

        while calls_used < budget and time.time() < deadline:
            # Build quadratic surrogate from recent points
            # Use only the most recent points (up to 2*dim+1)
            n_model = min(len(X_points), 2 * dim + 1)
            if n_model < dim + 1:
                # Not enough points, do a simple step
                direction = rng.standard_normal(dim)
                y = np.clip(x + trust_radius * direction, -5, 5)
                try:
                    fy = guarded(y)
                except Exception:
                    break
                X_points.append(y.copy())
                F_values.append(fy)
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    X_points[0] = x.copy()
                    F_values[0] = fx
                continue

            # Use the last n_model points
            X_m = np.array(X_points[-n_model:])
            F_m = np.array(F_values[-n_model:])

            # Center and scale for numerical stability
            X_center = X_m.mean(axis=0)
            X_scaled = X_m - X_center
            # Estimate scale
            scale = np.maximum(np.std(X_scaled, axis=0), 1e-6)
            X_norm = X_scaled / scale

            # Fit quadratic model: f(x) ≈ a + b^T x + x^T Q x
            # Design matrix for quadratic terms
            n = n_model
            d = dim
            n_quad = d + 1 + d * (d + 1) // 2  # bias + linear + quadratic terms

            # Build design matrix
            A = np.zeros((n, n_quad))
            A[:, 0] = 1.0  # bias
            A[:, 1:1+d] = X_norm  # linear terms

            # Quadratic terms (upper triangle including diagonal)
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    A[:, idx] = X_norm[:, i] * X_norm[:, j]
                    idx += 1

            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, F_m, rcond=None)
            except Exception:
                coeffs = np.zeros(n_quad)

            # Extract model parameters
            a = coeffs[0]
            b = coeffs[1:1+d]
            Q_flat = coeffs[1+d:]

            # Reconstruct Q matrix (symmetric)
            Q = np.zeros((d, d))
            idx = 0
            for i in range(d):
                for j in range(i, d):
                    Q[i, j] = Q_flat[idx]
                    Q[j, i] = Q_flat[idx]
                    idx += 1

            # Find minimizer of the quadratic model within the trust region
            # Minimize a + b^T x + x^T Q x
            # Gradient: b + 2 Q x = 0 => x = -0.5 Q^{-1} b
            try:
                if np.all(np.abs(Q) < 1e-10):
                    # Nearly flat, use gradient direction
                    if np.linalg.norm(b) > 1e-10:
                        step = -b / np.linalg.norm(b) * trust_radius
                    else:
                        step = np.zeros(d)
                else:
                    # Solve 2 Q x = -b
                    try:
                        step = np.linalg.solve(2 * Q, -b)
                    except np.linalg.LinAlgError:
                        step = np.linalg.lstsq(2 * Q, -b, rcond=None)[0]

                    # Clip to trust region
                    step_norm = np.linalg.norm(step)
                    if step_norm > trust_radius:
                        step = step / step_norm * trust_radius
            except Exception:
                step = np.zeros(d)

            # Proposed point
            x_proposed = np.clip(X_center + scale * step, -5, 5)

            # Evaluate
            try:
                fy = guarded(x_proposed)
            except Exception:
                break

            # Update trust region
            improvement = fx - fy
            predicted_improvement = F_m[0] - (a + b @ step + step @ Q @ step)

            if improvement > 0.25 * predicted_improvement:
                # Good step, expand trust region
                trust_radius = min(trust_radius * 1.5, max_radius)
            elif improvement > 0:
                # Some improvement, keep radius
                pass
            else:
                # No improvement, shrink trust region
                trust_radius = max(trust_radius * 0.5, min_radius)

            # Update best
            if fy < fx:
                fx = fy
                x = x_proposed.copy()

            # Add point to history
            X_points.append(x_proposed.copy())
            F_values.append(fy)

            # Keep history bounded
            max_history = 50
            if len(X_points) > max_history:
                X_points = X_points[-max_history:]
                F_values = F_values[-max_history:]

        return best_x, best_fx
