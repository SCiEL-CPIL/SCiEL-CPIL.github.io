"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and quadratic surrogate refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        n = self.budget

        # Initialize near origin (minimizer known to be in [-4,4]^5)
        x = rng.uniform(-3, 3, d)
        fx = func(x)

        # Keep history of evaluations for surrogate model
        history_x = [x.copy()]
        history_f = [fx]

        sigma = 2.0
        best_x = x.copy()
        best_f = fx

        calls_used = 1
        success_count = 0
        trail = 0.0
        window = 15  # trail window for 1/5 success rule

        for step in range(n - 1):
            # Check time budget
            if time.time() > deadline:
                break

            x_new = None

            # Try to use surrogate model after enough points
            if len(history_x) >= 15 and step % 3 == 0:
                # Fit a quadratic model to the last min(25, len) points
                recent_k = min(25, len(history_x))
                X = np.array(history_x[-recent_k:])
                F = np.array(history_f[-recent_k:])

                # Build feature matrix for quadratic: [1, x_i, x_i*x_j]
                # For d=5: 1 + 5 + 10 = 16 features
                n_params = 1 + d + d * (d + 1) // 2
                Phi = np.ones((recent_k, n_params))
                Phi[:, 1:1 + d] = X
                col = 1 + d
                for i in range(d):
                    for j in range(i, d):
                        Phi[:, col] = X[:, i] * X[:, j]
                        col += 1

                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, F, rcond=None)

                    # Extract gradient and Hessian
                    c1 = coeffs[1:1 + d]
                    H = np.zeros((d, d))
                    col = 1 + d
                    for i in range(d):
                        for j in range(i, d):
                            val = coeffs[col]
                            H[i, j] = val
                            if i != j:
                                H[j, i] = val
                            col += 1
                    # Scale Hessian: f ≈ c0 + c1^T x + x^T H x => grad = c1 + 2Hx
                    # Newton: solve (2H) x_new = -(c1)  =>  H x_new = -c1/2
                    # Actually: min c0 + c1^T x + x^T H x => grad = c1 + 2Hx = 0 => x = -H^{-1} c1 / 2
                    # But our parametrization: f = c0 + sum c_i x_i + sum c_ij x_i x_j
                    # grad_i = c_i + sum_j c_ij (1 if i==j else ... )
                    # Let's just use: predicted optimum = -H^{-1} * c1 / 2 if H is PD
                    # Simpler: solve H x* = -c1/2

                    eigvals = np.linalg.eigvalsh(H)
                    if np.all(eigvals > 1e-8):
                        try:
                            x_star = -np.linalg.solve(H, c1 / 2.0)
                            x_star = np.clip(x_star, -5, 5)
                            # Limit step from current x
                            step_vec = x_star - x
                            step_norm = np.linalg.norm(step_vec)
                            if step_norm > 3.0:
                                step_vec = step_vec / step_norm * 3.0
                            x_new = np.clip(x + step_vec, -5, 5)
                        except np.linalg.LinAlgError:
                            x_new = None
                    else:
                        x_new = None
                except Exception:
                    x_new = None

            if x_new is None:
                # ES step with 1/5 success rule
                x_new = np.clip(x + sigma * rng.standard_normal(d), -5, 5)

            fx_new = func(x_new)
            calls_used += 1

            # Record in history
            history_x.append(x_new.copy())
            history_f.append(fx_new)

            # Update best
            if fx_new < best_f:
                best_x = x_new.copy()
                best_f = fx_new

            # Adapt sigma using 1/5 success rule
            if fx_new < fx:
                x, fx = x_new, fx_new
                success_count += 1
            else:
                sigma *= 0.97

            # Periodically update sigma based on success ratio
            if (step + 1) % window == 0:
                ratio = success_count / window
                if ratio > 0.2:
                    sigma *= 1.2
                elif ratio < 0.2:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.01, 10.0)
                success_count = 0

            # Keep history bounded
            if len(history_x) > 60:
                history_x.pop(0)
                history_f.pop(0)

        return best_x, best_f
# EVOLVE-BLOCK-END
