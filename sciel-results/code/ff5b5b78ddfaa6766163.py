"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with quadratic surrogate model refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        n = self.budget

        # Initialize near origin (minimizer known to be in [-4,4]^5)
        x = rng.uniform(-2, 2, d)
        fx = func(x)

        # Keep history of evaluations for surrogate model
        history_x = [x.copy()]
        history_f = [fx]

        sigma = 1.5
        best_x = x.copy()
        best_f = fx

        calls_used = 1

        for step in range(n - 1):
            # Check time budget
            if time.time() > deadline:
                break

            # Try to use surrogate model after enough points
            if len(history_x) >= 12:
                # Fit a quadratic model to the last min(25, len) points
                # Use the most recent points for local accuracy
                recent_k = min(25, len(history_x))
                X = np.array(history_x[-recent_k:])
                F = np.array(history_f[-recent_k:])

                # Build feature matrix for quadratic: [1, x_i, x_i*x_j]
                # For d=5: 1 + 5 + 10 = 16 features
                def quad_features(xv):
                    xf = [1.0]
                    xf.extend(xv)
                    for i in range(d):
                        for j in range(i, d):
                            xf.append(xv[i] * xv[j])
                    return np.array(xf)

                try:
                    Phi = np.array([quad_features(xi) for xi in X])
                    # Solve least squares
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, F, rcond=None)

                    # Predict gradient and Hessian from coefficients
                    # f ≈ c0 + c1^T x + 0.5 x^T H x
                    # where c1 = coeffs[1:1+d]
                    # H is symmetric: H_ij = 2*coeffs[1+d+i+j] for i!=j, coeffs[1+d+i+i] for i=j
                    c1 = coeffs[1:1+d]
                    H = np.zeros((d, d))
                    for i in range(d):
                        for j in range(i, d):
                            idx = 1 + d + i + j
                            H[i, j] = 2.0 * coeffs[idx]
                            H[j, i] = H[i, j]

                    # Check if Hessian is positive definite (or at least useful)
                    eigvals = np.linalg.eigvalsh(H)
                    if np.all(eigvals > 1e-10):
                        # Newton step
                        try:
                            step_vec = np.linalg.solve(H, -c1)
                            # Limit step size
                            step_norm = np.linalg.norm(step_vec)
                            if step_norm > 2.0:
                                step_vec = step_vec / step_norm * 2.0
                            x_new = np.clip(x + step_vec, -5, 5)
                        except np.linalg.LinAlgError:
                            x_new = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                    else:
                        # Fallback to ES step
                        x_new = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                except Exception:
                    x_new = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            else:
                # Not enough points, use ES step
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

            # Adapt sigma
            if fx_new < fx:
                x, fx = x_new, fx_new
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Keep history bounded
            if len(history_x) > 50:
                history_x.pop(0)
                history_f.pop(0)

        return best_x, best_f
# EVOLVE-BLOCK-END
