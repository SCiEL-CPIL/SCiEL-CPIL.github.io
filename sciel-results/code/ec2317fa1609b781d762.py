"""Rewritten optimizer using a local quadratic surrogate model."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initialize with a few random points to build initial history
        # Start with a reasonable initial point
        x_best = rng.uniform(-3, 3, n)
        f_best = func(x_best)

        # History: list of (x, f) pairs
        history = [(x_best.copy(), f_best)]
        call_count = 1

        # Initial step size for random fallback
        sigma = 1.0

        # Minimum number of points needed for quadratic fit
        min_points = n * (n + 1) // 2 + 1  # 16 for n=5
        # Use up to 25 points for the fit
        max_points = 25

        while call_count < budget:
            # If we have enough points, try to fit a quadratic
            if len(history) >= min_points:
                # Select the max_points points with the lowest function values
                # This ensures the fit is local to the best region found so far
                sorted_history = sorted(history, key=lambda item: item[1])
                recent_best = sorted_history[:max_points]
                
                X = np.array([pt for pt, _ in recent_best])
                F = np.array([f for _, f in recent_best])

                # Fit quadratic: f(x) ≈ c + x^T b + x^T A x
                # Design matrix for quadratic terms
                # For each point x, create [1, x_1, ..., x_n, x_1^2, x_1*x_2, ..., x_n^2]
                quad_terms = []
                for xi in X:
                    row = [1.0]
                    row.extend(xi)
                    # Squared and cross terms
                    for i in range(n):
                        for j in range(i, n):
                            row.append(xi[i] * xi[j])
                    quad_terms.append(row)
                Q = np.array(quad_terms)

                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(Q, F, rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = None

                if coeffs is not None:
                    c = coeffs[0]
                    b = coeffs[1:1 + n]
                    
                    # Reconstruct A from the quadratic coefficients
                    A = np.zeros((n, n))
                    idx = 1 + n  # start index for quadratic terms
                    for i in range(n):
                        for j in range(i, n):
                            val = coeffs[idx]
                            if i == j:
                                A[i, i] = val
                            else:
                                A[i, j] = val / 2.0
                                A[j, i] = val / 2.0
                            idx += 1

                    # Find minimizer of the quadratic model
                    # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                    try:
                        # Add small regularization for stability
                        A_reg = A + 1e-6 * np.eye(n)
                        x_star = -0.5 * np.linalg.solve(A_reg, b)
                    except np.linalg.LinAlgError:
                        x_star = None

                    if x_star is not None:
                        # Clip to bounds
                        x_star = np.clip(x_star, -5, 5)

                        # Evaluate the predicted minimum directly
                        f_star = func(x_star)
                        call_count += 1
                        history.append((x_star.copy(), f_star))

                        if f_star < f_best:
                            x_best = x_star
                            f_best = f_star
                            # Increase sigma slightly if we made a big jump? 
                            # Or keep it stable. Let's increase slightly to allow further exploration if needed.
                            sigma = min(sigma * 1.1, 5.0)
                        else:
                            # The predicted minimum was not better.
                            # This can happen if the quadratic approximation is poor 
                            # or if we are very close to the minimum and noise/curvature matters.
                            # Fall back to a random step from the current best.
                            # We might also want to reduce sigma if we are stuck.
                            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
                            fy = func(y)
                            call_count += 1
                            history.append((y.copy(), fy))

                            if fy < f_best:
                                x_best = y
                                f_best = fy
                                sigma = min(sigma * 1.2, 5.0)
                            else:
                                sigma = max(sigma * 0.9, 0.01)
                        
                        continue

            # If quadratic fit failed or we don't have enough points,
            # do a random perturbation
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < f_best:
                x_best = y
                f_best = fy
                sigma = min(sigma * 1.2, 5.0)
            else:
                sigma = max(sigma * 0.9, 0.01)

            history.append((y.copy(), fy))

        return x_best, f_best
# EVOLVE-BLOCK-END
