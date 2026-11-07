# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Surrogate-model optimizer for the bent cigar (f12) function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Initial sampling for the surrogate model ---
        # We need at least dim + 1 points to fit a quadratic (with intercept).
        # For dim=5, we need 6 points. Let's start with a small set of diverse points.
        # Start at origin and add perturbations.
        points = []
        values = []

        def add_point(x):
            x = np.clip(x, -5, 5)
            fx = func(x)
            points.append(x)
            values.append(fx)
            return x, fx

        # Initial point
        x0 = np.zeros(dim)
        add_point(x0)

        # Add coordinate-wise perturbations to build a local basis
        for i in range(dim):
            for sign in [1, -1]:
                xi = np.zeros(dim)
                xi[i] = sign * 1.0
                add_point(xi)

        # We now have 1 + 2*dim = 11 points. Budget used: 11.
        # Remaining budget: budget - 11

        remaining = budget - len(points)
        if remaining <= 0:
            return np.array(points[0]), values[0]

        # --- Surrogate Model Fitting and Optimization Loop ---
        # We will fit a quadratic model to the most recent K points.
        # Quadratic model: f(x) = c0 + c^T x + x^T A x
        # Number of parameters: 1 + dim + dim*(dim+1)/2
        # For dim=5: 1 + 5 + 15 = 21 parameters.
        # We need at least 21 points for a full fit. If we have fewer, we use least squares.
        
        K = min(len(points), 25)  # Use last up to 25 points
        step_size = 0.5
        min_step = 1e-4
        max_iter = remaining

        for it in range(max_iter):
            # Select the last K points
            X = np.array(points[-K:])
            Y = np.array(values[-K:])
            
            # Build design matrix for quadratic model
            # Columns: [1, x1, ..., xd, x1^2, x1*x2, ..., xd^2]
            # We only need symmetric part of A, so we use x_i * x_j for i <= j
            d = dim
            n_params = 1 + d + d * (d + 1) // 2
            
            if K < n_params:
                # Not enough points for full quadratic. Use linear model?
                # Or just do a random step?
                # Let's fall back to a simple gradient descent step using finite differences
                # if we can't fit the model.
                # Actually, let's just pick a random point near the best found so far.
                best_idx = np.argmin(Y)
                x_best = X[best_idx]
                y = np.clip(x_best + 0.1 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                points.append(y)
                values.append(fy)
                continue

            # Design matrix
            Z = np.ones((K, n_params))
            Z[:, 1:1+d] = X
            idx = 1 + d
            for i in range(d):
                for j in range(i, d):
                    Z[:, idx] = X[:, i] * X[:, j]
                    idx += 1
            
            # Solve least squares
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(Z, Y, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback
                best_idx = np.argmin(Y)
                x_best = X[best_idx]
                y = np.clip(x_best + 0.1 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                points.append(y)
                values.append(fy)
                continue

            # Extract coefficients
            c0 = coeffs[0]
            c_lin = coeffs[1:1+d]
            c_quad_flat = coeffs[1+d:]
            
            # Reconstruct A (symmetric)
            A = np.zeros((d, d))
            idx = 0
            for i in range(d):
                for j in range(i, d):
                    val = c_quad_flat[idx]
                    A[i, j] = val
                    if i != j:
                        A[j, i] = val
                    idx += 1

            # Find minimizer of the quadratic model: c0 + c^T x + x^T A x
            # Gradient: c + 2 A x = 0 => x* = -0.5 A^{-1} c
            # If A is singular or indefinite, we need to be careful.
            # For a minimum, A should be positive definite.
            # Let's check eigenvalues.
            try:
                eigvals = np.linalg.eigvalsh(A)
                if np.min(eigvals) > 1e-8:
                    # Positive definite, unique minimum
                    x_pred = -0.5 * np.linalg.solve(A, c_lin)
                else:
                    # Not positive definite. Use pseudo-inverse or regularized solve.
                    # Add regularization
                    A_reg = A + 1e-3 * np.eye(d)
                    x_pred = -0.5 * np.linalg.solve(A_reg, c_lin)
            except np.linalg.LinAlgError:
                x_pred = -0.5 * np.linalg.lstsq(A, c_lin, rcond=None)[0]

            # Clip predicted point to the box
            x_pred = np.clip(x_pred, -5, 5)

            # We want to take a step towards x_pred.
            # Let's find the current best point from the history.
            best_idx = np.argmin(Y)
            x_best = X[best_idx]
            
            # Direction
            direction = x_pred - x_best
            norm_dir = np.linalg.norm(direction)
            
            if norm_dir < 1e-6:
                # Predicted point is very close to current best.
                # Take a small random step to explore.
                y = np.clip(x_best + step_size * rng.standard_normal(dim), -5, 5)
            else:
                # Move towards the predicted minimum.
                # Step size should be adaptive.
                # Let's take a fraction of the distance.
                alpha = min(1.0, step_size / norm_dir) if norm_dir > step_size else 1.0
                # Actually, if the predicted point is far, we might overshoot.
                # Let's just move a fixed step in the direction, or to the predicted point if it's close.
                # A common strategy: move to x_best + alpha * (x_pred - x_best)
                # If x_pred is inside the box and close, alpha=1 is good.
                # If x_pred is far, alpha < 1 is safer.
                
                # Let's use a step size that scales with the distance but capped.
                alpha = min(1.0, 0.5 * norm_dir / (norm_dir + 1.0))
                # Simpler: just move halfway to the predicted point, or a fixed step.
                # Let's try moving a fixed step in the direction of the predicted minimum.
                y = np.clip(x_best + step_size * direction / norm_dir, -5, 5)
                
                # Alternatively, if the predicted point is within a reasonable distance, go there.
                if norm_dir <= 2.0:
                    y = x_pred

            fy = func(y)
            points.append(y)
            values.append(fy)

            # Update step size based on improvement
            if fy < values[-K-1] if len(values) > K else False:
                # Improved
                pass
            # Simple adaptive step size: if we improved, keep step_size, else reduce?
            # Or increase if we are stuck?
            # Let's keep it simple for now.
            
            # If we are very close to the boundary of the search space, reduce step size?
            # Not necessary for now.

        # Return the best point found
        all_vals = np.array(values)
        best_idx = np.argmin(all_vals)
        return np.array(points[best_idx]), all_vals[best_idx]

# EVOLVE-BLOCK-END
