"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Local quadratic surrogate model optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = 5.0
        budget = self.budget

        # Storage for recent evaluations
        max_history = 50
        X = np.zeros((max_history, d))
        F = np.zeros(max_history)
        n_evals = 0

        def record(x, f):
            nonlocal n_evals
            n_evals += 1
            if n_evals <= max_history:
                X[n_evals - 1] = x
                F[n_evals - 1] = f
            else:
                # Shift history
                X[:-1] = X[1:]
                F[:-1] = F[1:]
                X[-1] = x
                F[-1] = f

        # Initial point
        x0 = np.zeros(d)
        f0 = func(x0)
        record(x0, f0)

        best_x = x0.copy()
        best_f = f0

        # Initial step size for exploration
        sigma = 1.0

        for _ in range(budget - 1):
            # Check if we have enough points to fit a quadratic
            # We need at least d+1 points for linear, but let's use a smaller set
            # For a quadratic in d dims, we need (d+1)(d+2)/2 points ideally, but that's too many.
            # Instead, we fit a quadratic to the last K points using least squares.
            # Let's use K = min(25, n_evals)
            K = min(25, n_evals)
            
            if K < d + 1:
                # Not enough points, use random perturbation
                # But we have at least 1 point. Let's try a few random steps until we have d+1
                # Actually, let's just do a random step
                y = np.clip(x0 + sigma * rng.standard_normal(d), -B, B)
                fy = func(y)
                record(y, fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                # Keep x0 as the center for now? Or move to y?
                # Let's keep the best point as the center for the next iteration's neighborhood?
                # The prompt says "fits a paraboloid to the last 25 function evaluations to predict the minimizer"
                # So the model is global over the recent history, not centered at x0 necessarily.
                # However, we need a starting point for the next step.
                # Let's update x0 to the best point found so far? Or the predicted minimizer?
                # "then evaluates a single step toward that predicted point"
                # So we calculate predicted minimizer x_pred. Then we evaluate x_new = x_current + alpha * (x_pred - x_current)?
                # Or just x_new = x_pred?
                # "evaluates a single step toward that predicted point" implies a step, not a jump.
                # But for smooth functions, jumping to the predicted minimizer is often good.
                # Let's try evaluating the predicted minimizer directly if it's within bounds.
                # If not, project it.
                
                # For now, with few points, let's just do a simple random walk to build history.
                # But we need to be efficient.
                # Let's stick to the plan: if K >= d+1, fit model.
                # Since K starts at 1, 2, ..., we will do random steps until K >= d+1 (6 points).
                # This is a bit wasteful but necessary to build the model.
                continue

            # Extract recent history
            Xr = X[n_evals - K:]
            Fr = F[n_evals - K:]
            
            # Fit quadratic model: f(x) ~ c + g^T x + 0.5 x^T H x
            # We can use least squares to fit the coefficients.
            # Design matrix A: columns are [1, x_i, x_i * x_i (symmetric part)]
            # Number of parameters: 1 + d + d*(d+1)/2 = (d+1)(d+2)/2
            # For d=5, params = 21.
            # We need K >= 21 for a full rank system, but K=25 is close.
            # If K < 21, the system is underdetermined or ill-conditioned.
            # We can use ridge regression or just fit a linear model if K is small.
            
            n_params = (d + 1) * (d + 2) // 2
            
            if K < n_params:
                # Not enough points for full quadratic. Fit linear model?
                # Linear model: f(x) ~ c + g^T x
                # Params: 1 + d = 6
                if K >= d + 1:
                    # Fit linear
                    A_lin = np.hstack([np.ones((K, 1)), Xr])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A_lin, Fr, rcond=None)
                        c = coeffs[0]
                        g = coeffs[1:]
                        # Minimizer of linear model is at infinity, so we can't use it directly.
                        # Instead, take a step in the negative gradient direction.
                        # x_new = x_center - alpha * g
                        # What is x_center? The centroid of the recent points?
                        x_center = np.mean(Xr, axis=0)
                        # Choose step size based on norm of g
                        step_size = 0.5 / (np.linalg.norm(g) + 1e-8)
                        y = np.clip(x_center - step_size * g, -B, B)
                        fy = func(y)
                        record(y, fy)
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                        x0 = y  # Move center to new point?
                        continue
                    except np.linalg.LinAlgError:
                        pass
                
                # Fallback to random step
                y = np.clip(x0 + sigma * rng.standard_normal(d), -B, B)
                fy = func(y)
                record(y, fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                continue

            # Fit full quadratic
            # Construct design matrix
            A = np.zeros((K, n_params))
            for i in range(K):
                x_i = Xr[i]
                A[i, 0] = 1.0
                A[i, 1:1+d] = x_i
                idx = 1 + d
                for j in range(d):
                    for k in range(j, d):
                        A[i, idx] = x_i[j] * x_i[k]
                        idx += 1
            
            # Ridge regression for stability
            lam = 1e-4
            try:
                A_reg = A.copy()
                # Add regularization to diagonal of A^T A
                # Alternatively, solve (A^T A + lam*I) coeffs = A^T F
                ATA = A.T @ A + lam * np.eye(n_params)
                ATb = A.T @ Fr
                coeffs = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, Fr, rcond=None)
                except np.linalg.LinAlgError:
                    y = np.clip(x0 + sigma * rng.standard_normal(d), -B, B)
                    fy = func(y)
                    record(y, fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    continue

            # Extract Hessian and gradient
            c = coeffs[0]
            g = coeffs[1:1+d].copy()
            # H is symmetric, stored in upper triangle
            H = np.zeros((d, d))
            idx = 1 + d
            for j in range(d):
                for k in range(j, d):
                    val = coeffs[idx]
                    H[j, k] = val
                    H[k, j] = val
                    idx += 1
            
            # The model is f(x) = c + g^T x + 0.5 x^T H x ?
            # Wait, my design matrix used x_i[j]*x_i[k].
            # So the term is sum_{j<=k} coeffs[idx] * x_j * x_k.
            # This corresponds to 0.5 x^T H x if H is symmetric and off-diagonals are doubled?
            # Let's check: 0.5 x^T H x = 0.5 * (sum_j H_jj x_j^2 + 2 sum_{j<k} H_jk x_j x_k)
            # = sum_j 0.5 H_jj x_j^2 + sum_{j<k} H_jk x_j x_k.
            # My expansion: sum_j c_jj x_j^2 + sum_{j<k} c_jk x_j x_k.
            # So H_jj = 2 * c_jj and H_jk = c_jk (for j!=k)?
            # No, if I define the model as c + g^T x + 0.5 x^T H x, then:
            # Coeff of x_j^2 is 0.5 H_jj.
            # Coeff of x_j x_k (j!=k) is 0.5 (H_jk + H_kj) = H_jk (since symmetric).
            # In my design matrix, the coefficient for x_j x_k (j<k) is coeffs[idx].
            # So coeffs[idx] = H_jk.
            # And coeffs[idx] for x_j^2 is 0.5 H_jj.
            # So H_jj = 2 * coeffs[idx_diag] and H_jk = coeffs[idx_off].
            
            H = np.zeros((d, d))
            idx = 1 + d
            for j in range(d):
                for k in range(j, d):
                    val = coeffs[idx]
                    if j == k:
                        H[j, j] = 2.0 * val
                    else:
                        H[j, k] = val
                        H[k, j] = val
                    idx += 1

            # Minimize the quadratic model
            # f(x) = c + g^T x + 0.5 x^T H x
            # Gradient: g + H x = 0 => x = -H^{-1} g
            try:
                x_pred = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # H is singular or ill-conditioned
                # Use pseudo-inverse or fall back to random
                try:
                    x_pred = np.linalg.lstsq(H, -g, rcond=None)[0]
                except np.linalg.LinAlgError:
                    y = np.clip(x0 + sigma * rng.standard_normal(d), -B, B)
                    fy = func(y)
                    record(y, fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    continue
            
            # Clip to bounds
            x_pred = np.clip(x_pred, -B, B)
            
            # Evaluate the predicted point
            fy = func(x_pred)
            record(x_pred, fy)
            
            if fy < best_f:
                best_f = fy
                best_x = x_pred.copy()
            
            # Update x0 to the new point for the next iteration?
            # The model is fitted to the last 25 points.
            # If we keep x0 as the center, we might drift away.
            # Let's set x0 to the best point found so far? Or the predicted point?
            # If the model is good, x_pred is close to the minimum.
            # Let's set x0 = x_pred.
            x0 = x_pred

        return best_x, best_f
# EVOLVE-BLOCK-END
