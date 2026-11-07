"""Combined optimizer: Quadratic model fitting for global localization, followed by Hessian-scaled (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that leverages the known quadratic structure of the Sphere function.
    
    Strategy:
    1. Sample a set of points to estimate the quadratic coefficients (Hessian and gradient).
    2. Solve for the stationary point (minimum) of the fitted quadratic.
    3. Use this point as the initial guess for a local search that is scaled by the inverse Hessian.
       This effectively performs a CMA-ES-like search with a fixed covariance matrix derived from the function's curvature.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def make_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Sampling for Quadratic Fit ---
        # We need at least n*(n+1)/2 + 1 points to fit a full quadratic.
        # For n=5, that's 15+1=16 points. Let's use 25 for robustness.
        
        n_samples = 25
        X = np.zeros((n_samples, n))
        F = np.zeros(n_samples)
        
        # Point 0: Origin (helps with intercept)
        X[0] = np.zeros(n)
        
        # Points 1..n: Axis-aligned points to estimate gradient and diagonal Hessian
        for i in range(n):
            X[i+1, i] = 2.0
            
        # Remaining points: Random points to capture off-diagonal Hessian terms
        for i in range(n+1, n_samples):
            X[i] = rng.uniform(-3, 3, n)

        for i in range(n_samples):
            if calls >= self.budget:
                break
            F[i] = make_call(X[i])
            
        # --- Phase 2: Quadratic Model Fitting ---
        # Model: f(x) = 0.5 * x^T H x + g^T x + c
        # We fit: f(x) ≈ w^T phi(x)
        # where phi(x) = [1, x_1, ..., x_n, x_1^2, ..., x_n^2, x_1x_2, ..., x_{n-1}x_n]
        
        def feature_map(x):
            # x is (n,)
            phi = np.zeros(1 + n + n + n*(n-1)//2)
            phi[0] = 1.0
            phi[1:1+n] = x
            phi[1+n:1+2*n] = x**2
            idx = 1 + 2*n
            for i in range(n):
                for j in range(i+1, n):
                    phi[idx] = x[i] * x[j]
                    idx += 1
            return phi

        Phi = np.array([feature_map(x) for x in X])
        
        # Solve least squares: Phi * w = F
        try:
            w, residuals, rank, sv = np.linalg.lstsq(Phi, F, rcond=None)
        except np.linalg.LinAlgError:
            w = None

        # Initialize Hessian and Gradient for the local search
        # If fit fails, fallback to identity Hessian and zero gradient
        H = np.eye(n)
        g = np.zeros(n)
        x_star = np.zeros(n)

        if w is not None:
            # Extract Hessian H and gradient g from weights w
            c = w[0]
            g = w[1:1+n].copy()
            
            H = np.zeros((n, n))
            for i in range(n):
                H[i, i] = 2.0 * w[1+n+i]
            
            idx = 1 + 2*n
            for i in range(n):
                for j in range(i+1, n):
                    H[i, j] = w[idx]
                    H[j, i] = w[idx]
                    idx += 1
            
            # Solve H * x_star = -g
            try:
                # Add small regularization for numerical stability
                H_reg = H + 1e-8 * np.eye(n)
                x_star = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                x_star = np.linalg.lstsq(H + 1e-8*np.eye(n), -g, rcond=None)[0]
            
            # Clip x_star to the box
            x_star = np.clip(x_star, -5, 5)
            
            # Evaluate the estimated minimum
            fx_star = make_call(x_star)
            
            # If the model was very poor, x_star might be worse than the best sample
            # In that case, we should restart from the best sample
            if fx_star > best_f:
                x_star = best_x.copy()
                # Re-estimate Hessian? No, just use identity for safety in local search
                # Or keep the Hessian but start from best_x
                # Keeping Hessian is risky if it's wrong. Let's use Identity if the fit was bad.
                # A simple check: if the model's prediction at x_star is much worse than actual, trust is low.
                # For now, if it's worse, we just use the best_x and a moderate sigma.
                H = np.eye(n) # Fallback to isotropic search if fit is bad

        else:
            # Fallback to best of initial samples
            best_init_idx = int(np.argmin(F))
            x_star = X[best_init_idx].copy()
            best_f = F[best_init_idx]
            best_x = x_star.copy()

        # --- Phase 3: Local Polish with Hessian-Scaled (1+1)-ES ---
        # Start from the estimated minimum
        x = x_star.copy()
        fx = make_call(x) # Ensure we have a fresh evaluation and updated best
        
        # Compute Cholesky decomposition of Hessian for efficient sampling
        # We want to sample z ~ N(0, H^{-1}).
        # Let H = L L^T. Then H^{-1} = (L^{-1})^T L^{-1}.
        # We can sample y ~ N(0, I), then z = L^{-T} y.
        # Or simpler: just use the eigen-decomposition or solve H z = y.
        # For n=5, solving is cheap.
        
        try:
            # Check if H is positive definite
            eigvals = np.linalg.eigvalsh(H)
            if np.any(eigvals < 1e-8):
                # Hessian not PD, fallback to identity
                H_inv = np.eye(n)
            else:
                H_inv = np.linalg.inv(H + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            H_inv = np.eye(n)
        
        # Initial sigma. Since we are at the minimum, start small.
        # The scale of the step in the original space is sigma * ||H^{-1} z||.
        # We want the step size to be small initially.
        sigma = 1e-3
        sigma_min = 1e-12
        sigma_max = 1.0
        
        # Track success rate for adaptive sigma
        # We use a simple 1/5 rule variant.
        success_count = 0
        total_steps = 0
        no_progress_calls = 0
        restart_threshold = 20
        max_restarts = 10
        restarts = 0

        # Pre-compute a random vector generator for efficiency? No, rng is fast.
        
        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate candidate
            # Sample z from N(0, H_inv)
            y = rng.standard_normal(n)
            try:
                z = np.linalg.solve(H + 1e-8 * np.eye(n), y)
            except np.linalg.LinAlgError:
                z = y # Fallback to isotropic

            # Scale by sigma
            step = sigma * z
            y_new = x + step
            fy_new = make_call(y_new)
            
            total_steps += 1

            if fy_new < fx:
                # Success
                x, fx = y_new, fy_new
                success_count += 1
                no_progress_calls = 0

                # Adapt sigma
                # If we are making progress, we can increase sigma slightly to move faster
                # But since we are close to the minimum, we should be careful.
                # A standard (1+1) ES increases sigma on success.
                sigma *= 1.1
                sigma = np.clip(sigma, sigma_min, sigma_max)
            else:
                # Failure
                sigma *= 0.9
                sigma = np.clip(sigma, sigma_min, sigma_max)
                no_progress_calls += 1

            # Restart logic if stuck
            if no_progress_calls >= restart_threshold and restarts < max_restarts:
                restarts += 1
                no_progress_calls = 0
                # Restart from best found so far
                x = best_x.copy()
                fx = best_f
                # Reset sigma to a moderate value
                sigma = 1e-2

        return best_x, best_f
# EVOLVE-BLOCK-END
