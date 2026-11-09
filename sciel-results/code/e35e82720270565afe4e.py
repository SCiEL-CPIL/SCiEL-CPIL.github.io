"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: LHS spread + CMA-ES + L-BFGS-B polish.
    
    Designed for BBOB-style Sphere function with hidden shift/rotation/offset.
    AOCC rewards early progress, so we start with a cheap global spread.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0  # Map to [-5, 5]

    def _df_trust_region(self, func, x0, budget, deadline, dim, call_count):
        """Derivative-Free Trust-Region solver using quadratic interpolation.
        
        For Sphere functions, this exploits the known quadratic structure
        to achieve super-linear convergence near the minimum.
        """
        n = dim
        x = x0.copy()
        fx = func(x)
        call_count[0] += 1
        
        # Maintain a set of interpolation points
        # Start with coordinate basis + current point
        points = [x.copy()]
        fvals = [fx]
        
        # Add coordinate directions
        for d in range(n):
            if call_count[0] >= budget or time.time() > deadline:
                break
            for sign in [1.0, -1.0]:
                y = x.copy()
                y[d] += sign * 0.1
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                call_count[0] += 1
                points.append(y.copy())
                fvals.append(fy)
        
        # Trust region radius
        delta = 1.0
        rng = np.random.default_rng(123)
        
        for iteration in range(1000):
            if time.time() > deadline or call_count[0] >= budget:
                break
            
            # Need at least n+1 points for a quadratic model
            if len(points) < n + 1:
                # Add more points
                for _ in range(n + 1 - len(points)):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n) * delta
                    y = x + z
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    call_count[0] += 1
                    points.append(y.copy())
                    fvals.append(fy)
                continue
            
            # Build quadratic model from interpolation points
            # Use the most recent n+1 points for stability
            m = min(len(points), 2 * n + 1)
            pts = np.array(points[-m:])
            fvs = np.array(fvals[-m:])
            
            # Center points at x for numerical stability
            A = pts - x  # (m, n)
            b = fvs - fx
            
            # Solve for quadratic coefficients: f(x) ≈ fx + g^T dx + 0.5 dx^T H dx
            # We need to fit: b ≈ g^T A_i + 0.5 A_i^T H A_i
            # This is a system of m equations in n + n(n+1)/2 unknowns
            # For simplicity, use least squares with regularization
            
            # Build design matrix for quadratic terms
            # Each row: [A_i, A_i_j * A_i_k for j<=k]
            n_quad = n + n * (n + 1) // 2
            M = np.zeros((m, n_quad))
            M[:, :n] = A
            quad_idx = n
            for j in range(n):
                for k in range(j, n):
                    M[:, quad_idx] = A[:, j] * A[:, k]
                    quad_idx += 1
            
            # Least squares fit
            try:
                coeffs, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback: simple gradient step
                grad_est = (fvs[1] - fvs[0]) / 0.1 if m >= 2 else np.zeros(n)
                x = np.clip(x - 0.01 * grad_est, -5.0, 5.0)
                fx = func(x)
                call_count[0] += 1
                continue
            
            # Extract gradient and Hessian
            g = coeffs[:n]
            H = np.zeros((n, n))
            idx = n
            for j in range(n):
                for k in range(j, n):
                    H[j, k] = coeffs[idx]
                    H[k, j] = coeffs[idx]
                    idx += 1
            
            # Ensure H is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.maximum(eigvals, 1e-8)
            H = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Compute trust-region step
            # Minimize: 0.5 * dx^T H dx + g^T dx  s.t. ||dx|| <= delta
            # Use conjugate gradient with trust region constraint
            # Simple approach: try Newton step, clip to trust region
            if np.linalg.matrix_rank(H) >= n:
                try:
                    dx = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    dx = -g / (np.linalg.norm(g) + 1e-10) * delta
            else:
                dx = -g / (np.linalg.norm(g) + 1e-10) * delta
            
            # Clip to trust region
            dx_norm = np.linalg.norm(dx)
            if dx_norm > delta:
                dx = dx * (delta / dx_norm)
            
            # Evaluate new point
            x_new = np.clip(x + dx, -5.0, 5.0)
            fx_new = func(x_new)
            call_count[0] += 1
            
            # Update trust region
            actual = fx_new - fx
            predicted = 0.5 * dx @ H @ dx + g @ dx
            rho = actual / (abs(predicted) + 1e-10)
            
            if rho > 0.25:
                # Good step
                x = x_new
                fx = fx_new
                delta = min(2 * delta, 5.0)
                points.append(x.copy())
                fvals.append(fx)
                # Remove oldest point if set is too large
                if len(points) > 2 * n + 1:
                    points.pop(0)
                    fvals.pop(0)
            elif rho > -0.1:
                # Acceptable step
                x = x_new
                fx = fx_new
                delta = 0.5 * delta
                points.append(x.copy())
                fvals.append(fx)
                if len(points) > 2 * n + 1:
                    points.pop(0)
                    fvals.pop(0)
            else:
                # Poor step, shrink trust region
                delta = 0.25 * delta
            
            # If delta is very small, we've converged
            if delta < 1e-6:
                break
        
        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = [0]
        
        def wrapped(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))
        
        # Phase 1: Latin Hypercube Sampling spread (cheap global exploration)
        # This gives us a good initial best quickly for AOCC
        n_lhs = 31
        rng = np.random.default_rng(42)
        lhs_samples = self._lhs_sample(n_lhs, dim, rng)
        
        for x in lhs_samples:
            if time.time() > deadline or call_count[0] >= budget:
                break
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
        
        # Phase 2: Derivative-Free Trust-Region from best sample (quadratic-exploiting)
        if time.time() < deadline - 30 and call_count[0] < budget - 50:
            try:
                x_tr, f_tr = self._df_trust_region(wrapped, best_x, budget - call_count[0], deadline - 30, dim, call_count)
                if f_tr < best_f:
                    best_f = f_tr
                    best_x = x_tr
            except Exception:
                pass
        
        # Phase 3: L-BFGS-B final polish (convex Sphere function) with tighter tolerance
        if time.time() < deadline - 5 and call_count[0] < budget - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        # Phase 4: Additional polish if we still have budget
        if time.time() < deadline - 2 and call_count[0] < budget - 2:
            try:
                from scipy.optimize import minimize
                res2 = minimize(wrapped, best_x, method="L-BFGS-B",
                                bounds=[(-5.0, 5.0)] * dim,
                                options={"maxiter": 100, "ftol": 1e-15, "gtol": 1e-12})
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = np.clip(res2.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
