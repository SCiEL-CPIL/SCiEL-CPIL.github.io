"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-guided trust-region optimizer with coordinate scan and multi-start ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Track best solution
        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ===== Phase 1: Coordinate-wise line search from origin =====
        # For BBOB functions with box [-5,5]^d and optimum in [-4,4]^d,
        # each coordinate independently can be optimized via 1D search.
        # We do a coarse-to-fine line search along each axis.

        x = np.zeros(n)
        fx = func(x)
        try_update(x, fx)

        # Coarse scan: check values at -4, -3, -2, -1, 0, 1, 2, 3, 4
        scan_points = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

        for coord in range(n):
            if not check_time() or eval_count >= budget:
                break

            # For this coordinate, find the best among scan points
            best_val = fx
            best_scan = x[coord]

            for sv in scan_points:
                if abs(sv - x[coord]) < 1e-10:
                    continue
                if not check_time() or eval_count >= budget:
                    break
                x[coord] = sv
                fv = func(x)
                try_update(x, fv)
                if fv < best_val:
                    best_val = fv
                    best_scan = sv

            x[coord] = best_scan
            fx = best_val

        # ===== Phase 2: Surrogate-guided trust-region search =====
        # Fit a quadratic surrogate on a Latin Hypercube sample, then use
        # the surrogate's gradient to guide local (1+1)-ES steps.
        
        if check_time() and eval_count < budget:
            # Build a Latin Hypercube sample in [-4, 4]^n
            n_lhs = 50  # Number of LHS points
            if eval_count + n_lhs > budget:
                n_lhs = max(10, budget - eval_count - 5)
            
            # Generate Latin Hypercube
            lhs_points = np.zeros((n_lhs, n))
            for d in range(n):
                perm = rng.permutation(n_lhs)
                for i in range(n_lhs):
                    lhs_points[i, d] = -4.0 + 8.0 * (perm[i] + rng.random()) / n_lhs
            
            # Evaluate LHS points
            lhs_values = np.zeros(n_lhs)
            for i in range(n_lhs):
                if not check_time() or eval_count >= budget:
                    break
                xi = lhs_points[i]
                fi = func(xi)
                try_update(xi, fi)
                lhs_values[i] = fi
            
            # Fit quadratic surrogate: f(x) ≈ c0 + c1*x + c2*(x^2) + c3*(x_i*x_j)
            # Design matrix for quadratic model
            def build_design_matrix(X):
                """Build design matrix for quadratic polynomial in n dimensions."""
                m, d = X.shape
                # Terms: 1, x_i, x_i^2, x_i*x_j (i<j)
                n_terms = 1 + d + d + d*(d-1)//2
                A = np.zeros((m, n_terms))
                A[:, 0] = 1.0
                A[:, 1:1+d] = X
                # x_i^2 terms
                idx = 1 + d
                for i in range(d):
                    A[:, idx+i] = X[:, i]**2
                # x_i*x_j terms
                idx = 1 + 2*d
                for i in range(d):
                    for j in range(i+1, d):
                        A[:, idx] = X[:, i] * X[:, j]
                        idx += 1
                return A
            
            # Collect all evaluated points for surrogate fitting
            # Use the LHS points plus the current best
            surrogate_X = lhs_points[:min(n_lhs, eval_count-1)]  # Exclude origin if counted
            surrogate_y = lhs_values[:len(surrogate_X)]
            
            if len(surrogate_X) >= n + 2:
                A = build_design_matrix(surrogate_X)
                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, surrogate_y, rcond=None)
                    
                    # Compute surrogate gradient at current best
                    if best_x is not None:
                        def surrogate_grad(x0):
                            """Compute gradient of quadratic surrogate at x0."""
                            grad = np.zeros(n)
                            # Linear terms: c[1:1+n]
                            grad += coeffs[1:1+n]
                            # Quadratic x_i^2 terms: 2*x_i * c[1+n:1+2*n]
                            grad += 2.0 * x0 * coeffs[1+n:1+2*n]
                            # Cross terms x_i*x_j: for each i, sum_j x_j * c[1+2*n + idx(i,j)]
                            idx = 1 + 2*n
                            for i in range(n):
                                for j in range(i+1, n):
                                    grad[i] += x0[j] * coeffs[idx]
                                    grad[j] += x0[i] * coeffs[idx]
                                    idx += 1
                            return grad
                        
                        grad = surrogate_grad(best_x)
                        grad_norm = np.linalg.norm(grad)
                        
                        if grad_norm > 1e-8:
                            # Use gradient direction for guided ES
                            # Trust-region step: move in -gradient direction with small sigma
                            sigma_tr = 0.3
                            x_tr = best_x.copy()
                            fx_tr = best_f
                            
                            for step in range(30):
                                if not check_time() or eval_count >= budget:
                                    break
                                # Move in negative gradient direction with some noise
                                direction = -grad / grad_norm
                                z = rng.standard_normal(n) * 0.3
                                y = np.clip(x_tr + sigma_tr * (direction + z), -5, 5)
                                fy = func(y)
                                try_update(y, fy)
                                
                                if fy < fx_tr:
                                    x_tr = y
                                    fx_tr = fy
                                    # Recompute gradient at new point
                                    grad = surrogate_grad(x_tr)
                                    grad_norm = np.linalg.norm(grad)
                                    if grad_norm < 1e-8:
                                        break
                                    sigma_tr = min(sigma_tr * 1.1, 2.0)
                                else:
                                    sigma_tr *= 0.8
                                    if sigma_tr < 1e-6:
                                        break
                        else:
                            # Gradient is zero, fall back to random exploration
                            x_es = best_x.copy()
                            fx_es = best_f
                            sigma = 0.3
                            no_improve = 0
                            for step in range(30):
                                if not check_time() or eval_count >= budget:
                                    break
                                z = rng.standard_normal(n)
                                y = np.clip(x_es + sigma * z, -5, 5)
                                fy = func(y)
                                try_update(y, fy)
                                if fy < fx_es:
                                    x_es = y
                                    fx_es = fy
                                    sigma = min(sigma * 1.2, 3.0)
                                    no_improve = 0
                                else:
                                    sigma *= 0.85
                                    no_improve += 1
                                if sigma < 1e-6 or no_improve >= 20:
                                    break
                except np.linalg.LinAlgError:
                    # Fallback to random ES
                    x_es = best_x.copy() if best_x is not None else np.zeros(n)
                    fx_es = best_f if best_x is not None else func(x_es)
                    try_update(x_es, fx_es)
                    sigma = 0.3
                    no_improve = 0
                    for step in range(30):
                        if not check_time() or eval_count >= budget:
                            break
                        z = rng.standard_normal(n)
                        y = np.clip(x_es + sigma * z, -5, 5)
                        fy = func(y)
                        try_update(y, fy)
                        if fy < fx_es:
                            x_es = y
                            fx_es = fy
                            sigma = min(sigma * 1.2, 3.0)
                            no_improve = 0
                        else:
                            sigma *= 0.85
                            no_improve += 1
                        if sigma < 1e-6 or no_improve >= 20:
                            break

        # ===== Phase 3: Multi-start (1+1)-ES with adaptive sigma =====
        # If budget remains, do several restarts from promising regions
        if check_time() and eval_count < budget:
            num_restarts = 5
            for restart_idx in range(num_restarts):
                if not check_time() or eval_count >= budget:
                    break

                if restart_idx == 0 and best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 0.5, -5, 5)
                elif restart_idx == 1:
                    x = rng.uniform(-3, 3, n)
                elif restart_idx == 2:
                    x = rng.uniform(-4, 4, n)
                else:
                    x = rng.uniform(-2, 2, n)

                fx = func(x)
                try_update(x, fx)

                # Local (1+1)-ES with adaptive sigma
                sigma = 0.5
                no_improve = 0
                max_no_improve = 40

                while eval_count < budget and check_time():
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    try_update(y, fy)

                    if fy < fx:
                        x = y
                        fx = fy
                        sigma = min(sigma * 1.25, 5.0)
                        no_improve = 0
                    else:
                        sigma *= 0.8
                        no_improve += 1

                    if sigma < 1e-6 or no_improve >= max_no_improve:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
