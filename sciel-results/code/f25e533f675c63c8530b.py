"""Optimizer for BBOB-style f11 (Discus) using quadratic fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return func(x)

        def time_ok():
            return time.time() < deadline - 2

        # Phase 1: Initial structured sampling for quadratic fit
        # Sample: center, ±h in each coordinate, and cross terms
        x0 = np.zeros(d)
        f0 = wrapped(x0)
        best_x = x0.copy()
        best_f = f0

        # Initial step size for exploration
        h0 = 2.0

        # Collect points for quadratic fit
        # We need at least 21 points for a 5D quadratic
        # Use: center + 2*5 coordinate + 5*4/2*2 cross = 1 + 10 + 20 = 31
        pts = [x0]
        fvals = [f0]

        # Coordinate points
        for i in range(d):
            for sign in [-1, 1]:
                if calls[0] >= budget or not time_ok():
                    break
                xi = x0.copy()
                xi[i] += sign * h0
                xi = np.clip(xi, -5, 5)
                fi = wrapped(xi)
                pts.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

        # Cross-term points (for off-diagonal Hessian)
        for i in range(d):
            for j in range(i + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        if calls[0] >= budget or not time_ok():
                            break
                        xi = x0.copy()
                        xi[i] += si * h0
                        xi[j] += sj * h0
                        xi = np.clip(xi, -5, 5)
                        fi = wrapped(xi)
                        pts.append(xi)
                        fvals.append(fi)
                        if fi < best_f:
                            best_f = fi
                            best_x = xi.copy()

        def fit_quadratic(pts, fvals):
            """Fit a quadratic to the given points. Returns (x_min, f_min_est)."""
            n = len(pts)
            if n < 21:
                return None, None
            
            # Design matrix for quadratic: [1, x, x_i*x_j]
            # For d=5: 1 + 5 + 15 = 21 features
            A = np.zeros((n, 21))
            for k, x in enumerate(pts):
                A[k, 0] = 1.0
                A[k, 1:6] = x
                # Quadratic terms
                idx = 6
                for i in range(d):
                    for j in range(i, d):
                        A[k, idx] = x[i] * x[j]
                        idx += 1
            
            # Least squares fit
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, np.array(fvals), rcond=None)
            except np.linalg.LinAlgError:
                return None, None
            
            # coeffs[0] = constant
            # coeffs[1:6] = linear terms (gradient/2)
            # coeffs[6:] = quadratic terms
            # f(x) = c0 + sum(ci * xi) + sum(cij * xi * xj)
            # Hessian: H[i][j] = 2*cij for i!=j, H[i][i] = 2*cii
            # Actually: f = c0 + l^T x + 0.5 * x^T H x
            # So l = coeffs[1:6], and H is built from coeffs[6:]
            
            l = coeffs[1:6].copy()
            H = np.zeros((d, d))
            idx = 6
            for i in range(d):
                for j in range(i, d):
                    cij = coeffs[idx]
                    H[i, j] = 2 * cij
                    if i != j:
                        H[j, i] = 2 * cij
                    idx += 1
            
            # Solve H * dx = -l for the minimum
            try:
                dx = np.linalg.solve(H, -l)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -l, rcond=None)[0]
                except:
                    return None, None
            
            x_min = np.zeros(d)
            # The minimum of the fitted quadratic
            # f(x) = c0 + l^T x + 0.5 x^T H x
            # Minimum at x* where H x* + l = 0 => x* = -H^{-1} l
            # But this is in the "global" coordinate system
            # Wait, I need to be careful. The fit is:
            # f(x) ≈ c0 + sum_i l_i x_i + sum_{i<=j} c_ij x_i x_j
            # This is NOT in the form c0 + l^T x + 0.5 x^T H x directly
            # Let me re-derive:
            # f(x) = c0 + l^T x + x^T M x  where M is upper-triangular from coeffs
            # The gradient is: l + 2*M*x + 2*M^T*x - 2*diag(M)*x... this is getting messy
            # Let me just use the standard form.
            
            # Actually, let me reformulate. The quadratic is:
            # f(x) = c0 + sum_i a_i x_i + sum_{i<=j} b_ij x_i x_j
            # Gradient: df/dx_i = a_i + sum_j b_ij x_j + sum_j b_ji x_j (with b_ij = b_ji for i!=j)
            # At minimum: a + (M + M^T) x = 0 where M has b_ij in upper triangle
            # (M + M^T) is the Hessian H
            # So H x* = -a, where a = coeffs[1:6]
            # And H = 2*M (where M is the matrix from b terms)
            
            # I already computed H above. Let me verify:
            # H[i][i] = 2 * b_ii
            # H[i][j] = 2 * b_ij for i != j
            # This is correct for the Hessian of the quadratic form.
            
            x_min = -np.linalg.solve(H, l) if np.linalg.matrix_rank(H) == d else None
            if x_min is None:
                try:
                    x_min = np.linalg.lstsq(H, -l, rcond=None)[0]
                except:
                    return None, None
            
            # Clip to box
            x_min = np.clip(x_min, -5, 5)
            
            # Estimate f at x_min
            f_min_est = coeffs[0] + l @ x_min + 0.5 * x_min @ H @ x_min
            
            return x_min, f_min_est

        # Fit initial quadratic and jump
        x_min, f_min_est = fit_quadratic(pts, fvals)
        if x_min is not None and calls[0] < budget and time_ok():
            f_jump = wrapped(x_min)
            if f_jump < best_f:
                best_f = f_jump
                best_x = x_min.copy()

        # Phase 2: Iterative local quadratic refinement
        # Around the best point, do smaller and smaller quadratic fits
        current_best = best_x.copy()
        current_best_f = best_f
        
        # Adaptive step size for local fits
        local_h = 0.5
        min_h = 0.01
        
        # Continue local refinement while we have budget
        while calls[0] < budget - 20 and time_ok():
            # Sample around current_best
            local_pts = [current_best]
            local_fvals = [current_best_f]
            
            # Coordinate points
            for i in range(d):
                for sign in [-1, 1]:
                    if calls[0] >= budget - 10 or not time_ok():
                        break
                    xi = current_best.copy()
                    xi[i] += sign * local_h
                    xi = np.clip(xi, -5, 5)
                    fi = wrapped(xi)
                    local_pts.append(xi)
                    local_fvals.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                        current_best = xi.copy()
                        current_best_f = fi
            
            # Cross terms
            for i in range(d):
                for j in range(i + 1, d):
                    for si in [-1, 1]:
                        for sj in [-1, 1]:
                            if calls[0] >= budget - 10 or not time_ok():
                                break
                            xi = current_best.copy()
                            xi[i] += si * local_h
                            xi[j] += sj * local_h
                            xi = np.clip(xi, -5, 5)
                            fi = wrapped(xi)
                            local_pts.append(xi)
                            local_fvals.append(fi)
                            if fi < best_f:
                                best_f = fi
                                best_x = xi.copy()
                                current_best = xi.copy()
                                current_best_f = fi
            
            # Fit local quadratic
            x_min, f_min_est = fit_quadratic(local_pts, local_fvals)
            
            if x_min is not None and calls[0] < budget - 5 and time_ok():
                f_jump = wrapped(x_min)
                if f_jump < best_f:
                    best_f = f_jump
                    best_x = x_min.copy()
                    current_best = x_min.copy()
                    current_best_f = f_jump
                    # Shrink step size when we make progress
                    local_h *= 0.5
                else:
                    # No improvement, shrink step size
                    local_h *= 0.7
            else:
                local_h *= 0.5
            
            if local_h < min_h:
                break
        
        # Phase 3: Final polish with small coordinate steps
        if calls[0] < budget and time_ok():
            polish_h = max(local_h * 0.1, 0.001)
            for _ in range(3):
                for i in range(d):
                    for sign in [-1, 1]:
                        if calls[0] >= budget or not time_ok():
                            break
                        xi = best_x.copy()
                        xi[i] += sign * polish_h
                        xi = np.clip(xi, -5, 5)
                        fi = wrapped(xi)
                        if fi < best_f:
                            best_f = fi
                            best_x = xi.copy()
                polish_h *= 0.5
        
        return best_x, best_f
# EVOLVE-BLOCK-END
