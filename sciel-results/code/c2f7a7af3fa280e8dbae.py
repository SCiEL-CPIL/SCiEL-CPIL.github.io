"""Rewritten optimizer for BBOB Sphere with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic adaptive grid refinement + quadratic model polish.
    
    Strategy:
    1. Coarse grid to locate the basin quickly (AOCC-friendly early progress).
    2. Adaptive refinement: iteratively refine the grid around the best point.
    3. Quadratic model fitting (Newton-style) for super-linear convergence
       on the Sphere function.
    4. Final L-BFGS-B polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _grid_search_coarse(self, func, dim, n_per_dim, call_count, budget, deadline):
        """Coarse uniform grid over [-5, 5]^dim.
        
        Returns best_x, best_f.
        """
        # Use a moderate grid size per dimension
        # For dim=5, n_per_dim=5 gives 3125 points; n_per_dim=4 gives 1024
        # We'll use adaptive: start with fewer if budget is tight
        n_pts = n_per_dim ** dim
        if n_pts > budget // 4:
            # Reduce n_per_dim
            while n_per_dim > 2:
                n_per_dim -= 1
                n_pts = n_per_dim ** dim
                if n_pts <= budget // 4:
                    break

        # Create grid
        coords = np.linspace(-5.0, 5.0, n_per_dim)
        grid = np.array(np.meshgrid(*([coords] * dim))).T.reshape(-1, dim)

        best_x = None
        best_f = np.inf

        for x in grid:
            if call_count[0] >= budget or time.time() > deadline:
                break
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
            call_count[0] += 1

        return best_x, best_f

    def _adaptive_refinement(self, func, x0, f0, dim, call_count, budget, deadline, max_levels=6):
        """Adaptive grid refinement around x0.
        
        At each level, create a local grid around the current best point
        with a shrinking radius, and refine.
        """
        x = x0.copy()
        fx = f0
        radius = 2.0  # initial local search radius

        for level in range(max_levels):
            if call_count[0] >= budget or time.time() > deadline:
                break

            # Number of points per dimension for this refinement level
            # Keep total points per level manageable
            n_per_dim = 5
            n_pts = n_per_dim ** dim
            # Don't exceed remaining budget for this level
            remaining = budget - call_count[0]
            if n_pts > remaining // (max_levels - level + 1):
                while n_per_dim > 2 and n_per_dim ** dim > remaining // (max_levels - level + 1):
                    n_per_dim -= 1
                n_pts = n_per_dim ** dim

            # Create local grid
            coords = np.linspace(x - radius, x + radius, n_per_dim)
            grid = np.array(np.meshgrid(*([coords] * dim))).T.reshape(-1, dim)

            level_best_x = x.copy()
            level_best_f = fx

            for gx in grid:
                if call_count[0] >= budget or time.time() > deadline:
                    break
                gx = np.clip(gx, -5.0, 5.0)
                gxf = float(func(gx))
                call_count[0] += 1
                if gxf < level_best_f:
                    level_best_f = gxf
                    level_best_x = gx.copy()

            # Update
            if level_best_f < fx:
                fx = level_best_f
                x = level_best_x
                # Shrink radius for next level
                radius *= 0.5
            else:
                # No improvement, shrink and continue
                radius *= 0.5

            if radius < 1e-6:
                break

        return x, fx

    def _quadratic_model_newton(self, func, x0, dim, call_count, budget, deadline):
        """Quadratic model fitting with Newton-style steps.
        
        For Sphere functions, the Hessian is constant, so a well-fit
        quadratic model converges in 1-2 Newton steps.
        """
        x = x0.copy()
        fx = float(func(x))
        call_count[0] += 1

        n = dim
        # Collect points for quadratic interpolation
        # Use 2n+1 points: x, x ± e_i * h for i=1..n
        h = 0.01
        points = [x.copy()]
        fvals = [fx]

        for d in range(n):
            if call_count[0] >= budget or time.time() > deadline:
                break
            for sign in [1.0, -1.0]:
                y = x.copy()
                y[d] += sign * h
                y = np.clip(y, -5.0, 5.0)
                fy = float(func(y))
                call_count[0] += 1
                points.append(y.copy())
                fvals.append(fy)

        # Now perform Newton iterations using the quadratic model
        for it in range(50):
            if call_count[0] >= budget or time.time() > deadline:
                break

            # Fit quadratic model from collected points
            m = len(points)
            if m < n + 1:
                break

            pts = np.array(points)
            fvs = np.array(fvals)

            # Center at x
            A = pts - x  # (m, n)
            b = fvs - fx  # (m,)

            # Build design matrix for quadratic: [dx, dx_j*dx_k]
            n_quad = n + n * (n + 1) // 2
            M = np.zeros((m, n_quad))
            M[:, :n] = A
            idx = n
            for j in range(n):
                for k in range(j, n):
                    M[:, idx] = A[:, j] * A[:, k]
                    idx += 1

            # Least squares fit
            try:
                coeffs, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                break

            g = coeffs[:n]
            H = np.zeros((n, n))
            idx = n
            for j in range(n):
                for k in range(j, n):
                    H[j, k] = coeffs[idx]
                    H[k, j] = coeffs[idx]
                    idx += 1

            # Regularize H to be positive definite
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.maximum(eigvals, 1e-6)
            H = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Newton step
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Fallback: gradient step
                grad_norm = np.linalg.norm(g)
                if grad_norm < 1e-10:
                    break
                dx = -g / (grad_norm ** 2) * 0.1

            # Limit step size
            dx_norm = np.linalg.norm(dx)
            max_step = 5.0
            if dx_norm > max_step:
                dx = dx * (max_step / dx_norm)

            x_new = np.clip(x + dx, -5.0, 5.0)
            fx_new = float(func(x_new))
            call_count[0] += 1

            if fx_new < fx:
                x = x_new
                fx = fx_new
                # Add new point to interpolation set
                points.append(x.copy())
                fvals.append(fx)
                if len(points) > 2 * n + 1:
                    points.pop(0)
                    fvals.pop(0)
            else:
                # Step didn't help, reduce step
                x = x - dx * 0.5
                x = np.clip(x, -5.0, 5.0)
                fx = float(func(x))
                call_count[0] += 1

            # Convergence check
            if np.linalg.norm(dx) < 1e-8:
                break

        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        # Phase 1: Coarse grid search for early progress
        # Use a moderate grid size
        n_coarse = 4  # 4^5 = 1024 points
        if n_coarse ** dim > budget // 5:
            n_coarse = 3  # 3^5 = 243 points

        best_x, best_f = self._grid_search_coarse(
            wrapped, dim, n_coarse, call_count, budget, deadline
        )

        # Phase 2: Adaptive refinement
        if time.time() < deadline - 30 and call_count[0] < budget - 50:
            try:
                x_ref, f_ref = self._adaptive_refinement(
                    wrapped, best_x, best_f, dim, call_count, budget, deadline - 30, max_levels=5
                )
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref
            except Exception:
                pass

        # Phase 3: Quadratic model Newton polish
        if time.time() < deadline - 10 and call_count[0] < budget - 10:
            try:
                x_new, f_new = self._quadratic_model_newton(
                    wrapped, best_x, dim, call_count, budget, deadline - 10
                )
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new
            except Exception:
                pass

        # Phase 4: L-BFGS-B final polish
        if time.time() < deadline - 2 and call_count[0] < budget - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 100, "ftol": 1e-15, "gtol": 1e-12, "maxls": 50})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
