"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with quadratic surrogate refinement for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 3.0
        calls_made = 1

        # Track recent evaluations for quadratic surrogate
        recent_points = []  # list of (x, f)
        max_recent = 12  # number of recent points to keep for quadratic fit

        # Main (1+1)-ES phase with adaptive step size
        for i in range(budget - 1):
            if time.time() > deadline:
                break

            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1

            # Store in recent points
            recent_points.append((y.copy(), fy))
            if len(recent_points) > max_recent:
                recent_points.pop(0)

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                # Adaptive step size: grow on success
                sigma = min(sigma * 1.15, 10.0)
            else:
                # Shrink on failure
                sigma = max(sigma * 0.85, 1e-7)

            # Try quadratic surrogate refinement every 10 steps if we have enough points
            if (i + 2) % 10 == 0 and len(recent_points) >= dim + 1:
                x_refined, f_refined, improved = self._quadratic_refinement(
                    func, recent_points, best_x, best_f, dim
                )
                if improved:
                    x, fx = x_refined, f_refined
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    # Reduce sigma since we're converging
                    sigma = min(sigma * 0.7, 1.0)

        # Final polish phase with decreasing sigma
        x = best_x.copy()
        fx = best_f
        remaining = budget - calls_made
        if remaining > 0:
            sigma = 0.1
            polish_budget = min(remaining, max(50, budget // 4))

            for i in range(polish_budget):
                if time.time() > deadline:
                    break

                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma = min(sigma * 1.03, 0.5)
                else:
                    sigma = max(sigma * 0.88, 1e-10)

                # Quadratic refinement during polish
                if (i + 1) % 5 == 0 and len(recent_points) >= dim + 1:
                    recent_points.append((y.copy(), fy))
                    if len(recent_points) > max_recent:
                        recent_points.pop(0)
                    x_refined, f_refined, improved = self._quadratic_refinement(
                        func, recent_points, best_x, best_f, dim
                    )
                    if improved:
                        x, fx = x_refined, f_refined
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                        sigma = min(sigma * 0.6, 0.1)

        return best_x, best_f

    def _quadratic_refinement(self, func, points, best_x, best_f, dim):
        """Try to fit a quadratic model and find its minimum."""
        try:
            # Use only the best subset of points for stability
            sorted_pts = sorted(points, key=lambda p: p[1])
            n_pts = min(len(sorted_pts), 2 * dim + 1)
            if n_pts < dim + 1:
                return best_x, best_f, False

            X = np.array([p[0] for p in sorted_pts[:n_pts]])
            F = np.array([p[1] for p in sorted_pts[:n_pts]])

            # Build quadratic model: f(x) ≈ c + g^T x + 0.5 x^T H x
            # Features: [1, x1, ..., xd, x1^2, x1*x2, ..., xd^2]
            n_feat = 1 + dim + dim * (dim + 1) // 2
            if n_pts < n_feat:
                # Fall back to linear model if not enough points
                A = np.hstack([np.ones((n_pts, 1)), X])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, F, rcond=None)
                except np.linalg.LinAlgError:
                    return best_x, best_f, False
                # Linear model: f ≈ c + g^T x, minimum at x = -c/g if ||g||>eps
                c = coeffs[0]
                g = coeffs[1:]
                if np.linalg.norm(g) < 1e-10:
                    return best_x, best_f, False
                x_star = np.clip(-c * g / np.dot(g, g), -5, 5)
                f_star = func(x_star)
                if f_star < best_f - 1e-10:
                    return x_star, f_star, True
                return best_x, best_f, False

            # Full quadratic fit
            A = np.zeros((n_pts, n_feat))
            for i in range(n_pts):
                A[i, 0] = 1
                A[i, 1:dim + 1] = X[i]
                # Quadratic terms
                for j in range(dim):
                    for k in range(j, dim):
                        A[i, dim + 1 + j * dim + k - j] = X[i, j] * X[i, k]
                # Note: this indexing is for the upper triangular part
                # Actually let's use a cleaner approach

            # Cleaner quadratic feature construction
            A = np.zeros((n_pts, n_feat))
            for i in range(n_pts):
                A[i, 0] = 1.0
                A[i, 1:dim + 1] = X[i]
                idx = dim + 1
                for j in range(dim):
                    for k in range(j, dim):
                        A[i, idx] = X[i, j] * X[i, k]
                        idx += 1

            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, F, rcond=None)
            except np.linalg.LinAlgError:
                return best_x, best_f, False

            # Extract: c, g, and quadratic coefficients
            c = coeffs[0]
            g = coeffs[1:dim + 1].copy()
            quad_coeffs = coeffs[dim + 1:]

            # Reconstruct Hessian H (symmetric): H[j,k] = 
            # For j==k: coefficient of x_j^2 is quad_coeffs[j*(dim+1)//2 ... ]
            # Actually, from the feature construction:
            # feature at index dim+1 + j*dim + k - j for j<=k
            # H[j,k] = H[k,j] = quad_coeffs[...], and for j==k, H[j,j] = quad_coeffs[...]
            # The quadratic term is sum_{j<=k} quad_coeff[j,k] * x_j * x_k
            # This equals 0.5 * x^T H x where H[j,k] = quad_coeff[j,k] for j!=k 
            # and H[j,j] = quad_coeff[j,j] ... wait, let me be more careful.

            # sum_{j<=k} q_{jk} x_j x_k = 0.5 x^T H x
            # where H[j,k] = q_{jk} for j!=k, H[j,j] = q_{jj}
            # Actually: x^T H x = sum_j H[j,j] x_j^2 + 2 sum_{j<k} H[j,k] x_j x_k
            # So 0.5 x^T H x = sum_j 0.5 H[j,j] x_j^2 + sum_{j<k} H[j,k] x_j x_k
            # We want: sum_{j<=k} q_{jk} x_j x_k = sum_j q_{jj} x_j^2 + sum_{j<k} q_{jk} x_j x_k
            # So: 0.5 H[j,j] = q_{jj} => H[j,j] = 2*q_{jj}
            # And: H[j,k] = q_{jk} for j<k

            H = np.zeros((dim, dim))
            idx = 0
            for j in range(dim):
                for k in range(j, dim):
                    qjk = quad_coeffs[idx]
                    if j == k:
                        H[j, j] = 2.0 * qjk
                    else:
                        H[j, k] = qjk
                        H[k, j] = qjk
                    idx += 1

            # Find minimum: solve (H) x_star = -g
            # If H is positive definite, x_star = -H^{-1} g
            try:
                # Check if H is roughly positive definite
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < 1e-8:
                    # Not positive definite, regularize
                    H_reg = H + np.eye(dim) * 1e-4
                    x_star = np.linalg.solve(H_reg, -g)
                else:
                    x_star = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Fallback: use gradient step
                if np.linalg.norm(g) < 1e-10:
                    return best_x, best_f, False
                x_star = best_x - g / (np.dot(g, g) + 1e-10)

            x_star = np.clip(x_star, -5, 5)
            f_star = func(x_star)

            if f_star < best_f - 1e-10:
                return x_star, f_star, True
            return best_x, best_f, False

        except Exception:
            return best_x, best_f, False
# EVOLVE-BLOCK-END
