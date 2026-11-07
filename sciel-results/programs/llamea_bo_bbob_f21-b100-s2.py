"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Halton exploration + quadratic model local search for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # --- Halton sequence generator (deterministic, space-filling) ---
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_seq(n, dim, bases):
            """Generate n points in [0,1)^dim using Halton sequence."""
            pts = []
            for i in range(1, n + 1):
                pt = np.array([halton(i, bases[d]) for d in range(dim)])
                pts.append(pt)
            return np.array(pts)

        # Map [0,1)^dim to [-5,5]^dim
        def to_box(pt):
            return pt * 10.0 - 5.0

        # --- Phase 1: Halton-based global exploration ---
        # Use first 80 Halton points for good space-filling coverage
        n_halton = 80
        bases = [2, 3, 5, 7, 11]  # first dim primes
        halton_pts_unit = halton_seq(n_halton, dim, bases)
        halton_pts = np.array([to_box(p) for p in halton_pts_unit])

        # Also include origin and a few strategic points
        extra_pts = [np.zeros(dim)]
        # Add midpoints of axes
        for d in range(dim):
            for s in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = s * 2.5
                extra_pts.append(pt)

        all_explore_pts = list(halton_pts) + extra_pts

        best_x = None
        best_f = np.inf
        eval_points = []  # (x, f) for model fitting

        for pt in all_explore_pts:
            if calls >= budget or time_left() < 1.0:
                break
            pt = np.clip(pt, -5, 5)
            f = eval_point(pt)
            eval_points.append((pt.copy(), f))
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_point(best_x)
            eval_points.append((best_x.copy(), best_f))

        # --- Phase 2: Sequential Quadratic Approximation local search ---
        # Fit quadratic models to recent evaluations and take steps toward predicted minima
        # Use the most recent M points for the model (adaptive window)

        M = 12  # number of points for quadratic model
        max_sqp_iters = 30
        sqp_step_scale = 1.0

        # Collect enough points for model fitting
        # If we have fewer than M points, use all
        for sqp_iter in range(max_sqp_iters):
            if calls >= budget - 5 or time_left() < 2.0:
                break

            # Use most recent M evaluated points for the quadratic model
            recent = eval_points[-M:]
            if len(recent) < dim + 1:
                # Not enough points for a meaningful model; do a local random-ish search
                # Use deterministic perturbation from best_x
                for d in range(dim):
                    for s in [-1, 1]:
                        if calls >= budget or time_left() < 1.0:
                            break
                        pt = best_x.copy()
                        pt[d] += s * 0.3 * sqp_step_scale
                        pt = np.clip(pt, -5, 5)
                        # Check if already evaluated
                        already = False
                        for ep, ef in eval_points[-20:]:
                            if np.allclose(ep, pt, atol=1e-8):
                                already = True
                                break
                        if not already:
                            f = eval_point(pt)
                            eval_points.append((pt.copy(), f))
                            if f < best_f:
                                best_f = f
                                best_x = pt.copy()
                continue

            X = np.array([p[0] for p in recent])
            F = np.array([p[1] for p in recent])

            # Fit quadratic model: f(x) ≈ c + g^T x + 0.5 * x^T H x
            # Design matrix for quadratic in dim dimensions
            n_params = 1 + dim + dim * (dim + 1) // 2
            if len(X) < n_params:
                # Not enough data; do local search
                for d in range(dim):
                    for s in [-1, 1]:
                        if calls >= budget or time_left() < 1.0:
                            break
                        pt = best_x.copy()
                        pt[d] += s * 0.2 * sqp_step_scale
                        pt = np.clip(pt, -5, 5)
                        already = False
                        for ep, ef in eval_points[-20:]:
                            if np.allclose(ep, pt, atol=1e-8):
                                already = True
                                break
                        if not already:
                            f = eval_point(pt)
                            eval_points.append((pt.copy(), f))
                            if f < best_f:
                                best_f = f
                                best_x = pt.copy()
                continue

            # Build design matrix
            A = np.ones((len(X), n_params))
            # Linear terms
            A[:, 1:1 + dim] = X
            # Quadratic terms
            col = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    A[:, col] = X[:, i] * X[:, j]
                    col += 1

            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(A, F, rcond=None)
            except np.linalg.LinAlgError:
                continue

            # Compute gradient and Hessian from coefficients
            g = coeffs[1:1 + dim].copy()
            H = np.zeros((dim, dim))
            col = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        H[i, i] = coeffs[col]
                    else:
                        H[i, j] = coeffs[col] / 2.0
                        H[j, i] = coeffs[col] / 2.0
                    col += 1

            # Make H positive semi-definite for stable step
            # Add small ridge if needed
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) < 1e-6:
                H = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T

            # Solve H * dx = -g
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                continue

            # Limit step size
            step_norm = np.linalg.norm(dx)
            if step_norm > sqp_step_scale:
                dx = dx * (sqp_step_scale / step_norm)

            # Try the proposed step
            x_new = best_x + dx
            x_new = np.clip(x_new, -5, 5)

            # Check if already evaluated
            already = False
            for ep, ef in eval_points[-30:]:
                if np.allclose(ep, x_new, atol=1e-8):
                    already = True
                    break

            if already:
                # Try a smaller step
                x_new = best_x + 0.5 * dx
                x_new = np.clip(x_new, -5, 5)
                already = False
                for ep, ef in eval_points[-30:]:
                    if np.allclose(ep, x_new, atol=1e-8):
                        already = True
                        break

            if already:
                continue

            if calls >= budget or time_left() < 1.0:
                break

            f_new = eval_point(x_new)
            eval_points.append((x_new.copy(), f_new))

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                # Successful step: maintain or slightly increase step scale
                sqp_step_scale = min(1.5, sqp_step_scale * 1.1)
            else:
                # Failed step: reduce step scale
                sqp_step_scale *= 0.7
                if sqp_step_scale < 0.01:
                    break

        # --- Phase 3: Final refinement with coordinate descent on best_x ---
        if calls < budget and time_left() > 2.0:
            # Fine coordinate descent from best_x
            for step_size in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
                if calls >= budget or time_left() < 1.0:
                    break
                improved = False
                for _pass in range(2):
                    if calls >= budget or time_left() < 1.0:
                        break
                    for d in range(dim):
                        if calls >= budget or time_left() < 0.5:
                            break
                        for s in [-1, 1]:
                            if calls >= budget or time_left() < 0.5:
                                break
                            pt = best_x.copy()
                            pt[d] += s * step_size
                            pt = np.clip(pt, -5, 5)
                            already = False
                            for ep, ef in eval_points[-30:]:
                                if np.allclose(ep, pt, atol=1e-8):
                                    already = True
                                    break
                            if already:
                                continue
                            f = eval_point(pt)
                            eval_points.append((pt.copy(), f))
                            if f < best_f:
                                best_f = f
                                best_x = pt.copy()
                                improved = True
                    if not improved:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
