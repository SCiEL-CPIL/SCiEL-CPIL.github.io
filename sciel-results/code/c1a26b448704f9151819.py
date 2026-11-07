"""Optimizer using ridge-aware surrogate modeling for BBOB f13 (Sharp Ridge).

Strategy:
1. Phase 1: Broad random sampling to get initial landscape information.
2. Phase 2: Build a quadratic surrogate model from the best evaluations.
   - Estimate the Hessian via finite differences of the surrogate.
   - Identify the ridge direction (smallest eigenvalue of Hessian).
   - Take Newton steps in the high-curvature directions, with damped steps along the ridge.
3. Phase 3: Local refinement with small random perturbations around the best point.

This approach explicitly handles the sharp ridge structure by separating
the ridge direction from the transverse directions.
"""
import time
import numpy as np


class Optimizer:
    """
    Ridge-aware optimizer for Sharp Ridge functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0
        all_points = []  # list of (x, f)

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            all_points.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Broad random sampling
        n_phase1 = min(80, budget)
        for _ in range(n_phase1):
            if evaluated >= budget or not elapsed_ok():
                break
            x_candidate = rng.uniform(-4.5, 4.5, size=dim)
            eval_point(x_candidate)

        if best_x is None or not elapsed_ok():
            return best_x, best_f

        # Phase 2: Ridge-aware optimization using surrogate model
        # We'll use the best points to build a local quadratic model
        # and identify the ridge direction.

        def build_quadratic_model(points, center, n_points=20):
            """Build a quadratic model from the best points near center."""
            # Select points closest to center with good fitness
            dists = [np.linalg.norm(p[0] - center) for p in points]
            sorted_idx = np.argsort(dists)
            selected = [points[i] for i in sorted_idx[:n_points]]

            if len(selected) < dim + 1:
                return None, None

            X = np.array([p[0] for p in selected])
            f = np.array([p[1] for p in selected])

            # Fit quadratic: f(x) ≈ f0 + g^T(x-c) + 0.5*(x-c)^T H (x-c)
            # Using least squares
            n = len(X)
            # Build feature matrix for quadratic terms
            # Features: 1, (x-c)_i, (x-c)_i*(x-c)_j for i<=j
            diff = X - center
            n_lin = dim
            n_quad = dim * (dim + 1) // 2
            n_feat = 1 + n_lin + n_quad

            A = np.zeros((n, n_feat))
            A[:, 0] = 1.0
            A[:, 1:1 + n_lin] = diff
            # Quadratic terms
            quad_idx = 1 + n_lin
            for i in range(dim):
                for j in range(i, dim):
                    A[:, quad_idx] = diff[:, i] * diff[:, j]
                    quad_idx += 1

            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
            except np.linalg.LinAlgError:
                return None, None

            f0 = coeffs[0]
            g = coeffs[1:1 + n_lin]
            # Reconstruct Hessian from quadratic coefficients
            H = np.zeros((dim, dim))
            quad_idx = 1 + n_lin
            for i in range(dim):
                for j in range(i, dim):
                    val = coeffs[quad_idx]
                    quad_idx += 1
                    H[i, j] = val
                    if i != j:
                        H[j, i] = val
            # The quadratic term is 0.5 * x^T H x, but we fit x_i*x_j directly
            # So H here is actually 2*H_true for off-diagonal, and H for diagonal
            # Actually, our fit is: sum_{i<=j} c_ij * x_i * x_j
            # This equals 0.5 * x^T H x where H_ij = c_ij + c_ji for i≠j, H_ii = c_ii
            # Wait, let me reconsider. The quadratic form is:
            # 0.5 * sum_i H_ii x_i^2 + sum_{i<j} H_ij x_i x_j
            # Our fit has: sum_i c_ii x_i^2 + sum_{i<j} c_ij x_i x_j
            # So H_ii = 2*c_ii, H_ij = c_ij for i≠j
            H2 = np.zeros((dim, dim))
            quad_idx = 1 + n_lin
            for i in range(dim):
                for j in range(i, dim):
                    val = coeffs[quad_idx]
                    quad_idx += 1
                    H2[i, j] = val
                    H2[j, i] = val
            # H_ii should be 2*val, H_ij should be val
            for i in range(dim):
                H2[i, i] = 2.0 * H2[i, i]

            return g, H2

        def ridge_aware_step(center, g, H, step_scale=1.0):
            """Compute a step that moves fast in high-curvature directions and slow along the ridge."""
            # Regularize Hessian to ensure positive definiteness
            H_reg = H + 1e-4 * np.eye(dim)

            try:
                eigvals, eigvecs = np.linalg.eigh(H_reg)
            except np.linalg.LinAlgError:
                # Fallback: simple gradient step
                return -g * step_scale

            # Sort eigenvalues
            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # The ridge direction is the one with the smallest eigenvalue
            # We want to move slowly along the ridge and fast in transverse directions
            step = np.zeros(dim)
            for i in range(dim):
                # Damping: smaller eigenvalue -> larger step (since we want to move against the gradient)
                # Newton step would be -H^{-1} g, so component is -g_i / eigval_i
                # But for the ridge (small eigval), this would be a huge step, which we don't want
                # Instead, we damp the step along the ridge direction
                if eigvals[i] < 1e-3:
                    # Ridge direction: use a damped gradient step
                    step += eigvecs[:, i] * (-g @ eigvecs[:, i]) * step_scale * 0.1
                else:
                    # Transverse direction: use Newton-like step
                    step += eigvecs[:, i] * (-g @ eigvecs[:, i]) / eigvals[i] * step_scale

            # Clip the step to avoid going out of bounds
            max_step = 2.0
            norm_step = np.linalg.norm(step)
            if norm_step > max_step:
                step = step * max_step / norm_step

            return step

        # Main ridge-aware optimization loop
        n_phase2 = budget - n_phase1
        stagnant = 0
        best_f_phase2 = best_f

        for iteration in range(min(100, n_phase2)):
            if evaluated >= budget or not elapsed_ok():
                break

            # Build model from best points
            g, H = build_quadratic_model(all_points, best_x, n_points=20)

            if g is None:
                # Fallback: random perturbation
                x_candidate = best_x + rng.normal(0, 0.5, size=dim)
                f_val = eval_point(x_candidate)
                if f_val < best_f_phase2:
                    best_f_phase2 = f_val
                    stagnant = 0
                else:
                    stagnant += 1
                continue

            # Compute ridge-aware step
            step = ridge_aware_step(best_x, g, H, step_scale=1.0)
            x_candidate = clip(best_x + step)
            f_val = eval_point(x_candidate)

            if f_val < best_f_phase2:
                best_f_phase2 = f_val
                stagnant = 0
            else:
                stagnant += 1

            # If stagnant for too long, try a larger random jump
            if stagnant > 10:
                stagnant = 0
                x_candidate = best_x + rng.normal(0, 1.0, size=dim)
                f_val = eval_point(x_candidate)

        # Phase 3: Local refinement
        if evaluated < budget and elapsed_ok() and best_x is not None:
            polish_budget = min(50, budget - evaluated)
            # Use very small steps
            for _ in range(polish_budget):
                if evaluated >= budget or not elapsed_ok():
                    break
                # Adaptive step size based on recent improvements
                if len(all_points) >= 2:
                    recent = all_points[-10:]
                    f_range = max(p[1] for p in recent) - min(p[1] for p in recent)
                    step_size = max(0.01, min(0.1, f_range * 0.01))
                else:
                    step_size = 0.05
                x_candidate = best_x + rng.normal(0, step_size, size=dim)
                eval_point(x_candidate)

        return best_x, best_f
