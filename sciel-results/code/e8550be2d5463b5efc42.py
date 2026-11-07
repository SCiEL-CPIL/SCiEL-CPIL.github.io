# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0

        best_x = None
        best_f = np.inf
        evaluated = 0
        all_pts = []
        all_vals = []

        def _ok():
            return time.time() < deadline and evaluated < budget

        def _eval(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            evaluated += 1
            all_pts.append(x.copy())
            all_vals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def _build_quadratic_model(pts, vals, center):
            """Build a quadratic model around center using local points."""
            n = len(pts)
            if n < dim + 1:
                return None
            pts = np.array(pts)
            vals = np.array(vals)
            # Center and scale
            d = pts - center
            # Use only points within a reasonable radius
            dists = np.linalg.norm(d, axis=1)
            mask = dists < 5.0
            if mask.sum() < dim + 1:
                return None
            d = d[mask]
            v = vals[mask]
            n = len(d)
            if n < dim + 1:
                return None
            # Build design matrix for quadratic: [1, d_i, d_i*d_i, d_i_j*d_i_k]
            # For dim=5: 1 + 5 + 15 + 10 = 31 params
            # Use simpler: 1 + 5 + 10 (symmetric quadratic)
            ones = np.ones((n, 1))
            linear = d
            # Quadratic terms (symmetric): d_i*d_j for i<=j
            quad_terms = []
            for i in range(dim):
                for j in range(i, dim):
                    quad_terms.append(d[:, i] * d[:, j])
            quad_arr = np.column_stack(quad_terms) if quad_terms else np.zeros((n, 0))
            A = np.column_stack([ones, linear, quad_arr])
            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
                return coeffs
            except np.linalg.LinAlgError:
                return None

        def _quad_predict(coeffs, x, center):
            """Predict value from quadratic model."""
            d = x - center
            ones = 1.0
            lin = coeffs[1:1 + dim] @ d
            quad = 0.0
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    quad += coeffs[idx] * d[i] * d[j]
                    idx += 1
            return coeffs[0] + lin + quad

        def _extract_hessian(coeffs, center):
            """Extract Hessian approximation from quadratic coefficients."""
            H = np.zeros((dim, dim))
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    val = coeffs[idx]
                    idx += 1
                    if i == j:
                        H[i, i] += 2.0 * val
                    else:
                        H[i, j] += val
                        H[j, i] += val
            return H

        def _ridge_direction(H, fallback=None):
            """Find the ridge direction (smallest eigenvalue of Hessian)."""
            try:
                eigvals, eigvecs = np.linalg.eigh(H)
                # Ridge direction is the eigenvector with smallest eigenvalue
                idx = np.argmin(eigvals)
                return eigvecs[:, idx], eigvals
            except np.linalg.LinAlgError:
                if fallback is not None:
                    return fallback, np.array([1.0] * dim)
                return np.ones(dim) / np.sqrt(dim), np.array([1.0] * dim)

        def _constrained_line_search(x0, direction, step0, max_steps=20):
            """Line search along a direction from x0."""
            x = x0.copy()
            fx = best_f
            step = step0
            for _ in range(max_steps):
                if not _ok():
                    break
                x_try = np.clip(x + step * direction, -5.0, 5.0)
                f_try = _eval(x_try)
                if f_try < fx - 1e-10:
                    fx = f_try
                    x = x_try
                    step *= 1.5
                else:
                    x_try2 = np.clip(x - step * direction, -5.0, 5.0)
                    f_try2 = _eval(x_try2)
                    if f_try2 < fx - 1e-10:
                        fx = f_try2
                        x = x_try2
                        direction = -direction
                        step *= 1.5
                    else:
                        step *= 0.5
                        if step < 1e-8:
                            break
            return x, fx, direction

        def _model_based_search(x0, alloc, seed_off):
            """Sequential model-based optimizer with ridge identification."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(1000 + seed_off)
            x = np.array(x0, dtype=float)
            fx = best_f

            # Phase A: Build initial local model with structured probes
            probe_pts = [x.copy()]
            probe_vals = [fx]

            # Probe along each axis
            probe_radius = 0.5
            for i in range(dim):
                if not _ok():
                    break
                for sign in (1, -1):
                    x_p = x.copy()
                    x_p[i] += sign * probe_radius
                    x_p = np.clip(x_p, -5.0, 5.0)
                    f_p = _eval(x_p)
                    probe_pts.append(x_p.copy())
                    probe_vals.append(f_p)

            # Build model
            coeffs = _build_quadratic_model(probe_pts, probe_vals, x)
            if coeffs is None:
                return

            H = _extract_hessian(coeffs, x)
            ridge_dir, eigvals = _ridge_direction(H)

            # Phase B: Constrained search along ridge direction
            step = 1.0
            for iteration in range(50):
                if not _ok():
                    break
                x, fx, ridge_dir = _constrained_line_search(x, ridge_dir, step, max_steps=10)
                if not _ok():
                    break

                # Rebuild model periodically to track ridge curvature changes
                if iteration % 5 == 4:
                    # Add some new probe points around current best
                    for _ in range(dim):
                        if not _ok():
                            break
                        z = r.standard_normal(dim)
                        z -= np.dot(z, ridge_dir) * ridge_dir  # orthogonalize
                        z /= (np.linalg.norm(z) + 1e-10)
                        x_p = np.clip(x + 0.3 * z, -5.0, 5.0)
                        f_p = _eval(x_p)
                        probe_pts.append(x_p.copy())
                        probe_vals.append(f_p)

                    # Keep only recent points for model
                    if len(probe_pts) > 50:
                        probe_pts = probe_pts[-50:]
                        probe_vals = probe_vals[-50:]

                    coeffs = _build_quadratic_model(probe_pts, probe_vals, x)
                    if coeffs is not None:
                        H = _extract_hessian(coeffs, x)
                        ridge_dir, eigvals = _ridge_direction(H)

                # Adaptive step: shrink if stuck
                step = max(step * 0.9, 0.01)

            # Phase C: Full model-based refinement
            for iteration in range(30):
                if not _ok():
                    break

                # Rebuild model with all recent points
                recent_pts = probe_pts[-min(len(probe_pts), 60):]
                recent_vals = probe_vals[-min(len(probe_vals), 60):]
                coeffs = _build_quadratic_model(recent_pts, recent_vals, x)
                if coeffs is None:
                    break

                H = _extract_hessian(coeffs, x)
                grad = coeffs[1:1 + dim].copy()

                # Check if at minimum
                if np.linalg.norm(grad) < 1e-6:
                    break

                # Newton-like step with ridge constraint
                # Try full Newton step
                try:
                    # Regularize Hessian for stability
                    H_reg = H + 1e-4 * np.eye(dim)
                    newton_step = -np.linalg.solve(H_reg, grad)
                except np.linalg.LinAlgError:
                    newton_step = -grad / (np.linalg.norm(grad) + 1e-10)

                # Constrain to ridge direction if Newton step is large
                step_norm = np.linalg.norm(newton_step)
                if step_norm > 2.0:
                    # Project onto ridge direction
                    ridge_dir, _ = _ridge_direction(H)
                    proj = np.dot(newton_step, ridge_dir) * ridge_dir
                    if np.linalg.norm(proj) > 0.1:
                        newton_step = proj
                        step_norm = np.linalg.norm(newton_step)

                # Limit step size
                if step_norm > 1.0:
                    newton_step = newton_step / step_norm

                x_try = np.clip(x + newton_step, -5.0, 5.0)
                f_try = _eval(x_try)

                if f_try < fx - 1e-10:
                    fx = f_try
                    x = x_try
                    probe_pts.append(x.copy())
                    probe_vals.append(fx)
                else:
                    # Try smaller step
                    for shrink in (0.5, 0.25, 0.1):
                        if not _ok():
                            break
                        x_try2 = np.clip(x + shrink * newton_step, -5.0, 5.0)
                        f_try2 = _eval(x_try2)
                        if f_try2 < fx - 1e-10:
                            fx = f_try2
                            x = x_try2
                            probe_pts.append(x.copy())
                            probe_vals.append(fx)
                            break

        def _nelder_mead(x0, alloc, seed_off):
            """Nelder-Mead simplex method for robust local optimization."""
            nonlocal evaluated, best_x, best_f
            if not _ok():
                return

            r = np.random.default_rng(2000 + seed_off)
            # Initialize simplex
            simplex = [x0.copy()]
            vals = [best_f]
            for i in range(dim):
                if not _ok():
                    break
                xi = x0.copy()
                xi[i] += 0.1
                xi = np.clip(xi, -5.0, 5.0)
                fi = _eval(xi)
                simplex.append(xi)
                vals.append(fi)

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = alloc

            for _ in range(max_iter):
                if not _ok():
                    break

                # Sort
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = _eval(x_r)

                if f_r < vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = _eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        vals[-1] = f_r
                elif f_r < vals[-2]:
                    simplex[-1] = x_r
                    vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = _eval(x_c)
                    if f_c < vals[-1]:
                        simplex[-1] = x_c
                        vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            vals[i] = _eval(simplex[i])

                # Check convergence
                if max(vals) - min(vals) < 1e-10:
                    break

        # === Main Strategy ===
        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration to find promising regions
        n_starts = 3
        for i in range(n_starts):
            if not _ok():
                break
            start = rng.uniform(-4.0, 4.0, size=dim)
            f_start = _eval(start)
            # Quick Nelder-Mead to find local basin
            _nelder_mead(start, 30, seed_off=i * 50)

        if best_x is None:
            return None, np.inf

        # Phase 2: Model-based ridge search from best point
        if _ok():
            _model_based_search(best_x, budget // 2, seed_off=0)

        # Phase 3: Nelder-Mead polish
        if best_x is not None and _ok():
            _nelder_mead(best_x, 40, seed_off=999)

        # Phase 4: Final model-based refinement
        if best_x is not None and _ok():
            _model_based_search(best_x, budget // 4, seed_off=100)

        # Phase 5: Final polish
        if best_x is not None and _ok():
            _nelder_mead(best_x, 20, seed_off=777)

        return best_x, best_f
# EVOLVE-BLOCK-END
