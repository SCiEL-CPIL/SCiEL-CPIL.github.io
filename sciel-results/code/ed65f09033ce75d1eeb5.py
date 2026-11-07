"""Optimizer combining CMA-ES with ridge-aware local refinement for BBOB f13 (Sharp Ridge).

Strategy:
1. Phase 1: Broad random sampling to locate the basin of attraction.
2. Phase 2: Multi-start CMA-ES to exploit the ridge structure via covariance adaptation.
3. Phase 3: Ridge-aware local refinement using a quadratic surrogate to fine-tune
   the solution, separating ridge and transverse directions.

This approach leverages CMA-ES's ability to adapt to anisotropic landscapes
and refines the result with explicit ridge awareness.
"""
import time
import numpy as np


class Optimizer:
    """
    Ridge-aware optimizer for Sharp Ridge functions using CMA-ES and surrogate refinement.
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
        n_phase1 = min(50, budget)
        for _ in range(n_phase1):
            if evaluated >= budget or not elapsed_ok():
                break
            x_candidate = rng.uniform(-4.5, 4.5, size=dim)
            eval_point(x_candidate)

        if best_x is None or not elapsed_ok():
            return best_x, best_f

        # CMA-ES implementation
        def cma_es(start_x, budget_alloc, seed_offset=0):
            nonlocal evaluated, best_x, best_f, deadline, elapsed_ok

            if evaluated >= budget or not elapsed_ok():
                return

            rng_local = np.random.default_rng(42 + seed_offset)

            # CMA-ES parameters
            pop_size = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # ~13 for dim=5
            sigma = 1.0
            mean = start_x.copy().astype(float)
            C = np.eye(dim)
            C_inv = np.eye(dim)

            # Adaptation rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Initial evaluations
            for _ in range(min(pop_size, budget_alloc)):
                if evaluated >= budget or not elapsed_ok():
                    break
                x_candidate = clip(mean + sigma * rng_local.standard_normal(dim))
                f_val = eval_point(x_candidate)

            # Main loop
            while evaluated < budget and elapsed_ok():
                if evaluated + pop_size > budget:
                    break

                # Generate population
                xs = []
                for _ in range(pop_size):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    # Sample from N(mean, sigma^2 * C)
                    z = rng_local.standard_normal(dim)
                    # Cholesky decomposition for sampling
                    try:
                        L = np.linalg.cholesky(C)
                        x_new = clip(mean + sigma * L @ z)
                    except np.linalg.LinAlgError:
                        x_new = clip(mean + sigma * z)
                    f_val = eval_point(x_new)
                    xs.append((x_new, f_val))

                if len(xs) < pop_size:
                    break

                # Sort by fitness
                xs.sort(key=lambda x: x[1])

                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * xs[i][0]

                # Update sigma
                old_mean = mean.copy()
                mean = new_mean

                # Compute direction for sigma adaptation
                try:
                    C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(dim)

                diff = (mean - old_mean) / (sigma * np.sqrt(mu_eff))
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * diff
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)
                if h_sigma < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1):
                    sigma *= np.exp(c_sigma / dim * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(dim - 1)))

                # Update covariance matrix
                diff_c = (mean - old_mean) / sigma
                p_c = (1 - c_c) * p_c + c_c * diff_c

                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2)) / \
                      ((1 + 2 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1))

                rank_one = np.outer(p_c, p_c)
                if h_c < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1):
                    rank_one += (1 - (1 - c_c)**2) * (c_c / (1 - c_c)) * np.eye(dim)

                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (xs[i][0] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    sigma = max(sigma * 0.5, 0.01)

            # After CMA-ES, do local refinement with smaller steps
            if evaluated < budget and elapsed_ok():
                local_sigma = sigma * 0.1
                for _ in range(min(20, budget - evaluated)):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    x_candidate = clip(mean + local_sigma * rng_local.standard_normal(dim))
                    f_val = eval_point(x_candidate)

        # Multi-start strategy
        n_starts = 2
        budget_per_start = (budget - n_phase1) // n_starts

        # First start: near the best found so far
        if best_x is not None:
            start1 = best_x + rng.normal(0, 0.5, size=dim)
        else:
            start1 = rng.uniform(-3.0, 3.0, size=dim)
        cma_es(start1, budget_per_start, seed_offset=0)

        # Second start: use remaining budget
        remaining = budget - evaluated
        if remaining > 0 and elapsed_ok():
            if best_x is not None:
                start2 = best_x + rng.normal(0, 0.3, size=dim)
            else:
                start2 = rng.uniform(-2.0, 2.0, size=dim)
            cma_es(start2, remaining, seed_offset=100)

        # Phase 3: Ridge-aware local refinement using quadratic surrogate
        if evaluated < budget and elapsed_ok() and best_x is not None:
            polish_budget = min(30, budget - evaluated)

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
            stagnant = 0
            best_f_phase3 = best_f

            for iteration in range(polish_budget):
                if evaluated >= budget or not elapsed_ok():
                    break

                # Build model from best points
                g, H = build_quadratic_model(all_points, best_x, n_points=20)

                if g is None:
                    # Fallback: random perturbation
                    x_candidate = best_x + rng.normal(0, 0.1, size=dim)
                    f_val = eval_point(x_candidate)
                    if f_val < best_f_phase3:
                        best_f_phase3 = f_val
                        stagnant = 0
                    else:
                        stagnant += 1
                    continue

                # Compute ridge-aware step
                step = ridge_aware_step(best_x, g, H, step_scale=0.5)
                x_candidate = clip(best_x + step)
                f_val = eval_point(x_candidate)

                if f_val < best_f_phase3:
                    best_f_phase3 = f_val
                    stagnant = 0
                else:
                    stagnant += 1

                # If stagnant for too long, try a larger random jump
                if stagnant > 5:
                    stagnant = 0
                    x_candidate = best_x + rng.normal(0, 0.5, size=dim)
                    f_val = eval_point(x_candidate)

        return best_x, best_f
