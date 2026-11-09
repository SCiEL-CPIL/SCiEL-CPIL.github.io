import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Halton sequence generator for quasi-random global sampling
        def halton_sequence(n, dim, base_start=2):
            """Generate n Halton sequence points in [0,1]^dim"""
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    base = base_start + j
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i - 1, j] = r
            return points

        # Phase 1: Global exploration with Halton sequence
        n_global = 40
        halton_pts = halton_sequence(n_global, dim)
        # Map from [0,1]^dim to [-4, 4]^dim (minimum guaranteed in [-4,4]^5)
        halton_pts = halton_pts * 8.0 - 4.0

        # Add the origin and a few structured points
        candidates = [np.zeros(dim)]
        # Add corners of [-4,4]^5 scaled down
        for _ in range(4):
            candidates.append(rng_uniform := np.random.default_rng(42 + _).uniform(-4, 4, dim))

        for p in candidates + list(halton_pts):
            if calls >= budget - 100 or time.time() > deadline:
                break
            safe_func(p)

        if not np.isfinite(best_f):
            return best_x, best_f

        # Phase 2: BFGS-based local optimization
        # Initialize BFGS inverse Hessian approximation H = I
        x_current = best_x.copy()
        f_current = best_f
        H = np.eye(dim)  # Inverse Hessian approximation
        step_count = 0

        # BFGS parameters
        trust_radius = 1.0
        min_trust_radius = 1e-4
        max_trust_radius = 5.0
        rho_min = 1e-8  # Minimum relative improvement
        max_bfgs_steps = 200
        bfgs_restart_count = 0

        for bfgs_iter in range(max_bfgs_steps):
            if calls >= budget - 10 or time.time() > deadline:
                break

            # Check if H is ill-conditioned or trust region too small
            eigvals = np.linalg.eigvalsh(H)
            cond = eigvals.max() / max(eigvals.min(), 1e-15)
            if cond > 1e10 or trust_radius < min_trust_radius:
                # Restart BFGS from best point
                x_current = best_x.copy()
                f_current = best_f
                H = np.eye(dim)
                trust_radius = 1.0
                bfgs_restart_count += 1
                if bfgs_restart_count > 5:
                    break
                continue

            # Compute search direction: d = -H * grad (approximate gradient via finite differences)
            # Use central differences for gradient estimation
            h_fd = 1e-5
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x_current.copy()
                x_plus[i] += h_fd
                x_minus = x_current.copy()
                x_minus[i] -= h_fd
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                f_plus = safe_func(x_plus)
                f_minus = safe_func(x_minus)
                if np.isfinite(f_plus) and np.isfinite(f_minus):
                    grad[i] = (f_plus - f_minus) / (2.0 * h_fd)
                elif np.isfinite(f_plus):
                    grad[i] = (f_plus - f_current) / h_fd
                elif np.isfinite(f_minus):
                    grad[i] = (f_current - f_minus) / h_fd

            if not np.all(np.isfinite(grad)):
                break

            # Check convergence
            if np.linalg.norm(grad, np.inf) < 1e-8:
                break

            # Compute search direction
            d = -H @ grad

            # Normalize and scale by trust radius
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-12:
                d = -grad
                d_norm = np.linalg.norm(d)
            if d_norm > trust_radius:
                d = d * (trust_radius / d_norm)

            # Line search with backtracking
            alpha = 1.0
            x_new = x_current + d
            f_new = None
            found_decrease = False

            for ls_iter in range(10):
                x_try = x_current + alpha * d
                x_try = np.clip(x_try, -5.0, 5.0)
                f_try = safe_func(x_try)
                if np.isfinite(f_try) and f_try < f_current - rho_min * alpha * np.linalg.norm(d) * np.linalg.norm(grad):
                    x_new = x_try
                    f_new = f_try
                    found_decrease = True
                    break
                alpha *= 0.5

            if not found_decrease:
                # No sufficient decrease found; shrink trust region
                trust_radius *= 0.5
                if trust_radius < min_trust_radius:
                    # Try a random perturbation to escape
                    x_pert = x_current + np.random.default_rng(bfgs_iter).normal(0, 0.1, dim)
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    f_pert = safe_func(x_pert)
                    if np.isfinite(f_pert) and f_pert < best_f:
                        x_current = x_pert
                        f_current = f_pert
                        H = np.eye(dim)
                        trust_radius = 0.5
                    else:
                        break
                continue

            # Update BFGS inverse Hessian
            s = x_new - x_current
            y = np.zeros(dim)
            # Recompute gradient at x_new for BFGS update
            for i in range(dim):
                x_plus = x_new.copy()
                x_plus[i] += h_fd
                x_minus = x_new.copy()
                x_minus[i] -= h_fd
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                f_plus = safe_func(x_plus)
                f_minus = safe_func(x_minus)
                if np.isfinite(f_plus) and np.isfinite(f_minus):
                    y[i] = (f_plus - f_minus) / (2.0 * h_fd)
                elif np.isfinite(f_plus):
                    y[i] = (f_plus - f_new) / h_fd
                elif np.isfinite(f_minus):
                    y[i] = (f_new - f_minus) / h_fd

            sy = s @ y
            if sy > 1e-10:
                # BFGS update for inverse Hessian
                Hy = H @ y
                rho = 1.0 / sy
                H = H + rho * (s @ s.T + (H @ y @ Hy.T) * rho - rho * (s @ Hy.T + Hy @ s.T))
                # Symmetrize
                H = (H + H.T) / 2.0

            # Update trust region
            if f_new < f_current - 0.5 * alpha * np.linalg.norm(d) * np.linalg.norm(grad):
                # Successful step, expand trust region
                trust_radius = min(trust_radius * 2.0, max_trust_radius)
            else:
                # Marginal step, keep trust region
                pass

            x_current = x_new
            f_current = f_new
            step_count += 1

            # Check for convergence
            if np.linalg.norm(s) < 1e-7:
                break

        # Phase 3: Final refinement with small coordinate steps
        remaining = budget - calls
        if remaining > 5:
            # Try small perturbations around best point
            rng_final = np.random.default_rng(999)
            for _ in range(min(remaining - 5, 30)):
                if calls >= budget - 2 or time.time() > deadline:
                    break
                # Small random perturbation
                scale = 0.01 * (1 + 0.1 * _)
                x_pert = best_x + rng_final.normal(0, scale, dim)
                x_pert = np.clip(x_pert, -5.0, 5.0)
                safe_func(x_pert)

        return best_x, best_f
