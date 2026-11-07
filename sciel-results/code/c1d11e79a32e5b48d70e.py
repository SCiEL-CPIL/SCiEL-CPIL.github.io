"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS global exploration + BFGS local refinement for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        def bfgs_local(x0, f0, max_evals=80):
            """Accelerated BFGS with finite-difference gradients and line search."""
            nonlocal best_x, best_f, total_calls

            n = dim
            x = x0.copy()
            f = f0

            # BFGS state
            H = np.eye(n)  # inverse Hessian approximation
            step_h = 1e-4  # finite difference step

            def compute_grad(x_cur):
                grad = np.zeros(n)
                for i in range(n):
                    xp = x_cur.copy()
                    xp[i] += step_h
                    xp = np.clip(xp, -5, 5)
                    fp = safe_call(xp)
                    xm = x_cur.copy()
                    xm[i] -= step_h
                    xm = np.clip(xm, -5, 5)
                    fm = safe_call(xm)
                    if fp is None or fm is None:
                        return None
                    grad[i] = (fp - fm) / (2 * step_h)
                return grad

            g = compute_grad(x)
            if g is None:
                return x, f

            # Line search (backtracking)
            def line_search(x_cur, g_cur, d, f_cur, max_ls_evals=10):
                alpha = 1.0
                c1 = 1e-4
                c2 = 0.9
                for _ in range(max_ls_evals):
                    if total_calls >= budget or time.time() > deadline:
                        return x_cur, f_cur, 0.0
                    x_new = x_cur + alpha * d
                    x_new = np.clip(x_new, -5, 5)
                    f_new = safe_call(x_new)
                    if f_new is None:
                        return x_cur, f_cur, 0.0
                    # Armijo condition
                    if f_new <= f_cur + c1 * alpha * np.dot(g_cur, d):
                        # Check curvature
                        g_new = compute_grad(x_new)
                        if g_new is not None and np.dot(g_new, d) <= c2 * np.dot(g_cur, d):
                            return x_new, f_new, alpha
                    alpha *= 0.5
                return x_cur, f_cur, 0.0

            for _ in range(max_evals):
                if time.time() > deadline or total_calls >= budget:
                    break

                # Check convergence
                if np.linalg.norm(g) < 1e-8:
                    break

                # Search direction
                d = -H @ g

                # Line search
                x_new, f_new, alpha = line_search(x, g, d, f)
                if alpha == 0.0:
                    # Line search failed, try steepest descent
                    d = -g
                    x_new, f_new, alpha = line_search(x, g, d, f)
                    if alpha == 0.0:
                        break

                # Update BFGS
                s = x_new - x
                g_new = compute_grad(x_new)
                if g_new is None:
                    break
                y = g_new - g
                sy = np.dot(s, y)
                if sy > 1e-10:
                    rho = 1.0 / sy
                    H = (np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * rho * sy * np.outer(s, s)
                    # Symmetrize
                    H = (H + H.T) / 2

                x = x_new
                f = f_new
                g = g_new

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            return x, f

        # Generate LHS (Latin Hypercube) stratified samples
        def generate_lhs(n_samples, dim, seed=42):
            rng = np.random.default_rng(seed)
            lhs = np.zeros((n_samples, dim))
            for i in range(n_samples):
                for j in range(dim):
                    # Stratified: divide [0,1] into n_samples strata, pick one point per stratum
                    stratum = (i + rng.random()) / n_samples
                    lhs[i, j] = stratum
            # Shuffle rows to decorrelate
            perm = rng.permutation(n_samples)
            lhs = lhs[perm]
            # Map from [0,1] to [-5, 5]
            lhs = lhs * 10.0 - 5.0
            return lhs

        # Phase 1: Global exploration with LHS grid
        n_global = 60  # number of global samples
        lhs_points = generate_lhs(n_global, dim)

        global_results = []
        for i in range(n_global):
            if time.time() > deadline or total_calls >= budget:
                break
            x = lhs_points[i]
            x = np.clip(x, -5, 5)
            f = safe_call(x)
            if f is not None:
                global_results.append((f, x.copy()))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # Sort global results by fitness
        global_results.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with BFGS from best global points
        # Select top candidates for local refinement
        n_local = min(len(global_results), 12)
        local_budget_per = max(5, (budget - total_calls) // max(n_local, 1))

        for i in range(n_local):
            if time.time() > deadline or total_calls >= budget:
                break
            f_init, x_init = global_results[i]
            remaining = budget - total_calls
            if remaining < 10:
                break
            bfgs_budget = min(local_budget_per, remaining)
            x_refined, f_refined = bfgs_local(x_init, f_init, max_evals=bfgs_budget)

            # If BFGS didn't improve, try a small perturbation restart
            if f_refined >= f_init - 1e-10 and time.time() < deadline and total_calls < budget:
                # Try a few small perturbations
                rng = np.random.default_rng(42 + i)
                for _ in range(3):
                    if time.time() > deadline or total_calls >= budget:
                        break
                    perturbation = rng.normal(0, 0.5, dim)
                    x_pert = np.clip(x_init + perturbation, -5, 5)
                    f_pert = safe_call(x_pert)
                    if f_pert is not None and f_pert < f_init:
                        x_refined2, f_refined2 = bfgs_local(x_pert, f_pert, max_evals=min(30, budget - total_calls))
                        if f_refined2 < f_refined:
                            x_refined, f_refined = x_refined2, f_refined2

        # Phase 3: Final fine-tuning with coordinate-wise search
        if best_x is not None and time.time() < deadline and total_calls < budget:
            for coord in range(dim):
                if time.time() > deadline or total_calls >= budget:
                    break
                # Try small steps in each direction
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    if time.time() > deadline or total_calls >= budget:
                        break
                    x_try = best_x.copy()
                    x_try[coord] += delta
                    x_try = np.clip(x_try, -5, 5)
                    f_try = safe_call(x_try)
                    if f_try is not None and f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
