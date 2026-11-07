"""Derivative-free Trust-Region BFGS with multi-start for BBOB f14."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            count_call()
            return func(np.clip(x, -5, 5))

        # Multi-start with deterministic seeds for reproducibility
        rng = np.random.default_rng(42)
        n_starts = min(5, budget // 100)  # at most 5 starts, at least 100 budget each
        if budget < 100:
            n_starts = 1

        # First, do a coarse grid / random sampling to find a good starting point
        # Use a small number of random points to locate the basin
        sample_budget = min(20, budget // 10)
        candidates = []
        for i in range(sample_budget):
            x = rng.uniform(-5, 5, dim)
            f = safe_func(x)
            if f is not None:
                candidates.append((f, x))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            if check_time():
                break

        if not candidates:
            return best_x, best_f if best_x is not None else (np.zeros(dim), 0.0)

        # Sort candidates and pick top ones for local refinement
        candidates.sort(key=lambda t: t[0])
        top_starts = [c[1] for c in candidates[:min(n_starts, len(candidates))]]

        # For each start, run DF-TBFGS
        for start_idx, x0 in enumerate(top_starts):
            if check_time() or calls_made >= budget:
                break

            # Allocate remaining budget for this local optimization
            remaining = budget - calls_made
            if remaining < 10:
                break

            # DF-TBFGS parameters
            eps = 1e-5  # finite difference step
            max_iter = min(remaining, 200)  # max iterations for this local search

            # BFGS Hessian inverse approximation
            H = np.eye(dim)
            x = x0.copy()
            fx = safe_func(x)
            if fx is None:
                continue
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Compute initial gradient using forward differences
            grad = np.zeros(dim)
            for j in range(dim):
                x_plus = x.copy()
                x_plus[j] += eps
                fp = safe_func(x_plus)
                if fp is None:
                    break
                grad[j] = (fp - fx) / eps
            else:
                # Run BFGS iterations
                for it in range(max_iter):
                    if check_time() or calls_made >= budget:
                        break

                    # Check convergence
                    if np.linalg.norm(grad) < 1e-8:
                        break

                    # Trust region radius
                    r = min(5.0, max(0.01, 0.5 * np.linalg.norm(grad)))

                    # Solve (H^{-1} grad) for direction
                    d = H.dot(grad)
                    dn = np.linalg.norm(d)
                    if dn < 1e-12:
                        break
                    d = -d / dn * r

                    # Try step
                    x_new = x + d
                    f_new = safe_func(x_new)
                    if f_new is None:
                        break

                    if f_new < fx:
                        # Successful step
                        s = x_new - x
                        ys = f_new - fx
                        # Update BFGS
                        Sy = s.dot(ys)
                        if Sy > 1e-10:
                            Hs = H.dot(s)
                            H = H - (np.outer(Hs, s) + np.outer(s, Hs)) / Sy + \
                                (np.outer(s, s) * (Sy + s.dot(H.dot(s)))) / (Sy * Sy)
                            # Ensure H is symmetric and positive definite-ish
                            H = 0.5 * (H + H.T)
                        x, fx = x_new, f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        # Recompute gradient
                        grad_new = np.zeros(dim)
                        for j in range(dim):
                            x_plus = x.copy()
                            x_plus[j] += eps
                            fp = safe_func(x_plus)
                            if fp is None:
                                break
                            grad_new[j] = (fp - fx) / eps
                        else:
                            grad = grad_new
                    else:
                        # Unsuccessful step, shrink trust region
                        # Simple backtracking
                        for alpha in [0.5, 0.25, 0.1]:
                            x_try = x + alpha * d
                            f_try = safe_func(x_try)
                            if f_try is None:
                                break
                            if f_try < fx:
                                s = x_try - x
                                ys = f_try - fx
                                Sy = s.dot(ys)
                                if Sy > 1e-10:
                                    Hs = H.dot(s)
                                    H = H - (np.outer(Hs, s) + np.outer(s, Hs)) / Sy + \
                                        (np.outer(s, s) * (Sy + s.dot(H.dot(s)))) / (Sy * Sy)
                                    H = 0.5 * (H + H.T)
                                x, fx = x_try, f_try
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                # Recompute gradient
                                grad_new = np.zeros(dim)
                                for j in range(dim):
                                    x_plus = x.copy()
                                    x_plus[j] += eps
                                    fp = safe_func(x_plus)
                                    if fp is None:
                                        break
                                    grad_new[j] = (fp - fx) / eps
                                else:
                                    grad = grad_new
                                break
                        else:
                            # All backtracks failed, reset H
                            H = np.eye(dim)
                            # Recompute gradient with smaller step
                            eps_new = eps * 0.5
                            grad_new = np.zeros(dim)
                            for j in range(dim):
                                x_plus = x.copy()
                                x_plus[j] += eps_new
                                fp = safe_func(x_plus)
                                if fp is None:
                                    break
                                grad_new[j] = (fp - fx) / eps_new
                            else:
                                grad = grad_new
                                eps = eps_new
                            if check_time():
                                break

        return best_x, best_f if best_x is not None else (np.zeros(dim), 0.0)
