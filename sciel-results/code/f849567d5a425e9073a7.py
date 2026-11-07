import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return np.inf
            calls_made += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        def bfgs_local(x0, max_iter=200):
            """Accelerated BFGS with numerical gradients."""
            nonlocal calls_made, best_f, best_x
            n = len(x0)
            x = x0.copy()
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

            # Numerical gradient
            def gradient(x):
                g = np.zeros(n)
                h = 1e-5
                for i in range(n):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h
                    xm[i] -= h
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fp = safe_call(xp)
                    fm = safe_call(xm)
                    if np.isfinite(fp) and np.isfinite(fm):
                        g[i] = (fp - fm) / (2 * h)
                    else:
                        g[i] = 0.0
                return g

            g = gradient(x)
            H = np.eye(n)  # Inverse Hessian approximation

            for it in range(max_iter):
                if calls_made >= total_budget or time.time() > deadline - 5:
                    break
                if np.linalg.norm(g, ord=np.inf) < 1e-8:
                    break

                # Search direction
                d = -H @ g

                # Line search (backtracking)
                alpha = 1.0
                f_new = f
                x_new = x
                for ls in range(30):
                    x_trial = x + alpha * d
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)
                    if f_trial < f - 1e-4 * alpha * np.dot(g, d):
                        f_new = f_trial
                        x_new = x_trial
                        break
                    alpha *= 0.5
                else:
                    break

                # Update
                s = x_new - x
                g_new = gradient(x_new)
                y = g_new - g

                if np.dot(s, y) <= 1e-10:
                    H = np.eye(n)
                else:
                    rho = 1.0 / np.dot(s, y)
                    H = ((np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s))

                x = x_new
                f = f_new
                g = g_new

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            return x, f

        # Phase 1: Deterministic LHS global exploration
        # Create a stratified Latin Hypercube sample for guaranteed coverage
        n_lhs = 40  # Number of LHS points
        rng = np.random.default_rng(123)

        # Generate LHS points: each dimension has one point per stratum
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Random permutation of strata
            perm = rng.permutation(n_lhs)
            # Uniform within each stratum
            strata = np.arange(n_lhs)
            lhs_points[:, d] = (strata + rng.uniform(size=n_lhs)) / n_lhs
            lhs_points[:, d] = np.clip(lhs_points[:, d] * 10 - 5, -5, 5)
            # Apply permutation
            lhs_points[:, d] = lhs_points[perm, d]

        # Also add origin and some structured points
        extra_points = [np.zeros(dim)]
        for d in range(dim):
            p1 = np.zeros(dim)
            p1[d] = 3.0
            p2 = np.zeros(dim)
            p2[d] = -3.0
            extra_points.extend([p1, p2])

        all_start_points = [lhs_points[i] for i in range(n_lhs)] + extra_points

        # Evaluate all starting points
        for p in all_start_points:
            if calls_made >= total_budget or time.time() > deadline - 30:
                break
            fx = safe_call(p)
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: BFGS local refinement from top candidates
        # Sort starting points by function value and refine the best ones
        # We need to track which points we've evaluated
        evaluated_points = []
        for p in all_start_points:
            if calls_made >= total_budget:
                break
            # We already evaluated these, so we know their values
            # But we need to re-evaluate to get current values if we want to sort
            # Actually, let's just collect the best few from our tracking
            pass

        # Better approach: collect all evaluated points and their values
        # Since we tracked best_x and best_f, let's just run BFGS from the best point
        # and also from a few other promising points

        # Run BFGS from the global best
        if calls_made < total_budget - 50 and time.time() < deadline - 10:
            bfgs_local(best_x)

        # Run BFGS from other promising LHS points (top 5 by proximity to best)
        if calls_made < total_budget - 50 and time.time() < deadline - 10:
            # Evaluate a subset of LHS points to find good local basins
            candidate_starts = []
            for p in lhs_points[:20]:  # First 20 LHS points
                if calls_made >= total_budget - 50:
                    break
                if time.time() > deadline - 10:
                    break
                # Check if we've already evaluated this point (approximately)
                # Just evaluate it to be safe
                fx = safe_call(p)
                if np.isfinite(fx):
                    candidate_starts.append((fx, p))

            # Sort by function value and take top 3
            candidate_starts.sort(key=lambda x: x[0])
            for _, p in candidate_starts[:3]:
                if calls_made >= total_budget - 30 or time.time() > deadline - 5:
                    break
                # Only run BFGS if this point is different from best_x
                if np.linalg.norm(p - best_x) > 0.1:
                    bfgs_local(p)

        # Phase 3: Final polish with BFGS from the global best
        if calls_made < total_budget - 10 and time.time() < deadline - 3:
            bfgs_local(best_x)

        return best_x, best_f
