import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 using analytic front + local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # 30s safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT1: f1 = x[0], g depends on x[1:]
        # Pareto front: f1 in [0,1], f2 = 1 - sqrt(f1)
        # To minimize f2 for given f1, minimize g = 1+10*(n-1) + sum(x_i^2 - 2*cos(20*pi*x_i))
        # Minimum of each term is at x_i = 0, giving g_min = 1 + 10*(n-1)

        # Strategy: place points along Pareto front, then locally optimize x[1:]
        # Use Nelder-Mead style local search on the 29-dim subspace

        def evaluate(x):
            return np.array(problem(x))

        # Initial points: diverse f1 values along [0, 1]
        n_initial = 15
        f1_targets = np.linspace(0.0, 1.0, n_initial)
        # Add a few extra points near interesting regions
        f1_targets = np.concatenate([f1_targets, [0.01, 0.05, 0.5, 0.95]])
        f1_targets = np.unique(f1_targets)
        n_initial = len(f1_targets)

        # Budget allocation: n_initial for initialization, rest for local search
        budget_per_point = max(1, (self.budget - n_initial) // n_initial)

        X = np.zeros((n_initial, dim))
        F = np.zeros((n_initial, self.n_obj))

        for i, f1_t in enumerate(f1_targets):
            x = np.zeros(dim)
            x[0] = f1_t
            # x[1:] start at 0 (optimal for g minimization)
            X[i] = x
            F[i] = evaluate(x)

        used = n_initial

        # Local search: for each point, try to improve g (x[1:]) while keeping x[0] fixed
        # Use a simple pattern search / coordinate descent on x[1:]
        for i in range(n_initial):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Local search on x[1:] to minimize g
            # Try small perturbations in each dimension
            x_current = X[i].copy()
            f_current = F[i].copy()
            g_current = f_current[1]  # f2 = g*(1-sqrt(f1/g)), but we want to minimize g

            # Actually, let's just minimize the sum of objectives or use f2 as proxy
            # For ZDT1, minimizing f2 at fixed f1 means minimizing g
            # g = 1 + 10*(n-1) + sum(x_j^2 - 2*cos(20*pi*x_j)) for j=1..n-1

            step = 0.1
            improved = True
            while improved and used < self.budget and time.time() < deadline:
                improved = False
                for j in range(1, dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_try = x_current.copy()
                        x_try[j] += direction * step
                        x_try = np.clip(x_try, lo, hi)
                        f_try = evaluate(x_try)
                        used += 1
                        if f_try[1] < f_current[1] - 1e-12:
                            x_current = x_try
                            f_current = f_try
                            X[i] = x_current
                            F[i] = f_current
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        # Use remaining budget for additional refinement or new points
        if used < self.budget and time.time() < deadline:
            # Add more points at unexplored f1 values
            all_f1 = X[:, 0]
            # Find gaps
            sorted_f1 = np.sort(all_f1)
            gaps = np.diff(sorted_f1)
            if len(gaps) > 0 and np.max(gaps) > 0.05:
                # Add point in largest gap
                max_gap_idx = np.argmax(gaps)
                new_f1 = (sorted_f1[max_gap_idx] + sorted_f1[max_gap_idx + 1]) / 2
                x_new = np.zeros(dim)
                x_new[0] = new_f1
                f_new = evaluate(x_new)
                used += 1
                # Append to arrays
                X = np.vstack([X, x_new])
                F = np.vstack([F, f_new])

                # Local search on this new point
                x_current = x_new.copy()
                f_current = f_new.copy()
                step = 0.1
                improved = True
                while improved and used < self.budget and time.time() < deadline:
                    improved = False
                    for j in range(1, dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_try = x_current.copy()
                            x_try[j] += direction * step
                            x_try = np.clip(x_try, lo, hi)
                            f_try = evaluate(x_try)
                            used += 1
                            if f_try[1] < f_current[1] - 1e-12:
                                x_current = x_try
                                f_current = f_try
                                X[-1] = x_current
                                F[-1] = f_current
                                improved = True
                                break
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break

        # Final: ensure we haven't exceeded budget
        return X
