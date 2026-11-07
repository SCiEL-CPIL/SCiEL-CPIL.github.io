import numpy as np
import scipy.optimize as opt
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Evaluation tracking
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        # Phase 1: Stratified random sampling for initial coverage
        # Use 30% of budget for initial diverse sampling
        n_random = min(int(budget * 0.3), 60)
        X_samples = []
        F_samples = []

        for i in range(n_random):
            if not time_ok():
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = evaluate(x)
            X_samples.append(x)
            F_samples.append(f)

        # Phase 2: Weighted-sum local search with multiple weight vectors
        # Generate diverse weight vectors on the simplex
        n_weights = 15
        weights = []
        
        # Vertices
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        
        # Edge midpoints
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                w = np.zeros(n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
        
        # Random Dirichlet samples
        while len(weights) < n_weights:
            w = rng.dirichlet([1, 1, 1])
            # Check for duplicates
            is_dup = False
            for uw in weights:
                if np.allclose(w, uw, atol=0.05):
                    is_dup = True
                    break
            if not is_dup:
                weights.append(w)
        
        # Normalize weights
        weights = [w / w.sum() for w in weights]

        # Allocate remaining budget to local search
        remaining_budget = budget - eval_count
        budget_per_weight = max(1, int(remaining_budget / len(weights)))

        for w in weights:
            if not time_ok():
                break

            # Multiple restarts per weight for robustness
            n_restarts = 3
            budget_per_restart = max(1, budget_per_weight // n_restarts)

            for r in range(n_restarts):
                if not time_ok():
                    break

                # Start from a random point or from a good sample
                if X_samples and rng.random() < 0.5:
                    idx = rng.integers(len(X_samples))
                    x0 = X_samples[idx].copy()
                else:
                    x0 = lo + (hi - lo) * rng.random(dim)

                # Define scalar objective
                def scalar_obj(x, weights=w):
                    if eval_count >= budget:
                        raise Exception("Budget exceeded")
                    f = problem(x)
                    return np.dot(weights, f)

                # Use L-BFGS-B for bounded continuous optimization
                bounds_opt = list(zip(lo, hi))

                try:
                    res = opt.minimize(
                        scalar_obj,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds_opt,
                        options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.success:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        X_samples.append(x_best)
                        F_samples.append(f_best)
                except Exception:
                    pass

                # Check time
                if not time_ok():
                    break

        # Phase 3: Final polish - use remaining budget for targeted local search
        # around the best non-dominated points found
        if time_ok() and eval_count < budget:
            if F_samples:
                F_arr = np.array(F_samples)
                # Find non-dominated points
                nondom_mask = np.ones(len(F_arr), dtype=bool)
                for i in range(len(F_arr)):
                    if not nondom_mask[i]:
                        continue
                    for j in range(len(F_arr)):
                        if i == j or not nondom_mask[j]:
                            continue
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            nondom_mask[i] = False
                            break

                # Take top non-dominated points for polishing
                nondom_indices = np.where(nondom_mask)[0]
                n_polish = min(5, len(nondom_indices), (budget - eval_count) // 10)

                for idx in nondom_indices[:n_polish]:
                    if not time_ok():
                        break

                    x0 = X_samples[idx].copy()

                    # Use different weight vectors for polishing
                    for w in weights[:3]:
                        if not time_ok():
                            break

                        def polish_obj(x, weights=w):
                            if eval_count >= budget:
                                raise Exception("Budget exceeded")
                            f = problem(x)
                            return np.dot(weights, f)

                        bounds_opt = list(zip(lo, hi))
                        try:
                            res = opt.minimize(
                                polish_obj,
                                x0,
                                method='L-BFGS-B',
                                bounds=bounds_opt,
                                options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-10}
                            )
                            if res.success:
                                x_best = np.clip(res.x, lo, hi)
                                f_best = evaluate(x_best)
                                X_samples.append(x_best)
                                F_samples.append(f_best)
                        except Exception:
                            pass

        return X_samples
