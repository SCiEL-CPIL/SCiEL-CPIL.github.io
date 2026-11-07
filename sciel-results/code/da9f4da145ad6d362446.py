# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-objective optimizer: LHS init, diversity-preserving EAs, and Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Record all evaluated points
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return x, f

        # Phase 1: Latin Hypercube initialization
        n_lhs = 20
        lhs_X = np.empty((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            lhs_X[:, d] = (perm + 0.5) / n_lhs
        lhs_X = lo + (hi - lo) * lhs_X

        for x in lhs_X:
            if eval_count >= self.budget or time.time() > deadline:
                break
            evaluate(x)

        # Phase 2: Diversity-preserving evolutionary search
        pop_size = min(15, self.budget - eval_count)
        if pop_size > 0 and eval_count < self.budget:
            # Initialize population from best non-dominated points
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated points
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            nd_idx = np.where(mask)[0]

            if len(nd_idx) > 0:
                # Select diverse subset
                if len(nd_idx) > pop_size:
                    # Pick spread points using hypervolume contribution
                    selected = []
                    remaining = list(nd_idx)
                    while len(selected) < pop_size and remaining:
                        best_idx = None
                        best_hv = -np.inf
                        for idx in remaining:
                            # Simple HV contribution: distance from others
                            if selected:
                                dists = np.linalg.norm(F_arr[idx] - F_arr[selected], axis=1)
                                contrib = np.min(dists) if len(dists) > 0 else 0
                            else:
                                contrib = -np.sum(F_arr[idx])
                            if contrib > best_hv:
                                best_hv = contrib
                                best_idx = idx
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                else:
                    selected = list(nd_idx)

                pop_X = np.array([X_arr[i] for i in selected])
                pop_F = np.array([F_arr[i] for i in selected])
            else:
                pop_X = X_arr[:pop_size].copy()
                pop_F = F_arr[:pop_size].copy()

            # Evolutionary loop
            while eval_count < self.budget and time.time() < deadline:
                # Find non-dominated in population
                mask = np.ones(len(pop_F), dtype=bool)
                for i in range(len(pop_F)):
                    for j in range(len(pop_F)):
                        if i != j and np.all(pop_F[j] <= pop_F[i]) and np.any(pop_F[j] < pop_F[i]):
                            mask[i] = False
                            break
                nd_mask = mask

                # Select parent (prefer non-dominated)
                if np.any(nd_mask):
                    parent_idx = rng.choice(np.where(nd_mask)[0])
                else:
                    parent_idx = rng.integers(len(pop_F))

                # Mutate
                sigma = 0.15
                child = pop_X[parent_idx] + sigma * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                cx, cf = evaluate(child)

                # Replace worst point in population
                if len(pop_F) < pop_size:
                    pop_X = np.vstack([pop_X, cx])
                    pop_F = np.vstack([pop_F, cf])
                else:
                    # Remove most dominated or worst-sum point
                    worst_idx = np.argmax(np.sum(pop_F, axis=1))
                    pop_X[worst_idx] = cx
                    pop_F[worst_idx] = cf

        # Phase 3: Nelder-Mead polish on top non-dominated points
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            nd_idx = np.where(mask)[0]

            # Pick top points by hypervolume contribution
            if len(nd_idx) > 0:
                # Sort by sum of objectives
                order = np.argsort(np.sum(F_arr[nd_idx], axis=1))
                top_n = min(5, len(nd_idx), (self.budget - eval_count) // 2)

                for idx in order[:top_n]:
                    if eval_count >= self.budget or time.time() > deadline:
                        break

                    x0 = X_arr[idx].copy()

                    def objective(x):
                        # Weighted sum with adaptive weights
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x))
                        return np.sum(f)

                    try:
                        res = minimize(objective, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                        if eval_count < self.budget:
                            cx = np.clip(res.x, lo, hi)
                            cf = np.array(problem(cx))
                            all_X.append(cx.copy())
                            all_F.append(cf.copy())
                            eval_count += 1
                    except Exception:
                        pass

        return all_X
# EVOLVE-BLOCK-END
