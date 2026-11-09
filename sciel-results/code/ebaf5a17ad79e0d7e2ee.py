"""Combined optimizer for WFG7: simple local search with final refinement."""
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using local search with refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim

        # Phase 1: Dense random sampling
        n_init = min(budget, 60)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        ref = np.array([3.0, 5.0, 7.0])

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def is_nondominated(f, F_all):
            for g in F_all:
                if np.all(g <= f) and np.any(g < f):
                    return False
            return True

        def non_dominated_set(F):
            idx = []
            for i, f in enumerate(F):
                if is_nondominated(f, F):
                    idx.append(i)
            return np.array(idx, dtype=int)

        # Phase 2: Build refined population from non-dominated points
        pop_size = 12
        nd_idx = non_dominated_set(F)
        if len(nd_idx) == 0:
            nd_idx = np.array([0])

        X_ref = X[nd_idx[:pop_size]].copy()
        F_ref = F[nd_idx[:pop_size]].copy()
        if len(X_ref) < pop_size:
            n_pad = pop_size - len(X_ref)
            X_pad = lo + (hi - lo) * rng.random((n_pad, dim))
            F_pad = np.array([problem(x) for x in X_pad])
            used += n_pad
            X_ref = np.vstack([X_ref, X_pad])
            F_ref = np.vstack([F_ref, F_pad])

        step_scale = 0.15 * (hi - lo)

        # Phase 3: Main local search loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select parent from non-dominated in refined pop
            nd_in_pop = []
            for i in range(len(F_ref)):
                if is_nondominated(F_ref[i], F_ref):
                    nd_in_pop.append(i)
            if not nd_in_pop:
                nd_in_pop = list(range(len(F_ref)))

            parent_idx = nd_in_pop[rng.integers(len(nd_in_pop))]
            parent = X_ref[parent_idx].copy()

            # Mutation with adaptive step
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction) + 1e-10
            scale = step_scale * (0.5 + 1.0 * rng.random())
            child = parent + scale * direction
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update population
            dominated_mask = np.array([not is_nondominated(F_ref[i], F_ref) for i in range(len(F_ref))])
            if np.any(dominated_mask):
                worst_idx = np.where(dominated_mask)[0][rng.integers(np.sum(dominated_mask))]
                X_ref[worst_idx] = child
                F_ref[worst_idx] = fc
            else:
                F_norm = (F_ref - np.min(F_ref, axis=0)) / (np.max(F_ref, axis=0) - np.min(F_ref, axis=0) + 1e-10)
                worst_idx = int(np.argmax(np.sum(F_norm, axis=1)))
                if np.any(fc < F_ref[worst_idx]):
                    X_ref[worst_idx] = child
                    F_ref[worst_idx] = fc

            # Periodic exploration
            if (budget - used) > 0 and (used % 20 == 0):
                exp = lo + (hi - lo) * rng.random(dim)
                fe = problem(exp)
                used += 1
                if is_nondominated(fe, F_ref):
                    ridx = rng.integers(len(F_ref))
                    X_ref[ridx] = exp
                    F_ref[ridx] = fe

        # Phase 4: Final local refinement around best non-dominated points
        all_F = np.vstack([F, F_ref])
        all_X = np.vstack([X, X_ref])

        final_nd = non_dominated_set(all_F)
        if len(final_nd) > 0:
            nd_points = all_F[final_nd]
            dists = np.sum((nd_points - ref) ** 2, axis=1)
            sorted_nd = final_nd[np.argsort(dists)]

            remaining = budget - used
            if remaining > 0:
                n_refine = min(len(sorted_nd), max(1, remaining // 10))
                for i in range(n_refine):
                    if used >= budget or time.time() > deadline:
                        break

                    orig_idx = sorted_nd[i]
                    base_x = all_X[orig_idx].copy()
                    base_f = all_F[orig_idx].copy()

                    local_budget = min(5, budget - used)
                    local_step = 0.03 * (hi - lo)

                    for j in range(local_budget):
                        if used >= budget or time.time() > deadline:
                            break

                        direction = rng.standard_normal(dim)
                        direction /= np.linalg.norm(direction) + 1e-10
                        step = local_step * (0.3 + 0.7 * rng.random())
                        candidate = base_x + step * direction
                        candidate = np.clip(candidate, lo, hi)

                        fc = problem(candidate)
                        used += 1

                        if dominates(fc, base_f):
                            base_x = candidate
                            base_f = fc
                            all_X[orig_idx] = base_x
                            all_F[orig_idx] = base_f

        return all_X
