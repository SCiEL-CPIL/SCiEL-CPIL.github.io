"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer with NSGA-II-like selection and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([3000.0, 0.06])

        # Initial population
        pop_size = min(30, budget // 3)
        pop_size = max(pop_size, 10)
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Hypervolume contribution helper for 2D
        def hv_contribution_2d(points, ref_pt):
            """Compute hypervolume contribution of a set of 2D points w.r.t. reference."""
            # Filter points strictly better than reference
            mask = np.all(points < ref_pt, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            prev_y = ref_pt[1]
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                if y < prev_y:
                    hv += (ref_pt[0] - x) * (prev_y - y)
                    prev_y = y
            return hv

        def get_non_dominated_front(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[i].append(j)
                    elif np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1]

        # Main evolution loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated front
            nd_mask = get_non_dominated_front(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # Fallback: use all
                nd_indices = np.arange(len(F))

            # Select parents from non-dominated front
            n_parents = min(5, len(nd_indices))
            parent_idx = np.random.choice(nd_indices, size=n_parents, replace=False)

            # Generate children via mutation
            children = []
            for pi in parent_idx:
                # Gaussian mutation with adaptive step
                step = 0.05 * (hi - lo)
                child = X[pi] + step * np.random.randn(dim)
                child = np.clip(child, lo, hi)
                children.append(child)

            # Evaluate children
            child_F = np.array([problem(c) for c in children])
            used += len(children)

            # Combine population and children
            X_all = np.vstack([X, children])
            F_all = np.vstack([F, child_F])

            # Select best pop_size individuals based on hypervolume contribution
            # Use a greedy approach: select points that maximize incremental HV
            selected_idx = []
            remaining = set(range(len(F_all)))
            current_hv = 0.0
            for _ in range(pop_size):
                best_idx = None
                best_gain = -1.0
                for idx in remaining:
                    # Compute HV with this point added to current selection
                    if selected_idx:
                        pts = np.vstack([F_all[selected_idx], F_all[idx]])
                    else:
                        pts = F_all[idx:idx+1]
                    hv = hv_contribution_2d(pts, ref)
                    gain = hv - current_hv
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is not None:
                    selected_idx.append(best_idx)
                    remaining.remove(best_idx)
                    if selected_idx:
                        pts = F_all[selected_idx]
                        current_hv = hv_contribution_2d(pts, ref)

            X = X_all[selected_idx]
            F = F_all[selected_idx]

            # Final polish: local search on non-dominated points
            if used + 10 < budget and time.time() < deadline - 30:
                nd_mask = get_non_dominated_front(F)
                nd_indices = np.where(nd_mask)[0]
                for idx in nd_indices[:5]:  # Polish up to 5 best points
                    if time.time() > deadline:
                        break
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        # Try small perturbations
                        for _ in range(4):
                            if time.time() > deadline:
                                break
                            pert = best_x + 0.01 * (hi - lo) * np.random.randn(dim)
                            pert = np.clip(pert, lo, hi)
                            pf = problem(pert)
                            used += 1
                            if np.all(pf <= best_f) and np.any(pf < best_f):
                                best_x = pert
                                best_f = pf
                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
