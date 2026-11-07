"""Multi-objective optimizer for RE21 with hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search for 2-objective RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([3000.0, 0.06])

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f

        def compute_hv_2d(points):
            """Compute 2D hypervolume for points dominated by reference."""
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            max_y = ref[1]
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                if y < max_y:
                    hv += (ref[0] - x) * (max_y - y)
                    max_y = y
            return hv

        def non_dominated_indices(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Phase 1: Stratified random initialization with emphasis on corners
        rng = np.random.default_rng(42)
        n_init = 30

        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            for d in range(dim):
                if (i + d) % 3 == 0:
                    x[d] = lo[d]
                elif (i + d) % 3 == 1:
                    x[d] = hi[d]
                else:
                    x[d] = lo[d] + rng.random() * (hi[d] - lo[d])
            evaluate(x)

        # Add random points
        n_rand = 15
        for _ in range(n_rand):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # Phase 2: Multi-start local refinement using Nelder-Mead-like approach
        # Focus on points near the Pareto front
        F_arr = np.array(all_F)
        nd_idx = non_dominated_indices(F_arr)

        # Sort non-dominated points by proximity to ideal point
        ideal = np.min(F_arr, axis=0)
        if len(nd_idx) > 0:
            nd_F = F_arr[nd_idx]
            dists = np.sum((nd_F - ideal) ** 2, axis=1)
            # Pick the best few for local refinement
            n_refine = min(8, len(nd_idx), (budget - used) // 8)
            best_local = nd_idx[np.argsort(dists)[:n_refine]]

            for start_idx in best_local:
                if used >= budget or time.time() > deadline:
                    break

                # Local search around this point
                best_x = all_X[start_idx].copy()
                best_f = all_F[start_idx].copy()

                # Adaptive step size
                base_step = 0.1 * (hi - lo)
                for step_iter in range(7):
                    if used >= budget or time.time() > deadline:
                        break

                    # Try perturbations in each dimension
                    for d in range(dim):
                        if used >= budget or time.time() > deadline:
                            break

                        for direction in [-1, 1]:
                            if used >= budget or time.time() > deadline:
                                break

                            candidate = best_x.copy()
                            candidate[d] += direction * base_step * (0.5 + 0.5 * rng.random())
                            candidate = np.clip(candidate, lo, hi)

                            f_cand = evaluate(candidate)

                            # Accept if it improves the hypervolume contribution
                            # Simple check: if it's better in at least one objective
                            # and not worse in the other
                            if (f_cand[0] < best_f[0] and f_cand[1] <= best_f[1]) or \
                               (f_cand[1] < best_f[1] and f_cand[0] <= best_f[0]):
                                best_x = candidate.copy()
                                best_f = f_cand.copy()

                    # Shrink step size
                    base_step *= 0.6

        # Phase 3: Final random sampling to fill remaining budget
        while used < budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
