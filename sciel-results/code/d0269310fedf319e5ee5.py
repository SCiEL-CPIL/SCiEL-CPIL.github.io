"""Multi-objective optimizer for RE34: Hypervolume-guided search with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search with structured initialization and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # ---------- Helpers ----------
        def hv2(sub, ref2):
            """2D hypervolume (minimization)."""
            if len(sub) == 0:
                return 0.0
            sub = sub[np.all(sub < ref2, axis=1)]
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                w0 = min(x, ref2[0]) - prev
                if w0 <= 0:
                    continue
                ybest = np.min(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                if ybest >= ref2[1]:
                    prev = x
                    continue
                total += w0 * (ref2[1] - ybest)
                prev = x
            return total

        def hv3(P, ref3):
            """3D hypervolume (minimization)."""
            if len(P) == 0:
                return 0.0
            mask = np.all(P < ref3, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                w0 = min(x, ref3[0]) - prev
                if w0 <= 0:
                    continue
                sub = P[i:, 1:]
                total += w0 * hv2(sub, ref3[1:])
                prev = x
            return total

        def pareto_indices_fast(F):
            """Return indices of non-dominated points using vectorized comparison."""
            n = len(F)
            if n == 0:
                return []
            # F: (n, m), m = number of objectives
            # Point i is dominated if exists j: F[j] <= F[i] and F[j] < F[i]
            # Vectorized: for each i, check if any j dominates it
            # Use broadcasting: F[:, np.newaxis, :] vs F[np.newaxis, :, :]
            # But this is O(n^2 * m) which is fine for n<=200
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                # Check if any other point dominates i
                # A point j dominates i if: all(F[j] <= F[i]) and any(F[j] < F[i])
                diffs = F - F[i]  # (n, m)
                leq = np.all(diffs <= 0, axis=1)  # (n,)
                lt = np.any(diffs < 0, axis=1)  # (n,)
                dominates = leq & lt
                dominates[i] = False  # i doesn't dominate itself
                if np.any(dominates):
                    dominated[i] = True
            return [i for i in range(n) if not dominated[i]]

        # ---------- Main Loop ----------
        X = []
        F = []
        evaluated_points = []  # For deduplication

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))
            evaluated_points.append(np.asarray(x, dtype=float))

        def is_duplicate(x, threshold=1e-4):
            """Check if point is too close to any previously evaluated point."""
            if not evaluated_points:
                return False
            dists = np.min(np.linalg.norm(np.array(evaluated_points) - x, axis=1))
            return dists < threshold

        # 1. Structured Initial Sampling
        # Start with the known good corner [1,1,1,1,1]
        x_corner = lo.copy()
        add(x_corner, problem(x_corner))

        # Add structured spread along multiple axes
        # RE34: f1 dominated by x1, so vary x1 while keeping others at lower bound
        n_structured = 25
        for i in range(n_structured):
            if len(X) >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x1 = lo[0] + (hi[0] - lo[0]) * i / max(n_structured - 1, 1)
            x = np.array([x1, lo[1], lo[2], lo[3], lo[4]])
            x = np.clip(x, lo, hi)
            if not is_duplicate(x):
                add(x, problem(x))

        # Add structured spread along x2
        n_structured2 = 10
        for i in range(n_structured2):
            if len(X) >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x2 = lo[1] + (hi[1] - lo[1]) * i / max(n_structured2 - 1, 1)
            x = np.array([lo[0], x2, lo[2], lo[3], lo[4]])
            x = np.clip(x, lo, hi)
            if not is_duplicate(x):
                add(x, problem(x))

        # Add a few random points for diversity
        n_random_init = 15
        for _ in range(n_random_init):
            if len(X) >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x = lo + (hi - lo) * rng.random(dim)
            if not is_duplicate(x):
                add(x, problem(x))

        used = len(X)
        scale = 0.2
        best_hv = 0.0
        stagnation = 0
        hv_compute_interval = 5  # Compute HV every 5 iterations to save time
        last_hv_compute = 0

        # ---------- Main Search Loop ----------
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Only compute HV periodically to save computation
            if used - last_hv_compute >= hv_compute_interval or stagnation > 20:
                F_arr = np.array(F)
                current_hv = hv3(F_arr, ref)
                if current_hv > best_hv + 1e-6:
                    best_hv = current_hv
                    stagnation = 0
                else:
                    stagnation += 1
                last_hv_compute = used
            else:
                # Approximate stagnation without full HV computation
                if used - last_hv_compute > 10:
                    stagnation += 1

            # Determine exploration vs exploitation
            if stagnation > 30:
                explore_prob = 0.5
                step_mult = 2.0
            elif stagnation > 10:
                explore_prob = 0.3
                step_mult = 1.5
            else:
                explore_prob = 0.1
                step_mult = 1.0

            # Occasionally jump to a completely random point when stuck
            if stagnation > 50 and rng.random() < 0.3:
                x = lo + (hi - lo) * rng.random(dim)
                if not is_duplicate(x):
                    fc = problem(x)
                    add(x, fc)
                    used += 1
                    scale = max(0.3, scale * 1.5)
                    continue
                else:
                    # Try a different random point
                    x = lo + (hi - lo) * rng.random(dim)
                    fc = problem(x)
                    add(x, fc)
                    used += 1
                    scale = max(0.3, scale * 1.5)
                    continue

            # Get Pareto front
            F_arr = np.array(F)
            pf_idx = pareto_indices_fast(F_arr)
            if not pf_idx:
                pf_idx = list(range(len(F_arr)))

            # Select parent
            # Strategy: Pick from Pareto front, weighted by "importance"
            cand = F_arr[pf_idx]
            sums = np.sum(cand, axis=1)
            range_sums = sums.max() - sums.min() + 1e-9
            w = np.exp(-2.0 * (sums - sums.min()) / range_sums)
            w /= w.sum()
            pidx = pf_idx[int(rng.choice(len(pf_idx), p=w))]

            # Mutation with multiple strategies
            strategy = rng.random()
            if strategy < 0.3:
                # Local refinement: small step
                step = scale * 0.5 * step_mult
                delta = step * (hi - lo) * rng.standard_normal(dim)
            elif strategy < 0.6:
                # Standard mutation
                step = scale * step_mult
                delta = step * (hi - lo) * rng.standard_normal(dim)
            elif strategy < 0.8:
                # Coordinate-wise mutation: perturb only 1-2 dimensions
                step = scale * step_mult
                n_perturb = rng.integers(1, 3)
                dims_to_perturb = rng.choice(dim, size=n_perturb, replace=False)
                delta = np.zeros(dim)
                for d in dims_to_perturb:
                    delta[d] = step * (hi[d] - lo[d]) * rng.standard_normal()
            else:
                # Large jump: mix with random point
                step = scale * step_mult * 1.5
                delta = step * (hi - lo) * rng.standard_normal(dim)

            child = X[pidx] + delta
            child = np.clip(child, lo, hi)

            if is_duplicate(child):
                # Try a slightly different perturbation
                delta2 = 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = child + delta2
                child = np.clip(child, lo, hi)

            fc = problem(child)
            add(child, fc)
            used += 1

            # Adaptive Scale
            if stagnation > 10:
                scale = min(0.5, scale * 1.1)
            else:
                scale = max(0.03, scale * 0.995)

        return X
# EVOLVE-BLOCK-END
