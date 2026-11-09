"""Multi-objective optimizer on WFG4: NSGA-II + memetic local search + hypervolume archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fast_nondominated_sort(F):
    """Return list of fronts (each a list of indices), front 0 = non-dominated."""
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    fronts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                S[i].append(j)
                nd[j] += 1
            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                nd[i] += 1
    front0 = [i for i in range(n) if nd[i] == 0]
    fronts.append(front0)
    k = 0
    while fronts[k]:
        nxt = []
        for p in fronts[k]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
            k += 1
    return fronts


def _crowding_distance(F, idx):
    """Crowding distance for points in idx (list of indices into F)."""
    m = len(idx)
    if m <= 2:
        return np.full(m, np.inf)
    cd = np.zeros(m)
    Fsub = F[idx]
    n_obj = Fsub.shape[1]
    for d in range(n_obj):
        order = np.argsort(Fsub[:, d])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        rng = Fsub[order[-1], d] - Fsub[order[0], d]
        if rng > 0:
            for i in range(1, m - 1):
                cd[order[i]] += (Fsub[order[i + 1], d] - Fsub[order[i - 1], d]) / rng
    return cd


def _hv_contribution(f, archive_F):
    """Approximate hypervolume contribution of point f w.r.t. archive_F and ref [3,5,7]."""
    ref = np.array([3.0, 5.0, 7.0])
    # contribution = volume of the box from f to ref that is not covered by archive
    # simple approximation: box volume minus overlaps is complex; use a cheap proxy:
    # product of (ref - f) for dims where f < ref, scaled by (1 - coverage)
    if np.any(f >= ref):
        return 0.0
    vol = np.prod(ref - f)
    # penalty if dominated by any archive point
    for af in archive_F:
        if np.all(af <= f) and np.any(af < f):
            return 0.0
    return vol


class Optimizer:
    """NSGA-II with memetic local search and hypervolume-aware archive on WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        rng = np.random.default_rng(12345)
        used = 0

        def evaluate(x):
            nonlocal used
            f = np.asarray(problem(np.asarray(x, dtype=float)), dtype=float)
            used += 1
            return f

        # --- Initialization: spread across the box, plus a few near-front guesses ---
        pop_size = min(24, budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        # Add a few points biased toward lower objective values (near origin in decision space)
        n_bias = max(3, pop_size // 4)
        X_bias = lo + (hi - lo) * (0.15 * rng.random((n_bias, dim)))
        X = np.vstack([X, X_bias])
        pop_size = len(X)
        F = np.array([evaluate(x) for x in X])

        # --- Memetic local search: bounded line search (gradient-free) ---
        def local_search(x, f, max_steps=6):
            """Coordinate/line search around x to reduce objectives. Returns (x_best, f_best)."""
            x_best = x.copy()
            f_best = f.copy()
            step = 0.05 * (hi - lo)
            for _ in range(max_steps):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for sign in (1, -1):
                        xn = x_best.copy()
                        xn[d] += sign * step
                        if xn[d] < lo[d] or xn[d] > hi[d]:
                            continue
                        fn = evaluate(xn)
                        if np.all(fn <= f_best) and np.any(fn < f_best):
                            x_best = xn
                            f_best = fn
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-5 * np.max(hi - lo):
                        break
            return x_best, f_best

        # Apply local search to a few of the best (lowest-sum) points
        order = np.argsort(np.sum(F, axis=1))
        n_ls = min(4, pop_size)
        for idx in order[:n_ls]:
            if time.time() > deadline:
                break
            X[idx], F[idx] = local_search(X[idx], F[idx])

        # --- NSGA-II loop ---
        while used < budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = _fast_nondominated_sort(F)

            # Build next population: take front 0 fully, then fill with crowding
            next_idx = list(fronts[0])
            for front in fronts[1:]:
                if len(next_idx) >= pop_size:
                    break
                cd = _crowding_distance(F, front)
                order_c = np.argsort(-cd)
                for j in order_c:
                    if len(next_idx) >= pop_size:
                        break
                    next_idx.append(front[j])

            # Parents: pick from front 0 (non-dominated)
            parents_idx = fronts[0] if fronts[0] else list(range(pop_size))

            # Generate children via binary tournament + crossover + mutation
            n_child = pop_size - len(next_idx)
            if n_child <= 0:
                # Population is full of non-dominated; still do a local search round
                if time.time() > deadline:
                    break
                for idx in np.random.permutation(pop_size)[:max(1, pop_size // 4)]:
                    if time.time() > deadline:
                        break
                    X[idx], F[idx] = local_search(X[idx], F[idx])
                continue

            for _ in range(n_child):
                if time.time() > deadline or used >= budget:
                    break
                # Tournament selection
                def tournament():
                    a, b = rng.integers(0, len(parents_idx), size=2)
                    fa, fb = F[parents_idx[a]], F[parents_idx[b]]
                    if np.all(fa <= fb) and np.any(fa < fb):
                        return parents_idx[a]
                    if np.all(fb <= fa) and np.any(fb < fa):
                        return parents_idx[b]
                    return parents_idx[a] if rng.random() < 0.5 else parents_idx[b]

                p1 = tournament()
                p2 = tournament()
                # Simulated binary crossover (SBX)
                child = X[p1].copy()
                eta = 20.0
                for d in range(dim):
                    if X[p1][d] != X[p2][d]:
                        u = rng.random()
                        if u <= 0.5:
                            alpha = (2 * u) ** (1 / (eta + 1))
                        else:
                            alpha = (2 * (1 - u)) ** (1 / (eta + 1))
                        child[d] = 0.5 * ((1 + alpha) * X[p1][d] + (1 - alpha) * X[p2][d])
                # Gaussian mutation
                child = child + 0.1 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = evaluate(child)

                # Insert child into population, keeping non-dominated structure
                # Simple approach: replace the point with the lowest hypervolume contribution
                X = np.vstack([X, child])
                F = np.vstack([F, fc])
                pop_size = len(X)

                # Truncate: keep non-dominated, then by crowding, up to pop_size
                if pop_size > 24:
                    fronts_t = _fast_nondominated_sort(F)
                    keep = list(fronts_t[0])
                    for front in fronts_t[1:]:
                        if len(keep) >= 24:
                            break
                        cd = _crowding_distance(F, front)
                        order_c = np.argsort(-cd)
                        for j in order_c:
                            if len(keep) >= 24:
                                break
                            keep.append(front[j])
                    X = X[keep]
                    F = F[keep]
                    pop_size = len(X)

            # Periodic local search on best points
            if used % 10 == 0 and time.time() < deadline:
                order = np.argsort(np.sum(F, axis=1))
                for idx in order[:max(2, pop_size // 5)]:
                    if time.time() > deadline or used >= budget:
                        break
                    X[idx], F[idx] = local_search(X[idx], F[idx])

        # Final: return the non-dominated set (hypervolume-relevant)
        fronts = _fast_nondominated_sort(F)
        best_idx = list(fronts[0]) if fronts[0] else list(range(len(F)))
        return X[best_idx]
# EVOLVE-BLOCK-END
