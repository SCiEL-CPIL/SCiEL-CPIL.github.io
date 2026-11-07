"""Multi-objective optimizer for RE37: DE + weight-grid exploration + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with weight-grid init, NM polish, deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 305.0  # safety margin before 330s kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        d, n_obj, budget = self.dim, self.n_obj, self.budget

        def ok():
            return time.time() < deadline

        # ---- weight-grid initial population (structured, covers objective space) ----
        # 3 objectives -> use a structured grid of weights
        def weight_grid(n_obj, n_per):
            pts = []
            # include vertices
            for i in range(n_obj):
                v = np.zeros(n_obj); v[i] = 1.0
                pts.append(v)
            # include center
            pts.append(np.ones(n_obj) / n_obj)
            # include 2-objective edges midpoints
            for i in range(n_obj):
                for j in range(i + 1, n_obj):
                    v = np.zeros(n_obj); v[i] = 0.5; v[j] = 0.5
                    pts.append(v)
            # include some interior random-ish structured points
            for k in range(1, n_per):
                frac = k / n_per
                for i in range(n_obj):
                    for j in range(i + 1, n_obj):
                        v = np.zeros(n_obj)
                        v[i] = frac
                        v[j] = 1.0 - frac
                        pts.append(v)
            return np.array(pts)

        W = weight_grid(n_obj, 6)
        # Normalize weights
        W = W / np.linalg.norm(W, axis=1, keepdims=True)

        # Use a subset of weights to define initial population
        pop_size = min(24, budget // 4)
        # Take first pop_size weights (or wrap)
        idx = np.arange(pop_size) % len(W)
        W_init = W[idx]

        # For each weight, find a good point by scalarizing: minimize w.f
        # Use a quick local search (Nelder-Mead) per weight to get a diverse start
        X = np.zeros((pop_size, d))
        F = np.zeros((pop_size, n_obj))
        used = 0

        def nm_minimize(obj_fn, x0, max_iter=20, tol=1e-6):
            # Simple Nelder-Mead for a scalar function
            x = np.array(x0, dtype=float)
            n = len(x)
            simplex = [x.copy()]
            for i in range(n):
                xi = x.copy()
                xi[i] += 0.1 * (hi[i] - lo[i])
                simplex.append(xi)
            simplex = np.array(simplex)
            fvals = np.array([obj_fn(s) for s in simplex])
            used_count = len(simplex)
            for it in range(max_iter):
                if not ok():
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = obj_fn(xr); used_count += 1
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = obj_fn(xe); used_count += 1
                    if fe < fr:
                        simplex[-1] = xe; fvals[-1] = fe
                    else:
                        simplex[-1] = xr; fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr; fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = obj_fn(xc); used_count += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc; fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = obj_fn(simplex[i]); used_count += 1
                if np.max(np.abs(fvals - fvals[0])) < tol:
                    break
            best = np.argmin(fvals)
            return simplex[best], fvals[best], used_count

        for i in range(pop_size):
            if not ok():
                break
            w = W_init[i]

            def scalar_obj(x, w=w):
                return float(np.dot(w, problem(x)))

            x0 = lo + (hi - lo) * 0.5
            xb, fb, cnt = nm_minimize(scalar_obj, x0, max_iter=15)
            used += cnt
            X[i] = xb
            F[i] = np.array(problem(xb))
            used += 1

        # ---- DE loop ----
        CR = 0.9
        F_mut = 0.8

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_mask(F):
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        mask[i] = False
                        break
            return mask

        while used < budget and ok():
            # Select 3 distinct parents
            p1, p2, p3 = rng.choice(pop_size, size=3, replace=False)
            v = X[p1] + F_mut * (X[p2] - X[p3])
            # Binomial crossover
            r = rng.random(d)
            jrand = rng.integers(d)
            child = np.where((r < CR) | (np.arange(d) == jrand), v, X[p1])
            child = np.clip(child, lo, hi)
            fc = np.array(problem(child))
            used += 1

            # Replace worst (by sum) if child is non-dominated or improves
            # Simple: replace worst
            worst = int(np.argmax(np.sum(F, axis=1)))
            # Only replace if child is at least as good in some sense
            if not dominates(F[worst], child):
                X[worst] = child
                F[worst] = fc

        # ---- Polish non-dominated points with NM on hypervolume contribution ----
        # Identify non-dominated points and do a quick NM polish on each to tighten front
        nd_mask = non_dominated_mask(F)
        nd_idx = np.where(nd_mask)[0]

        # Sort nd_idx by sum (worst first) to polish most impactful
        order = np.argsort(np.sum(F[nd_idx], axis=1))
        nd_idx = nd_idx[order]

        # Polish a limited number of points
        max_polish = min(8, len(nd_idx))
        for idx_i in range(max_polish):
            if not ok() or used >= budget:
                break
            i = nd_idx[idx_i]
            # Objective: minimize sum (to tighten) while staying feasible
            def obj_fn(x, i=i):
                f = problem(x)
                return float(np.sum(f))
            xb, fb, cnt = nm_minimize(obj_fn, X[i], max_iter=20)
            used += cnt
            F[i] = np.array(problem(xb))
            X[i] = xb
            used += 1

        return X
# EVOLVE-BLOCK-END
