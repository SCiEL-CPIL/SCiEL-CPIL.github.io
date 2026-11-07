"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with LHS initialization and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # LHS initialization for good spread
        n_init = min(50, self.budget // 4)
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_init)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.random(n_init)) / n_init
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_init

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            ncount = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        ncount[i] += 1
            front0 = [i for i in range(n) if ncount[i] == 0]
            fronts.append(front0)
            k = 0
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        ncount[j] -= 1
                        if ncount[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            cd = np.zeros(len(indices))
            for m in range(F.shape[1]):
                vals = F[indices, m]
                sv = np.argsort(vals)
                cd[sv[0]] = np.inf
                cd[sv[-1]] = np.inf
                rng_ = vals[sv[-1]] - vals[sv[0]]
                if rng_ > 1e-15:
                    for k in range(1, len(indices) - 1):
                        cd[sv[k]] += (vals[sv[k + 1]] - vals[sv[k - 1]]) / rng_
            return cd

        def nelder_mead(problem, x0, bounds, max_iter=50):
            """Simple Nelder-Mead for a single scalar objective (minimize sum of objectives)."""
            dim = len(x0)
            lo = np.array([b[0] for b in bounds], dtype=float)
            hi = np.array([b[1] for b in bounds], dtype=float)
            
            def obj(x):
                x = np.clip(x, lo, hi)
                f = problem(x)
                return np.sum(f)
            
            simplex = [x0.copy()]
            for d in range(dim):
                dx = x0.copy()
                dx[d] += 0.05 * (hi[d] - lo[d])
                simplex.append(np.clip(dx, lo, hi))
            
            simplex = np.array(simplex)
            fvals = np.array([obj(s) for s in simplex])
            used_nm = dim + 1
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            
            for _ in range(max_iter):
                if used_nm >= self.budget - used - 5:
                    break
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = obj(xr)
                used_nm += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lo, hi)
                    fe = obj(xe)
                    used_nm += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = obj(xc)
                    used_nm += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = obj(simplex[i])
                            used_nm += 1
                        if used_nm >= self.budget - used - 5:
                            break
                
                if np.max(np.abs(simplex - simplex[0])) < 1e-8:
                    break
            
            best_idx = np.argmin(fvals)
            used += used_nm
            return simplex[best_idx]

        pop_size = 25
        while used < self.budget and time.time() < deadline:
            # Create offspring via binary tournament + crossover + mutation
            fronts = non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for r, fr in enumerate(fronts):
                for idx in fr:
                    ranks[idx] = r

            children_X = []
            children_F = []
            n_off = min(pop_size, self.budget - used)
            for _ in range(n_off):
                if time.time() > deadline:
                    break
                # Binary tournament selection
                i1, i2 = rng.integers(0, len(F), size=2)
                if ranks[i1] < ranks[i2]:
                    parent1 = i1
                elif ranks[i2] < ranks[i1]:
                    parent1 = i2
                else:
                    parent1 = i1 if rng.random() < 0.5 else i2
                i3, i4 = rng.integers(0, len(F), size=2)
                if ranks[i3] < ranks[i4]:
                    parent2 = i3
                elif ranks[i4] < ranks[i3]:
                    parent2 = i4
                else:
                    parent2 = i3 if rng.random() < 0.5 else i4

                # Simulated binary crossover
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            child[d] = X[parent1, d]
                        else:
                            child[d] = X[parent2, d]
                else:
                    child = X[parent1].copy()

                # Adaptive mutation
                mut_rate = 0.4
                sigma = 0.15 + 0.25 * (1.0 - used / float(self.budget))
                for d in range(self.dim):
                    if rng.random() < mut_rate:
                        child[d] += sigma * span[d] * rng.standard_normal()
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            cX = np.array(children_X)
            cF = np.array(children_F)

            # Combine and select
            allX = np.vstack([X, cX])
            allF = np.vstack([F, cF])
            fronts = non_dominated_sort(allF)

            new_idx = []
            for fr in fronts:
                if len(new_idx) + len(fr) <= pop_size:
                    new_idx.extend(fr)
                else:
                    need = pop_size - len(new_idx)
                    cd = crowding_distance(allF, fr)
                    order = np.argsort(-cd)
                    new_idx.extend([fr[i] for i in order[:need]])
                    break

            X = allX[new_idx]
            F = allF[new_idx]

        # Final polish: local search around best non-dominated points
        if used < self.budget - 5 and time.time() < deadline:
            fronts = non_dominated_sort(F)
            # Get top non-dominated points
            if fronts and len(fronts[0]) > 0:
                nd_indices = fronts[0][:min(5, len(fronts[0]))]
                for idx in nd_indices:
                    if used >= self.budget - 5 or time.time() > deadline:
                        break
                    x0 = X[idx].copy()
                    x_new = nelder_mead(problem, x0, bounds, max_iter=30)
                    if x_new is not None:
                        f_new = problem(x_new)
                        used += 1  # already counted in nelder_mead, but ensure consistency
                        # Add to population if non-dominated
                        dominated = False
                        for j in range(len(F)):
                            if dominates(F[j], f_new):
                                dominated = True
                                break
                        if not dominated:
                            X = np.vstack([X, x_new])
                            F = np.vstack([F, f_new])

        return X
# EVOLVE-BLOCK-END
