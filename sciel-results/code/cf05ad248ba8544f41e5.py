"""Multi-objective optimizer for RE34 with deadline guard, LHS init, fast non-dominated sort, crowding, and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        rng = np.random.default_rng(0)

        # LHS initialization
        def lhs(n, dim):
            pts = []
            for d in range(dim):
                perm = np.arange(n)
                rng.shuffle(perm)
                pts.append((perm + rng.random(n)) / n)
            return np.column_stack(pts)

        pop_size = min(40, budget // 4)
        X = lo + range_ * lhs(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_count = [0] * n
            front = [set() for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    front[0].add(i)
            k = 0
            while front[k]:
                next_front = set()
                for i in front[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.add(j)
                k += 1
                front[k] = next_front
            return front

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            idx_list = list(indices)
            dist = np.zeros(len(idx_list))
            F_sub = F[idx_list]
            for m in range(n_obj):
                vals = F_sub[:, m]
                mn, mx = vals.min(), vals.max()
                if mx == mn:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return dist

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            fronts = fast_non_dominated_sort(F)
            
            # Build ranked list
            ranked = []
            for f in fronts:
                if not f:
                    break
                cd = crowding_distance(F, f)
                sorted_idx = sorted(f, key=lambda i: cd[i], reverse=True)
                ranked.extend(sorted_idx)

            # Select parents from top fronts
            parent_pool = ranked[:min(len(ranked), 2 * pop_size)]
            
            # Tournament selection
            def tournament(k=3):
                cands = rng.integers(0, len(parent_pool), size=k)
                best = cands[0]
                for c in cands[1:]:
                    # compare by rank position
                    if parent_pool[c] < parent_pool[best]:
                        best = c
                return parent_pool[best]

            # Generate children
            n_children = min(pop_size, budget - used)
            children_X = []
            children_F = []
            for _ in range(n_children):
                p1 = tournament()
                p2 = tournament()
                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    child = np.zeros(dim)
                    for d in range(dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta + 1))
                        child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                else:
                    child = X[p1].copy()
                # Mutation
                if rng.random() < 0.9:
                    pm = 1.0 / dim
                    for d in range(dim):
                        if rng.random() < pm:
                            u = rng.random()
                            if u <= 0.5:
                                delta = (2 * u) ** (1 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                            child[d] += delta * (hi[d] - lo[d])
                child = np.clip(child, lo, hi)
                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine and select
            X_all = np.vstack([X, children_X])
            F_all = np.vstack([F, children_F])
            fronts = fast_non_dominated_sort(F_all)
            
            new_X = []
            new_F = []
            for f in fronts:
                if not f:
                    break
                if len(new_X) + len(f) <= pop_size:
                    new_X.extend([X_all[i] for i in f])
                    new_F.extend([F_all[i] for i in f])
                else:
                    cd = crowding_distance(F_all, f)
                    remaining = pop_size - len(new_X)
                    sorted_idx = sorted(f, key=lambda i: cd[i], reverse=True)
                    for i in sorted_idx[:remaining]:
                        new_X.append(X_all[i])
                        new_F.append(F_all[i])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Final polish with Nelder-Mead on non-dominated points
        if time.time() < deadline - 10.0:
            fronts = fast_non_dominated_sort(F)
            nd_indices = list(fronts[0]) if fronts[0] else [int(np.argmin(np.sum(F, axis=1)))]
            
            for idx in nd_indices[:10]:
                if time.time() > deadline - 5.0:
                    break
                if used >= budget:
                    break
                
                x0 = X[idx].copy()
                f0 = F[idx].copy()
                
                for m in range(n_obj):
                    if time.time() > deadline - 3.0:
                        break
                    if used >= budget:
                        break
                    
                    def obj(x):
                        return problem(x)[m]
                    
                    try:
                        res = minimize(obj, x0, method='Nelder-Mead', 
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                        if res.fun < f0[m]:
                            # Check if still non-dominated or improves
                            new_x = res.x
                            new_f = np.array([problem(new_x)[k] for k in range(n_obj)])
                            used += 1
                            # Replace if better in at least one objective and not worse in others
                            if np.any(new_f < f0) and np.all(new_f <= f0 + 1e-10):
                                X[idx] = new_x
                                F[idx] = new_f
                                x0 = new_x
                                f0 = new_f
                    except:
                        pass

        return X
# EVOLVE-BLOCK-END
