"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


class Optimizer:
    """Multi-objective optimizer using LHS initialization, NSGA-II, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Latin Hypercube Sampling for initial population
        pop_size = min(24, budget // 8)
        pop_size = max(pop_size, 8)

        # LHS initialization
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = (perm + rng.random(pop_size)) / pop_size
        X = lo + (hi - lo) * X

        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # set of points dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1

            # Assign remaining points to last front
            if len(fronts) < n:
                remaining = [i for i in range(n) if rank[i] == 0 and i not in fronts[0]]
                if remaining:
                    for i in remaining:
                        rank[i] = len(fronts)
                    fronts.append(remaining)

            return fronts, rank

        def crowding_distance(F, front):
            if len(front) <= 2:
                return {i: float('inf') for i in front}

            dist = {i: 0.0 for i in front}
            for m in range(n_obj):
                vals = sorted([(F[i][m], i) for i in front])
                dist[vals[0][1]] = float('inf')
                dist[vals[-1][1]] = float('inf')
                fmin = vals[0][0]
                fmax = vals[-1][0]
                if fmax - fmin < 1e-12:
                    continue
                for k in range(1, len(vals) - 1):
                    dist[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / (fmax - fmin)
            return dist

        # Main evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            fronts, rank = fast_non_dominated_sort(F)

            # Selection: binary tournament on (rank, crowding distance)
            # Compute crowding distances for the first front
            cd = {}
            for front in fronts:
                cd.update(crowding_distance(F, front))

            def tournament():
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if cd.get(i, 0) > cd.get(j, 0) else j

            # Generate children
            n_children = min(pop_size, budget - used)
            children_X = []
            children_F = []

            for _ in range(n_children):
                if time.time() > deadline:
                    break

                p1 = tournament()
                p2 = tournament()

                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            # Simulated binary crossover
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 15))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 15))
                            if X[p1][d] <= X[p2][d]:
                                child[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                            else:
                                child[d] = 0.5 * ((1 + beta) * X[p2][d] + (1 - beta) * X[p1][d])
                        else:
                            child[d] = X[p1][d] if rng.random() < 0.5 else X[p2][d]
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 0.9:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (2 + 20)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            X_all = np.vstack([X, children_X])
            F_all = np.vstack([F, children_F])

            # Non-dominated sort on combined population
            fronts_all, rank_all = fast_non_dominated_sort(F_all)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd_front = crowding_distance(F_all, front)
                    sorted_front = sorted(front, key=lambda i: cd_front.get(i, 0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    break

            if len(new_X) < pop_size:
                # Fill with best remaining
                for front in fronts_all:
                    for idx in front:
                        if len(new_X) >= pop_size:
                            break
                        if not any(np.allclose(X_all[idx], nx) for nx in new_X):
                            new_X.append(X_all[idx])
                            new_F.append(F_all[idx])
                    if len(new_X) >= pop_size:
                        break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        # Local refinement phase: use remaining budget for local search on best points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Sort by sum of objectives, take top few for local refinement
            order = np.argsort(np.sum(F, axis=1))
            top_indices = order[:min(5, len(F))]

            for idx in top_indices:
                if time.time() > deadline or remaining <= 0:
                    break

                # Local search around this point
                best_x = X[idx].copy()
                best_f = F[idx].copy()

                # Adaptive step size
                step = 0.05 * (hi - lo)

                for ls_iter in range(min(20, remaining)):
                    if time.time() > deadline:
                        break

                    # Try random perturbations
                    improved = False
                    for _ in range(3):
                        perturbation = rng.standard_normal(dim)
                        new_x = best_x + step * perturbation
                        new_x = np.clip(new_x, lo, hi)
                        new_f = problem(new_x)
                        remaining -= 1

                        # Check if new point dominates or is non-dominated with better sum
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x
                            best_f = new_f
                            improved = True
                            break
                        elif np.sum(new_f) < np.sum(best_f) - 1e-6:
                            # Accept if significantly better in aggregate
                            if not any(np.all(f <= new_f) and np.any(f < new_f) for f in F):
                                best_x = new_x
                                best_f = new_f
                                improved = True
                                break

                    if improved:
                        step *= 0.8
                    else:
                        step *= 0.5
                        if step < 0.001 * (hi - lo).min():
                            break

                # Update population
                X[idx] = best_x
                F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
