"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II inspired multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a good fraction of budget
        pop_size = max(30, min(80, budget // 3))

        def lhs_sample(n, dim, lo, hi, rng):
            """Latin Hypercube Sampling for diverse initialization."""
            samples = np.zeros((n, dim))
            for j in range(dim):
                perm = rng.permutation(n)
                samples[:, j] = (perm + rng.random(n)) / n
            return lo + samples * (hi - lo)

        # Initial population: LHS for diversity
        X = lhs_sample(pop_size, dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominance_matrix(F):
            """Compute dominance matrix: M[i,j]=1 if i dominates j."""
            n = len(F)
            M = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            M[i, j] = True
            return M

        def non_dominated_sort(F):
            """NSGA-II non-dominated sorting. Returns list of fronts (indices)."""
            n = len(F)
            M = dominance_matrix(F)
            domination_count = np.sum(M, axis=1)  # how many dominate i
            dominated_set = [np.where(M[i])[0].tolist() for i in range(n)]

            fronts = []
            current_front = np.where(domination_count == 0)[0].tolist()
            fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)

            n = len(front_indices)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-15:
                    dist[:] = np.inf
                    continue
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_val - min_val)
            return dist

        def evaluate_child(parents, rng, lo, hi, dim):
            """Generate a child via mutation from a parent."""
            # Adaptive mutation: start with larger steps, shrink over time
            scale = 0.15 * (hi - lo)
            child = parents + scale * rng.standard_normal(dim)
            # Occasionally do a larger jump for exploration
            if rng.random() < 0.1:
                child = parents + 0.5 * (hi - lo) * rng.standard_normal(dim)
            return np.clip(child, lo, hi)

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Sort population into fronts
            fronts = non_dominated_sort(F)

            # Select parents from the best front(s)
            # Use tournament selection among non-dominated solutions
            non_dom_indices = fronts[0]
            if len(non_dom_indices) < 2:
                non_dom_indices = list(range(len(F)))

            # Pick two parents via tournament
            def tournament_select(indices):
                a = indices[rng.integers(len(indices))]
                b = indices[rng.integers(len(indices))]
                # Prefer lower objective sum as tie-breaker
                if np.sum(F[a]) <= np.sum(F[b]):
                    return a
                return b

            p1 = tournament_select(non_dom_indices)
            p2 = tournament_select(non_dom_indices)

            # Crossover: blend with mutation
            alpha = rng.random()
            child = alpha * X[p1] + (1 - alpha) * X[p2]
            # Add mutation
            scale = 0.1 * (hi - lo)
            child += scale * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual (highest sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Periodic diversity injection: replace a random individual with a new LHS point
            if used % 20 == 0 and used < budget - 5:
                if time.time() > deadline:
                    break
                new_point = lo + (hi - lo) * rng.random(dim)
                new_f = problem(new_point)
                used += 1
                # Replace a randomly chosen non-best individual
                best_idx = int(np.argmin(np.sum(F, axis=1)))
                candidates = [i for i in range(len(F)) if i != best_idx]
                replace_idx = candidates[rng.integers(len(candidates))]
                X[replace_idx] = new_point
                F[replace_idx] = new_f

        return X
# EVOLVE-BLOCK-END
