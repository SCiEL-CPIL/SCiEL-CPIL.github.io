"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using Island Model MODE (Multi-Objective Differential Evolution)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        n_islands = 4
        pop_size = 15  # 15 * 4 = 60 total evaluations per generation cycle
        total_pop = n_islands * pop_size

        # Initialize islands
        islands_X = []
        islands_F = []
        used = 0

        for _ in range(n_islands):
            X = lo + (hi - lo) * rng.random((pop_size, self.dim))
            F = np.array([problem(x) for x in X])
            used += pop_size
            islands_X.append(X)
            islands_F.append(F)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    fronts.append([i])
            front_idx = 0
            while front_idx < len(fronts):
                next_front = []
                for i in fronts[front_idx]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front_idx += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            cd = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                sorted_idx = sorted(indices, key=lambda i: F[i, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max - f_min > 1e-15:
                    for k in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[k]] += (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / (f_max - f_min)
            return cd

        def select_next_generation(X, F, pop_size):
            """Select next generation based on non-dominated sorting and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                    break
            return np.array(next_X), np.array(next_F)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            for i in range(n_islands):
                if used >= self.budget:
                    break

                X = islands_X[i]
                F = islands_F[i]

                # Differential Evolution Step
                offspring_X = []
                offspring_F = []

                for _ in range(pop_size):
                    # Select 3 distinct random parents
                    p1, p2, p3 = rng.choice(pop_size, size=3, replace=False)
                    
                    # DE/rand/1/bin strategy
                    # V = X[p1] + F * (X[p2] - X[p3])
                    F_factor = 0.8
                    V = X[p1] + F_factor * (X[p2] - X[p3])
                    
                    # Boundary handling: reflect or clip
                    # Clipping is simpler and effective for [0,1]
                    V = np.clip(V, lo, hi)

                    # Mutation with small probability to add diversity
                    if rng.random() < 0.1:
                        V += 0.1 * rng.standard_normal(self.dim)
                        V = np.clip(V, lo, hi)

                    offspring_X.append(V)
                    
                    if used < self.budget:
                        f_val = problem(V)
                        offspring_F.append(f_val)
                        used += 1
                    else:
                        # If budget exhausted, we can't evaluate, but we need to fill arrays for logic?
                        # Actually, if budget is exhausted, we should break out.
                        # But we are in a loop creating offspring. Let's break if no budget.
                        # The check `if used < self.budget` before append handles it.
                        # However, if we run out of budget mid-island, we should stop processing this island.
                        break

                if not offspring_X:
                    continue

                offspring_X = np.array(offspring_X)
                offspring_F = np.array(offspring_F)

                # Combine parents and offspring
                combined_X = np.vstack([X, offspring_X])
                combined_F = np.vstack([F, offspring_F])

                # Select next generation
                new_X, new_F = select_next_generation(combined_X, combined_F, pop_size)
                islands_X[i] = new_X
                islands_F[i] = new_F

            # Migration: After each full cycle of islands, migrate best individuals
            # Simple ring migration: each island sends its best (by hypervolume contribution or just non-dominated front size) to next
            # For simplicity and speed, we can just swap a few random individuals or the best non-dominated ones.
            # Let's do a simple migration of the best individual (lowest sum of objectives) from each island to the next.
            for i in range(n_islands):
                if used >= self.budget:
                    break
                # Find "best" individual in island i (simple scalarization)
                # Or just take the first individual from the non-dominated front
                fronts = fast_non_dominated_sort(islands_F[i])
                if fronts:
                    best_idx = fronts[0][0]
                    # Send to next island
                    next_i = (i + 1) % n_islands
                    # Replace the worst individual in next island?
                    # Or just replace a random one?
                    # Let's replace the one with the highest crowding distance? No, that's expensive.
                    # Let's replace a random individual in the next island to maintain diversity.
                    if rng.random() < 0.5: # 50% chance to migrate to save time? No, migration is free (no eval).
                        # Replace random index in next island
                        replace_idx = rng.integers(pop_size)
                        islands_X[next_i][replace_idx] = islands_X[i][best_idx]
                        islands_F[next_i][replace_idx] = islands_F[i][best_idx]

        # Return the best set of points found (all non-dominated points from all islands)
        all_X = np.vstack(islands_X)
        all_F = np.vstack(islands_F)
        
        # Filter to only non-dominated points for return? 
        # The contract says "Every returned objective vector is recorded". 
        # The return value is ignored. So we can return anything, but usually it's the final population.
        # However, the scoring uses "all recorded vectors". 
        # The evaluator records *every* objective vector returned by problem(x).
        # So the return value of __call__ does not affect the score directly, only the calls to problem() do.
        # But the contract says "return value of __call__ is ignored".
        # So I will just return the last island's population or something.
        
        return all_X
# EVOLVE-BLOCK-END
