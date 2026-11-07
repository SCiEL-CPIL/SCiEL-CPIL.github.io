"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        
        # Budget management
        used = 0
        population = [] # List of (x, f)

        def time_left():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if used >= self.budget or not time_left():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            used += 1
            return (x, f)

        def non_dominated_sort(pop):
            """
            Perform non-dominated sorting.
            Returns a list of fronts, where each front is a list of indices into pop.
            """
            n = len(pop)
            if n == 0:
                return [], []
            S = [[] for _ in range(n)]  # Dominated by i
            n_dominates = [0] * n  # Number of solutions dominated by i
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    fi = pop[i][1]
                    fj = pop[j][1]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(fj <= fi) and np.any(fj < fi):
                        S[j].append(i)
                        n_dominates[j] += 1

            fronts = []
            current_front = []
            for i in range(n):
                if n_dominates[i] == 0:
                    rank[i] = 0
                    current_front.append(i)

            if current_front:
                fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                
            return fronts, rank

        def crowding_distance(pop, front_indices):
            """
            Calculate crowding distance for a set of indices in pop.
            Returns a dictionary mapping index to distance.
            """
            distances = {i: 0.0 for i in front_indices}
            if len(front_indices) <= 2:
                return distances
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_indices = sorted(front_indices, key=lambda i: pop[i][1][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = pop[sorted_indices[0]][1][m]
                f_max = pop[sorted_indices[-1]][1][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                
                for k in range(1, len(sorted_indices) - 1):
                    i = sorted_indices[k]
                    i_prev = sorted_indices[k - 1]
                    i_next = sorted_indices[k + 1]
                    distances[i] += (pop[i_next][1][m] - pop[i_prev][1][m]) / f_range
            
            return distances

        def select_tournament(pop, ranks, distances, k=2):
            """
            Tournament selection.
            """
            candidates = rng.choice(len(pop), size=k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if ranks[c] < ranks[best]:
                    best = c
                elif ranks[c] == ranks[best]:
                    # If same rank, prefer larger crowding distance
                    if distances.get(c, 0) > distances.get(best, 0):
                        best = c
            return pop[best]

        # Initialization
        pop_size = 20
        
        # Phase 1: Initialize population
        for _ in range(pop_size):
            if used >= self.budget or not time_left():
                break
            x = lo + (hi - lo) * rng.random(dim)
            res = evaluate(x)
            if res:
                population.append(res)
        
        if not population:
            return []

        # Phase 2: Evolution
        while used < self.budget and time_left():
            # Sort population
            fronts, ranks = non_dominated_sort(population)
            
            # Calculate crowding distance for all fronts
            distances = {}
            for front in fronts:
                d = crowding_distance(population, front)
                distances.update(d)

            # Generate offspring
            offspring = []
            for _ in range(pop_size):
                if used >= self.budget or not time_left():
                    break
                parent1 = select_tournament(population, ranks, distances)
                parent2 = select_tournament(population, ranks, distances)
                
                # Crossover: Simple blend
                if rng.random() < 0.9:
                    alpha = rng.random()
                    child1 = alpha * parent1[0] + (1 - alpha) * parent2[0]
                    child2 = (1 - alpha) * parent1[0] + alpha * parent2[0]
                else:
                    child1 = parent1[0].copy()
                    child2 = parent2[0].copy()
                
                # Mutation: Gaussian
                mutation_rate = 0.1
                mutation_strength = 0.1 # Relative to range
                for d in range(dim):
                    if rng.random() < mutation_rate:
                        child1[d] += rng.normal(0, (hi[d] - lo[d]) * mutation_strength)
                    if rng.random() < mutation_rate:
                        child2[d] += rng.normal(0, (hi[d] - lo[d]) * mutation_strength)
                
                # Clip
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                res1 = evaluate(child1)
                if res1:
                    offspring.append(res1)
                
                res2 = evaluate(child2)
                if res2:
                    offspring.append(res2)
            
            if not offspring:
                break

            # Combine and Select
            combined = population + offspring
            fronts, ranks = non_dominated_sort(combined)
            
            # Calculate distances for combined
            distances = {}
            for front in fronts:
                d = crowding_distance(combined, front)
                distances.update(d)
            
            # Select best pop_size
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    new_population.extend(combined[i] for i in front)
                else:
                    # Need to select from this front
                    # Sort by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: distances[i], reverse=True)
                    needed = pop_size - len(new_population)
                    new_population.extend(combined[i] for i in front_sorted[:needed])
                    break
            
            population = new_population

        return population
# EVOLVE-BLOCK-END
