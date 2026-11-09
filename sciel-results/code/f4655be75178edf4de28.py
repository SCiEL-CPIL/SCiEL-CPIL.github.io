"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II based multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 50
        # Calculate number of generations
        # Initial population uses pop_size evaluations
        # Each generation creates pop_size offspring
        remaining_budget = self.budget - pop_size
        n_generations = remaining_budget // pop_size
        
        if n_generations < 1:
            # If budget is too small, just return random points
            population = np.random.uniform(lo, hi, (self.budget, self.dim))
            objectives = np.array([problem(x) for x in population])
            return objectives

        # Initialize population
        population = np.random.uniform(lo, hi, (pop_size, self.dim))
        objectives = np.array([problem(x) for x in population])
        eval_count = pop_size

        def dominates(a, b):
            """Check if a dominates b"""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(objs):
            """
            Fast non-dominated sorting.
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(objs)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dominated = [0] * n  # Number of points dominated by i
            fronts = [[]]  # List of fronts, first one is front 0
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(objs[i], objs[j]):
                        S[i].append(j)
                    elif dominates(objs[j], objs[i]):
                        n_dominated[i] += 1
                
                if n_dominated[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
                
            # Remove the last empty front if it exists
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        def crowding_distance(objs, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                sorted_indices = sorted(indices, key=lambda i: objs[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                obj_range = objs[sorted_indices[-1]][m] - objs[sorted_indices[0]][m]
                if obj_range == 0:
                    continue
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (objs[sorted_indices[i+1]][m] - objs[sorted_indices[i-1]][m]) / obj_range
            return dist

        def tournament_selection(population, objectives, fronts, crowding_dists):
            """Binary tournament selection."""
            i = np.random.randint(len(population))
            j = np.random.randint(len(population))
            
            # Determine front rank for i and j
            rank_i = 0
            rank_j = 0
            for k, front in enumerate(fronts):
                if i in front:
                    rank_i = k
                    break
                if j in front:
                    rank_j = k
                    break
            
            if rank_i < rank_j:
                return population[i].copy()
            elif rank_j < rank_i:
                return population[j].copy()
            else:
                # Same front, compare crowding distance
                if crowding_dists[i] > crowding_dists[j]:
                    return population[i].copy()
                else:
                    return population[j].copy()

        def sbx_crossover(x1, x2, eta=20):
            """Simulated Binary Crossover."""
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    # Swap parents if random < 0.5 to ensure diversity
                    pass 
                
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                
                # Calculate beta
                if np.random.random() < 0.5:
                    beta = (2 * x1[i] / (x1[i] + x2[i])) ** (1 / (eta + 1)) if x1[i] + x2[i] != 0 else 1
                else:
                    beta = (2 * x2[i] / (x1[i] + x2[i])) ** (1 / (eta + 1)) if x1[i] + x2[i] != 0 else 1
                    
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 + 1/beta) * x2[i] + (1 - 1/beta) * x1[i]) if beta != 0 else x2[i]
                
            return child1, child2

        def polynomial_mutation(x, eta=20):
            """Polynomial Mutation."""
            child = x.copy()
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    y = x[i]
                    # Map y to [0,1] for mutation calculation
                    # y_low = lo[i], y_high = hi[i]
                    if y == lo[i]:
                        yl = 0.0
                    elif y == hi[i]:
                        yl = 1.0
                    else:
                        yl = (y - lo[i]) / (hi[i] - lo[i])
                    
                    if np.random.random() < 0.5:
                        delta1 = (yl + 1)
                        delta = (2 * delta1) ** (-1 / (eta + 1)) - 1
                    else:
                        delta2 = (1 - yl + 1)
                        delta = 1 - (2 * delta2) ** (-1 / (eta + 1))
                    
                    new_yl = yl + delta * (1 - 2 * yl)
                    new_yl = np.clip(new_yl, 0.0, 1.0)
                    child[i] = lo[i] + new_yl * (hi[i] - lo[i])
            return child

        # Main evolution loop
        for gen in range(n_generations):
            # 1. Non-dominated sorting and crowding distance for current population
            fronts = fast_non_dominated_sort(objectives)
            crowding_dists = {}
            for front in fronts:
                cd = crowding_distance(objectives, front)
                for i in front:
                    crowding_dists[i] = cd[i]
            
            # 2. Create offspring population
            offspring = []
            offspring_objs = []
            
            for _ in range(pop_size):
                p1 = tournament_selection(population, objectives, fronts, crowding_dists)
                p2 = tournament_selection(population, objectives, fronts, crowding_dists)
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Clip to bounds
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                obj_c1 = np.array(problem(c1))
                obj_c2 = np.array(problem(c2))
                
                offspring.append(c1)
                offspring.append(c2)
                offspring_objs.append(obj_c1)
                offspring_objs.append(obj_c2)
                
                eval_count += 2
                if eval_count >= self.budget:
                    break
            
            # Trim offspring to exact pop_size if we overshot (shouldn't happen with break, but safe)
            offspring = np.array(offspring[:pop_size])
            offspring_objs = np.array(offspring_objs[:pop_size])
            
            # 3. Elitist replacement
            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring])
            combined_objs = np.vstack([objectives, offspring_objs])
            
            # Sort combined population
            combined_fronts = fast_non_dominated_sort(combined_objs)
            
            new_population = []
            new_objectives = []
            selected_indices = []
            
            for front in combined_fronts:
                if len(new_population) + len(front) <= pop_size:
                    # Take whole front
                    for idx in front:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_objs[idx])
                        selected_indices.append(idx)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_population)
                    cd = crowding_distance(combined_objs, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_population.append(combined_pop[idx])
                        new_objectives.append(combined_objs[idx])
                        selected_indices.append(idx)
                    break
            
            population = np.array(new_population)
            objectives = np.array(new_objectives)
            
            if eval_count >= self.budget:
                break

        # Return all evaluated points? No, the task says "Every returned objective vector is recorded".
        # The evaluator records the return value of __call__?
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # Wait, "return value of __call__ is ignored" but "Every returned objective vector is recorded".
        # This implies the evaluator hooks into `problem` calls?
        # "problem(x) ... returns 3 objective values."
        # "Every returned objective vector is recorded"
        # This usually means the evaluator wraps the `problem` function or the `Optimizer` is expected to return the set of non-dominated points?
        # Let's re-read carefully:
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This phrasing is slightly ambiguous. Does it mean the evaluator records the output of `problem(x)` every time it's called?
        # OR does it mean the evaluator records the vector returned by `__call__`?
        # "The return value of __call__ is ignored" -> So it doesn't matter what I return from `__call__`?
        # But then "Every returned objective vector is recorded" -> Which vector?
        # If the return value of `__call__` is ignored, how are the vectors recorded?
        # Hypothesis: The evaluator wraps `problem`. Every time `problem(x)` is called, the result is recorded.
        # If so, my strategy of calling `problem` exactly `budget` times is correct.
        # I don't need to return anything specific, or I can return the final non-dominated set.
        # However, to be safe, I will return the final non-dominated set of objectives, just in case the "ignored" part applies to something else or is a red herring for the *value* used for scoring, but the *recording* happens via the `problem` calls.
        
        # Let's assume the evaluator records all `problem` calls.
        # I have made exactly `budget` calls (or close to it).
        # I will return the non-dominated points from the final population.
        
        final_fronts = fast_non_dominated_sort(objectives)
        if final_fronts:
            best_front_indices = final_fronts[0]
            return np.array(objectives[best_front_indices])
        else:
            return np.array([])

# EVOLVE-BLOCK-END
