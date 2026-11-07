# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        mid = (hi + lo) / 2.0

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # WFG4 analysis:
        # The WFG4 problem is a constrained multi-objective problem.
        # For the standard 3-objective WFG4 with 12 variables:
        # - The Pareto front is a line segment (or curve) in objective space.
        # - Key insight: WFG4 has a "constraint" that the sum of variables 
        #   must be at least some value for the problem to be feasible, but
        #   the standard formulation uses a transformation.
        #
        # Actually, let me reconsider. WFG4 in the DTLZ/WFG suite:
        # - It's a 3-objective problem with a convex Pareto front.
        # - The Pareto front is a curve in the (f1, f2, f3) space.
        # - Reference point is [3.0, 5.0, 7.0].
        #
        # Strategy: Use NSGA-II with:
        # 1. Good initial population spread across the bounds
        # 2. Non-dominated sorting
        # 3. Crowding distance for diversity
        # 4. SBX crossover + polynomial mutation
        # 5. Budget management

        def non_dominated_sort(F):
            """
            F: (N, M) array of objective values
            Returns: list of fronts, each front is a list of indices
            """
            N = len(F)
            M = F.shape[1] if F.ndim > 1 else 1
            if M == 1:
                M = 1
            else:
                M = F.shape[1]
            
            # domination[i] = list of indices that dominate i
            dominated = [[] for _ in range(N)]
            # nd_count[i] = number of solutions that dominate i
            nd_count = np.zeros(N, dtype=int)
            
            fronts = []
            
            for i in range(N):
                for j in range(i + 1, N):
                    # Check if i dominates j
                    i_dom_j = np.all(F[i] <= F[j]) and np.any(F[i] < F[j])
                    j_dom_i = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                    if i_dom_j:
                        dominated[i].append(j)
                        nd_count[j] += 1
                    elif j_dom_i:
                        dominated[j].append(i)
                        nd_count[i] += 1
            
            # First front: solutions with nd_count == 0
            current_front = [i for i in range(N) if nd_count[i] == 0]
            if not current_front:
                # All solutions dominate each other? Return all in one front
                return [[i for i in range(N)]]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated[i]:
                        nd_count[j] -= 1
                        if nd_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # If some solutions were never added (shouldn't happen), add them
            added = set()
            for front in fronts:
                for idx in front:
                    added.add(idx)
            remaining = [i for i in range(N) if i not in added]
            if remaining:
                fronts.append(remaining)
            
            return fronts

        def crowding_distance(F, front_indices):
            """
            Compute crowding distance for solutions in front_indices.
            F: (N, M) array
            front_indices: list of indices
            Returns: dict mapping index -> crowding distance
            """
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            distances = {i: 0.0 for i in front_indices}
            M = F.shape[1] if F.ndim > 1 else 1
            
            for m in range(M):
                obj_vals = F[front_indices, m]
                sorted_idx = np.argsort(obj_vals)
                sorted_front = [front_indices[i] for i in sorted_idx]
                
                min_val = obj_vals[sorted_idx[0]]
                max_val = obj_vals[sorted_idx[-1]]
                
                if max_val == min_val:
                    continue
                
                # Edge points get infinite distance
                distances[sorted_front[0]] = float('inf')
                distances[sorted_front[-1]] = float('inf')
                
                for k in range(1, len(sorted_front) - 1):
                    distances[sorted_front[k]] += (
                        obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]
                    ) / (max_val - min_val)
            
            return distances

        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            """Simulated Binary Crossover"""
            n = len(x1)
            x3 = np.zeros(n)
            x4 = np.zeros(n)
            
            for i in range(n):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if np.random.rand() < 0.5:
                            beta = 1.0 + (2.0 * min(x1[i], x2[i]) - lo[i]) / (x1[i] - x2[i])
                            alpha = 2.0 - beta ** (-1.0 - 2.0 * eta)
                            rand = np.random.rand()
                            if rand <= 1.0 / alpha:
                                beta_q = (rand * alpha) ** (1.0 / (1.0 + 2.0 * eta))
                            else:
                                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (1.0 + 2.0 * eta))
                            x3[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))
                            x4[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))
                        else:
                            beta = 1.0 + (2.0 * (hi[i] - max(x1[i], x2[i])) / (x1[i] - x2[i]))
                            alpha = 2.0 - beta ** (-1.0 - 2.0 * eta)
                            rand = np.random.rand()
                            if rand <= 1.0 / alpha:
                                beta_q = (rand * alpha) ** (1.0 / (1.0 + 2.0 * eta))
                            else:
                                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (1.0 + 2.0 * eta))
                            x3[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))
                            x4[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))
                    else:
                        x3[i] = x1[i]
                        x4[i] = x2[i]
                else:
                    x3[i] = x2[i]
                    x4[i] = x1[i]
            
            x3 = np.clip(x3, lo, hi)
            x4 = np.clip(x4, lo, hi)
            return x3, x4

        def polynomial_mutation(x, lo, hi, eta=20.0, prob_mut=None):
            """Polynomial Mutation"""
            if prob_mut is None:
                prob_mut = 1.0 / len(x)
            
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < prob_mut:
                    if x_mut[i] <= lo[i] or x_mut[i] >= hi[i]:
                        x_mut[i] = np.random.uniform(lo[i], hi[i])
                    else:
                        delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                        delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                        rand = np.random.rand()
                        if rand < 0.5:
                            xy = 1.0 - delta1
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (1.0 + 2.0 * eta))
                            delta_q = (val ** (1.0 / (1.0 + 2.0 * eta)) - 1.0)
                            x_mut[i] = x_mut[i] + delta_q * (hi[i] - lo[i])
                        else:
                            xy = 1.0 - delta2
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (1.0 + 2.0 * eta))
                            delta_q = 1.0 - (val ** (1.0 / (1.0 + 2.0 * eta)))
                            x_mut[i] = x_mut[i] + delta_q * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            
            return x_mut

        # Initialize population
        pop_size = min(100, budget // 2)
        if pop_size < 10:
            pop_size = 10
        
        # Use Latin Hypercube or stratified sampling for better initial spread
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        
        # Evaluate initial population
        F = np.zeros((pop_size, n_obj))
        all_X = []
        all_F = []
        evals_used = 0
        
        for i in range(pop_size):
            if evals_used >= budget or time.time() > deadline:
                break
            f = problem(X[i])
            F[i] = f
            evals_used += 1
            all_X.append(X[i].copy())
            all_F.append(np.array(f))
        
        if evals_used == 0:
            # Fallback: at least one evaluation
            x0 = np.random.uniform(lo, hi)
            f0 = problem(x0)
            all_X.append(x0.copy())
            all_F.append(np.array(f0))
            return all_X
        
        # NSGA-II loop
        # Sort population by non-dominated front and crowding distance
        def sort_population(X_pop, F_pop):
            fronts = non_dominated_sort(F_pop)
            sorted_indices = []
            for front in fronts:
                cd = crowding_distance(F_pop, front)
                front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                sorted_indices.extend(front_sorted)
            return sorted_indices

        def tournament_select(X_pop, F_pop, pop_size_current, k=2):
            """Tournament selection based on rank and crowding distance"""
            sorted_idx = sort_population(X_pop, F_pop)
            rank = {idx: i for i, idx in enumerate(sorted_idx)}
            
            # Get crowding distances for each front
            fronts = non_dominated_sort(F_pop)
            cd_map = {}
            for front in fronts:
                cd = crowding_distance(F_pop, front)
                cd_map.update(cd)
            
            selected = []
            for _ in range(pop_size_current):
                candidates = np.random.randint(0, len(X_pop), size=k)
                best = candidates[0]
                for c in candidates[1:]:
                    if rank[c] < rank[best]:
                        best = c
                    elif rank[c] == rank[best]:
                        if cd_map.get(c, 0) > cd_map.get(best, 0):
                            best = c
                selected.append(best)
            return selected

        # Main loop
        while evals_used < budget and time.time() < deadline:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(pop_size, budget - evals_used)
            
            for _ in range(n_offspring):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Select parents
                parents = tournament_select(X, F, len(X), k=2)
                p1 = X[parents[0]]
                p2 = X[parents[1]]
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2, lo, hi, eta=20.0)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi, eta=20.0, prob_mut=1.0/dim)
                c2 = polynomial_mutation(c2, lo, hi, eta=20.0, prob_mut=1.0/dim)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            # Evaluate offspring
            for x_off in offspring_X:
                if evals_used >= budget or time.time() > deadline:
                    break
                f_off = problem(x_off)
                evals_used += 1
                offspring_F.append(f_off)
                all_X.append(x_off.copy())
                all_F.append(np.array(f_off))
            
            if not offspring_F:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)
            
            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    cd = crowding_distance(combined_F, front)
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in front_sorted[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            if not new_X:
                break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return all_X
# EVOLVE-BLOCK-END
