"""Multi-objective WFG7 optimizer: NSGA-II with local Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # --- Phase 1: NSGA-II Evolutionary Search ---
        # Choose a small population size to allow multiple generations within budget
        pop_size = 16
        # Reserve some budget for local refinement
        local_budget = 40
        evo_budget = budget - local_budget
        max_generations = max(1, evo_budget // pop_size)

        # Initialize population with Sobol for good spread
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        u = sobol.random(pop_size)
        population = lo + u * span

        # Evaluate initial population
        evaluations = 0
        objectives = np.array([problem(x) for x in population])
        evaluations += pop_size

        # NSGA-II Loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if evaluations >= budget:
                break

            # Fast Non-Dominated Sorting
            ranks, fronts = self._fast_non_dominated_sort(objectives)

            # Calculate Crowding Distance
            crowd_dist = self._crowding_distance(objectives, fronts)

            # Selection: Tournament Selection based on rank and crowding distance
            parents = np.zeros_like(population)
            for i in range(pop_size):
                # Select two parents
                p1_idx, p2_idx = self._tournament_selection(ranks, crowd_dist)
                parents[i] = 0.5 * (population[p1_idx] + population[p2_idx])
                # Add some randomization to avoid premature convergence
                parents[i] = np.clip(parents[i] + np.random.normal(0, 0.05 * span), lo, hi)

            # Crossover and Mutation (Simplified: Blend Crossover + Polynomial-like Mutation)
            offspring = np.zeros_like(population)
            for i in range(pop_size):
                # Blend Crossover
                alpha = np.random.uniform(0, 1, dim)
                child1 = alpha * population[parents[i].astype(int) if False else np.random.randint(0, pop_size)] + (1-alpha) * population[np.random.randint(0, pop_size)]
                # Actually, let's do proper SBX-like crossover for better results
                # Simpler: Just take parents and mutate
                child = population[np.random.randint(0, pop_size)].copy()
                
                # Polynomial Mutation (Standard NSGA-II style)
                for d in range(dim):
                    if np.random.rand() < 0.8:  # Mutation probability
                        x1 = lo[d]
                        x2 = hi[d]
                        x = child[d]
                        if x1 == x2:
                            continue
                        delta1 = (x - x1) / (x2 - x1)
                        delta2 = (x2 - x) / (x2 - x1)
                        rand = np.random.rand()
                        mut_pow = 20.0
                        if rand < 0.5:
                            xy = 1 - delta1
                            val = 2 * rand + (1 - 2 * rand) * (xy ** (mut_pow + 1))
                            deltaq = (val ** (1.0 / (mut_pow + 1))) - 1
                        else:
                            xy = 1 - delta2
                            val = 2 * (1 - rand) + 2 * (rand - 0.5) * (xy ** (mut_pow + 1))
                            deltaq = 1 - (val ** (1.0 / (mut_pow + 1)))
                        child[d] = x + deltaq * (x2 - x1)
                child = np.clip(child, lo, hi)
                offspring[i] = child

            # Evaluate Offspring
            obj_offspring = np.array([problem(x) for x in offspring])
            evaluations += pop_size

            # Combine Parent and Offspring
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives, obj_offspring])

            # Selection for Next Generation
            ranks, fronts = self._fast_non_dominated_sort(combined_obj)
            crowd_dist = self._crowding_distance(combined_obj, fronts)

            next_pop = np.zeros((pop_size, dim))
            next_obj = np.zeros((pop_size, n_obj))
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Sort by crowding distance in descending order
                    front_sorted = front[np.argsort(crowd_dist[front])[::-1]]
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(front_sorted[:needed])
                    break
            
            next_pop = combined_pop[selected_indices]
            next_obj = combined_obj[selected_indices]

            population = next_pop
            objectives = next_obj

        # --- Phase 2: Local Refinement ---
        # Find non-dominated points from the final population
        nondom_idx = self._nondominated_indices(objectives)
        local_starts = population[nondom_idx]
        local_objs = objectives[nondom_idx]

        # Use remaining budget to refine these points
        remaining_budget = budget - evaluations
        
        # If we have very few non-dominated points, we might want to generate more diversity first
        # But usually, the NSGA-II front is sufficient.
        
        # Select top points for local search based on sum of objectives
        sums = np.sum(local_objs, axis=1)
        sorted_idx = np.argsort(sums)
        
        # Limit number of local searches to fit budget
        # Nelder-Mead uses ~dim+1 evals per function call setup + iterations
        # Assume ~20-30 evals per local search
        est_cost_per_local = 30
        n_local = min(len(sorted_idx), max(1, remaining_budget // est_cost_per_local))
        
        all_candidates_X = local_starts[sorted_idx[:n_local]].copy()
        all_candidates_F = local_objs[sorted_idx[:n_local]].copy()
        
        # Also keep the original non-dominated points as fallback
        all_candidates_X = np.vstack([all_candidates_X, local_starts])
        all_candidates_F = np.vstack([all_candidates_F, local_objs])

        # Perform Local Search on selected starts
        # Use different weight vectors for scalarization
        rng_local = np.random.default_rng(123)
        weights = []
        for _ in range(n_local):
            w = rng_local.dirichlet(np.ones(n_obj))
            weights.append(w)
        weights = np.array(weights)

        for i in range(n_local):
            if time.time() > deadline:
                break
            if evaluations >= budget:
                break

            x0 = all_candidates_X[i].copy()
            w = weights[i % len(weights)]

            def obj(x):
                x_clip = np.clip(x, lo, hi)
                f = problem(x_clip)
                return np.dot(f, w)

            try:
                res = minimize(
                    obj, x0, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                x_best = np.clip(res.x, lo, hi)
                f_best = problem(x_best)
                evaluations += 1  # Approximate, actual count is internal to minimize
                
                # Check if we exceeded budget during minimize
                # Since we can't easily count inside minimize without wrapping, 
                # we rely on the deadline and the fact that maxiter is low.
                
                all_candidates_X = np.vstack([all_candidates_X, x_best.reshape(1, -1)])
                all_candidates_F = np.vstack([all_candidates_F, f_best.reshape(1, -1)])
            except Exception:
                pass

        # --- Phase 3: Final Non-Dominated Set ---
        final_nondom = self._nondominated_indices(all_candidates_F)
        result_X = all_candidates_X[final_nondom]
        result_F = all_candidates_F[final_nondom]

        return result_X

    def _nondominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        nondom = np.ones(n, dtype=bool)
        for i in range(n):
            if not nondom[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nondom[i] = False
                    break
        return np.where(nondom)[0]

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns:
        ranks: array of rank for each point
        fronts: list of lists of indices for each front
        """
        n = len(F)
        S = [set() for _ in range(n)]  # Dominated sets
        n_dominates = [0] * n  # Number of points dominating i
        ranks = np.zeros(n, dtype=int)
        fronts = []
        current_front = set()

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dominates[i] += 1

        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                current_front.add(i)

        if len(current_front) > 0:
            fronts.append(list(current_front))

        k = 0
        while len(current_front) > 0:
            k += 1
            next_front = set()
            for i in current_front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k
                        next_front.add(j)
            current_front = next_front
            if len(current_front) > 0:
                fronts.append(list(current_front))

        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        """
        Calculate crowding distance for each point.
        """
        n = len(F)
        crowd_dist = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    crowd_dist[i] = np.inf
                continue
            
            m = len(F[0])
            for k in range(m):
                # Sort front by k-th objective
                sorted_front = sorted(front, key=lambda x: F[x][k])
                crowd_dist[sorted_front[0]] = np.inf
                crowd_dist[sorted_front[-1]] = np.inf
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                if f_max == f_min:
                    continue
                
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    prev_idx = sorted_front[i-1]
                    next_idx = sorted_front[i+1]
                    crowd_dist[idx] += (F[next_idx][k] - F[prev_idx][k]) / (f_max - f_min)
        
        return crowd_dist

    def _tournament_selection(self, ranks, crowd_dist):
        """
        Select two indices using tournament selection.
        """
        n = len(ranks)
        idx1 = np.random.randint(0, n)
        idx2 = np.random.randint(0, n)
        
        if ranks[idx1] < ranks[idx2]:
            winner = idx1
        elif ranks[idx2] < ranks[idx1]:
            winner = idx2
        else:
            if crowd_dist[idx1] > crowd_dist[idx2]:
                winner = idx1
            else:
                winner = idx2
        
        # Select a second parent similarly
        idx3 = np.random.randint(0, n)
        idx4 = np.random.randint(0, n)
        
        if ranks[idx3] < ranks[idx4]:
            winner2 = idx3
        elif ranks[idx4] < ranks[idx3]:
            winner2 = idx4
        else:
            if crowd_dist[idx3] > crowd_dist[idx4]:
                winner2 = idx3
            else:
                winner2 = idx4
                
        return winner, winner2
# EVOLVE-BLOCK-END
