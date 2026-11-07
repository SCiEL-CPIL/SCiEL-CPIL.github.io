import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0  # Safety margin
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # All evaluated points (X, F)
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def fast_nondominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # Number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # If p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # If q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
            
            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = rank[p] + 1
                            next_front.append(q)
                current_front = next_front
                
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for a front.
            Returns a dictionary mapping index to distance.
            """
            if not front_indices:
                return {}
            
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            
            for k in range(n_obj):
                # Sort front by k-th objective
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])
                
                # Boundary points have infinite distance
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                
                # Normalize by range
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / f_range
            
            return dist

        def tournament_select(pop, fronts, crowd_dists):
            """
            Binary tournament selection based on rank and crowding distance.
            pop: list of indices
            fronts: list of fronts (lists of indices)
            crowd_dists: dict of index -> crowding distance
            """
            i = np.random.randint(len(pop))
            j = np.random.randint(len(pop))
            
            # Find rank for i and j
            # Since fronts are small, we can just check which front they belong to
            # But it's faster to precompute rank for all points in the union population
            # For simplicity in this small budget, we can just compare their positions in the sorted structure
            # However, standard NSGA-II uses a comparison function.
            
            # Let's implement a helper to get rank and distance
            # To avoid O(N) search for rank every time, we pass a rank_map
            pass # Placeholder, we will use a rank_map in the main loop

        def compare(i, j, rank_map, crowd_dists):
            """
            Returns True if i is better than j.
            """
            rank_i = rank_map.get(i, float('inf'))
            rank_j = rank_map.get(j, float('inf'))
            
            if rank_i < rank_j:
                return True
            elif rank_i > rank_j:
                return False
            else:
                # Same rank, higher crowding distance is better
                dist_i = crowd_dists.get(i, 0.0)
                dist_j = crowd_dists.get(j, 0.0)
                return dist_i > dist_j

        # --- Main Algorithm ---

        # Population size for NSGA-II
        # Budget is 200. 
        # Initial population: 50
        # Generations: (200 - 50) / 50 = 3 generations? 
        # Actually, NSGA-II usually evaluates N offspring per generation.
        # Let's use a smaller population to allow more generations or just use the whole budget as a pool.
        # Given the small budget (200), a standard generational NSGA-II might be too coarse.
        # Alternative: Use a "pool" based approach where we maintain a Pareto archive and generate offspring from it.
        # This is essentially (mu+lambda) ES with Pareto selection, which is often better for small budgets.
        
        # Let's stick to a robust (mu+lambda) strategy with explicit non-dominated sorting and crowding.
        # mu = 20 (archive size)
        # lambda = 20 (offspring per generation)
        # Total evals = 20 (init) + generations * 20
        
        mu = 20
        lam = 20
        
        # 1. Initialize population
        pop_X = []
        pop_F = []
        
        for _ in range(mu):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            pop_X.append(x)
            pop_F.append(f)
            
        # 2. Evolution loop
        gen = 0
        while used < budget and time.time() < deadline:
            gen += 1
            offspring_X = []
            offspring_F = []
            
            # Generate offspring
            for _ in range(lam):
                if used >= budget or time.time() > deadline:
                    break
                
                # Select parents from current population using tournament
                # We need to sort current population first to get ranks
                # For efficiency with small mu, we can just do a simple tournament on the existing list
                # But we need ranks. Let's compute them once per generation for the current population.
                
                # To avoid recomputing ranks for every parent selection (expensive), 
                # we can just pick 2 random parents and crossover/mutate.
                # Standard NSGA-II uses ranks. Let's compute ranks for current pop.
                
                # Optimization: Since mu is small (20), computing ranks is fast.
                # But doing it for every offspring is wasteful. 
                # Let's compute ranks for the current population once before generating offspring.
                # Wait, the population changes? No, in standard NSGA-II, we select from P_t to create Q_t, then combine.
                
                # Let's restructure: 
                # 1. Sort P_t (current pop) to get ranks and crowding.
                # 2. Generate Q_t (offspring) using tournament on P_t.
                # 3. Combine P_t and Q_t.
                # 4. Sort combined to select P_{t+1}.
                
                # This is cleaner.
                
                pass # Logic moved to outer scope

            # --- Revised Loop Structure ---
            
            # Step A: Sort current population (pop_F)
            fronts = fast_nondominated_sort(np.array(pop_F))
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank
            
            # Calculate crowding distances for each front
            crowd_dists = {}
            for front in fronts:
                dists = crowding_distance(np.array(pop_F), front)
                crowd_dists.update(dists)
            
            # Step B: Generate offspring
            for _ in range(lam):
                if used >= budget or time.time() > deadline:
                    break
                
                # Tournament selection
                i = np.random.randint(len(pop_X))
                j = np.random.randint(len(pop_X))
                
                if compare(i, j, rank_map, crowd_dists):
                    parent1 = pop_X[i]
                else:
                    parent1 = pop_X[j]
                
                i = np.random.randint(len(pop_X))
                j = np.random.randint(len(pop_X))
                
                if compare(i, j, rank_map, crowd_dists):
                    parent2 = pop_X[i]
                else:
                    parent2 = pop_X[j]
                
                # Crossover (SBX - Simulated Binary Crossover simplified or just blend)
                # For low dim, simple blend or uniform is fine. Let's use simple blend.
                alpha = np.random.random(dim)
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Mutation
                # Gaussian mutation
                mutation_scale = 0.2 * (hi - lo)
                child = child + np.random.randn(dim) * mutation_scale
                child = np.clip(child, lo, hi)
                
                f = evaluate(child)
                offspring_X.append(child)
                offspring_F.append(f)
            
            # Step C: Combine and Select
            # Combine parents and offspring
            combined_X = pop_X + offspring_X
            combined_F = pop_F + offspring_F
            
            # Sort combined population
            fronts = fast_nondominated_sort(np.array(combined_F))
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank
            
            crowd_dists = {}
            for front in fronts:
                dists = crowding_distance(np.array(combined_F), front)
                crowd_dists.update(dists)
            
            # Select next population of size mu
            new_pop_X = []
            new_pop_F = []
            
            for rank, front in enumerate(fronts):
                if len(new_pop_X) + len(front) <= mu:
                    # Take all
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                else:
                    # Need to select some from this front
                    remaining = mu - len(new_pop_X)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: crowd_dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                    break
            
            pop_X = new_pop_X
            pop_F = new_pop_F

        return all_X
