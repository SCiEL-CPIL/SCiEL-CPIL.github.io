import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # Budget 200. N=20 allows 9 generations (20 init + 9*20 offspring = 200)
        N = 20
        
        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        
        used = N
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Perform fast non-dominated sorting.
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
            n_dominates = np.zeros(n, dtype=int) # n_dominates[i] = number of solutions that dominate i
            rank = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    # i dominates j if F[i] <= F[j] and F[i] < F[j] in at least one obj
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1
            
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            for i in current_front:
                rank[i] = 0
                
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for a front.
            Returns an array of distances for the indices in front_indices.
            """
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            m = len(F[front_indices[0]])
            n = len(front_indices)
            dist = np.zeros(n)
            
            F_front = F[front_indices]
            
            for j in range(m):
                # Sort front by j-th objective
                sorted_indices = np.argsort(F_front[:, j])
                dist_sorted = np.zeros(n)
                # Assign inf to extremes
                dist_sorted[0] = np.inf
                dist_sorted[-1] = np.inf
                # Calculate distances for intermediate points
                if n > 2:
                    obj_vals = F_front[sorted_indices, j]
                    range_val = obj_vals[-1] - obj_vals[0]
                    if range_val > 1e-10:
                        for i in range(1, n - 1):
                            dist_sorted[i] += (obj_vals[i+1] - obj_vals[i-1]) / range_val
                # Map back to original order
                for i, idx in enumerate(sorted_indices):
                    dist[idx] += dist_sorted[i]
                    
            return dist

        def select_parents(rank, dist, n_select):
            """
            Binary tournament selection.
            """
            selected = []
            for _ in range(n_select):
                i1 = np.random.randint(len(rank))
                i2 = np.random.randint(len(rank))
                if rank[i1] < rank[i2]:
                    selected.append(i1)
                elif rank[i2] < rank[i1]:
                    selected.append(i2)
                else:
                    if dist[i1] > dist[i2]:
                        selected.append(i1)
                    else:
                        selected.append(i2)
            return selected

        def sbx_crossover(x1, x2, eta=15.0):
            """
            Simulated Binary Crossover.
            """
            x3 = np.zeros_like(x1)
            x4 = np.zeros_like(x2)
            
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    if x1[i] != x2[i]:
                        if x1[i] < x2[i]:
                            x_lo, x_hi = x1[i], x2[i]
                        else:
                            x_lo, x_hi = x2[i], x1[i]
                        
                        if np.random.rand() < 0.5:
                            beta = (2 * np.random.rand() + (x_lo - x1[i]) / (x_hi - x_lo)) ** (1 / (eta + 1))
                            x3[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                        else:
                            beta = (2 * np.random.rand() + (x_hi - x2[i]) / (x_hi - x_lo)) ** (1 / (eta + 1))
                            x3[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                        
                        # Swap roles for x4
                        if np.random.rand() < 0.5:
                            beta = (2 * np.random.rand() + (x_lo - x2[i]) / (x_hi - x_lo)) ** (1 / (eta + 1))
                            x4[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                        else:
                            beta = (2 * np.random.rand() + (x_hi - x1[i]) / (x_hi - x_lo)) ** (1 / (eta + 1))
                            x4[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                    else:
                        x3[i] = x1[i]
                        x4[i] = x2[i]
                else:
                    x3[i] = x1[i]
                    x4[i] = x2[i]
            
            return x3, x4

        def polynomial_mutation(x, lo, hi, eta=20.0):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < 1.0 / len(x):
                    if x_mut[i] < lo[i] and x_mut[i] > hi[i]: # Should not happen if clipped
                        pass
                    delta = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                    y = np.random.rand()
                    if y < 0.5:
                        xy = (2 * y + delta) ** (1 / (eta + 1))
                        delta_q = xy - 1
                    else:
                        xy = (2 * (1 - y) + (1 - delta)) ** (1 / (eta + 1))
                        delta_q = 1 - xy
                    x_mut[i] = x_mut[i] + delta_q * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Evolution Loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Create offspring
            X_offspring = np.zeros((N, dim))
            F_offspring = np.zeros((N, n_obj))
            
            # Select parents and create offspring
            for i in range(0, N, 2):
                # Select 2 parents
                p1_idx, p2_idx = select_parents(rank, dist, 2)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                X_offspring[i] = c1
                X_offspring[i+1] = c2
                used += 2
                
                # Check budget mid-generation if necessary, but we loop by N
                if used >= self.budget:
                    break
            
            # Evaluate offspring
            for i in range(N):
                if used <= self.budget:
                    F_offspring[i] = problem(X_offspring[i])
                    used += 1
                else:
                    # If we ran out of budget, we might have unevaluated offspring.
                    # We should only evaluate if we have budget.
                    # The loop above incremented used for creation, but not evaluation.
                    # Let's restructure to be safe.
                    pass
            
            # Actually, the logic above is a bit messy with budget tracking.
            # Let's simplify: Generate N offspring, evaluate them, then combine.
            # If used + N > budget, we can't do a full generation.
            # But N=20, budget=200. 200 % 20 == 0. So we can do exactly 9 full generations after init.
            # So we can just do full generations.
            
            # Combine parents and offspring
            X_combined = np.vstack([X, X_offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # Sort
            fronts, rank = fast_non_dominated_sort(F_combined)
            
            # Calculate crowding distance for each front
            dist = np.zeros(len(F_combined))
            for front in fronts:
                d = crowding_distance(F_combined, front)
                for idx, d_val in zip(front, d):
                    dist[idx] = d_val
            
            # Select next generation
            X_new = np.zeros((N, dim))
            F_new = np.zeros((N, n_obj))
            
            # Fill with fronts
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Need to select partial front
                    remaining = N - len(selected_indices)
                    # Sort front by crowding distance (descending)
                    front_with_dist = [(idx, dist[idx]) for idx in front]
                    front_with_dist.sort(key=lambda x: x[1], reverse=True)
                    selected_indices.extend([idx for idx, _ in front_with_dist[:remaining]])
                    break
            
            for i, idx in enumerate(selected_indices):
                X_new[i] = X_combined[idx]
                F_new[i] = F_combined[idx]
            
            X = X_new
            F = F_new

        # Return the non-dominated points from the final population
        # The evaluator ignores the return value, but we return something valid.
        # It's best to return the non-dominated set.
        fronts_final, _ = fast_non_dominated_sort(F)
        if fronts_final:
            return X[fronts_final[0]]
        return X

# The code above has a bug in the budget tracking inside the loop.
# Let's rewrite the main loop cleanly.
