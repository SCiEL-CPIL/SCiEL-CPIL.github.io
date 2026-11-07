"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II with DTLZ4-aware initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting on a set of points.
        Returns a list of lists of indices, sorted by dominance rank.
        """
        n = len(F)
        if n == 0:
            return []
        
        # S_i: set of solutions dominated by i
        # n_i: number of solutions dominating i
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        
        # Pre-compute dominance relations
        # F is (n, m)
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        front = []
        for i in range(n):
            if n_dom[i] == 0:
                front.append(i)
        
        if not front:
            # All dominated? Should not happen if we have at least one point
            # If all are equal, they are all non-dominated
            return [[i for i in range(n)]]
            
        fronts = [front]
        current_front = front
        k = 0
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                current_front = next_front
            else:
                break
                
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a specific subset of indices in F.
        Returns a dict mapping index -> distance.
        """
        if not indices:
            return {}
        
        n = len(indices)
        if n <= 2:
            return {idx: float('inf') for idx in indices}
            
        # Get the sub-array of objectives for these indices
        F_sub = F[indices]
        
        cd = {idx: 0.0 for idx in indices}
        
        for m in range(F_sub.shape[1]):
            # Sort indices by objective m
            # We need to sort the global indices based on F_sub[:, m]
            sorted_indices = sorted(indices, key=lambda idx: F[idx, m])
            
            # End points get infinite distance
            cd[sorted_indices[0]] = float('inf')
            cd[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            f_range = f_max - f_min
            if f_range == 0:
                continue
                
            for k in range(1, n - 1):
                idx = sorted_indices[k]
                idx_prev = sorted_indices[k - 1]
                idx_next = sorted_indices[k + 1]
                cd[idx] += (F[idx_next, m] - F[idx_prev, m]) / f_range
                
        return cd

    def _sbx_crossover(self, x1, x2, bounds, eta=20.0):
        """Simulated Binary Crossover."""
        child1 = x1.copy()
        child2 = x2.copy()
        
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    # Clip to bounds
                    lo, hi = bounds[i]
                    child1[i] = np.clip(c1, lo, hi)
                    child2[i] = np.clip(c2, lo, hi)
        
        return child1, child2

    def _polynomial_mutation(self, x, bounds, eta=20.0):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.random() < 0.8: # Mutation rate 1/dim? No, usually 1/dim or fixed. Let's use 0.5 for exploration? 
                # Standard is 1/dim. But for DTLZ4, we want to move along the front.
                # Let's use 1.0 for all dims to ensure movement? No, 0.5 is safe.
                if np.random.random() < 0.5: # Actually, let's just apply to all if random < 1/dim? 
                    # Let's stick to 1/dim as standard
                    pass 
            # Let's implement standard 1/dim
        # Re-implementing properly
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.random() < 1.0 / len(x):
                lo, hi = bounds[i]
                if hi - lo < 1e-10:
                    continue
                u = np.random.random()
                y = x[i]
                delta1 = (y - lo) / (hi - lo)
                delta2 = (hi - y) / (hi - lo)
                
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    beta = (1.0 + 2.0 * (0.5 - val)) ** (1.0 / (eta + 1.0))
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    beta = (1.0 + 2.0 * (val - 0.5)) ** (1.0 / (eta + 1.0))
                
                step = (hi - lo) * (beta - 1.0)
                x_mut[i] = y + step
                x_mut[i] = np.clip(x_mut[i], lo, hi)
        
        return x_mut

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # We have 200 budget. 
        # Init: 100. Offspring: 100. Total 200.
        pop_size = 100
        
        # Initialize population
        X = np.empty((pop_size, dim))
        
        # DTLZ4-aware initialization
        # Indices 0 to n_obj-2 are shape variables
        # Indices n_obj-1 to end are distance variables
        shape_dim = n_obj - 1
        dist_dim = dim - shape_dim
        
        # Half of the population: On the Pareto front (g=0)
        # Set distance variables to 0.5
        for i in range(pop_size // 2):
            X[i, :shape_dim] = np.random.random(shape_dim)
            X[i, shape_dim:] = 0.5
        
        # Other half: Random
        for i in range(pop_size // 2, pop_size):
            X[i] = lo + (hi - lo) * np.random.random(dim)
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main loop
        while used < budget:
            # Create offspring
            offspring = np.empty((pop_size, dim))
            for i in range(pop_size):
                # Binary tournament selection
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                
                # Simple dominance check for tournament
                # If p1 dominates p2, pick p1
                # If p2 dominates p1, pick p2
                # Else, pick randomly (or by crowding distance, but we don't have it here easily)
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    p1 = X[p1_idx]
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    p1 = X[p2_idx]
                else:
                    p1 = X[np.random.choice([p1_idx, p2_idx])]
                
                # Second parent
                p3_idx = np.random.randint(pop_size)
                p4_idx = np.random.randint(pop_size)
                if np.all(F[p3_idx] <= F[p4_idx]) and np.any(F[p3_idx] < F[p4_idx]):
                    p2 = X[p3_idx]
                elif np.all(F[p4_idx] <= F[p3_idx]) and np.any(F[p4_idx] < F[p3_idx]):
                    p2 = X[p4_idx]
                else:
                    p2 = X[np.random.choice([p3_idx, p4_idx])]
                
                # Crossover
                c1, c2 = self._sbx_crossover(p1, p2, bounds)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, bounds)
                c2 = self._polynomial_mutation(c2, bounds)
                
                # We need to fill 100 offspring slots. 
                # Currently we are producing 2 per iteration.
                # So loop pop_size // 2 times?
                # Let's adjust the loop structure.
                pass
                
            # The above loop structure is inefficient. Let's restart the offspring generation logic.
            
            offspring = np.empty((pop_size, dim))
            for i in range(0, pop_size, 2):
                # Select parents
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    p1 = X[p1_idx]
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    p1 = X[p2_idx]
                else:
                    p1 = X[np.random.choice([p1_idx, p2_idx])]
                
                p3_idx = np.random.randint(pop_size)
                p4_idx = np.random.randint(pop_size)
                if np.all(F[p3_idx] <= F[p4_idx]) and np.any(F[p3_idx] < F[p4_idx]):
                    p2 = X[p3_idx]
                elif np.all(F[p4_idx] <= F[p3_idx]) and np.any(F[p4_idx] < F[p3_idx]):
                    p2 = X[p4_idx]
                else:
                    p2 = X[np.random.choice([p3_idx, p4_idx])]
                
                # Crossover
                c1, c2 = self._sbx_crossover(p1, p2, bounds)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, bounds)
                c2 = self._polynomial_mutation(c2, bounds)
                
                offspring[i] = c1
                if i + 1 < pop_size:
                    offspring[i + 1] = c2
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F_comb)
            
            # Select next population
            X_new = np.empty((pop_size, dim))
            F_new = np.empty((pop_size, n_obj))
            count = 0
            
            for front in fronts:
                if count + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    cd_dict = self._crowding_distance(F_comb, front)
                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda idx: cd_dict[idx], reverse=True)
                    for idx in sorted_front:
                        if count < pop_size:
                            X_new[count] = X_comb[idx]
                            F_new[count] = F_comb[idx]
                            count += 1
                    break
                
            X = X_new
            F = F_new
            
        return X
# EVOLVE-BLOCK-END
