"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Population size
        # Budget is 200. 
        # If pop_size = 40, init = 40. Remaining = 160.
        # Generations = 160 / 40 = 4.
        # Total evals = 40 + 4*40 = 200.
        pop_size = 40
        
        # Initial Population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II Parameters
        eta_c = 15.0
        eta_m = 20.0
        prob_c = 0.9
        prob_m = 1.0 / self.dim

        def non_dominated_sort(F):
            """
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated by
            n_d = [0] * n  # Number of dominated
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                        
                if n_d[i] == 0:
                    fronts[0].append(i)
                    
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices in the same front.
            """
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                # Set boundary distances to inf
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_min = F[sorted_idx[0]][m]
                f_max = F[sorted_idx[-1]][m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    dist[idx] += (F[sorted_idx[i+1]][m] - F[sorted_idx[i-1]][m]) / range_f
            return dist

        def sbx_crossover(x1, x2):
            if rng.random() > prob_c:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        beta = 1.0
                    else:
                        if rng.random() <= 0.5:
                            alpha = (x1[i] - lo[i]) / (x1[i] - x2[i] + 1e-14)
                            beta = 1 + 2 * alpha
                        else:
                            alpha = (hi[i] - x2[i]) / (x2[i] - x1[i] + 1e-14)
                            beta = 1 + 2 * alpha
                        
                        if rng.random() <= 1.0 / (beta + 1.0):
                            beta_q = (beta * rng.random()) ** (1.0 / (eta_c + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - beta * rng.random())) ** (1.0 / (eta_c + 1.0))
                    
                    c1[i] = 0.5 * ((1 + beta_q) * x1[i] + (1 - beta_q) * x2[i])
                    c2[i] = 0.5 * ((1 - beta_q) * x1[i] + (1 + beta_q) * x2[i])
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def polynomial_mutation(x):
            for i in range(self.dim):
                if rng.random() < prob_m:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-14)
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                        delta_q = (val ** (1.0 / (eta_m + 1.0)) - 1.0)
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                        delta_q = 1.0 - (val ** (1.0 / (eta_m + 1.0)))
                    
                    x[i] += delta_q * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Evolution Loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Sort current population
            fronts = non_dominated_sort(F)
            
            # Create offspring
            offspring_X = []
            for _ in range(pop_size):
                # Select parents using tournament selection
                p1_idx = self._tournament_select(F, fronts)
                p2_idx = self._tournament_select(F, fronts)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parents and offspring
            X_comb = np.vstack((X, offspring_X))
            F_comb = np.vstack((F, offspring_F))
            
            # Select next generation
            fronts_comb = non_dominated_sort(F_comb)
            
            X_next = []
            F_next = []
            for front in fronts_comb:
                if len(X_next) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    # Need to truncate
                    need = pop_size - len(X_next)
                    # Calculate crowding distance for this front
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(need):
                        idx = front[sorted_front[i]]
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)

        return X

    def _tournament_select(self, F, fronts):
        """Tournament selection based on rank and crowding distance."""
        # Assign ranks
        rank = [0] * len(F)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Calculate crowding distances for each front
        cd = [0.0] * len(F)
        for front in fronts:
            if len(front) > 2:
                front_cd = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = front_cd[i]
            else:
                for idx in front:
                    cd[idx] = float('inf')
        
        # Tournament
        t1 = np.random.randint(0, len(F))
        t2 = np.random.randint(0, len(F))
        
        if rank[t1] < rank[t2]:
            return t1
        elif rank[t2] < rank[t1]:
            return t2
        else:
            if cd[t1] > cd[t2]:
                return t1
            else:
                return t2

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in the same front.
        """
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n
        
        dist = [0.0] * n
        for m in range(self.n_obj):
            # Sort indices by objective m
            sorted_idx = sorted(indices, key=lambda i: F[i][m])
            # Set boundary distances to inf
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            
            f_min = F[sorted_idx[0]][m]
            f_max = F[sorted_idx[-1]][m]
            range_f = f_max - f_min
            if range_f == 0:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_idx[i]
                dist[idx] += (F[sorted_idx[i+1]][m] - F[sorted_idx[i-1]][m]) / range_f
        return dist

# EVOLVE-BLOCK-END
