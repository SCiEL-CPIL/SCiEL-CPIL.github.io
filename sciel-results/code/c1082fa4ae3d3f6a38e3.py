import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size 40 gives 5 generations for budget 200
        N = 40
        T = self.budget // N
        
        # Initialize
        X = lo + range_ * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        while used < self.budget:
            # Calculate ranks and crowding for current population
            ranks, crowding = self._non_dominated_sorting(F)
            
            # Generate offspring
            Q = np.zeros((N, self.dim))
            for i in range(N):
                p1 = self._tournament_select(X, ranks, crowding)
                p2 = self._tournament_select(X, ranks, crowding)
                
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                Q[2*i] = c1
                Q[2*i+1] = c2
                
            # Evaluate offspring
            Fq = np.array([problem(x) for x in Q])
            used += N
            
            # Combine parents and offspring
            R = np.vstack([X, Q])
            Fr = np.vstack([F, Fq])
            
            # Non-dominated sorting for selection
            ranks_r, crowding_r = self._non_dominated_sorting(Fr)
            
            # Selection: Sort by rank, then crowding distance
            # We want the best N individuals
            # Sort indices by (rank, -crowding)
            sort_keys = [(ranks_r[i], -crowding_r[i], i) for i in range(len(R))]
            sort_keys.sort()
            selected_indices = [k[2] for k in sort_keys[:N]]
            
            X = R[selected_indices]
            F = Fr[selected_indices]
            
        return X

    def _tournament_select(self, X, ranks, crowding):
        i = np.random.randint(len(X))
        j = np.random.randint(len(X))
        
        if ranks[i] < ranks[j]:
            return X[i]
        elif ranks[j] < ranks[i]:
            return X[j]
        else:
            if crowding[i] > crowding[j]:
                return X[i]
            else:
                return X[j]

    def _sbx_crossover(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        eta_cr = 15.0
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_cr + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_cr + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        # Polynomial Mutation
        eta_m = 20.0
        pm_rate = 1.0 / self.dim
        
        for i in range(self.dim):
            if np.random.rand() < pm_rate:
                delta_1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta_2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1.0 - delta_1
                    val = 1.0 - xy ** (eta_m + 1)
                    deltaq = (2 * val) ** (1.0 / (eta_m + 1))
                else:
                    xy = 1.0 - delta_2
                    val = 1.0 - xy ** (eta_m + 1)
                    deltaq = (2 * val) ** (1.0 / (eta_m + 1))
                
                x[i] += (hi[i] - lo[i]) * deltaq
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

    def _non_dominated_sorting(self, F):
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N)
        
        # Fast Non-Dominated Sort
        S = [set() for _ in range(N)]
        n = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(i+1, N):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n[i] += 1
        
        # Front 0
        front = [i for i in range(N) if n[i] == 0]
        k = 0
        current_front = front
        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = k
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
            
        # Crowding Distance
        fronts_dict = {}
        for i in range(N):
            r = ranks[i]
            if r not in fronts_dict:
                fronts_dict[r] = []
            fronts_dict[r].append(i)
        
        for r, members in fronts_dict.items():
            if len(members) <= 2:
                for i in members:
                    crowding[i] = float('inf')
                continue
            
            for m in range(self.n_obj):
                sorted_members = sorted(members, key=lambda x: F[x][m])
                crowding[sorted_members[0]] = float('inf')
                crowding[sorted_members[-1]] = float('inf')
                
                f_min = F[sorted_members[0]][m]
                f_max = F[sorted_members[-1]][m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for idx in range(1, len(sorted_members)-1):
                    i = sorted_members[idx]
                    prev_i = sorted_members[idx-1]
                    next_i = sorted_members[idx+1]
                    crowding[i] += (F[next_i][m] - F[prev_i][m]) / range_f
                    
        return ranks, crowding
