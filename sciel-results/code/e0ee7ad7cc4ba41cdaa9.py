import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 10
        # Ensure we don't exceed budget
        # Init cost: N
        # Each generation cost: N
        # Total = N + G * N <= budget
        # G = (budget - N) // N
        if self.budget < N:
             # Fallback to random search if budget is tiny
             X = lo + (hi - lo) * rng.random((self.budget, self.dim))
             F = np.array([problem(x) for x in X])
             return X

        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)] # S[i] = indices dominated by i
            n_d = [0] * n # n_d[i] = number of points dominating i
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
            
            current_front = []
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = self.n_obj
            dist = {i: 0.0 for i in front_indices}
            
            for k in range(m):
                # Sort front by objective k
                sorted_indices = sorted(front_indices, key=lambda i: F[i, k])
                # Assign infinite distance to best and worst
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                range_k = f_max - f_min
                if range_k == 0:
                    continue
                
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    dist[idx] += (F[idx_next, k] - F[idx_prev, k]) / range_k
            
            return dist

        # Helper: Binary Tournament Selection
        def tournament_select(X, F, rank):
            i = rng.integers(N)
            j = rng.integers(N)
            if rank[i] < rank[j]:
                return X[i]
            elif rank[j] < rank[i]:
                return X[j]
            else:
                # If same rank, use crowding distance? 
                # For simplicity in tournament, we can just pick random if tied,
                # or compute CD on the fly. 
                # Standard NSGA-II uses CD for selection.
                # To avoid computing CD for every tournament, we usually pre-compute CD for the current population.
                # But here we are selecting from P. Let's pre-compute CD for P at the start of each gen.
                # I will modify the main loop to handle this.
                return X[i] # Fallback

        # Main Loop
        for gen in range(G):
            # Compute ranks and crowding distances for current population P
            fronts, rank = non_dominated_sort(F)
            
            # Pre-compute crowding distance for each front
            cd = {}
            for front in fronts:
                cd_front = crowding_distance(F, front)
                for i in front:
                    cd[i] = cd_front[i]
            
            # Selection function with CD
            def select_one():
                i = rng.integers(N)
                j = rng.integers(N)
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    if cd[i] > cd[j]:
                        return i
                    elif cd[j] > cd[i]:
                        return j
                    else:
                        return i # Random tie break

            # Create offspring Q
            Q = np.zeros((N, self.dim))
            for i in range(N):
                p1 = X[select_one()]
                p2 = X[select_one()]
                
                # SBX Crossover
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5: # p_c = 0.5 usually, or 1.0. Let's use 1.0 for each gene? 
                        # Standard SBX is applied to the pair.
                        # Let's use standard SBX with p_c=1.0 for the vector.
                        pass
                
                # Let's implement SBX properly for the whole vector
                # u = rng.random(self.dim)
                # beta = np.where(u <= 0.5, (2*u)**(1/(eta_c+1)), (2*(1-u))**(-1/(eta_c+1)))
                # child = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
                # child = np.clip(child, lo, hi)
                
                # Polynomial Mutation
                # for d in range(self.dim):
                #     if rng.random() < 1.0/self.dim:
                #         y = rng.random()
                #         delta = (2*y)**(1/(eta_m+1)) - 1 if y <= 0.5 else 1 - (2*(1-y))**(1/(eta_m+1))
                #         child[d] += delta * (hi[d] - lo[d])
                #         child[d] = np.clip(child[d], lo[d], hi[d])
                
                # Simpler: Gaussian Mutation
                # child = p1 + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                # child = np.clip(child, lo, hi)
                
                # Let's use SBX + Poly Mutation as it's standard NSGA-II
                u = rng.random(self.dim)
                eta_c = 20.0
                beta = np.where(u <= 0.5, (2*u)**(1/(eta_c+1)), (2*(1-u))**(-1/(eta_c+1)))
                child = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
                
                # Mutation
                eta_m = 20.0
                p_m = 1.0 / self.dim
                for d in range(self.dim):
                    if rng.random() < p_m:
                        y = rng.random()
                        if y <= 0.5:
                            delta = (2*y)**(1/(eta_m+1)) - 1
                        else:
                            delta = 1 - (2*(1-y))**(1/(eta_m+1))
                        child[d] += delta * (hi[d] - lo[d])
                        child[d] = np.clip(child[d], lo[d], hi[d])
                        
                Q[i] = child
            
            F_Q = np.array([problem(x) for x in Q])
            evals += N
            
            # Combine P and Q
            R_X = np.vstack((X, Q))
            R_F = np.vstack((F, F_Q))
            
            # Non-dominated sort on R
            fronts_R, rank_R = non_dominated_sort(R_F)
            
            # Select next P
            X_new = np.zeros((N, self.dim))
            F_new = np.zeros((N, self.n_obj))
            count = 0
            
            for front in fronts_R:
                if count + len(front) <= N:
                    for i in front:
                        X_new[count] = R_X[i]
                        F_new[count] = R_F[i]
                        count += 1
                else:
                    # Need to select from this front
                    needed = N - count
                    cd_front = crowding_distance(R_F, front)
                    # Sort by CD descending
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    for i in sorted_front[:needed]:
                        X_new[count] = R_X[i]
                        F_new[count] = R_F[i]
                        count += 1
                    break
                
                if count >= N:
                    break
            
            X = X_new
            F = F_new
            
        return X
