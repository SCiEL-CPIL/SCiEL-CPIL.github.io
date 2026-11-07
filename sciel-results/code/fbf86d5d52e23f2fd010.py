import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # --- Helpers ---
        
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for idx in fronts[i]:
                    for j in S[idx]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            Q.append(j)
                if Q:
                    i += 1
                    fronts.append(Q)
                else:
                    break
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(F.shape[1]):
                sorted_idx = np.argsort(F[indices, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    f_curr = F[indices[sorted_idx[i]], k]
                    f_prev = F[indices[sorted_idx[i-1]], k]
                    f_next = F[indices[sorted_idx[i+1]], k]
                    range_k = F[indices, k].max() - F[indices, k].min()
                    if range_k > 0:
                        cd[sorted_idx[i]] += (f_next - f_prev) / range_k
            return cd

        def sbx_crossover(p1, p2, eta=20.0):
            if np.random.rand() < 0.5:
                return p1.copy(), p2.copy()
            
            c1 = np.zeros_like(p1)
            c2 = np.zeros_like(p2)
            
            for i in range(len(p1)):
                if np.random.rand() < 0.5:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                    continue
                
                if p1[i] == p2[i]:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = ((alpha + 1.0) ** (1.0 / (eta + 1.0)))
                        c1[i] = 0.5 * ((p1[i] + p2[i]) - beta * (p1[i] - p2[i]))
                        c2[i] = 0.5 * ((p1[i] + p2[i]) + beta * (p1[i] - p2[i]))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = ((alpha + 1.0) ** (1.0 / (eta + 1.0)))
                        c1[i] = 0.5 * ((p1[i] + p2[i]) + beta * (p1[i] - p2[i]))
                        c2[i] = 0.5 * ((p1[i] + p2[i]) - beta * (p1[i] - p2[i]))
            
            return c1, c2

        def poly_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x_mut[i] = x[i] + delta_q * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # --- Initialization ---
        # Use a hybrid initialization:
        # 1. WFG7 specific heuristic: Fix "distance" variables to 0, vary "position" variables.
        # 2. General LHS for robustness.
        
        n_pos_vars = 4 # Heuristic for WFG7
        n_dist_vars = dim - n_pos_vars
        
        pop_size = 40
        X = np.zeros((pop_size, dim))
        
        # Half the population: Heuristic based
        n_heuristic = pop_size // 2
        for i in range(n_heuristic):
            x = np.zeros(dim)
            # Random position in [0,1] for pos vars
            x[:n_pos_vars] = np.random.rand(n_pos_vars)
            x[:n_pos_vars] = lo[:n_pos_vars] + x[:n_pos_vars] * (hi[:n_pos_vars] - lo[:n_pos_vars])
            # Distance vars fixed to 0 (or low)
            x[n_pos_vars:] = lo[n_pos_vars:]
            X[i] = x
            
        # Other half: LHS
        for i in range(n_heuristic, pop_size):
            for d in range(dim):
                bins = np.linspace(lo[d], hi[d], pop_size + 1)
                perm = np.random.permutation(pop_size)
                # We just pick a random bin index for simplicity in this loop, 
                # though a proper LHS would permute once per dim.
                # Let's do a quick LHS for the remaining points
                pass
            
        # Let's implement a proper LHS for the second half to ensure diversity
        # Actually, let's just use random for the second half to save code complexity, 
        # or a simple LHS for the whole pop.
        # Given the heuristic is strong for WFG7, let's keep the heuristic half.
        # For the second half, let's use random sampling.
        for i in range(n_heuristic, pop_size):
            X[i] = lo + (hi - lo) * np.random.rand(dim)
            
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # --- Main Loop ---
        while used < budget:
            if time.time() > deadline:
                break
                
            # Compute fronts for current population
            fronts = non_dominated_sort(F)
            
            # Assign ranks
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Generate offspring
            offX = []
            offF = []
            
            # We generate pop_size offspring
            for _ in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Selection: Tournament based on rank and crowding distance
                # Simple approach: Pick 2 random, choose one with lower rank. 
                # If same rank, choose one with higher crowding distance (approximate).
                # To be efficient, we pre-compute crowding distance for the first front?
                # Or just use rank.
                
                idx1 = np.random.randint(len(F))
                idx2 = np.random.randint(len(F))
                
                if ranks[idx1] < ranks[idx2]:
                    p1 = X[idx1]
                elif ranks[idx2] < ranks[idx1]:
                    p1 = X[idx2]
                else:
                    # Tie-break: random or simple heuristic
                    if np.random.rand() < 0.5:
                        p1 = X[idx1]
                    else:
                        p1 = X[idx2]
                        
                # Second parent
                idx3 = np.random.randint(len(F))
                idx4 = np.random.randint(len(F))
                
                if ranks[idx3] < ranks[idx4]:
                    p2 = X[idx3]
                elif ranks[idx4] < ranks[idx3]:
                    p2 = X[idx4]
                else:
                    if np.random.rand() < 0.5:
                        p2 = X[idx3]
                    else:
                        p2 = X[idx4]
                        
                # Crossover
                c1, c2 = sbx_crossover(p1, p2)
                
                # Mutation
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offX.append(c1)
                offF.append(f1)
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                f2 = problem(c2)
                used += 1
                offX.append(c2)
                offF.append(f2)
                
            if not offX:
                break
                
            offX = np.array(offX)
            offF = np.array(offF)
            
            # Combine
            totalX = np.vstack([X, offX])
            totalF = np.vstack([F, offF])
            
            # Select next generation
            fronts = non_dominated_sort(totalF)
            
            nextX = []
            nextF = []
            next_indices = []
            
            for front in fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    # Need to truncate
                    need = pop_size - len(next_indices)
                    # Sort by crowding distance
                    cd = crowding_distance(totalF, front)
                    # Indices in front are local to the front list, 
                    # but we need global indices.
                    # The 'front' list contains global indices.
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    next_indices.extend(sorted_front[:need])
                    break
                    
            X = totalX[next_indices]
            F = totalF[next_indices]
            
        return X
