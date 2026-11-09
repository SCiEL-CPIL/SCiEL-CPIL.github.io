"""Improved program using NSGA-II with fast non-dominated sorting for multi-objective optimization."""
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
        deadline = start_time + 290  # Safety margin before 300s/330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Determine population size
        N = 30
        if self.budget < N:
            N = max(2, self.budget)
        
        # Check if we can do at least 1 generation
        G = (self.budget - N) // N
        if G < 1:
            # Just do random sampling
            X = lo + (hi - lo) * np.random.random((N, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize Population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Track all evaluated points for final hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        for g in range(G):
            # Check time limit
            if time.time() > deadline:
                break
            
            # Create Offspring
            Q = np.empty((N, self.dim))
            Fq = np.empty((N, self.n_obj))
            
            # Non-dominated sorting on P
            rank, crowding = self._fast_non_dominated_sort(F)
            
            offspring_count = 0
            for i in range(N // 2):
                if time.time() > deadline:
                    break
                
                # Select 2 parents via tournament
                p1_idx = self._tournament(rank, crowding, N, 2)
                p2_idx = self._tournament(rank, crowding, N, 2)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                # Evaluate
                Fq[offspring_count] = problem(c1)
                Q[offspring_count] = c1
                used += 1
                offspring_count += 1
                if used > self.budget:
                    break
                    
                Fq[offspring_count] = problem(c2)
                Q[offspring_count] = c2
                used += 1
                offspring_count += 1
                if used > self.budget:
                    break
            
            if offspring_count == 0:
                break
                
            if offspring_count < N:
                Q = Q[:offspring_count]
                Fq = Fq[:offspring_count]
            
            # Combine P and Q
            X_combined = np.vstack([X, Q])
            F_combined = np.vstack([F, Fq])
            
            # Non-dominated sort on combined
            rank_c, crowding_c = self._fast_non_dominated_sort(F_combined)
            
            # Select next population P from R
            order = np.lexsort((-crowding_c, rank_c))
            next_indices = order[:N]
            
            X = X_combined[next_indices]
            F = F_combined[next_indices]
            
            # Record all evaluated points
            all_X.append(Q.copy())
            all_F.append(Fq.copy())
            
        return X

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting (Deb et al. 2002) with crowding distance.
        Returns:
            rank: array of ranks (0 for non-dominated, 1 for next, etc.)
            crowding: array of crowding distances
        """
        n = len(F)
        rank = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Build domination relations
        S = [[] for _ in range(n)]  # S[i] = list of indices dominated by i
        n_d = np.zeros(n, dtype=int)  # Number of solutions dominating i
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        # Front 0
        front = [i for i in range(n) if n_d[i] == 0]
        for i in front:
            rank[i] = 0
        
        front_idx = 0
        while len(front) > 0:
            # Calculate crowding distance for current front
            self._calculate_crowding(F, front, crowding)
            
            # Find next front
            next_front = []
            for i in front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            
            front_idx += 1
            front = next_front
            
        return rank, crowding

    def _calculate_crowding(self, F, front, crowding):
        """Calculate crowding distance for a given front."""
        m = len(front)
        if m <= 2:
            for idx in front:
                crowding[idx] = np.inf
            return
        
        for k in range(self.n_obj):
            f_vals = F[front, k]
            f_min = np.min(f_vals)
            f_max = np.max(f_vals)
            if f_max - f_min == 0:
                continue
            
            # Sort front by objective k
            sorted_front = sorted(front, key=lambda idx: F[idx, k])
            
            # Boundary points get infinite crowding
            crowding[sorted_front[0]] = np.inf
            crowding[sorted_front[-1]] = np.inf
            
            # Intermediate points
            for i in range(1, m - 1):
                idx_curr = sorted_front[i]
                idx_prev = sorted_front[i - 1]
                idx_next = sorted_front[i + 1]
                dist = (F[idx_next, k] - F[idx_prev, k]) / (f_max - f_min)
                crowding[idx_curr] += dist

    def _tournament(self, rank, crowding, N, k):
        """
        Tournament selection.
        Returns index of the winner.
        """
        idxs = np.random.choice(N, size=k, replace=False)
        best_idx = idxs[0]
        best_rank = rank[best_idx]
        best_crowd = crowding[best_idx]
        
        for idx in idxs[1:]:
            r = rank[idx]
            c = crowding[idx]
            if r < best_rank:
                best_idx = idx
                best_rank = r
                best_crowd = c
            elif r == best_rank:
                if c > best_crowd:
                    best_idx = idx
                    best_crowd = c
                    
        return best_idx

    def _sbx_crossover(self, p1, p2, lo, hi):
        """
        Simulated Binary Crossover.
        """
        c1 = p1.copy()
        c2 = p2.copy()
        
        eta = 15.0
        prob = 0.9
        
        for i in range(self.dim):
            if np.random.random() < prob:
                if abs(p1[i] - p2[i]) < 1e-10:
                    continue
                
                x1 = min(p1[i], p2[i])
                x2 = max(p1[i], p2[i])
                swap = p1[i] > p2[i]
                
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                c1_val = 0.5 * ((x1 + x2) - (x2 - x1) * beta)
                c2_val = 0.5 * ((x1 + x2) + (x2 - x1) * beta)
                
                c1_val = np.clip(c1_val, lo[i], hi[i])
                c2_val = np.clip(c2_val, lo[i], hi[i])
                
                if swap:
                    c1[i] = c1_val
                    c2[i] = c2_val
                else:
                    c1[i] = c2_val
                    c2[i] = c1_val
                    
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """
        Polynomial Mutation.
        """
        eta = 20.0
        pm = 1.0 / self.dim
        
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.random() < pm:
                delta_1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-10)
                delta_2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-10)
                
                u = np.random.random()
                if u < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy) ** (eta + 1.0)
                    delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy) ** (eta + 1.0)
                    delta_q = 1.0 - val ** (1.0 / (eta + 1.0))
                    
                x_mut[i] += delta_q * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
                
        return x_mut
# EVOLVE-BLOCK-END
