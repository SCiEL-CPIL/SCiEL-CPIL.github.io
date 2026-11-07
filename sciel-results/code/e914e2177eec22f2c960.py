"""Multi-objective optimizer for WFG9 combining NSGA-II structure with adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with deadline guard for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Check if a dominates b."""
        return np.all(a <= b) and np.any(a < b)

    def _fast_nondominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_count = np.zeros(n, dtype=int)  # n[i] = number of points dominating i
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_count[i] += 1
        
        front = [i for i in range(n) if n_count[i] == 0]
        if not front:
            return [list(range(n))]
            
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            if not next_front:
                break
            fronts.append(next_front)
            front = next_front
            
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Calculate crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, float('inf'))
            
        cd = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            # Sort by objective k
            sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][k])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            min_val = F[front_indices[sorted_idx[0]]][k]
            max_val = F[front_indices[sorted_idx[-1]]][k]
            range_val = max_val - min_val
            
            if range_val < 1e-14:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                next_idx = sorted_idx[i + 1]
                cd[idx] += (F[front_indices[next_idx]][k] - F[front_indices[prev_idx]][k]) / range_val
                
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        """Simulated Binary Crossover."""
        child = p1.copy()
        if np.random.random() > 0.5:
            p1, p2 = p2, p1
            
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = p1[i], p2[i]
                    if x1 > x2:
                        x1, x2 = x2, x1
                    
                    rand = np.random.random()
                    if rand <= 1.0 / (eta + 1.0):
                        beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * (x1 + x2 - beta * (x2 - x1))
                    c2 = 0.5 * (x1 + x2 + beta * (x2 - x1))
                    
                    child[i] = c1 if np.random.random() < 0.5 else c2
                    
        return np.clip(child, lo, hi)

    def _poly_mutation(self, x, lo, hi, eta, pm):
        """Polynomial Mutation."""
        x = x.copy()
        for i in range(self.dim):
            if np.random.random() < pm:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-14)
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)
                rand = np.random.random()
                if rand < 0.5:
                    xy = 1.0 - (1.0 - delta1) ** (eta + 1.0)
                else:
                    xy = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                
                mut_pow = 1.0 / (eta + 1.0)
                xq = x[i] + (hi[i] - lo[i]) * (xy - 0.5)
                x[i] = np.clip(xq, lo[i], hi[i])
        return x

    def _tournament_select(self, F, ranks, cd, k=2):
        """Tournament selection based on rank and crowding distance."""
        n = len(F)
        candidates = np.random.randint(0, n, k)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                best = c
        return best

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        if budget < 10:
            # Very small budget: just random sampling
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X
        
        # Use a smaller population for more generations, then refine
        # Start with N=10, allowing 20 generations
        N = 10
        
        if budget < 2 * N:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X
        
        # Initialize: mix of random and structured points
        X = np.zeros((N, dim))
        # First half: stratified sampling
        n_strat = N // 2
        for i in range(n_strat):
            for d in range(dim):
                frac = (i + 0.5) / n_strat
                X[i, d] = lo[d] + frac * (hi[d] - lo[d]) + (hi[d] - lo[d]) * 0.1 * (np.random.random() - 0.5)
        # Second half: random
        for i in range(n_strat, N):
            for d in range(dim):
                X[i, d] = np.random.uniform(lo[d], hi[d])
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # Keep track of all non-dominated points found so far
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # NSGA-II parameters
        pc = 0.9
        eta_c = 15.0
        eta_m = 20.0
        pm = 1.0 / dim
        
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break
            
            gen += 1
            progress = used / budget
            
            # Calculate ranks and crowding distance on current population
            fronts = self._fast_nondominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            cd = np.zeros(N)
            
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                if len(front) > 2:
                    cd[front] = self._crowding_distance(F, front)
                else:
                    cd[front] = float('inf')
            
            # Generate offspring
            n_off = min(N, budget - used)
            if n_off <= 0:
                break
                
            offspring_X = np.empty((n_off, dim))
            offspring_F = np.empty((n_off, n_obj))
            
            # Adaptive parameters: increase exploitation over time
            current_pm = pm * (1.5 - 0.5 * progress)  # 1.5*pm -> 1.0*pm
            current_eta_m = eta_m * (1.0 + 0.5 * progress)  # increase eta for finer steps
            
            for i in range(n_off):
                if time.time() > deadline:
                    break
                if used + i >= budget:
                    n_off = i
                    break
                    
                # Select parents using tournament
                p1_idx = self._tournament_select(F, ranks, cd)
                p2_idx = self._tournament_select(F, ranks, cd)
                
                # Crossover
                if np.random.random() < pc:
                    child = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi, eta_c)
                else:
                    child = X[p1_idx].copy()
                
                # Mutation
                child = self._poly_mutation(child, lo, hi, current_eta_m, current_pm)
                
                fc = problem(child)
                offspring_X[i] = child
                offspring_F[i] = fc
                used += 1
                
            if n_off == 0:
                break
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X[:n_off]])
            F_comb = np.vstack([F, offspring_F[:n_off]])
            
            # Select next generation using non-dominated sorting
            fronts = self._fast_nondominated_sort(F_comb)
            cd_comb = np.zeros(len(X_comb))
            
            selected = []
            for rank, front in enumerate(fronts):
                if len(selected) + len(front) <= N:
                    selected.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    cd_comb[front] = self._crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd_comb[i], reverse=True)
                    needed = N - len(selected)
                    selected.extend(sorted_front[:needed])
                    break
            
            X = X_comb[selected]
            F = F_comb[selected]
            
            # Track all non-dominated points
            all_X.append(X.copy())
            all_F.append(F.copy())
            
            # If we have budget remaining and are near the end, do local refinement
            if used >= budget - 20 and time.time() < deadline - 30:
                # Pick the best non-dominated points and do local search
                fronts_final = self._fast_nondominated_sort(F)
                if fronts_final:
                    best_front = fronts_final[0]
                    # Do local perturbation around best points
                    local_n = min(len(best_front), (budget - used) // 2)
                    for i in range(local_n):
                        if time.time() > deadline:
                            break
                        if used >= budget:
                            break
                        idx = best_front[i % len(best_front)]
                        # Small random perturbation
                        x_local = X[idx].copy()
                        for d in range(dim):
                            if np.random.random() < 0.5:
                                x_local[d] += (hi[d] - lo[d]) * 0.02 * (np.random.random() - 0.5)
                        x_local = np.clip(x_local, lo, hi)
                        f_local = problem(x_local)
                        used += 1
                        # Add to population if it's non-dominated or improves
                        X_comb2 = np.vstack([X, x_local.reshape(1, -1)])
                        F_comb2 = np.vstack([F, f_local.reshape(1, -1)])
                        fronts2 = self._fast_nondominated_sort(F_comb2)
                        cd2 = np.zeros(len(X_comb2))
                        selected2 = []
                        for rank2, front2 in enumerate(fronts2):
                            if len(selected2) + len(front2) <= N:
                                selected2.extend(front2)
                            else:
                                cd2[front2] = self._crowding_distance(F_comb2, front2)
                                sorted_front2 = sorted(front2, key=lambda i: cd2[i], reverse=True)
                                needed2 = N - len(selected2)
                                selected2.extend(sorted_front2[:needed2])
                                break
                        X = X_comb2[selected2]
                        F = F_comb2[selected2]
                        all_X.append(X.copy())
                        all_F.append(F.copy())
        
        return X
# EVOLVE-BLOCK-END
