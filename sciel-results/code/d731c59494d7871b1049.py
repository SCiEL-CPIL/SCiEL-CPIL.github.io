"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """NSGA-II multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        # Budget 200. 
        # Let's use N=100 for strong initial coverage, then 1 generation of offspring.
        # Or N=50 for 3 generations.
        # WFG7 is deceptive. Large initial population helps find the front.
        # Let's try N=100.
        N = 100
        
        # 1. Initialization using Sobol sequence for better spread
        # Generate N points in [0, 1]^dim
        sobol_engine = qmc.Sobol(d=self.dim, scramble=True)
        # We need to ensure we have enough points. 
        # Sobol requires power of 2 for best quality, but works for any N.
        # Let's generate N points.
        X_init = sobol_engine.random(N)
        X = lo + X_init * range_
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        if used >= self.budget:
            return X
            
        # NSGA-II Loop
        # Parameters
        pc = 0.9  # Crossover probability
        pm = 1.0 / self.dim  # Mutation probability per gene
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            d = np.zeros(n, dtype=int)
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                    elif dominates(F[q], F[p]):
                        d[p] += 1
                if d[p] == 0:
                    fronts.append([p])
                    
            i = 0
            while i < len(fronts):
                current = fronts[i]
                next_front = []
                for p in current:
                    for q in S[p]:
                        d[q] -= 1
                        if d[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, 1e9)
            
            cd = np.zeros(n)
            for m in range(self.n_obj):
                # Extract values for this objective
                vals = F[indices, m]
                # Sort indices by this objective
                sorted_idx = np.argsort(vals)
                # Boundary points get infinite distance
                cd[sorted_idx[0]] = 1e9
                cd[sorted_idx[-1]] = 1e9
                
                min_val = vals[sorted_idx[0]]
                max_val = vals[sorted_idx[-1]]
                span = max_val - min_val
                
                if span < 1e-18:
                    continue
                    
                for k in range(1, n - 1):
                    idx = sorted_idx[k]
                    idx_prev = sorted_idx[k-1]
                    idx_next = sorted_idx[k+1]
                    cd[idx] += (vals[idx_next] - vals[idx_prev]) / span
            return cd

        def select_parents(F, X):
            n = len(F)
            fronts = fast_non_dominated_sort(F)
            selected = []
            
            for fr in fronts:
                if len(selected) + len(fr) <= n:
                    selected.extend(fr)
                else:
                    room = n - len(selected)
                    cd = crowding_distance(F, fr)
                    order = np.argsort(-cd)
                    selected.extend(fr[order[:room]])
                    break
            
            return selected

        def sbx_crossover(a, b):
            if np.random.rand() < pc:
                c1 = a.copy()
                c2 = b.copy()
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        if abs(a[i] - b[i]) < 1e-14:
                            continue
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                        
                        c1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
                        c2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
                return c1, c2
            else:
                return a.copy(), b.copy()

        def polynomial_mutation(x):
            for i in range(self.dim):
                if np.random.rand() < pm:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    # Normalize mutation to range
                    x[i] = x[i] + delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # Main Evolutionary Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Select Parents
            parents = select_parents(F, X)
            
            # 2. Generate Offspring
            offspring = []
            n_offspring = min(N, self.budget - used)
            
            for _ in range(n_offspring // 2):
                # Binary Tournament Selection
                p1_idx = np.random.randint(len(parents))
                p2_idx = np.random.randint(len(parents))
                
                # Simple tournament: pick 2 random, pick better
                def tournament(pop_idx, F, X, n=2):
                    candidates = np.random.choice(pop_idx, n, replace=False)
                    best = candidates[0]
                    for c in candidates[1:]:
                        if dominates(F[c], F[best]):
                            best = c
                        elif not dominates(F[best], F[c]):
                            # If incomparable, keep first or random
                            pass
                    return best
                
                p1 = tournament(parents, F, X)
                p2 = tournament(parents, F, X)
                
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring.append(c1)
                offspring.append(c2)
                
                if len(offspring) >= n_offspring:
                    break
            
            if not offspring:
                break
                
            offspring = np.array(offspring)
            
            # 3. Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring])
            used += len(offspring)
            
            # 4. Combine and Select
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_off])
            
            keep = select_parents(F_combined, X_combined)
            X = X_combined[keep]
            F = F_combined[keep]
            
            # Check if we should stop early if no improvement or time is up
            if time.time() > deadline:
                break

        return X
# EVOLVE-BLOCK-END
