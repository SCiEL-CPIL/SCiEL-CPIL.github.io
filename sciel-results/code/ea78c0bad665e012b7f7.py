import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget 200. 
        # If pop=20, we do 10 generations.
        # If pop=10, we do 20 generations. Smaller pop might allow more generations (more exploration steps) but less diversity per step.
        # For 12D, 20 is a safe bet.
        pop_size = 20
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        eta_c = 20.0
        eta_m = 20.0
        pc = 0.9
        pm = 1.0 / self.dim
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            Q = np.empty_like(X)
            for i in range(pop_size):
                # Tournament Selection
                p1 = self._tournament_select(F, rng)
                p2 = self._tournament_select(F, rng)
                
                # Crossover (SBX)
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation (Polynomial)
                c1 = self._poly_mutate(c1, lo, hi, eta_m, pm, rng)
                c2 = self._poly_mutate(c2, lo, hi, eta_m, pm, rng)
                
                Q[i] = c1
                # We only have budget for pop_size children per generation to fit exactly?
                # If we generate 2*pop_size, we use 40 evals per gen. 200/40 = 5 gens.
                # If we generate pop_size, we use 20 evals per gen. 200/20 = 10 gens.
                # NSGA-II usually uses mu=pop, lambda=pop.
                # So we generate pop_size children.
                
                # Wait, standard NSGA-II generates lambda offspring. 
                # If lambda = pop_size, we evaluate pop_size new points.
                
            # Evaluate offspring
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            # Combine
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            # Non-dominated sort and crowding distance
            X, F = self._nsga2_select(P, F_p, pop_size, rng)
            
        return X

    def _tournament_select(self, F, rng, k=2):
        idxs = rng.integers(0, len(F), size=k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return F[best_idx].copy() # Return objective vector? No, need decision vector.
        # Correction: Tournament selection on population X, using F for dominance.
        # I need to pass X as well or store indices.
        
    # Let's restructure to keep X and F together properly.
    
    def _dominates(self, a, b):
        # a dominates b if a <= b and a < b in at least one obj
        return np.all(a <= b) and np.any(a < b)

    def _nsga2_select(self, P, F_p, target_size, rng):
        # P is decision vectors, F_p is objective vectors
        # 1. Non-dominated sorting
        ranks, fronts = self._non_dominated_sort(F_p)
        
        # 2. Crowding distance
        # We only need crowding distance for the last front that is partially selected
        # But standard implementation calculates for all or just the last front.
        # Let's calculate for all for simplicity, or just the last one.
        
        next_X = []
        next_F = []
        
        for front in fronts:
            if len(next_X) + len(front) <= target_size:
                next_X.extend(P[front])
                next_F.extend(F_p[front])
            else:
                # Partial front
                # Calculate crowding distance for individuals in this front
                # Map indices in front to indices in P
                front_indices = front
                # Sort by crowding distance
                # Crowding distance is calculated in objective space
                # We need the objective vectors of the individuals in this front
                F_front = F_p[front_indices]
                
                # Calculate crowding distance
                cd = self._crowding_distance(F_front)
                
                # Sort indices by cd (descending)
                sorted_indices = np.argsort(-cd)
                
                # Take the best (highest cd) until target_size is reached
                needed = target_size - len(next_X)
                for idx in sorted_indices[:needed]:
                    global_idx = front_indices[idx]
                    next_X.append(P[global_idx])
                    next_F.append(F_p[global_idx])
                    
                break
                
        return np.array(next_X), np.array(next_F)

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)] # Dominated sets
        n_dominating = [0] * n
        rank = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dominating[i] += 1
                    
        current_front = []
        for i in range(n):
            if n_dominating[i] == 0:
                rank[i] = 0
                current_front.append(i)
                
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1
            
        return rank, fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
            
        cd = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            # Sort by k-th objective
            sorted_idx = np.argsort(F[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            
            f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range
                
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if 2 * rng.random() <= (1 / (eta + 1)):
                        beta = (2 * rng.random()) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - rng.random()))) ** (1 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _poly_mutate(self, x, lo, hi, eta, rate, rng):
        for i in range(len(x)):
            if rng.random() < rate:
                delta = rng.random()
                if delta < 0.5:
                    val = (2 * delta) ** (1 / (eta + 1))
                    delta_1 = (1 - val)
                else:
                    val = (2 * (1 - delta)) ** (1 / (eta + 1))
                    delta_1 = (val - 1)
                    
                # Calculate new value
                # Standard polynomial mutation formula
                yl = (x[i] - lo[i]) / (hi[i] - lo[i])
                yu = (hi[i] - x[i]) / (hi[i] - lo[i])
                
                if delta_1 < 0:
                    # x decreases
                    x[i] = x[i] - delta_1 * (hi[i] - lo[i]) # Wait, delta_1 is negative here?
                    # Let's use the standard formula from Deb 2001
                    # If delta < 0.5, delta1 = (2*delta)^(1/(eta+1)), delta' = 1 - delta1
                    # If delta > 0.5, delta1 = (2*(1-delta))^(1/(eta+1)), delta' = delta1 - 1
                    # new_x = x + (hi-lo) * (delta' * (some factor)?)
                    # Actually, the standard formula is:
                    # if delta < 0.5:
                    #   delta1 = (2*delta)^(1/(eta+1))
                    #   delta' = 1 - delta1
                    # else:
                    #   delta1 = (2*(1-delta))^(1/(eta+1))
                    #   delta' = delta1 - 1
                    # 
                    # new_x = x + (hi - lo) * delta' * (0 if ... else ...)
                    # Actually, it's simpler to just add a small random step and clip, 
                    # or use the exact polynomial distribution.
                    
                    # Let's stick to a simpler Gaussian mutation for reliability and speed, 
                    # scaled by the range.
                    # Gaussian mutation: x + sigma * N(0,1)
                    # sigma = 0.1 * (hi - lo)
                    pass
                    
        # Let's replace _poly_mutate with a simpler, robust Gaussian mutation
        # This avoids complex implementation errors in polynomial mutation.
        
        # Re-implementing _poly_mutate as Gaussian for robustness
        # But wait, I already defined it. Let's modify the method body in the final code.
        
        return x

    # Let's rewrite the mutation to be Gaussian in the final code block for clarity and robustness.
    # The SBX crossover is good.
    
    # One issue: `_tournament_select` in the draft above was incomplete.
    # I need to select from X (decision vectors) based on F (objectives).
    
    def _tournament_select(self, X, F, rng, k=2):
        idxs = rng.integers(0, len(F), size=k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    # Update __call__ to use the fixed _tournament_select
    
    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        eta_c = 20.0
        pc = 0.9
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            Q = np.empty_like(X)
            for i in range(pop_size):
                p1 = self._tournament_select(X, F, rng)
                p2 = self._tournament_select(X, F, rng)
                
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Gaussian Mutation
                # Scale: 0.1 of range
                # Rate: 1/dim
                sigma = 0.1 * (hi - lo)
                c1 = c1 + sigma * rng.standard_normal(self.dim)
                c2 = c2 + sigma * rng.standard_normal(self.dim)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                Q[i] = c1
                # We discard c2 to keep pop size constant and fit budget.
                # Standard NSGA-II generates lambda=mu children.
                
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            X, F = self._nsga2_select(P, F_p, pop_size, rng)
            
        return X

    # The rest of the helper methods (_dominates, _nsga2_select, _non_dominated_sort, _crowding_distance, _sbx_crossover) remain as defined in the draft, 
    # with _poly_mutate removed/replaced by inline Gaussian.
