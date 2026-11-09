import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        # Reserve 50 evaluations for local search
        budget_ns = self.budget - 50
        if budget_ns < 2 * N:
            budget_ns = self.budget
            local_budget = 0
        else:
            local_budget = 50
            
        G = max(1, (budget_ns - N) // N)
        
        # Sobol Initialization
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2 for best quality, but works for any.
        # We need N points.
        # Next power of 2 >= N
        n_sobol = 1
        while n_sobol < N:
            n_sobol *= 2
        samples = sampler.random(n_sobol)
        X = lo + (hi - lo) * samples[:N]
        
        F = np.array([problem(x) for x in X])
        
        # NSGA-II
        for gen in range(G):
            if time.time() > deadline - 5: # Leave time for local search
                break
                
            # Selection
            Q = np.zeros((N, self.dim))
            # Precompute ranks and crowding distances for selection?
            # Standard NSGA-II:
            # 1. Combine P and Q? No, P is current, Q is offspring.
            # Select from P to create Q.
            
            # Calculate ranks and CD for P
            fronts = self._non_dominated_sort(F)
            # Assign ranks
            ranks = np.zeros(N, dtype=int)
            for i, front in enumerate(fronts):
                ranks[front] = i
                
            # Calculate Crowding Distance for each front
            cd = np.zeros(N)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = self._crowding_distance(F[front])
                    # Map back? _crowding_distance returns local array.
                    # Let's make _crowding_distance take indices or handle mapping.
                    # Current impl returns array of size len(front).
                    # Need to map to global indices.
                    pass # I will fix the helper to return global or handle in loop.
                    
            # Actually, simpler: Just use tournament on ranks.
            # For CD, we need it for the final selection of the combined population.
            # For parent selection, rank is usually sufficient, or rank+CD.
            
            for i in range(0, N, 2):
                p1 = self._tournament_select(X, F, ranks)
                p2 = self._tournament_select(X, F, ranks)
                
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                Q[i] = c1
                if i + 1 < N:
                    Q[i+1] = c2
            
            FQ = np.array([problem(x) for x in Q])
            
            # Combine
            R_X = np.vstack((X, Q))
            R_F = np.vstack((F, FQ))
            
            # Non-dominated sort R
            fronts_r = self._non_dominated_sort(R_F)
            
            # Select next P
            next_X = []
            next_F = []
            for front in fronts_r:
                if len(next_X) + len(front) <= N:
                    next_X.append(R_X[front])
                    next_F.append(R_F[front])
                else:
                    remaining = N - len(next_X)
                    subset_F = R_F[front]
                    # Calculate CD for this front
                    cd_front = self._crowding_distance(subset_F)
                    sorted_indices = np.argsort(-cd_front)
                    selected_indices = sorted_indices[:remaining]
                    next_X.append(R_X[front][selected_indices])
                    next_F.append(R_F[front][selected_indices])
                    break
            
            X = np.vstack(next_X)
            F = np.vstack(next_F)
            
        # Local Search
        if local_budget > 0 and time.time() < deadline - 2:
            # Identify best points
            # Sort by rank, then CD?
            fronts_final = self._non_dominated_sort(F)
            # Take first front (Pareto)
            if len(fronts_final) > 0:
                pareto_indices = fronts_final[0]
                # If too many, take top few based on CD or just all
                # We have local_budget evals.
                # Let's pick min(len(pareto_indices), local_budget // 5) points to refine
                n_refine = min(len(pareto_indices), local_budget // 5)
                if n_refine > 0:
                    # Shuffle to pick diverse ones? Or just take first n_refine
                    refine_indices = pareto_indices[:n_refine]
                    evals_per_point = local_budget // n_refine
                    
                    for idx in refine_indices:
                        if time.time() > deadline - 1:
                            break
                        x_best = X[idx].copy()
                        f_best = F[idx].copy()
                        
                        # Simple local search: Gaussian perturbation
                        for k in range(evals_per_point):
                            if time.time() > deadline - 1:
                                break
                            # Step size adaptive? Start with 10% of range
                            step = 0.1 * (hi - lo)
                            x_trial = x_best + np.random.normal(0, step)
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = np.array(problem(x_trial))
                            
                            # Check if f_trial is non-dominated by current best set?
                            # Or just if it dominates f_best?
                            # If it dominates f_best, replace.
                            if np.all(f_trial <= f_best) and np.any(f_trial < f_best):
                                x_best = x_trial
                                f_best = f_trial
                                X[idx] = x_best
                                F[idx] = f_best
                            # Optionally, if it's non-dominated by the whole set, add it?
                            # Adding to population is complex here. Just improving the best is safer.
                            
        return X

    def _tournament_select(self, X, F, ranks):
        idx1 = np.random.randint(len(X))
        idx2 = np.random.randint(len(X))
        
        if ranks[idx1] < ranks[idx2]:
            return X[idx1]
        elif ranks[idx2] < ranks[idx1]:
            return X[idx2]
        else:
            # Tie: use random or CD (not computed globally here, so random)
            return X[np.random.choice([idx1, idx2])]

    def _sbx_crossover(self, p1, p2, lo, hi):
        eta_c = 20
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = 1.0 / (2.0 * u)
                        beta_q = beta ** (1.0 / (eta_c + 1))
                        c1[i] = 0.5 * ((1 + beta_q) * x1 + (1 - beta_q) * x2)
                    else:
                        beta = 1.0 / (2.0 * (1.0 - u))
                        beta_q = beta ** (1.0 / (eta_c + 1))
                        c1[i] = 0.5 * ((1 + beta_q) * x2 + (1 - beta_q) * x1)
                    
                    # c2 is the other child
                    # Standard SBX:
                    # If we calculated c1 using x1, x2, c2 is usually the symmetric one?
                    # Actually, the formula for c2 is similar but uses the other branch?
                    # No, usually we generate c1 and c2 from the same parents.
                    # A common implementation:
                    # if u < 0.5:
                    #   beta1 = ...
                    #   c1 = ...
                    # else:
                    #   beta2 = ...
                    #   c2 = ...
                    # But we need both c1 and c2.
                    # Let's just use the same logic for c2 with a new random u?
                    # Or use the property that c1 + c2 = x1 + x2? (Only for uniform/arithmetic)
                    # For SBX, c1 and c2 are independent samples from the distribution.
                    # So we should generate c2 independently.
                    
                    u2 = np.random.rand()
                    if u2 <= 0.5:
                        beta2 = 1.0 / (2.0 * u2)
                        beta_q2 = beta2 ** (1.0 / (eta_c + 1))
                        c2[i] = 0.5 * ((1 + beta_q2) * x1 + (1 - beta_q2) * x2)
                    else:
                        beta2 = 1.0 / (2.0 * (1.0 - u2))
                        beta_q2 = beta2 ** (1.0 / (eta_c + 1))
                        c2[i] = 0.5 * ((1 + beta_q2) * x2 + (1 - beta_q2) * x1)
                    
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
                
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _polynomial_mutation(self, x, lo, hi):
        eta_m = 20
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                y = x[i]
                delta1 = (y - lo[i]) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0
                delta2 = (hi[i] - y) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0
                mut_pow = 1.0 / (eta_m + 1)
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy) ** (mut_pow + 1)
                    deltaq = val ** (1.0 / (mut_pow + 1)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy) ** (mut_pow + 1)
                    deltaq = val ** (1.0 / (mut_pow + 1)) - 1.0
                
                y = y + deltaq * (hi[i] - lo[i])
                x[i] = np.clip(y, lo[i], hi[i])
        return x

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_d = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1
        
        fronts = []
        current_front = np.where(n_d == 0)[0]
        
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        next_front.append(j)
            current_front = np.array(next_front)
            
        return fronts

    def _crowding_distance(self, F):
        m = len(F)
        if m <= 2:
            return np.full(m, np.inf)
        
        cd = np.zeros(m)
        for k in range(self.n_obj):
            sorted_indices = np.argsort(F[:, k])
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            if f_max - f_min > 1e-14:
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / (f_max - f_min)
        return cd
