import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        def eval_point(x):
            return np.array(problem(x))
        
        # Phase 1: Initial Population
        n_init = 30
        n_gen = 4
        n_offspring = 30
        
        # Generate LHS
        X_pop = np.zeros((n_init, dim))
        for i in range(dim):
            bin_edges = np.linspace(lo[i], hi[i], n_init + 1)
            samples = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) * np.random.rand(n_init)
            np.random.shuffle(samples)
            X_pop[:, i] = samples
            
        F_pop = np.array([eval_point(x) for x in X_pop])
        
        eval_count = n_init
        
        # Phase 2: NSGA-II Generations
        for g in range(n_gen):
            if time.time() > deadline - 10.0:
                break
                
            # Calculate ranks and crowding for current pop
            fronts = self._fast_non_dominated_sort(F_pop)
            ranks = np.zeros(n_init, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            cd = self._crowding_distance(F_pop, fronts)
            
            # Generate offspring
            X_off = np.zeros((n_offspring, dim))
            F_off = np.zeros((n_offspring, n_obj))
            
            eta_c = 20.0
            eta_m = 20.0
            prob_c = 0.9
            
            idx = 0
            while idx < n_offspring:
                if time.time() > deadline - 5.0:
                    break
                if eval_count + 1 > budget:
                    break
                    
                p1_idx = self._tournament(ranks, cd)
                p2_idx = self._tournament(ranks, cd)
                
                p1 = X_pop[p1_idx]
                p2 = X_pop[p2_idx]
                
                if np.random.rand() < prob_c:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                    
                c1 = self._polynomial_mutation(c1, lo, hi, eta_m)
                c2 = self._polynomial_mutation(c2, lo, hi, eta_m)
                
                if idx < n_offspring and eval_count + 1 <= budget:
                    X_off[idx] = c1
                    F_off[idx] = eval_point(c1)
                    eval_count += 1
                    idx += 1
                    
                if idx < n_offspring and eval_count + 1 <= budget:
                    X_off[idx] = c2
                    F_off[idx] = eval_point(c2)
                    eval_count += 1
                    idx += 1
                    
            # Combine and Select
            X_comb = np.vstack([X_pop, X_off[:idx]])
            F_comb = np.vstack([F_pop, F_off[:idx]])
            
            fronts = self._fast_non_dominated_sort(F_comb)
            ranks = np.zeros(len(X_comb), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            cd = self._crowding_distance(F_comb, fronts)
            
            # Selection: Keep top n_init
            # We need to select n_init individuals from the combined set
            # Standard NSGA-II selection
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= n_init:
                    selected_indices.extend(front)
                else:
                    # Sort front by crowding distance
                    front_list = list(front)
                    front_list.sort(key=lambda i: cd[i], reverse=True)
                    needed = n_init - len(selected_indices)
                    selected_indices.extend(front_list[:needed])
                    break
            
            X_pop = X_comb[selected_indices]
            F_pop = F_comb[selected_indices]
            
        # Phase 3: Local Search with remaining budget
        remaining_budget = budget - eval_count
        if remaining_budget > 0 and len(F_pop) > 0:
            # Find best 5 non-dominated points (Front 0)
            fronts = self._fast_non_dominated_sort(F_pop)
            if fronts:
                best_front = list(fronts[0])
                # Sort by crowding distance to pick diverse ones? Or just take first 5
                # Let's take the 5 with highest crowding distance
                cd = self._crowding_distance(F_pop, fronts)
                best_front.sort(key=lambda i: cd[i], reverse=True)
                candidates = best_front[:5]
                
                # Local search: random perturbation
                step_size = 0.1 * (hi - lo)
                
                for idx in candidates:
                    if remaining_budget <= 0:
                        break
                    best_x = X_pop[idx].copy()
                    best_f = F_pop[idx].copy()
                    
                    # Try a few perturbations
                    for _ in range(min(remaining_budget, 10)):
                        if remaining_budget <= 0:
                            break
                        new_x = best_x + np.random.randn(dim) * step_size * np.random.rand(dim)
                        new_x = np.clip(new_x, lo, hi)
                        new_f = eval_point(new_x)
                        remaining_budget -= 1
                        eval_count += 1
                        
                        # Check if new_f is non-dominated by best_f
                        # If new_f is better than best_f in all objectives, or non-dominated
                        # Simple check: if new_f is non-dominated by the current best set?
                        # To keep it simple, if new_f is non-dominated by best_f, accept it?
                        # No, we want to improve the front.
                        # If new_f dominates best_f, replace.
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_f = new_f
                            best_x = new_x
                        # Or if it's non-dominated, we might want to keep it?
                        # But we only have one slot per candidate in the final output list?
                        # We can update the population if it's better.
                        
                    # Update population
                    X_pop[idx] = best_x
                    F_pop[idx] = best_f

        return X_pop

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
            
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                    
        front = set()
        for i in range(n):
            if n_dom[i] == 0:
                front.add(i)
        if front:
            fronts.append(front)
                
        k = 0
        while k < len(fronts):
            next_front = set()
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    def _crowding_distance(self, F, fronts):
        n = len(F)
        cd = np.zeros(n)
        
        for front in fronts:
            m = len(front)
            if m <= 2:
                for idx in front:
                    cd[idx] = np.inf
                continue
                
            for k in range(self.n_obj):
                sorted_idx = sorted(front, key=lambda i: F[i][k])
                min_val = F[sorted_idx[0]][k]
                max_val = F[sorted_idx[-1]][k]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                if max_val == min_val:
                    continue
                    
                for i in range(1, m - 1):
                    idx = sorted_idx[i]
                    cd[idx] += (F[sorted_idx[i+1]][k] - F[sorted_idx[i-1]][k]) / (max_val - min_val)
                    
        return cd

    def _tournament(self, ranks, cd):
        i1 = np.random.randint(0, len(ranks))
        i2 = np.random.randint(0, len(ranks))
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            else:
                return i2

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = (1.0 + 1.0/eta) ** (alpha + 1.0)
                        ratio = (1.0 + (p2[i]-p1[i])/(hi[i]-lo[i])) / beta
                        if ratio > 0:
                            c1[i] = 0.5 * (p1[i] + p2[i] - (p2[i]-p1[i]) * (ratio - 1.0))
                            c2[i] = 0.5 * (p1[i] + p2[i] + (p2[i]-p1[i]) * (ratio - 1.0))
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta = (1.0 + 1.0/eta) ** (alpha + 1.0)
                            ratio = (1.0 + (p1[i]-p2[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c1[i] = 0.5 * (p1[i] + p2[i] + (p2[i]-p1[i]) * (ratio - 1.0))
                                c2[i] = 0.5 * (p1[i] + p2[i] - (p2[i]-p1[i]) * (ratio - 1.0))
                            else:
                                c1[i] = p1[i]
                                c2[i] = p2[i]
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = (1.0 + 1.0/eta) ** (alpha + 1.0)
                        ratio = (1.0 + (p1[i]-p2[i])/(hi[i]-lo[i])) / beta
                        if ratio > 0:
                            c1[i] = 0.5 * (p1[i] + p2[i] + (p1[i]-p2[i]) * (ratio - 1.0))
                            c2[i] = 0.5 * (p1[i] + p2[i] - (p1[i]-p2[i]) * (ratio - 1.0))
                        else:
                            alpha = 2.0 * u
                            beta = (1.0 + 1.0/eta) ** (alpha + 1.0)
                            ratio = (1.0 + (p2[i]-p1[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c1[i] = 0.5 * (p1[i] + p2[i] - (p1[i]-p2[i]) * (ratio - 1.0))
                                c2[i] = 0.5 * (p1[i] + p2[i] + (p1[i]-p2[i]) * (ratio - 1.0))
                            else:
                                c1[i] = p1[i]
                                c2[i] = p2[i]
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta):
        x_m = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                y = x_m[i] + delta * (hi[i] - lo[i])
                if y < lo[i]:
                    y = lo[i]
                elif y > hi[i]:
                    y = hi[i]
                x_m[i] = y
        return x_m
