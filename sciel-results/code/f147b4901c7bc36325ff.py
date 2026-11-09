import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # State
        X = []
        F = []
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            fx = np.array(problem(x))
            X.append(x)
            F.append(fx)
            eval_count += 1
            return fx
            
        def check_time():
            return time.time() < deadline - 5.0

        # 1. Initial Sampling: LHS
        n_init = 40
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            bin_edges = np.linspace(lo[i], hi[i], n_init + 1)
            samples = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) * np.random.rand(n_init)
            np.random.shuffle(samples)
            X_init[:, i] = samples
            
        for i in range(n_init):
            if not check_time():
                break
            eval_point(X_init[i])
            
        X_arr = np.array(X)
        F_arr = np.array(F)
        
        # Helper: Non-dominated sort (simple, for small populations)
        def non_dominated_sort(F_pop):
            n = len(F_pop)
            if n == 0:
                return []
            S = [set() for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F_pop[i] <= F_pop[j]) and np.any(F_pop[i] < F_pop[j]):
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

        def crowding_distance(F_pop, fronts):
            n = len(F_pop)
            cd = np.zeros(n)
            for front in fronts:
                m = len(front)
                if m <= 2:
                    for idx in front:
                        cd[idx] = np.inf
                    continue
                for k in range(n_obj):
                    sorted_idx = sorted(front, key=lambda i: F_pop[i][k])
                    min_val = F_pop[sorted_idx[0]][k]
                    max_val = F_pop[sorted_idx[-1]][k]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    if max_val == min_val:
                        continue
                    for i in range(1, m - 1):
                        idx = sorted_idx[i]
                        cd[idx] += (F_pop[sorted_idx[i+1]][k] - F_pop[sorted_idx[i-1]][k]) / (max_val - min_val)
            return cd

        def tournament_select(ranks, cd):
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

        def sbx_crossover(p1, p2, lo, hi, eta=20.0):
            c1 = p1.copy()
            c2 = p2.copy()
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if abs(p1[i] - p2[i]) > 1e-14:
                        u = np.random.rand()
                        if u <= 0.5:
                            alpha = 2.0 * u
                            beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                            ratio = (1.0 + (p2[i]-p1[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c1[i] = 0.5 * (p1[i] + p2[i] - (p2[i]-p1[i]) * (ratio - 1.0))
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                            ratio = (1.0 + (p1[i]-p2[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c1[i] = 0.5 * (p1[i] + p2[i] + (p2[i]-p1[i]) * (ratio - 1.0))
                        
                        # c2 is the other child
                        if u <= 0.5:
                            alpha = 2.0 * (1.0 - u)
                            beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                            ratio = (1.0 + (p1[i]-p2[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c2[i] = 0.5 * (p1[i] + p2[i] + (p1[i]-p2[i]) * (ratio - 1.0))
                        else:
                            alpha = 2.0 * u
                            beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                            ratio = (1.0 + (p2[i]-p1[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c2[i] = 0.5 * (p1[i] + p2[i] - (p2[i]-p1[i]) * (ratio - 1.0))
            return c1, c2

        def polynomial_mutation(x, lo, hi, eta=20.0):
            x_m = x.copy()
            for i in range(dim):
                if np.random.rand() < 1.0 / dim:
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

        # 2. Main Loop: NSGA-II style
        while eval_count < budget and check_time():
            # Sort current population
            fronts = non_dominated_sort(F_arr)
            ranks = np.zeros(len(F_arr), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            cd = crowding_distance(F_arr, fronts)
            
            # Select parents
            p1_idx = tournament_select(ranks, cd)
            p2_idx = tournament_select(ranks, cd)
            
            p1 = X_arr[p1_idx]
            p2 = X_arr[p2_idx]
            
            # Crossover
            if np.random.rand() < 0.9:
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
            else:
                c1 = p1.copy()
                c2 = p2.copy()
                
            # Mutation
            c1 = polynomial_mutation(c1, lo, hi)
            c2 = polynomial_mutation(c2, lo, hi)
            
            # Evaluate
            f1 = eval_point(c1)
            if f1 is None:
                break
            f2 = eval_point(c2)
            if f2 is None:
                break
            
            # Update arrays
            X_arr = np.vstack([X_arr, c1])
            F_arr = np.vstack([F_arr, f1])
            if f2 is not None:
                 X_arr = np.vstack([X_arr, c2])
                 F_arr = np.vstack([F_arr, f2])
            
            # Truncation: Keep best N (N = budget)
            if len(F_arr) > budget:
                # Re-sort
                fronts = non_dominated_sort(F_arr)
                ranks = np.zeros(len(F_arr), dtype=int)
                for i, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = i
                cd = crowding_distance(F_arr, fronts)
                
                # Sort by rank then crowding distance (descending)
                # We want to keep the best ones.
                # Create a list of indices sorted by (rank, -cd)
                indices = np.arange(len(F_arr))
                # Simple sort: stable sort by rank, then stable sort by -cd?
                # Actually, we just need to pick the top 'budget' ones.
                # Let's create a combined score for sorting: rank is primary, cd is secondary (higher is better).
                # Since we want to remove the worst, we sort ascending by rank, and for same rank, descending by cd.
                # Then we take the first 'budget' elements.
                
                # Efficient way:
                # Sort indices by rank
                idx_sorted = np.argsort(ranks, kind='stable')
                # Within same rank, sort by -cd (descending cd)
                # This is a bit tricky with numpy argsort stability across groups.
                # Let's just do a simple python sort for clarity since N is small (<= 400)
                sorted_indices = sorted(range(len(F_arr)), key=lambda i: (ranks[i], -cd[i]))
                
                # Keep first 'budget'
                keep_indices = sorted_indices[:budget]
                
                X_arr = X_arr[keep_indices]
                F_arr = F_arr[keep_indices]
                
        return X_arr
