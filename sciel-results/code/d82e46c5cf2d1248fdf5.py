import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We should stop at 320s to be safe.
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Pop size 20 as per insights
        pop_size = 20
        
        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), np.array([])
            
            ranks = np.zeros(n, dtype=int)
            crowding = np.zeros(n)
            
            front = set(range(n))
            rank = 0
            while len(front) > 0:
                front_list = list(front)
                nd_in_front = set()
                for i in front_list:
                    is_nd = True
                    for j in front_list:
                        if i == j:
                            continue
                        if np.all(F[i] >= F[j]) and np.any(F[i] > F[j]):
                            is_nd = False
                            break
                    if is_nd:
                        nd_in_front.add(i)
                
                for i in nd_in_front:
                    ranks[i] = rank
                
                if len(nd_in_front) > 1:
                    for k in range(n_obj):
                        sorted_front = sorted(nd_in_front, key=lambda x: F[x, k])
                        f_min = F[sorted_front[0], k]
                        f_max = F[sorted_front[-1], k]
                        if f_max <= f_min:
                            continue
                        for idx, i in enumerate(sorted_front):
                            if idx == 0 or idx == len(sorted_front) - 1:
                                crowding[i] = np.inf
                            else:
                                crowding[i] += (F[sorted_front[idx+1], k] - F[sorted_front[idx-1], k]) / (f_max - f_min)
                
                rank += 1
                front = front - nd_in_front
            
            return ranks, crowding
        
        def tournament_select(X, F, ranks, crowding, k=2):
            # Select based on rank then crowding
            # Create a combined score for tournament? Or just pick random k and compare.
            # Comparing ranks and crowding directly is better.
            indices = np.random.randint(0, len(F), size=k)
            best_idx = indices[0]
            for i in indices[1:]:
                # Lower rank is better
                if ranks[i] < ranks[best_idx]:
                    best_idx = i
                elif ranks[i] == ranks[best_idx]:
                    # Higher crowding is better
                    if crowding[i] > crowding[best_idx]:
                        best_idx = i
            return X[best_idx]
        
        def sbx_crossover(x1, x2, eta=20.0):
            x = np.zeros_like(x1)
            for i in range(len(x1)):
                if x1[i] == x2[i]:
                    x[i] = x1[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    if np.random.rand() < 0.5:
                        x[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                    else:
                        x[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
            return x
        
        def polynomial_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(len(x_mut)):
                if np.random.rand() < 1.0 / len(x_mut):
                    range_i = hi[i] - lo[i]
                    if range_i < 1e-14:
                        continue
                    y = (x_mut[i] - lo[i]) / range_i
                    y = np.clip(y, 0, 1)
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (1 + 2 * y) ** (eta + 1.0)
                        delta = (1.0 / delta1) ** (1.0 / (eta + 1.0)) - 1
                    else:
                        delta2 = (1 + 2 * (1 - y)) ** (eta + 1.0)
                        delta = 1 - (1.0 / delta2) ** (1.0 / (eta + 1.0))
                    if delta >= 0:
                        x_mut[i] = lo[i] + (y + delta * (1 - y)) * range_i
                    else:
                        x_mut[i] = lo[i] + (y + delta * y) * range_i
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            X_off = []
            F_off = []
            
            # We generate pop_size offspring
            for _ in range(pop_size):
                if evaluations >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select parents
                p1 = tournament_select(X, F, ranks, crowding)
                p2 = tournament_select(X, F, ranks, crowding)
                
                # Crossover
                child = sbx_crossover(p1, p2)
                
                # Mutation
                child = polynomial_mutation(child)
                
                # Evaluate
                f_child = np.array(problem(child))
                evaluations += 1
                X_off.append(child)
                F_off.append(f_child)
            
            if len(X_off) == 0:
                break
            
            X_off = np.array(X_off)
            F_off = np.array(F_off)
            
            # Combine
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Sort
            ranks, crowding = non_dominated_sort(F_all)
            
            # Select best pop_size
            # Lexsort: rank ascending, crowding descending
            sort_idx = np.lexsort((-crowding, ranks))
            X = X_all[sort_idx[:pop_size]]
            F = F_all[sort_idx[:pop_size]]
            
            # Update ranks/crowding for the new population for next tournament
            # We need ranks and crowding for the *current* population X, F
            # The non_dominated_sort above was on F_all. We need to re-calculate or extract.
            # Extracting is faster.
            # But for simplicity and since pop is small (40), re-calculating is fine.
            # Actually, we can just re-run non_dominated_sort on the new X, F.
            ranks, crowding = non_dominated_sort(F)
        
        return np.zeros((1, dim))
