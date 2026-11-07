"""Model-and-optimize for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective NSGA-II:
    1) Initialize with space-filling Halton sequence.
    2) Use NSGA-II (Non-dominated Sorting Genetic Algorithm II) for the main search.
    3) This handles the multi-objective nature directly without scalarization.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- Helper: Halton Sequence ---
        def halton(dim, num):
            seq = np.zeros((num, dim))
            for i in range(num):
                for d in range(dim):
                    f = 1.0
                    r = 0.0
                    base = d + 2
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    seq[i, d] = r
            return seq

        # --- Helper: Non-dominated Sort ---
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    current_front.append(i)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        # --- Helper: Crowding Distance ---
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                min_val = np.min(F_sub[:, k])
                max_val = np.max(F_sub[:, k])
                if max_val - min_val < 1e-10:
                    continue
                sorted_idx = np.argsort(F_sub[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / (max_val - min_val)
            
            return cd

        # --- Helper: SBX Crossover ---
        def sbx_crossover(x1, x2, prob=0.9, eta=15.0):
            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            if rng.random() > prob:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for i in range(len(x1)):
                if x1[i] == x2[i]:
                    continue
                if rng.random() < 0.5:
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 1.0 / (eta + 1.0):
                            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                        c2[i] = 0.5 * ((1.0 + 1.0/beta) * x2[i] + (1.0 - 1.0/beta) * x1[i])
                    else:
                        c1[i], c2[i] = c2[i], c1[i]
            
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        # --- Helper: Polynomial Mutation ---
        def polynomial_mutation(x, prob=1.0/dim, eta=20.0):
            x = np.asarray(x).copy()
            for i in range(len(x)):
                if rng.random() < prob:
                    u = rng.random()
                    if u <= 0.5:
                        delta1 = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta1 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta1 * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # --- Main Logic ---
        
        # 1. Initialize Population
        pop_size = 10
        if self.budget < pop_size * 2:
            X_pop = np.random.uniform(lo, hi, size=(self.budget, dim))
            F_pop = np.array([eval_real(x) for x in X_pop])
            return None
            
        X_init = halton(dim, pop_size)
        F_init = np.array([eval_real(x) for x in X_init])
        
        # 2. Evolutionary Loop
        while self.eval_count < self.budget and time.time() < deadline:
            X_offspring = []
            F_offspring = []
            
            for _ in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                    
                idx1 = rng.integers(0, len(X_init))
                idx2 = rng.integers(0, len(X_init))
                
                p1 = X_init[idx1]
                p2 = X_init[idx2]
                
                c1, c2 = sbx_crossover(p1, p2)
                
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                f1 = eval_real(c1)
                if self.eval_count >= self.budget:
                    break
                f2 = eval_real(c2)
                
                X_offspring.append(c1)
                F_offspring.append(f1)
                X_offspring.append(c2)
                F_offspring.append(f2)
                
            if not X_offspring:
                break

            X_offspring = np.array(X_offspring)
            F_offspring = np.array(F_offspring)
            
            X_comb = np.vstack([X_init, X_offspring])
            F_comb = np.vstack([F_init, F_offspring])
            
            fronts = non_dominated_sort(F_comb)
            
            X_next = []
            F_next = []
            count = 0
            
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    count += len(front)
                else:
                    need = pop_size - count
                    cd = crowding_distance(F_comb, front)
                    pairs = [(cd[i], front[i]) for i in range(len(front))]
                    pairs.sort(reverse=True)
                    for _, idx in pairs[:need]:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    count += need
                    break
            
            X_init = np.array(X_next)
            F_init = np.array(F_next)
            
        return None
# EVOLVE-BLOCK-END
