"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(100, self.budget // 2)
        # Crossover/mutation rates
        cx_rate = 0.9
        mut_rate = 1.0 / dim

        # Initial population
        rng = np.random.default_rng()
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Check deadline
        if time.time() > deadline:
            return X

        def fast_non_dominated_sort(F):
            """Return list of fronts, each containing indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1
            
            front = [i for i in range(n) if nd[i] == 0]
            if not front:
                return [[i for i in range(n)]]
            fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F[front_indices, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F[front_indices[sorted_idx[0]], m]
                f_max = F[front_indices[sorted_idx[-1]], m]
                if f_max > f_min:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[front_indices[sorted_idx[i+1]], m] - 
                                               F[front_indices[sorted_idx[i-1]], m]) / (f_max - f_min)
            return cd

        def tournament_select(fronts, F, k=2):
            """Select one individual via binary tournament on (front, crowding)."""
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distances per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    front_cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = front_cd[i]
                else:
                    for idx in front:
                        cd[idx] = np.inf
            
            # Tournament
            i = rng.integers(len(F))
            j = rng.integers(len(F))
            if rank[i] < rank[j]:
                return i
            elif rank[j] < rank[i]:
                return j
            else:
                return i if cd[i] > cd[j] else j

        def simulate_binary_crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() > cx_rate:
                return x1.copy(), x2.copy()
            
            x1c = x1.copy()
            x2c = x2.copy()
            
            for j in range(dim):
                if rng.random() < 0.5:
                    continue
                if abs(x1[j] - x2[j]) > 1e-14:
                        if x1[j] < x2[j]:
                            x_lo, x_hi = x1[j], x2[j]
                            idx_lo, idx_hi = 0, 1
                        else:
                            x_lo, x_hi = x2[j], x1[j]
                            idx_lo, idx_hi = 1, 0
                        
                        beta = 1.0
                        if rng.random() < 0.5:
                            beta = (0.5 * (x1c[idx_lo] - x1c[idx_hi]) / (x_lo - x_hi)) ** (1.0 / 2.1)
                        else:
                            beta = (0.5 * (x1c[idx_hi] - x1c[idx_lo]) / (x_hi - x_lo)) ** (1.0 / 2.1)
                        
                        x1c[idx_lo] = 0.5 * (x1c[idx_lo] + x1c[idx_hi]) - beta * 0.5 * (x1c[idx_hi] - x1c[idx_lo])
                        x1c[idx_hi] = 0.5 * (x1c[idx_lo] + x1c[idx_hi]) + beta * 0.5 * (x1c[idx_hi] - x1c[idx_lo])
            
            return x1c, x2c

        def polynomial_mutation(x):
            """Polynomial mutation."""
            xm = x.copy()
            for j in range(dim):
                if rng.random() < mut_rate:
                    y = xm[j]
                    yl, yu = lo[j], hi[j]
                    delta1 = (y - yl) / (yu - yl) if yu > yl else 0
                    delta2 = (yu - y) / (yu - yl) if yu > yl else 0
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (2.1 + 1.0))
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + (2.0 * rand - 1.0) * (xy ** (2.1 + 1.0))
                    deltaq = (val - 1.0) * (yu - yl)
                    xm[j] = y + deltaq
            return np.clip(xm, lo, hi)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Create offspring population
            offspring_X = []
            for _ in range(pop_size):
                p1 = tournament_select(fronts, F)
                p2 = tournament_select(fronts, F)
                c1, c2 = simulate_binary_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                used += 2
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            if len(offspring_X) == 0:
                break
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            
            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(new_X)
                    front_F = combined_F[front]
                    front_cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-front_cd)
                    for i in range(remaining):
                        idx = front[sorted_by_cd[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
