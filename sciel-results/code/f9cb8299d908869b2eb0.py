"""Multi-objective optimizer for RE21 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        range_ = hi - lo
        
        # Population size
        P = 20
        # Number of generations to fit budget
        # Init P + Gens * P = Budget
        # 200 = 20 + 9 * 20
        n_gens = (self.budget - P) // P
        
        rng = np.random.default_rng(0)
        
        # State
        X = []
        F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Fast non-dominated sorting."""
            n = len(F)
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].add(q)
                    elif dominates(F[q], F[p]):
                        n_dominated[p] += 1
            
            for p in range(n):
                if n_dominated[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dominated[q] -= 1
                        if n_dominated[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            
            # Remove the last empty front if any
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if not indices:
                return {}
            
            n = len(indices)
            if n <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            m = self.n_obj
            
            for k in range(m):
                # Sort by k-th objective
                sorted_indices = sorted(indices, key=lambda i: F[i][k])
                # Assign infinity to boundary points
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                # Normalize
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / f_range
            
            return dist

        # Initialize population
        for _ in range(P):
            x = lo + range_ * rng.random(self.dim)
            x, f = evaluate(x)
            X.append(x)
            F.append(f)
            
        # Main loop
        for g in range(n_gens):
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(P):
                # Selection: Tournament
                # Pick 2 random parents
                p1_idx = rng.integers(len(X))
                p2_idx = rng.integers(len(X))
                
                # Simple domination check for tournament
                if dominates(F[p1_idx], F[p2_idx]):
                    parent1_idx = p1_idx
                elif dominates(F[p2_idx], F[p1_idx]):
                    parent1_idx = p2_idx
                else:
                    parent1_idx = p1_idx if rng.random() < 0.5 else p2_idx
                    
                # Second parent
                p3_idx = rng.integers(len(X))
                p4_idx = rng.integers(len(X))
                if dominates(F[p3_idx], F[p4_idx]):
                    parent2_idx = p3_idx
                elif dominates(F[p4_idx], F[p3_idx]):
                    parent2_idx = p4_idx
                else:
                    parent2_idx = p3_idx if rng.random() < 0.5 else p4_idx
                
                x1 = X[parent1_idx]
                x2 = X[parent2_idx]
                
                # SBX Crossover
                if rng.random() < 0.9:
                    # Simulated Binary Crossover
                    u = rng.random(self.dim)
                    beta1 = np.where(u < 0.5, (2 * u) ** (1 / 21), (2 * (1 - u)) ** (1 / 21))
                    beta2 = 1 / beta1
                    x_child1 = 0.5 * ((1 + beta1) * x1 + (1 - beta1) * x2)
                    x_child2 = 0.5 * ((1 + beta2) * x1 + (1 - beta2) * x2)
                else:
                    x_child1 = x1.copy()
                    x_child2 = x2.copy()
                
                # Polynomial Mutation
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 21) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 21)
                        x_child1[d] += delta * (hi[d] - lo[d])
                        
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 21) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 21)
                        x_child2[d] += delta * (hi[d] - lo[d])
                
                x_child1, f_child1 = evaluate(x_child1)
                x_child2, f_child2 = evaluate(x_child2)
                
                offspring_X.append(x_child1)
                offspring_F.append(f_child1)
                offspring_X.append(x_child2)
                offspring_F.append(f_child2)
                
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break
                
            # Combine parents and offspring
            all_X = X + offspring_X
            all_F = F + offspring_F
            
            # Non-dominated sort
            fronts = non_dominated_sort(all_F)
            
            # Select next generation
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= P:
                    # Take all
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need to select only some from this front
                    # Calculate crowding distance
                    dist = crowding_distance(all_F, front)
                    # Sort by distance descending
                    sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                    needed = P - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break
            
            X = new_X
            F = new_F
            
        return X
# EVOLVE-BLOCK-END
