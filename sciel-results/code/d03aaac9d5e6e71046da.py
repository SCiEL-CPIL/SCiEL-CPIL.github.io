"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sort.
        Returns a list of lists, where each inner list contains indices of points
        in that front. Front 0 is the best.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # Domination sets
        n_dominated = [0] * n  # Number of points dominated by i
        fronts = [[]]  # List of fronts

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominated[i] += 1
            if n_dominated[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of points given by indices.
        Returns a dictionary mapping index to distance.
        """
        if len(indices) <= 2:
            return {i: float('inf') for i in indices}
        
        m = F.shape[1]
        cd = {i: 0.0 for i in indices}
        
        for k in range(m):
            # Sort indices by objective k
            sorted_indices = sorted(indices, key=lambda x: F[x, k])
            
            # End points have infinite distance
            cd[sorted_indices[0]] = float('inf')
            cd[sorted_indices[-1]] = float('inf')
            
            range_k = F[sorted_indices[-1], k] - F[sorted_indices[0], k]
            if range_k == 0:
                continue
                
            for i in range(1, len(sorted_indices) - 1):
                cd[sorted_indices[i]] += (F[sorted_indices[i + 1], k] - F[sorted_indices[i - 1], k]) / range_k
                
        return cd

    def _tournament_select(self, ranks, cds, count):
        """
        Binary tournament selection.
        ranks: dict mapping index to rank
        cds: dict mapping index to crowding distance
        count: number of parents to select
        """
        selected = []
        n = len(ranks)
        indices = list(ranks.keys())
        
        for _ in range(count):
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            idx1 = indices[i1]
            idx2 = indices[i2]
            
            # Compare rank first
            if ranks[idx1] < ranks[idx2]:
                selected.append(idx1)
            elif ranks[idx2] < ranks[idx1]:
                selected.append(idx2)
            else:
                # Tie-break by crowding distance
                if cds[idx1] > cds[idx2]:
                    selected.append(idx1)
                else:
                    selected.append(idx2)
                    
        return selected

    def _sbx_crossover(self, x1, x2, bounds, eta=20.0):
        """
        Simulated Binary Crossover.
        Returns two children.
        """
        dim = len(x1)
        c1 = np.zeros(dim)
        c2 = np.zeros(dim)
        
        for j in range(dim):
            if np.random.random() < 0.5:
                # No crossover
                c1[j] = x1[j]
                c2[j] = x2[j]
            else:
                if abs(x1[j] - x2[j]) < 1e-14:
                    c1[j] = x1[j]
                    c2[j] = x2[j]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1[j] = 0.5 * ((1.0 + beta) * x1[j] + (1.0 - beta) * x2[j])
                    c2[j] = 0.5 * ((1.0 - beta) * x1[j] + (1.0 + beta) * x2[j])
                    
                    # Clip to bounds
                    c1[j] = np.clip(c1[j], bounds[j][0], bounds[j][1])
                    c2[j] = np.clip(c2[j], bounds[j][0], bounds[j][1])
                    
        return c1, c2

    def _mutation(self, x, bounds, eta=20.0):
        """
        Polynomial Mutation.
        """
        dim = len(x)
        x_mut = x.copy()
        rate = 1.0 / dim
        
        for j in range(dim):
            if np.random.random() < rate:
                low, high = bounds[j]
                delta = high - low
                if delta == 0:
                    continue
                u = np.random.random()
                if u <= 0.5:
                    delta1 = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta1 = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x_mut[j] += delta1 * delta
                x_mut[j] = np.clip(x_mut[j], low, high)
                
        return x_mut

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 30
        
        # Initialize population
        X = np.random.uniform(lo, hi, (N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Main loop
        while used < budget:
            # Determine how many offspring to generate
            n_offspring = min(N, budget - used)
            
            if n_offspring == 0:
                break
                
            # Generate offspring
            Y = np.zeros((n_offspring, dim))
            FY = np.zeros((n_offspring, n_obj))
            
            # Prepare selection data for current population
            fronts = self._fast_non_dominated_sort(F)
            ranks = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Crowding distance for each front
            cds = {}
            for front in fronts:
                front_cds = self._crowding_distance(F, front)
                for idx in front:
                    cds[idx] = front_cds[idx]
            
            for i in range(n_offspring):
                # Select parents
                p1_idx = self._tournament_select(ranks, cds, 1)[0]
                p2_idx = self._tournament_select(ranks, cds, 1)[0]
                
                # Crossover
                c1, c2 = self._sbx_crossover(X[p1_idx], X[p2_idx], bounds)
                
                # We only need one child per evaluation to save budget?
                # Or we can generate two and evaluate both?
                # Standard NSGA-II generates N offspring from N parents.
                # Each offspring takes 2 parents.
                # So for n_offspring evaluations, we need n_offspring offspring.
                # We generate 2 children per crossover, but we only evaluate 1?
                # No, we should evaluate both if we have budget, or just one?
                # To maximize diversity, usually we generate 2 and keep both in the pool.
                # But we have a strict budget.
                # Let's generate 1 child per iteration to control budget exactly.
                # This is less efficient than NSGA-II but fits the budget.
                
                # Actually, let's just take c1.
                child = c1
                
                # Mutation
                child = self._mutation(child, bounds)
                
                # Evaluate
                f_child = problem(child)
                Y[i] = child
                FY[i] = f_child
                used += 1
                
            # Combine X and Y
            X_new = np.vstack([X, Y])
            F_new = np.vstack([F, FY])
            
            # Select next population
            fronts = self._fast_non_dominated_sort(F_new)
            
            X_next = np.zeros((N, dim))
            F_next = np.zeros((N, n_obj))
            
            count = 0
            for front in fronts:
                if count + len(front) <= N:
                    # All points in this front fit
                    for idx in front:
                        X_next[count] = X_new[idx]
                        F_next[count] = F_new[idx]
                        count += 1
                else:
                    # We need to select from this front
                    # Calculate crowding distance for this front
                    front_cds = self._crowding_distance(F_new, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda idx: front_cds[idx], reverse=True)
                    
                    remaining = N - count
                    for idx in sorted_front[:remaining]:
                        X_next[count] = X_new[idx]
                        F_next[count] = F_new[idx]
                        count += 1
                    break
                
            X = X_next
            F = F_next
            
        return X
# EVOLVE-BLOCK-END
