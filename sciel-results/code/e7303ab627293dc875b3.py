"""Rewritten optimizer for WFG9 using MOEA/D."""
# EVOLVE-BLOCK-START
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
        
        # Population size / Number of subproblems
        # Use a simplex lattice for 3 objectives with resolution 5: C(5+3-1, 3-1) = C(7,2) = 21
        N = 21
        if N > self.budget:
            N = self.budget
            
        # Generate deterministic simplex lattice weight vectors
        weights = self._simplex_lattice(N, self.n_obj)
        
        # Initialize population with deterministic LHS
        X = lo + (hi - lo) * self._deterministic_lhs(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point z*
        z_star = np.min(F, axis=0)
        
        # Define neighbors based on weight vector proximity
        neighbors = self._define_neighbors(weights, N)
        
        # Helper: Tchebycheff aggregation
        def tchebycheff(f, weight, z):
            return np.max(weight * np.abs(f - z))
            
        # Calculate initial Tchebycheff values
        g_vals = np.array([tchebycheff(F[i], weights[i], z_star) for i in range(N)])
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=30.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation with lower rate for high dimensions
        def polynomial_mutate(x, eta=30.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.3:  # Lower mutation rate for 12D
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x_mut[i] += delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Shuffle order to avoid bias
            order = np.random.permutation(N)
            
            for i in order:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select parents from neighborhood of i
                if len(neighbors[i]) < 2:
                    p1_idx = i
                    p2_idx = i
                else:
                    p1_idx, p2_idx = np.random.choice(neighbors[i], size=2, replace=False)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = polynomial_mutate(c1)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f1)
                
                # Calculate Tchebycheff for child with respect to subproblem i
                g_child_i = tchebycheff(f1, weights[i], z_star)
                
                # Update subproblem i
                if g_child_i < g_vals[i]:
                    X[i] = c1
                    F[i] = f1
                    g_vals[i] = g_child_i
                    
                    # Update neighbors
                    for j in neighbors[i]:
                        g_child_j = tchebycheff(f1, weights[j], z_star)
                        if g_child_j < g_vals[j]:
                            X[j] = c1
                            F[j] = f1
                            g_vals[j] = g_child_j
                            
        # Return the non-dominated front from the final population
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0: return []
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            front = [set() for _ in range(n)]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
                if n_doms[p] == 0:
                    ranks[p] = 0
                    front[0].add(p)
                    
            i = 0
            while front[i]:
                next_front = set()
                for p in front[i]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.add(q)
                i += 1
                if next_front:
                    front[i] = next_front
                else:
                    break
            return [list(f) for f in front if f]

        fronts = fast_non_dominated_sort(F)
        if fronts:
            return F[fronts[0]]
        else:
            return F
    
    def _simplex_lattice(self, N, n_obj):
        """Generate deterministic simplex lattice weight vectors."""
        if n_obj == 3:
            # For 3 objectives, use a triangular grid
            # Find resolution h such that C(h+2, 2) >= N
            h = 0
            while (h + 2) * (h + 1) // 2 < N:
                h += 1
            
            weights = []
            for i in range(h + 1):
                for j in range(h + 1 - i):
                    k = h - i - j
                    weights.append([i / h, j / h, k / h])
            
            weights = np.array(weights[:N])
            # Normalize to ensure they sum to 1
            weights = weights / weights.sum(axis=1, keepdims=True)
            return weights
        else:
            # Fallback: random weights for non-3-objective problems
            weights = np.random.random((N, n_obj))
            weights /= weights.sum(axis=1, keepdims=True)
            return weights
    
    def _deterministic_lhs(self, n, dim):
        """Generate deterministic Latin Hypercube Sampling points."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Use a deterministic permutation based on the dimension
            perm = np.arange(n)
            # Apply a simple deterministic shuffle based on dimension
            for i in range(n - 1):
                j = (i * 7 + d * 13) % (n - i - 1) + i + 1
                perm[i], perm[j] = perm[j], perm[i]
            points[:, d] = (perm + 0.5) / n
        return points
    
    def _define_neighbors(self, weights, N):
        """Define neighbors based on weight vector proximity."""
        neighbors = []
        # Find 5 nearest neighbors for each weight vector
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            dists[i] = np.inf
            # Take 5 nearest neighbors
            n_neighbors = min(5, N - 1)
            neighbor_indices = np.argsort(dists)[:n_neighbors]
            neighbors.append(list(neighbor_indices))
        return neighbors
# EVOLVE-BLOCK-END
