import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20  # Population size
        T = 5   # Neighborhood size
        
        # 1. Initialization
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # 2. Weight Vectors
        # Generate N random weight vectors on the simplex
        weights = np.random.rand(N, self.n_obj)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # 3. Neighborhood Definition
        # Find T closest neighbors for each weight vector
        neighbors = np.zeros((N, T), dtype=int)
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            # Exclude self
            dists[i] = np.inf
            # Get indices of T smallest distances
            neighbors[i] = np.argsort(dists)[:T]
            
        # 4. Ideal Point
        z_star = np.min(F, axis=0)
        
        # 5. Evolutionary Loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Pick a random subproblem
            k = np.random.randint(N)
            
            # Select parents from neighborhood of k
            # Choose 2 distinct indices from neighbors[k]
            # If T < 2, handle differently, but T=5 is fine
            parent_indices = np.random.choice(neighbors[k], size=2, replace=False)
            p1 = X[parent_indices[0]]
            p2 = X[parent_indices[1]]
            
            # Crossover: Simulated Binary Crossover (SBX)
            # eta = 20 is standard
            eta = 20
            child = np.zeros_like(p1)
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    child[j] = p1[j]
                else:
                    child[j] = p2[j]
                    
            # Apply SBX
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(p1[j] - p2[j]) > 1e-14:
                        u = np.random.rand()
                        beta = 0
                        if u <= 1 / (eta + 1):
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                    else:
                        child[j] = p1[j]
                        
            # Mutation: Polynomial Mutation
            # Rate 1/dim, eta_m = 20
            eta_m = 20
            for j in range(self.dim):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    
                    range_j = hi[j] - lo[j]
                    child[j] += delta * range_j
                    child[j] = np.clip(child[j], lo[j], hi[j])
            
            # Evaluate Child
            fc = problem(child)
            evals += 1
            
            # Update Ideal Point
            z_star = np.minimum(z_star, fc)
            
            # Replacement Strategy
            # Calculate Tchebycheff indicator for child with weight lambda_k
            # g_te(f, lambda) = max_i { lambda_i * (f_i - z_star_i) }
            
            def tcheb(f_vec, w_vec):
                return np.max(w_vec * (f_vec - z_star))
            
            g_child = tcheb(fc, weights[k])
            
            # Find the worst individual in the neighborhood N(k) U {k}
            # We compare based on the weight vector of the subproblem k?
            # Standard MOEA/D: Replace the individual x_j in N(k) U {k} such that
            # g_te(f(x_j), lambda^k) is maximized, IF g_child < that max.
            
            candidates = np.unique(np.concatenate(([k], neighbors[k])))
            
            worst_idx = -1
            worst_val = np.inf
            
            for j in candidates:
                g_j = tcheb(F[j], weights[k])
                if g_j > worst_val:
                    worst_val = g_j
                    worst_idx = j
                    
            if g_child < worst_val:
                X[worst_idx] = child
                F[worst_idx] = fc
                
        return X
