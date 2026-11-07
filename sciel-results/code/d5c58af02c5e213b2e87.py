import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        N = 50
        if self.budget < N:
            X = lo + rng.random((self.budget, self.dim)) * span
            F = np.array([problem(x) for x in X])
            return X

        # Generate weight vectors
        weights = []
        for _ in range(N):
            w = rng.random(self.n_obj)
            w = w / np.sum(w)
            weights.append(w)
        weights = np.array(weights)
        
        # Determine neighborhoods (T=10)
        T = 10
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighborhoods = []
        for i in range(N):
            order = np.argsort(dists[i])
            neighbors = [j for j in order if j != i][:T]
            neighborhoods.append(neighbors)

        # Initialize population
        X = lo + rng.random((N, self.dim)) * span
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Ideal point
        z_star = np.min(F, axis=0)
        
        # Helper: Tchebycheff aggregation
        def tchebycheff(f, w, z):
            return np.max(w * np.abs(f - z))
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta_c, lo, hi, rng):
            child1 = np.zeros_like(x1)
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (2 * u) ** ((eta_c + 1) / 2)
                    else:
                        alpha = (2 * (1 - u)) ** ((eta_c + 1) / 2)
                    
                    c1 = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                    child1[i] = np.clip(c1, lo[i], hi[i])
            return child1

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta_m, lo, hi, rng):
            y = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    y[i] += delta_q * (hi[i] - lo[i])
                    y[i] = np.clip(y[i], lo[i], hi[i])
            return y

        eta_c = 20.0
        eta_m = 20.0
        
        while evals < self.budget:
            for i in range(N):
                if evals >= self.budget:
                    break
                
                # Select parents
                if rng.random() < 0.9:
                    candidates = neighborhoods[i]
                else:
                    candidates = list(range(N))
                
                p1_idx, p2_idx = rng.choice(candidates, 2, replace=False)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover and Mutation
                child = sbx_crossover(p1, p2, eta_c, lo, hi, rng)
                child = polynomial_mutation(child, eta_m, lo, hi, rng)
                
                # Evaluate
                F_child = problem(child)
                evals += 1
                
                # Update z_star
                z_star = np.minimum(z_star, F_child)
                
                # Update population
                g_i = tchebycheff(F[i], weights[i], z_star)
                g_child_i = tchebycheff(F_child, weights[i], z_star)
                
                if g_child_i < g_i:
                    X[i] = child
                    F[i] = F_child
                else:
                    # Check neighbors
                    for j in neighborhoods[i]:
                        g_j = tchebycheff(F[j], weights[j], z_star)
                        g_child_j = tchebycheff(F_child, weights[j], z_star)
                        if g_child_j < g_j:
                            X[j] = child
                            F[j] = F_child
                            break

        return X
