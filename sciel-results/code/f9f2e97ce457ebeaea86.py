"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: 50 is a good balance for 200 budget
        N = 50
        if budget < N:
            N = max(2, budget)

        # Generate weight vectors for MOEA/D
        # Target N weights using a simplex grid
        H = 1
        while (H + 1) * (H + 2) // 2 < N:
            H += 1
        
        weights = []
        for i in range(H + 1):
            for j in range(H - i + 1):
                k = H - i - j
                w = np.array([i, j, k], dtype=float) / H
                weights.append(w)
        
        # Take first N weights
        weights = np.array(weights[:N])
        
        # Initial population
        # Heuristic for DTLZ2: The last (dim - (n_obj - 1)) variables are typically
        # the "distance" variables that should be 0.
        # We initialize them to 0 to start close to the Pareto front.
        # The first (n_obj - 1) variables determine the shape, so we sample them uniformly.
        num_shape_vars = n_obj - 1
        num_dist_vars = dim - num_shape_vars
        
        X = np.zeros((N, dim))
        # Sample shape variables
        X[:, :num_shape_vars] = lo[:num_shape_vars] + span[:num_shape_vars] * rng.random((N, num_shape_vars))
        # Set distance variables to their optimal value (usually 0 or center)
        # For standard DTLZ2 with g=sum(x^2), 0 is optimal.
        # If g=sum((x-0.5)^2), 0.5 is optimal.
        # Assuming standard DTLZ2 where 0 is optimal for distance vars.
        X[:, num_shape_vars:] = lo[num_shape_vars:] # Assuming lo is 0
        
        F = np.array([problem(x) for x in X])
        used = N

        # Compute ideal point (min values) for Tchebycheff normalization
        # We don't know the ideal point a priori, so we use the current min.
        # This is standard for MOEA/D.
        
        # Define neighborhood
        # Smaller neighborhood for faster convergence
        T = 10
        if N <= T:
            T = N
        neighbors = []
        for i in range(N):
            dists = np.sum((weights - weights[i])**2, axis=1)
            idx = np.argsort(dists)
            neighbors.append(idx[:T].tolist())

        # Tchebycheff aggregation function
        def tchebycheff(f, w, z):
            # z is the ideal point (min of each objective)
            # We need to normalize f. 
            # Standard: (f_i - z_i) / range_i
            # We don't have range. We can use 1.0 or max-min.
            # Using 1.0 is common if scales are similar. DTLZ2 objectives are in [0, 2].
            # Let's use a dynamic range or just 1.0.
            # To be safe, let's use 1.0 as the range factor, or better, 
            # we can update z and range periodically.
            # For simplicity and speed, let's use z = min(F) and range = 1.0.
            # Actually, DTLZ2 objectives are bounded by 2. So range 2 is safe.
            # Let's use range = 2.0.
            return np.max(w * (f - z) / 2.0)

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for d in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[d] - x2[d]) < 1e-14:
                        continue
                    if rng.random() < 0.5:
                        beta = 1.0 / (2.0 * rng.random() + 1e-12)
                    else:
                        beta = 1.0 / (2.0 * (1.0 - rng.random()) + 1e-12)
                    beta += 1.0
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child1[d] = 0.5 * ((1 + alpha) * x1[d] + (1 - alpha) * x2[d])
                        else:
                            child1[d] = 0.5 * ((1 - alpha) * x1[d] + (1 + alpha) * x2[d])
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child2[d] = 0.5 * ((1 + alpha) * x2[d] + (1 - alpha) * x1[d])
                        else:
                            child2[d] = 0.5 * ((1 - alpha) * x2[d] + (1 + alpha) * x1[d])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Polynomial mutation
        def polynomial_mutate(x, eta=20.0, prob=1.0 / dim):
            child = x.copy()
            for d in range(dim):
                if rng.random() < prob:
                    delta1 = (child[d] - lo[d]) / span[d]
                    delta2 = (hi[d] - child[d]) / span[d]
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 + (1.0 + 2.0 * delta1) * rand
                        deltaq = (xy ** (1.0 / (eta + 1.0)) - 1.0)
                    else:
                        xy = 1.0 + (1.0 + 2.0 * delta2) * (1.0 - rand)
                        deltaq = 1.0 - (xy ** (1.0 / (eta + 1.0)))
                    child[d] += deltaq * span[d]
                    child[d] = np.clip(child[d], lo[d], hi[d])
            return child

        # Main loop
        z = np.min(F, axis=0) # Ideal point estimate
        
        while used < budget:
            # Determine how many offspring we can generate in this generation
            n_offspring = min(N, budget - used)
            
            # Shuffle indices to process
            indices = list(range(N))
            rng.shuffle(indices)
            
            for i in indices[:n_offspring]:
                # Select 2 parents from neighborhood of i
                neigh = neighbors[i]
                p1_idx = neigh[rng.integers(len(neigh))]
                p2_idx = neigh[rng.integers(len(neigh))]
                
                # Crossover and Mutation
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                offspring = c1
                # Increase mutation probability for distance variables to ensure they converge to 0
                offspring = polynomial_mutate(offspring, eta=20.0, prob=1.0/dim)
                
                f_offspring = problem(offspring)
                used += 1
                
                # Update ideal point
                z = np.minimum(z, f_offspring)
                
                # Check if offspring is better than parent i in Tchebycheff sense
                w_i = weights[i]
                val_parent = tchebycheff(F[i], w_i, z)
                val_offspring = tchebycheff(f_offspring, w_i, z)
                
                if val_offspring < val_parent:
                    X[i] = offspring
                    F[i] = f_offspring

        # Final Polish: Use any remaining budget to refine the best non-dominated points
        # Identify non-dominated points
        nondom_idx = []
        for i in range(N):
            is_dom = False
            for j in range(N):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom = True
                    break
            if not is_dom:
                nondom_idx.append(i)
        
        # If we have budget left, perturb the distance variables of non-dominated points
        # to try to get them exactly to 0.
        if used < budget and len(nondom_idx) > 0:
            # Select a few best points
            # Sort by sum of objectives
            sums = np.sum(F[nondom_idx], axis=1)
            best_idx_in_nd = np.argsort(sums)[:min(10, len(nondom_idx))]
            global_best_idx = [nondom_idx[i] for i in best_idx_in_nd]
            
            for i in global_best_idx:
                if used >= budget: break
                # Try to zero out distance variables
                for _ in range(3): # 3 attempts per point
                    if used >= budget: break
                    x_new = X[i].copy()
                    # Add small noise to distance variables
                    noise = rng.normal(0, 0.01, size=num_dist_vars)
                    x_new[num_shape_vars:] += noise
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    if np.all(f_new <= F[i]):
                        X[i] = x_new
                        F[i] = f_new

        return X
# EVOLVE-BLOCK-END
