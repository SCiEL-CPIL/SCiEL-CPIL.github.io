"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D based optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Population size
        N = 100
        
        # Generate weight vectors using simplex lattice
        # For 3 objectives, we generate points on the simplex
        # We want a good spread.
        # Simple method: generate random points and normalize, or use grid.
        # Grid method for 3D:
        # Let H be the number of divisions.
        # H=10 gives roughly 66 points. H=15 gives 136.
        # We need exactly N weights.
        # We will generate a set of candidate weights and pick N evenly spaced ones or just random ones on simplex.
        # Random on simplex is easier to implement robustly for arbitrary N.
        # But structured is better for MOEA/D.
        # Let's use a simple grid approach and subsample if needed.
        
        def generate_weights(n_obj, N):
            # Generate all combinations of i, j, k such that i+j+k = H
            # Then map to weights w_i = i/H, etc.
            # We need to find H such that the number of points is >= N, then subsample.
            # Number of points for H is (H+1)(H+2)/2.
            # For N=100, H=13 gives 105 points.
            H = int(np.ceil((-3 + np.sqrt(9 + 8*N)) / 2.0))
            
            points = []
            for i in range(H + 1):
                for j in range(H - i + 1):
                    k = H - i - j
                    points.append([i, j, k])
            
            points = np.array(points, dtype=float) / H
            # We have len(points) >= N
            # Shuffle and take first N? Or sort by distance to centroid?
            # Sorting by distance to centroid (1/n_obj, ...) ensures good spread.
            centroid = np.ones(n_obj) / n_obj
            dists = np.linalg.norm(points - centroid, axis=1)
            # We want points that are well spread. 
            # Actually, for MOEA/D, the specific distribution matters less than the neighborhood.
            # Let's just take the first N after shuffling to ensure diversity in weights?
            # Or better: sort by one coordinate to ensure coverage?
            # Let's just shuffle and take N.
            idx = rng.permutation(len(points))
            return points[idx[:N]]

        weights = generate_weights(self.n_obj, N)
        
        # Initialize population
        # Use Latin Hypercube Sampling for better initial spread
        def lhs(dim, n):
            # Split [0,1] into n bins, pick one random point in each bin for each dimension
            # This ensures good spread
            x = np.zeros((n, dim))
            for d in range(dim):
                bins = np.linspace(0, 1, n + 1)
                for i in range(n):
                    x[i, d] = rng.uniform(bins[i], bins[i+1])
            return x
            
        X = lo + (hi - lo) * lhs(self.dim, N)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Identify neighbors for each weight vector
        # Distance is Euclidean distance between weight vectors
        # Select K nearest neighbors
        K = 20
        dists = np.linalg.norm(weights[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
        neighbor_idx = np.argsort(dists, axis=1)[:, :K]
        
        # Main loop
        # Adaptive mutation: Cauchy distribution is good for multi-objective
        # Scale factor: 0.5
        # Mutation rate: 1.0 (always mutate)
        
        # We will iterate until budget is reached.
        # Each iteration: select one parent index i, choose a neighbor j, 
        # create child, evaluate, and update subpopulation if it improves the fitness.
        
        # Fitness function for MOEA/D:
        # g(x, w) = sum(w_k * (f_k - z_k)^p) or similar.
        # Simplest: weighted sum with ideal point z.
        # z_k = min(F[:, k])
        
        z = np.min(F, axis=0)
        
        def fitness(x, w):
            # Weighted sum
            # To handle different scales, we could normalize, but WFG objectives are usually comparable.
            # Let's use the standard Tchebycheff or Weighted Sum.
            # Weighted Sum is simpler and works well with hypervolume if weights are chosen well.
            # However, Weighted Sum cannot find non-convex parts.
            # Tchebycheff: max_k (w_k * |f_k - z_k|)
            # Let's use Tchebycheff as it handles non-convexity better.
            # Actually, for WFG7, the front is non-convex.
            # Let's use the weighted sum of the objectives, but maybe scaled.
            # A common choice is g(x,w) = sum w_k f_k.
            # Let's stick to Weighted Sum for simplicity and speed, but ensure weights are diverse.
            return np.dot(w, x)
            
        # Wait, the fitness function should take the *solution* x and compute the *objective* values?
        # No, in MOEA/D, we usually evaluate the objective function F(x) and then compute the scalar fitness.
        # But we are limited in evaluations.
        # So for each new child x_c:
        # 1. Evaluate F_c = problem(x_c)
        # 2. For each neighbor i, check if g(F_c, w_i) < g(F_i, w_i)
        # 3. If so, replace X_i, F_i.
        
        # To speed up, we only check neighbors.
        
        # Precompute current fitness values for all individuals
        # G[i] = g(F[i], weights[i])
        
        def compute_fitness(F_mat, weights_mat):
            # F_mat: (N, n_obj)
            # weights_mat: (N, n_obj)
            # Return (N,) array of scalar fitness
            # Using Weighted Sum
            return np.sum(F_mat * weights_mat, axis=1)
            
        G = compute_fitness(F, weights)
        
        # Update ideal point z
        # z = np.min(F, axis=0) # Already done
        
        # Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select a random parent index i
            i = rng.integers(N)
            
            # Select a neighbor j from i's neighborhood
            # With probability 0.9, pick from neighbors; else random
            if rng.random() < 0.9:
                j = neighbor_idx[i, rng.integers(K)]
            else:
                j = rng.integers(N)
            
            # Crossover: SBX (Simulated Binary Crossover)
            # eta = 20
            eta = 20
            c1 = X[i]
            c2 = X[j]
            
            # SBX
            # u = rng.random(self.dim)
            # beta = np.zeros(self.dim)
            # for d in range(self.dim):
            #     if u[d] <= 0.5:
            #         beta[d] = (2 * u[d]) ** (1.0 / (eta + 1))
            #     else:
            #         beta[d] = (2 * (1 - u[d])) ** (1.0 / (eta + 1))
            # child1 = 0.5 * ((1 + beta) * c1 + (1 - beta) * c2)
            # child2 = 0.5 * ((1 - beta) * c1 + (1 + beta) * c2)
            
            # Vectorized SBX
            u = rng.random(self.dim)
            beta = np.zeros(self.dim)
            mask = u <= 0.5
            beta[mask] = (2 * u[mask]) ** (1.0 / (eta + 1))
            beta[~mask] = (2 * (1 - u[~mask])) ** (1.0 / (eta + 1))
            
            # Only generate one child to save budget? 
            # Standard MOEA/D generates 1 or 2. 
            # Since budget is tight (200), and N=100, we have 100 generations.
            # Generating 1 child per generation is safer.
            # Let's generate 1 child.
            child = 0.5 * ((1 + beta) * c1 + (1 - beta) * c2)
            
            # Mutation: Cauchy
            # Standard Cauchy has heavy tails, good for exploration.
            # Scale it by the range of the variable.
            # cauchy = rng.standard_cauchy(self.dim)
            # child = child + 0.5 * (hi - lo) * cauchy
            # Clip to bounds
            
            # However, Cauchy can produce very large values.
            # Let's use Gaussian with adaptive sigma?
            # Or just simple Gaussian.
            # Let's use Gaussian with sigma = 0.1 * range.
            # Or better, use the difference between parents?
            # Let's stick to Gaussian for stability.
            mutation_sigma = 0.1
            child = child + mutation_sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            F_child = problem(child)
            used += 1
            
            # Update ideal point z
            z = np.minimum(z, F_child)
            
            # Check if child is better than parent i or its neighbors
            # We need to recompute G for neighbors?
            # Or just check the scalar fitness for each neighbor.
            
            # g(child, w_k) = sum(w_k * F_child)
            # Compare with G[k]
            
            # We only check neighbors of i
            neigh = neighbor_idx[i]
            
            # Calculate fitness of child for all neighbors
            # G_child_neigh = sum(weights[neigh] * F_child, axis=1)
            G_child_neigh = np.sum(weights[neigh] * F_child, axis=1)
            
            # Find where G_child_neigh < G[neigh]
            # If so, replace
            
            # Also, we should check if the child dominates any point in the population?
            # MOEA/D usually relies on the decomposition.
            # But to maintain diversity, we might need to check for dominance.
            # However, with 200 budget, checking dominance against 100 points for every child is O(100*3) which is fine.
            # But MOEA/D replacement strategy is specific:
            # "If g(x_c, w_k) <= g(x_k, w_k), replace x_k with x_c"
            
            # Let's implement this.
            improved_mask = G_child_neigh < G[neigh]
            
            if np.any(improved_mask):
                # Replace all improved neighbors? 
                # Usually, we replace one or all. Replacing all might be too aggressive?
                # Standard MOEA/D replaces the specific subpopulation if it is better for that weight.
                # Let's replace all that are improved.
                idx_to_replace = neigh[improved_mask]
                
                # Update X and F
                for idx in idx_to_replace:
                    X[idx] = child
                    F[idx] = F_child
                    G[idx] = np.dot(weights[idx], F_child)
                    
        return X
# EVOLVE-BLOCK-END
