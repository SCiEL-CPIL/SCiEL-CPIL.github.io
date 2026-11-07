"""Rewritten multi-objective optimizer for WFG4 using MOEA/D."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Determine population size
        # N=40, Generations=4 -> 40 + 4*40 = 200
        N = 40
        
        # Generate weight vectors uniformly on the simplex
        # For 3 objectives, we can use a grid or random sampling.
        # A simple way: generate random points and normalize.
        # Or use a structured grid. With N=40, random is fine but might be sparse.
        # Let's use a deterministic grid for better coverage.
        # For 3 objectives, we can use a 2D grid projected to the simplex.
        # Let's just use random weights for simplicity and robustness, or a standard simplex grid.
        # Standard simplex grid for n_obj=3, N=40.
        # We can generate points by taking a square grid and mapping to triangle?
        # Easier: Random weights, then normalize.
        weights = rng.random((N, n_obj))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Ideal point
        z_id = np.min(F, axis=0)
        
        # MOEA/D parameters
        eta_c = 20.0
        eta_m = 20.0
        p_c = 1.0
        p_m = 1.0 / dim
        T = 5 # Neighborhood size
        
        # Calculate neighborhood for each weight vector
        # Distance between weight vectors
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        # Find T closest neighbors for each i
        neighborhoods = []
        for i in range(N):
            # Sort by distance, take first T (excluding self? usually including self is fine or just neighbors)
            sorted_idx = np.argsort(dists[i])
            # Exclude self if it's at 0 distance? dists[i,i]=0.
            # Take first T+1 to ensure T neighbors if self is included? 
            # Usually, neighborhood includes the self index or just neighbors.
            # Let's take T neighbors excluding self.
            neighbors = []
            for idx in sorted_idx:
                if idx != i:
                    neighbors.append(idx)
                if len(neighbors) == T:
                    break
            neighborhoods.append(neighbors)

        # MOEA/D Loop
        while used + N <= budget:
            new_X = np.empty((N, dim))
            new_F = np.empty((N, n_obj))
            
            for i in range(N):
                # Select parents
                # Parent 1: from neighborhood of i
                # Parent 2: from entire population (or neighborhood)
                # Standard MOEA/D: 1 from neighborhood, 1 from population.
                # Let's pick 1 from neighborhood and 1 from population to encourage diversity.
                
                if len(neighborhoods[i]) > 0:
                    p1_idx = neighborhoods[i][rng.integers(len(neighborhoods[i]))]
                else:
                    p1_idx = i
                
                p2_idx = rng.integers(N)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover
                if rng.random() < p_c:
                    c1, c2 = self._sbx_crossover(rng, p1, p2, lo, hi, eta_c)
                    # We only need one child per weight vector i in standard MOEA/D
                    # But SBX produces two. We can use c1 for i and maybe c2 for a neighbor?
                    # Standard MOEA/D: produce one child, evaluate, update.
                    # If we produce two, we might waste evaluations or need to assign them.
                    # Let's just use c1.
                    child = c1
                else:
                    child = p1.copy()
                
                # Mutation
                child = self._poly_mutation(rng, child, lo, hi, eta_m, p_m)
                
                # Evaluate
                f_new = np.array(problem(child), dtype=float)
                used += 1
                
                # Update ideal point
                z_id = np.minimum(z_id, f_new)
                
                # Scalarization: Tchebycheff
                # g(f, lambda) = max(lambda_i * |f_i - z_id_i|)
                # Note: z_id is updated before calculation? 
                # Usually z_id is the current best.
                # If we update z_id first, the new point might be compared against a target that includes itself.
                # This is a common implementation detail. 
                # Let's use the z_id from the start of the generation for stability, 
                # or update it. Updating it makes the target move.
                # Let's use the current z_id (which includes the new point if it improved).
                # Actually, if f_new is better, it becomes z_id.
                # Let's compute scalar value using the z_id BEFORE updating for this specific check?
                # Or just use the global z_id which is now updated.
                # If f_new is the new best, |f_new - z_id| = 0 for that objective.
                # This is fine.
                
                scalar_val = np.max(weights[i] * np.abs(f_new - z_id))
                
                # Current scalar value for solution i
                # We need to store the scalar values or recompute?
                # It's better to store them.
                # Let's maintain an array S of scalar values.
                # Initialize S
                if not hasattr(self, '_S'):
                    self._S = np.zeros(N)
                    for k in range(N):
                        self._S[k] = np.max(weights[k] * np.abs(F[k] - z_id))
                
                # Update if better
                if scalar_val < self._S[i]:
                    X[i] = child
                    F[i] = f_new
                    self._S[i] = scalar_val
                else:
                    # Keep old solution
                    # We don't replace X[i] and F[i]
                    pass
                    
            # After generation, we might want to update S for all points if z_id changed significantly?
            # Or just rely on the local updates.
            # The X and F arrays now contain the best solutions found for each weight vector.
            
        return X

    def _sbx_crossover(self, rng, p1, p2, lo, hi, eta_c):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() <= 1.0 / (eta_c + 1.0):
                        beta = 2.0 * rng.random()
                    else:
                        beta = 2.0 * (1.0 - rng.random())
                    
                    beta = 1.0 / (beta * eta_c + 1.0)
                    
                    c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                    
                    if c1[i] > hi[i]:
                        c1[i] = hi[i]
                    if c1[i] < lo[i]:
                        c1[i] = lo[i]
                    if c2[i] > hi[i]:
                        c2[i] = hi[i]
                    if c2[i] < lo[i]:
                        c2[i] = lo[i]
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        return c1, c2

    def _poly_mutation(self, rng, x, lo, hi, eta_m, p_m):
        x_mut = x.copy()
        for i in range(len(x)):
            if rng.random() < p_m:
                if abs(hi[i] - lo[i]) < 1e-14:
                    continue
                
                r = rng.random()
                if r < 0.5:
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                    mut_pow = 2.0 / (eta_m + 1.0) * (delta1 + 1.0)
                    if r <= 1.0 / mut_pow:
                        delta = (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta = (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] - delta * (hi[i] - lo[i])
                else:
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                    mut_pow = 2.0 / (eta_m + 1.0) * (delta2 + 1.0)
                    if (1.0 - r) <= 1.0 / mut_pow:
                        delta = (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta = (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] + delta * (hi[i] - lo[i])
                    
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut
# EVOLVE-BLOCK-END
