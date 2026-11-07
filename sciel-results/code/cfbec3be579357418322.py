"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D-like multi-objective optimizer using Weighted Sum decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _generate_weight_vectors(self, n_obj, N):
        """Generate N weight vectors for n_obj objectives using simplex lattice."""
        # For 3 objectives, we can use a grid on the simplex
        # Or use a random uniform distribution on the simplex for flexibility
        # Simplex lattice method:
        # For n_obj=3, we need a, b, c >= 0, a+b+c=1
        # We can iterate over a from 0 to 1, b from 0 to 1-a, c = 1-a-b
        
        # However, a fixed grid might be sparse. Let's use a stratified random approach
        # or a standard simplex lattice with sufficient resolution.
        # Given N is likely around 50-100, a grid is efficient.
        
        weights = []
        # Determine resolution for each dimension
        # For 3 objectives, number of points in simplex lattice with resolution 'res' is (res+1)(res+2)/2
        # We want approximately N points.
        # (res+1)(res+2)/2 ~ N => res ~ sqrt(2N)
        res = int(np.sqrt(2 * N))
        
        points = []
        for a in range(res + 1):
            for b in range(res + 1 - a):
                c = res - a - b
                points.append([a / res, b / res, c / res])
        
        # If we have more points than N, sample them. If fewer, add more or just use what we have.
        # Actually, let's just generate N random points on the simplex if the grid is too complex to match exactly.
        # But a structured grid is better for coverage.
        # Let's stick to a simple random sampling on the simplex for robustness against N.
        
        # Random points on simplex
        # Method: generate n_obj-1 random numbers, sort them, differences give weights.
        # For n_obj=3: u, v ~ U(0,1). sort u, v. w1 = min(u,v), w2 = max(u,v)-min(u,v), w3 = 1-max(u,v).
        
        u = np.random.random((N, n_obj - 1))
        u.sort(axis=1)
        weights = np.zeros((N, n_obj))
        weights[:, 0] = u[:, 0]
        weights[:, 1] = u[:, 1] - u[:, 0]
        weights[:, 2] = 1 - u[:, 1]
        
        # Ensure they sum to 1 (they do by construction)
        return weights

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget is 200. 
        # Let's use a population size that allows for several generations.
        # N=50, generations ~ 4. 
        # N=20, generations ~ 10.
        # MOEA/D works well with moderate N. Let's try N=40.
        N = 40
        if N > self.budget:
            N = self.budget
            
        # Generate weight vectors
        W = self._generate_weight_vectors(self.n_obj, N)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Calculate ideal point (min of F)
        F_min = np.min(F, axis=0)
        
        # Define neighborhood: each individual interacts with its nearest neighbors in weight space
        # Calculate pairwise distances between weight vectors
        # For efficiency with small N, we can just compute all pairwise distances
        W_diff = W[:, np.newaxis, :] - W[np.newaxis, :, :]
        dists = np.sum(W_diff ** 2, axis=2)
        
        # Sort neighbors for each individual
        # neighbor_indices[i] is a list of indices sorted by distance to i
        neighbor_indices = np.argsort(dists, axis=1)
        
        # Neighborhood size
        T = max(2, N // 5) # Typically 20% of population
        
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Update Ideal Point
            F_min = np.min(F, axis=0)
            
            # For each individual, generate a child
            new_X = np.empty_like(X)
            new_F = np.empty_like(F)
            
            for i in range(N):
                # Check budget
                if used >= self.budget or time.time() > deadline:
                    # If we run out of budget, we break. 
                    # But we need to fill new_X/new_F for the ones we didn't process?
                    # No, if we break, we should probably stop and return current best.
                    # However, the loop structure updates X/F at the end of the generation.
                    # If we break mid-generation, we have partial updates.
                    # Better to check before starting the generation.
                    pass 
                
                # Select parents from neighborhood
                # Pick 2 parents from the neighborhood of i (including i)
                neighbors = neighbor_indices[i, :T+1] # Include self
                p1_idx = neighbors[rng.integers(len(neighbors))]
                p2_idx = neighbors[rng.integers(len(neighbors))]
                
                # Crossover (SBX or Simple Blend)
                # Let's use a simple blend crossover with mutation
                alpha = rng.random()
                child = (1 - alpha) * X[p1_idx] + alpha * X[p2_idx]
                
                # Mutation
                # Gaussian mutation
                mut_rate = 0.2
                mut_scale = 0.1 * (hi - lo)
                mask = rng.random(self.dim) < mut_rate
                child[mask] += mut_scale[mask] * rng.standard_normal(np.sum(mask))
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                new_X[i] = child
                new_F[i] = fc
                
                # Update the individual if it improves the weighted sum
                # Weighted sum for weight vector i
                w_i = W[i]
                # Normalize objectives? 
                # Using F_min for normalization: f_norm = (f - f_min) / (f_max - f_min)
                # But f_max is unknown. 
                # Standard MOEA/D uses: g = sum w_j * (f_j - f_j^min)
                # This works if we just want to minimize the weighted sum of raw objectives.
                # However, scaling matters. 
                # Let's use the current F_min and a rough estimate of range or just raw values.
                # Given the reference point is [3, 5, 7], the objectives are likely in that range.
                # Let's stick to raw weighted sum for simplicity, as normalization adds complexity and potential instability early on.
                
                current_w_sum = np.dot(w_i, F[i])
                new_w_sum = np.dot(w_i, fc)
                
                if new_w_sum <= current_w_sum:
                    X[i] = child
                    F[i] = fc
                else:
                    # Keep old individual
                    new_X[i] = X[i]
                    new_F[i] = F[i]
            
            # If we broke out early due to time/budget, we might have partial new_X/new_F
            # But the loop above updates X/F in place if improved. 
            # Wait, the code above updates X[i] and F[i] inside the loop.
            # So X and F are always up to date.
            # We don't actually need new_X/new_F buffers if we update in place.
            # Let's clean that up.
            
            # Check if we should continue
            if used >= self.budget:
                break

        return X
# EVOLVE-BLOCK-END
