"""Improved program using MOEA/D for multi-objective optimization."""
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
        deadline = start_time + 290  # Safety margin before 300s/330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size
        # For MOEA/D, a larger population helps cover the front better.
        # Budget is 200. 
        # Init: N evals.
        # Loop: Each iteration updates N individuals? Or a subset?
        # Standard MOEA/D iterates through all individuals.
        # If we update all N every generation, it's N evals per gen.
        # Total: N + G*N <= 200.
        # Let's try N=20. G = 9.
        # Or N=10. G = 19.
        # N=20 is usually good for 3 objectives.
        
        N = 20
        if self.budget < N:
            N = self.budget
            
        # Initialize weight vectors
        # For 3 objectives, we can use a simplex lattice or random weights.
        # Simplex lattice is better for coverage.
        weights = self._generate_weight_vectors(N, self.n_obj)
        
        # Ideal point (min of each objective) - updated dynamically
        ideal_point = np.full(self.n_obj, np.inf)
        
        # Initialize Population
        # Use stratified initialization based on weights?
        # Or just random. Random is safer if weights are not perfectly aligned.
        # Let's use random initialization to avoid bias if weight generation is imperfect.
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Update ideal point
        ideal_point = np.min(F, axis=0)
        
        # Neighborhood list
        # For each individual i, find K nearest neighbors based on weight vectors.
        # K is typically 2T where T is the number of neighbors.
        # Let's use K = 5 or 10.
        K = 10
        if N < K:
            K = N
        neighbors = self._build_neighborhood(weights, K)
        
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Iterate through each individual
            for i in range(N):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select parent from neighborhood
                # With probability 0.9, select from neighborhood, else from whole population
                if np.random.random() < 0.9:
                    neigh_idx = np.random.choice(neighbors[i])
                else:
                    neigh_idx = np.random.randint(0, N)
                    
                # Select another parent (same logic)
                if np.random.random() < 0.9:
                    neigh_idx2 = np.random.choice(neighbors[i])
                else:
                    neigh_idx2 = np.random.randint(0, N)
                    
                p1 = X[neigh_idx]
                p2 = X[neigh_idx2]
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                # c2 is discarded in standard MOEA/D single-offspring update per individual slot?
                # Usually, we produce one offspring per individual update.
                # Let's just evaluate c1.
                
                # Evaluate
                F_new = problem(c1)
                used += 1
                
                # Update ideal point
                ideal_point = np.minimum(ideal_point, F_new)
                
                # Update individual i if it improves the objective associated with weight i
                # Using Tchebycheff scalarization
                # g(x, w) = max_j (w_j * |f_j(x) - z_j^*) / w_j)
                # Here z* is ideal point.
                
                # Calculate scalar value for new offspring
                # Handle w_j = 0 case (shouldn't happen with simplex)
                w_i = weights[i]
                
                # Normalize objectives by ideal point? 
                # Standard Tchebycheff: max_j ( w_j * |f_j(x) - z_j^*| )
                # If w_j is 0, term is 0.
                
                # We need to compare the new offspring with the current X[i]
                # And potentially with neighbors to update them if they are worse?
                # Standard MOEA/D:
                # 1. Check if offspring is better than x_i according to w_i. If yes, replace.
                # 2. Check neighbors. If offspring is better than neighbor x_k according to w_k, replace x_k.
                
                # Calculate Tchebycheff value for new offspring with weight w_i
                val_new_i = self._tchebycheff(F_new, ideal_point, w_i)
                
                # Calculate Tchebycheff value for current x_i with weight w_i
                val_curr_i = self._tchebycheff(F[i], ideal_point, w_i)
                
                # If new is better, update X[i]
                if val_new_i < val_curr_i:
                    X[i] = c1
                    F[i] = F_new
                    
                    # Update neighbors if they are worse
                    for k in neighbors[i]:
                        if k == i:
                            continue
                        # Check if new offspring is better than X[k] according to w_k
                        w_k = weights[k]
                        val_new_k = self._tchebycheff(F_new, ideal_point, w_k)
                        val_curr_k = self._tchebycheff(F[k], ideal_point, w_k)
                        
                        if val_new_k < val_curr_k:
                            X[k] = c1
                            F[k] = F_new

        return X

    def _generate_weight_vectors(self, N, n_obj):
        """
        Generate N weight vectors on the simplex.
        For n_obj=3, use a grid on the 2D simplex.
        """
        if n_obj == 2:
            # Simple line
            weights = np.zeros((N, 2))
            for i in range(N):
                weights[i, 0] = i / (N - 1)
                weights[i, 1] = 1 - weights[i, 0]
            return weights
        
        if n_obj == 3:
            # Grid on triangle
            # Sum of i1 + i2 + i3 = H
            # We need approx N points.
            # Let H be such that (H+1)(H+2)/2 ~ N
            H = int(np.sqrt(2 * N))
            # Adjust H to get close to N
            # Actually, let's just generate random points and project?
            # Or use a fixed grid and subsample.
            
            # Let's try a simple random generation with normalization
            # This might have poor coverage.
            # Better: Generate on a grid.
            points = []
            # Find H such that number of points >= N
            while True:
                num_points = (H + 1) * (H + 2) // 2
                if num_points >= N:
                    break
                H += 1
                
            for i1 in range(H + 1):
                for i2 in range(H - i1 + 1):
                    i3 = H - i1 - i2
                    points.append([i1, i2, i3])
            
            points = np.array(points, dtype=float)
            # Normalize to sum to 1
            sums = points.sum(axis=1, keepdims=True)
            weights = points / sums
            
            # If we have more than N, subsample uniformly
            if len(weights) > N:
                indices = np.random.choice(len(weights), N, replace=False)
                weights = weights[indices]
            
            # If we have less (shouldn't happen with H logic), pad with random?
            # But H logic ensures >= N.
            
            return weights
        
        # Fallback for other n_obj
        weights = np.random.random((N, n_obj))
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _build_neighborhood(self, weights, K):
        """
        Build neighborhood list based on distance between weight vectors.
        """
        N = len(weights)
        neighbors = []
        # Calculate pairwise distances
        # dist[i, j] = ||w_i - w_j||
        # Using numpy broadcasting
        diff = weights[:, np.newaxis, :] - weights[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        
        for i in range(N):
            # Sort indices by distance
            idxs = np.argsort(dist[i])
            # Take first K (excluding self? usually self is included or not, doesn't matter much)
            # Let's take K nearest including self, but ensure self is in there?
            # Standard is K nearest neighbors.
            neighbors.append(idxs[:K])
            
        return neighbors

    def _tchebycheff(self, f, z, w):
        """
        Calculate Tchebycheff scalarization value.
        g(x, w) = max_j ( w_j * |f_j(x) - z_j| )
        """
        # Handle w_j = 0
        # If w_j is 0, the term is 0.
        # If w_j > 0, term is w_j * |f_j - z_j|
        
        # Vectorized calculation
        # Avoid division by zero if we were normalizing by w, but here we just multiply.
        # Wait, standard Tchebycheff is max( w_j * |f_j - z_j| ).
        # Some formulations use |f_j - z_j| / w_j. 
        # The task says "minimizing all 3 objectives".
        # Let's stick to max(w_j * |f_j - z_j|).
        
        terms = w * np.abs(f - z)
        return np.max(terms)

    def _sbx_crossover(self, p1, p2, lo, hi):
        """
        Simulated Binary Crossover.
        """
        c1 = p1.copy()
        c2 = p2.copy()
        
        eta = 20.0
        prob = 0.9
        
        for i in range(self.dim):
            if np.random.random() < prob:
                if p1[i] == p2[i]:
                    continue
                
                # Determine beta
                if p1[i] < p2[i]:
                    x1, x2 = p1[i], p2[i]
                    swap = False
                else:
                    x1, x2 = p2[i], p1[i]
                    swap = True
                    
                # Check boundaries
                if 2 * x1 <= lo[i] or 2 * x1 >= hi[i]:
                    beta = 1.0
                elif 2 * x2 <= lo[i] or 2 * x2 >= hi[i]:
                    beta = 1.0
                else:
                    beta = (x2 - x1) / (hi[i] - lo[i])
                    if beta > 0:
                        u = np.random.random()
                        if u <= 0.5:
                            beta_q = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                        
                        c1_val = 0.5 * ((x1 + x2) - (x2 - x1) * beta_q)
                        c2_val = 0.5 * ((x1 + x2) + (x2 - x1) * beta_q)
                        
                        if swap:
                            c1[i] = c1_val
                            c2[i] = c2_val
                        else:
                            c1[i] = c2_val
                            c2[i] = c1_val
                        
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """
        Polynomial Mutation.
        """
        eta = 20.0
        pm = 1.0 / self.dim
        
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.random() < pm:
                if x[i] < lo[i]:
                    continue
                if x[i] > hi[i]:
                    continue
                    
                delta_1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta_2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                
                u = np.random.random()
                if u < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy) ** (eta + 1.0)
                    delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy) ** (eta + 1.0)
                    delta_q = 1.0 - val ** (1.0 / (eta + 1.0))
                    
                x_mut[i] += delta_q * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
                
        return x_mut
# EVOLVE-BLOCK-END
