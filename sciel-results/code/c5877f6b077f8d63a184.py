"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for WFG7 using MOEA/D-DE strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Parameters for MOEA/D-DE
        # Budget is 200. We need to be careful with overhead.
        # Pop size 20 is reasonable for 200 evals (10 generations).
        pop_size = 20
        K = 5  # Neighborhood size
        
        # Generate weight vectors for decomposition (Tchebycheff)
        # For 3 objectives, we generate weight vectors on a simplex
        # Using a recursive or grid-based approach for simplicity and coverage
        # Here we use a simple uniform distribution of weights summing to 1
        # For 3 objectives, weights (w1, w2, w3) where w1+w2+w3=1
        # We can generate these by sampling or using a known grid.
        # Given the small pop size, we can use a set of predefined weights
        # or generate them randomly but normalized.
        # Let's generate 20 weight vectors uniformly on the simplex.
        
        # Simple way to generate weight vectors for 3 objectives:
        # Sample 2 random numbers u, v in [0,1]
        # Sort them: u1 <= u2
        # w1 = u1, w2 = u2 - u1, w3 = 1 - u2
        # This gives a uniform distribution on the simplex.
        
        weights = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            u = np.sort(np.random.random(2))
            weights[i, 0] = u[0]
            weights[i, 1] = u[1] - u[0]
            weights[i, 2] = 1.0 - u[1]
            
        # Normalize weights to ensure they are in [0,1] and sum to 1
        # They already sum to 1, but we ensure no negatives
        weights = np.clip(weights, 0, 1)
        
        # Find neighbors for each individual
        neighbors = []
        for i in range(pop_size):
            # Calculate distance to all other weight vectors
            dists = np.sum((weights - weights[i])**2, axis=1)
            # Sort indices by distance
            sorted_indices = np.argsort(dists)
            # Take K nearest neighbors (excluding self)
            neigh = sorted_indices[1:K+1]
            neighbors.append(neigh)
            
        # Initialize population
        X = lo + range_ * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_points = [tuple(f) for f in F]
        
        # MOEA/D-DE loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a random parent index
            m = np.random.randint(pop_size)
            
            # Select two distinct parents from the neighborhood of m
            neigh_m = neighbors[m]
            if len(neigh_m) < 2:
                # Fallback if neighborhood is too small (shouldn't happen with K=5, pop=20)
                p1_idx = np.random.choice([i for i in range(pop_size) if i != m])
                p2_idx = np.random.choice([i for i in range(pop_size) if i != m and i != p1_idx])
            else:
                p1_idx, p2_idx = np.random.choice(neigh_m, size=2, replace=False)
                
            # Differential Evolution crossover
            # y = x_p1 + C * (x_p2 - x_p1)
            C = 0.5
            y = X[p1_idx] + C * (X[p2_idx] - X[p1_idx])
            
            # Mutation
            # Add some random perturbation to ensure exploration
            # sigma is scaled by range
            sigma = 0.1
            y = y + sigma * range_ * np.random.standard_normal(self.dim)
            
            # Clip to bounds
            y = np.clip(y, lo, hi)
            
            # Evaluate
            fy = problem(y)
            used += 1
            all_points.append(tuple(fy))
            
            # Update mechanism
            # In MOEA/D, we update the subproblem m if the new solution y
            # improves the Tchebycheff indicator for subproblem m
            # or if it dominates the current solution in the neighborhood.
            
            # Tchebycheff indicator for subproblem m
            # g(x, w) = max_j { w_j * |f_j(x) - z_j^*| }
            # We don't have ideal points z^* yet, so we use a simplified update rule:
            # Replace X[m] with y if y is better than X[m] in terms of the Tchebycheff value
            # or if y dominates X[m].
            
            # Calculate Tchebycheff value for X[m] and y with weight w_m
            w_m = weights[m]
            
            # Since we don't have ideal points, we use a proxy:
            # Compare the weighted sum or just use domination.
            # A common simplified update is to replace if y dominates X[m]
            # or if the Tchebycheff value of y is lower than X[m] (assuming ideal points are 0 for simplicity in early stages, 
            # but that's not right. Ideal points should be the min values seen so far).
            
            # Let's maintain ideal points z^*
            # This requires updating z^* globally.
            
            # Alternative simpler strategy: Replace X[m] if y is non-dominated with respect to X[m]
            # or if it improves the Tchebycheff indicator using current ideal points.
            
            # Let's compute ideal points from current population F
            # This is expensive if done every iteration, but pop_size is small (20).
            # We can update ideal points incrementally.
            
            # For simplicity and speed, let's use a domination-based replacement
            # combined with a Tchebycheff check using a fixed reference (e.g., min values seen so far).
            
            # To avoid recalculating ideal points every time, let's just use domination
            # and a simple weighted sum if no domination exists.
            
            # Check if y dominates X[m]
            dominates = np.all(fy <= F[m]) and np.any(fy < F[m])
            
            # Check if X[m] dominates y
            dominated = np.all(F[m] <= fy) and np.any(F[m] < fy)
            
            if dominates:
                X[m] = y
                F[m] = fy
            elif not dominated:
                # Incomparable. Use Tchebycheff indicator.
                # We need ideal points. Let's compute them quickly.
                # Since pop_size is small, this is O(N*M) which is 20*3 = 60 ops.
                ideal = np.min(F, axis=0)
                
                # Tchebycheff value for X[m]
                g_m = np.max(w_m * np.abs(F[m] - ideal))
                # Tchebycheff value for y
                g_y = np.max(w_m * np.abs(fy - ideal))
                
                if g_y < g_m:
                    X[m] = y
                    F[m] = fy
            # Else, keep X[m]
            
        return X
# EVOLVE-BLOCK-END
