"""Optimized program for WFG9 using decomposition-based MOEA/D with Tchebycheff scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D Multi-Objective Optimizer using Tchebycheff decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Number of weight vectors on the simplex
        # For 3 objectives, we use a systematic grid
        # Let's use roughly 25-30 subproblems for good coverage with budget=200
        # With budget 200, we can do about 10-15 generations if we use 20-20 subproblems
        # Let's use 21 subproblems (7x3 grid or similar)
        
        # Generate weight vectors on the simplex for 3 objectives
        # Using a systematic approach: for i+j+k = H, w_i = i/H, w_j = j/H, w_k = k/H
        # Let H = 6, so we get (6+1 choose 3-1)/2! = 28 vectors... let's use H=5 -> 21 vectors
        H = 5
        weights = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                weights.append([i / H, j / H, k / H])
        weights = np.array(weights)
        n_sub = len(weights)
        
        # If budget is small, reduce subproblems
        if self.budget < 100:
            H = 3
            weights = []
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    weights.append([i / H, j / H, k / H])
            weights = np.array(weights)
            n_sub = len(weights)

        # Initialize population: one individual per subproblem
        X = lo + range_ * np.random.rand(n_sub, self.dim)
        F = np.array([problem(x) for x in X])
        evaluations = n_sub

        # Ideal point (minimum of each objective) and nadir point (maximum)
        # Initialize with current extremes
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)

        # Update ideal/nadir with small margins for normalization
        # We'll update these dynamically

        # Main loop
        # Each generation: for each subproblem, generate one offspring
        # So each generation uses n_sub evaluations
        # Total generations = (budget - n_sub) // n_sub

        while evaluations + n_sub <= self.budget:
            if time.time() > deadline:
                break

            # Update ideal and nadir
            ideal = np.min(F, axis=0)
            nadir = np.max(F, axis=0)
            
            # Avoid division by zero
            norm = nadir - ideal
            norm[norm < 1e-10] = 1.0

            # For each subproblem, create an offspring
            for i in range(n_sub):
                if time.time() > deadline:
                    break
                if evaluations >= self.budget:
                    break

                # Select neighborhood: for simplicity, pick a random parent from the entire population
                # or a nearby one. Let's pick 2 parents: one is the best neighbor, one is random
                # For MOEA/D, typically we select from a neighborhood.
                # Let's define neighborhood as indices with similar weights.
                # For speed and simplicity with small pop, let's just pick 2 random parents,
                # but bias towards the current best for this subproblem.
                
                # Parent 1: The current solution for this subproblem (exploitation)
                # Parent 2: A random solution (exploration) or best from neighborhood
                p1 = X[i]
                p2 = X[np.random.randint(n_sub)]
                
                # Crossover (SBX)
                child = self._sbx_crossover(p1, p2, self.dim)
                
                # Mutation (Polynomial)
                child = self._poly_mutate(child, lo, hi, self.dim)
                
                # Evaluate
                fc = problem(child)
                evaluations += 1
                
                # Update the individual if it improves the Tchebycheff indicator
                # Compute Tchebycheff indicator for the child and the current individual
                # g(x, w) = max_j { w_j * |f_j(x) - z_j^*| }
                
                # Use ideal point as reference
                # Normalize objectives to [0,1] using ideal and nadir
                f_norm_child = (fc - ideal) / norm
                f_norm_parent = (F[i] - ideal) / norm
                
                # Clip to [0,1] for stability
                f_norm_child = np.clip(f_norm_child, 0, 1)
                f_norm_parent = np.clip(f_norm_parent, 0, 1)
                
                w = weights[i]
                
                g_child = np.max(w * f_norm_child)
                g_parent = np.max(w * f_norm_parent)
                
                # Accept if child is better (lower g)
                if g_child < g_parent:
                    X[i] = child
                    F[i] = fc
        
        return X

    def _sbx_crossover(self, p1, p2, dim):
        """Simulated Binary Crossover."""
        eta_c = 20.0
        child = np.zeros(dim)
        for i in range(dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    if np.random.rand() <= 0.5:
                        # Avoid division by zero
                        beta = (0.5 * (p1[i] / (p2[i] + 1e-10))) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (0.5 * (p2[i] / (p1[i] + 1e-10))) ** (1.0 / (eta_c + 1))
                    beta = max(beta, 1e-14)
                    child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            else:
                child[i] = p1[i]
        return child

    def _poly_mutate(self, x, lo, hi, dim):
        """Polynomial Mutation."""
        eta_m = 20.0
        x_new = np.zeros(dim)
        for i in range(dim):
            if np.random.rand() < 0.5:
                x_new[i] = x[i]
            else:
                if hi[i] - lo[i] < 1e-14:
                    x_new[i] = x[i]
                else:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                    rand = np.random.rand()
                    if rand <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                        deltaq = val ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                        deltaq = 1.0 - val ** (1.0 / (eta_m + 1.0))
                    x_new[i] = x[i] + deltaq * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
        return x_new
# EVOLVE-BLOCK-END
