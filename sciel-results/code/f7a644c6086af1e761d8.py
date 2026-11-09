"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-DE with WFG7-aware initialization for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weights(self, n_obj, N):
        """Generate weight vectors for decomposition."""
        if n_obj == 2:
            weights = np.zeros((N, 2))
            for i in range(N):
                weights[i, 0] = i / (N - 1)
                weights[i, 1] = 1 - weights[i, 0]
        else:
            # Simplex-lattice for 3 objectives
            weights = []
            for i in range(N + 1):
                for j in range(N + 1 - i):
                    k = N - i - j
                    weights.append([i / N, j / N, k / N])
            weights = np.array(weights)
            # If we need exactly N weights, subsample
            if len(weights) >= N:
                indices = np.linspace(0, len(weights) - 1, N, dtype=int)
                weights = weights[indices]
            else:
                # Pad by repeating
                while len(weights) < N:
                    weights = np.vstack([weights, weights[:N - len(weights)]])
        return weights

    def _wfg7_aware_init(self, N, lo, hi, rng):
        """WFG7-aware initialization: bias shape variables toward Pareto front region."""
        X = np.zeros((N, self.dim))
        
        # For WFG7, variables 0-2 are position variables, 3-11 are shape variables
        # The Pareto front is achieved when shape variables are near specific values
        # Typically, shape variables should be near 0.5 or specific patterns
        
        for i in range(N):
            # Position variables: uniform random
            X[i, :3] = lo[:3] + (hi[:3] - lo[:3]) * rng.random(3)
            
            # Shape variables: biased toward center of range
            for j in range(3, self.dim):
                # Mix of uniform and center-biased
                if rng.random() < 0.5:
                    X[i, j] = (lo[j] + hi[j]) / 2 + (hi[j] - lo[j]) * 0.3 * rng.standard_normal()
                else:
                    X[i, j] = lo[j] + (hi[j] - lo[j]) * rng.random()
            
            # Clip to bounds
            X[i] = np.clip(X[i], lo, hi)
        
        return X

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 25
        
        # Generate weight vectors
        weights = self._generate_weights(self.n_obj, N)
        
        # WFG7-aware initialization
        rng = np.random.default_rng(0)
        X = self._wfg7_aware_init(N, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Normalize objectives for scalarization
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range < 1e-10] = 1.0
        F_norm = (F - F_min) / F_range
        
        # Adaptive DE parameters
        F_mut = 0.8
        CR = 0.9
        
        # Track best hypervolume points
        best_points = list(zip(F.tolist()))
        
        # Main loop
        generation = 0
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            generation += 1
            
            # Adaptive mutation: reduce F_mut in later generations
            F_mut = 0.8 - 0.3 * (generation / max(1, self.budget // N))
            F_mut = max(0.4, min(0.8, F_mut))
            
            # Generate offspring using MOEA/D-DE
            X_off = np.empty_like(X)
            F_off = np.empty_like(F)
            
            for i in range(N):
                if evals >= self.budget or time.time() > deadline:
                    break
                
                # Select neighbors based on weight vector proximity
                # Find 3 nearest neighbors in weight space
                weight_diffs = np.linalg.norm(weights - weights[i], axis=1)
                weight_diffs[i] = np.inf  # Exclude self
                neighbor_indices = np.argsort(weight_diffs)[:3]
                
                # Select 3 parents from neighbors
                p1, p2, p3 = X[neighbor_indices[0]], X[neighbor_indices[1]], X[neighbor_indices[2]]
                
                # Mutation
                v = p1 + F_mut * (p2 - p3)
                
                # Crossover
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                evals += 1
                
                X_off[i] = child
                F_off[i] = fc
                
                # Update best points
                best_points.append(fc.tolist())
            
            if evals >= self.budget:
                break
            
            # Update population using scalarized fitness
            X_next = np.empty_like(X)
            F_next = np.empty_like(F)
            
            # Update normalization
            F_all = np.vstack([F, F_off])
            F_min = F_all.min(axis=0)
            F_max = F_all.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range < 1e-10] = 1.0
            
            for i in range(N):
                # Scalarize parent and offspring
                F_parent_norm = (F[i] - F_min) / F_range
                F_child_norm = (F_off[i] - F_min) / F_range
                
                # Tchebycheff scalarization
                g_parent = np.max(weights[i] * F_parent_norm)
                g_child = np.max(weights[i] * F_child_norm)
                
                # Keep the better one
                if g_child <= g_parent:
                    X_next[i] = X_off[i]
                    F_next[i] = F_off[i]
                else:
                    X_next[i] = X[i]
                    F_next[i] = F[i]
            
            X = X_next
            F = F_next
        
        return X
# EVOLVE-BLOCK-END
