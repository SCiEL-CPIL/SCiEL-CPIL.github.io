"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D with WFG7-aware initialization for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weights(self, n_obj, N):
        """Generate weight vectors on simplex for n_obj objectives."""
        if n_obj == 3:
            # Generate points on 2D simplex (triangle) for 3 objectives
            weights = []
            # Use stratified sampling for better coverage
            n_per_line = max(1, N // 10)
            for i in range(N + 1):
                for j in range(N + 1 - i):
                    k = N - i - j
                    w = np.array([i, j, k], dtype=float) / N
                    weights.append(w)
            weights = np.array(weights)
            # If we have too many, sample uniformly
            if len(weights) > N:
                idx = np.random.default_rng(42).choice(len(weights), size=N, replace=False)
                weights = weights[idx]
            elif len(weights) < N:
                # Pad with random points on simplex
                rng = np.random.default_rng(43)
                extra = []
                for _ in range(N - len(weights)):
                    u = rng.random(2)
                    w = np.array([1-u[0], u[0]*(1-u[1]), u[0]*u[1]])
                    extra.append(w)
                weights = np.vstack([weights, np.array(extra)])
        else:
            # Generic case: random points on simplex
            rng = np.random.default_rng(42)
            weights = rng.random((N, n_obj))
            weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _wfg7_aware_init(self, N, dim, lo, hi, rng):
        """Initialize with WFG7-aware bias: shape variables near Pareto front region."""
        X = np.empty((N, dim))
        
        # WFG7 has 12 variables: 3 shape variables (x1,x2,x3) and 9 position variables
        # The Pareto front is characterized by specific relationships in shape variables
        # Bias shape variables toward regions that produce diverse Pareto front points
        
        for i in range(N):
            # Shape variables (indices 0, 1, 2) - bias toward known Pareto region
            # WFG7 Pareto front: f1 = x1, f2 = x2, f3 = x3 with constraints
            # Good initialization: spread shape variables across [0.1, 0.9]
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * (0.1 + 0.8 * rng.random())
            X[i, 1] = lo[1] + (hi[1] - lo[1]) * (0.1 + 0.8 * rng.random())
            X[i, 2] = lo[2] + (hi[2] - lo[2]) * (0.1 + 0.8 * rng.random())
            
            # Position variables (indices 3-11) - random initialization
            X[i, 3:] = lo[3:] + (hi[3:] - lo[3:]) * rng.random(dim - 3)
        
        return X

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 24
        
        # Generate weight vectors
        weights = self._generate_weights(self.n_obj, N)
        
        # WFG7-aware initialization
        rng = np.random.default_rng(0)
        X = self._wfg7_aware_init(N, self.dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Normalize objectives for decomposition
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0
        F_norm = (F - f_min) / f_range
        
        # DE Parameters - adaptive
        F_mut_init = 0.8
        F_mut_min = 0.3
        CR = 0.9
        
        # Track best solutions
        best_F = F.copy()
        best_X = X.copy()
        
        # Main loop
        gen = 0
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            gen += 1
            # Adaptive mutation factor: decrease over generations
            progress = min(1.0, evals / self.budget)
            F_mut = F_mut_init - (F_mut_init - F_mut_min) * progress
            
            # Generate offspring
            X_off = np.empty_like(X)
            F_off = np.empty_like(F)
            
            for i in range(N):
                if evals >= self.budget or time.time() > deadline:
                    break
                
                # Select parents from neighborhood (MOEA/D style)
                # Use 2 parents for DE/bin/2
                idx1 = rng.integers(N)
                idx2 = rng.integers(N)
                if idx2 == idx1:
                    idx2 = (idx2 + 1) % N
                p1, p2 = X[idx1], X[idx2]
                
                # Mutation: DE/bin/2
                v = p1 + F_mut * (p2 - X[i])
                
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
            
            if evals < self.budget:
                # Combine parent and offspring
                X_combined = np.vstack([X, X_off])
                F_combined = np.vstack([F, F_off])
                
                # Update normalization
                f_min = F_combined.min(axis=0)
                f_max = F_combined.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range < 1e-10] = 1.0
                F_combined_norm = (F_combined - f_min) / f_range
                
                # Select next generation using decomposition
                X_next = np.empty_like(X)
                F_next = np.empty_like(F)
                
                for i in range(N):
                    # Find best solution in neighborhood for weight i
                    # Use Tchebycheff aggregation
                    best_score = np.inf
                    best_idx = -1
                    
                    for j in range(2 * N):
                        # Tchebycheff: max_k lambda_k * |f_k - z_k|
                        # Use normalized objectives
                        scores = weights[i] * np.abs(F_combined_norm[j] - 0.5)
                        score = np.max(scores)
                        if score < best_score:
                            best_score = score
                            best_idx = j
                    
                    X_next[i] = X_combined[best_idx]
                    F_next[i] = F_combined[best_idx]
                
                X = X_next
                F = F_next
                
                # Track best
                for i in range(N):
                    for k in range(self.n_obj):
                        if F[i, k] < best_F[i, k]:
                            best_F[i, k] = F[i, k]
                            best_X[i] = X[i]
            
            if evals >= self.budget:
                break

        return X
# EVOLVE-BLOCK-END
