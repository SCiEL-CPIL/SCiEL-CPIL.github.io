"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using decomposition-based approach with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        # Create weight vectors for decomposition
        n_weights = 50
        weights = []
        for i in range(n_weights):
            w = rng.dirichlet(np.ones(self.n_obj))
            weights.append(w)
        weights = np.array(weights)
        
        # Initialize population
        pop_size = min(20, self.budget // 10)
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best solutions found
        all_X = [x.copy() for x in X]
        all_F = [f.copy() for f in F]
        
        # Adaptive mutation rate
        base_mut_rate = 0.3
        exploration_rate = 0.1
        
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Find non-dominated solutions
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Select parent with probability proportional to non-domination
            if len(parents) > 0:
                parent = parents[rng.integers(len(parents))]
            else:
                parent = X[rng.integers(len(X))]
            
            # Adaptive mutation: combine local and global exploration
            if rng.random() < exploration_rate:
                # Large jump (global exploration)
                child = lo + (hi - lo) * rng.random(n)
            else:
                # Local mutation with adaptive step size
                progress = used / self.budget
                step_size = base_mut_rate * (1 - 0.5 * progress)
                child = parent + step_size * (hi - lo) * rng.standard_normal(n)
                child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archive
            all_X.append(child.copy())
            all_F.append(fc.copy())
            
            # Replace worst solution in population based on sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return all_X
# EVOLVE-BLOCK-END
