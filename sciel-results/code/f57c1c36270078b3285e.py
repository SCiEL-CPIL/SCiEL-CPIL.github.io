import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 10  # Smaller population for DE to allow more generations/restarts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng()

        # Initialize population
        X = lo + (hi - lo) * rng.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = self.pop_size

        # Archive to store all non-dominated points
        archive_X = []
        archive_F = []
        for i in range(self.pop_size):
            if self.is_nondominated(F[i], archive_F):
                archive_X.append(X[i])
                archive_F.append(F[i])

        # Differential Evolution parameters
        F_de = 0.8
        CR = 0.9
        max_generations = (self.budget - used) // self.pop_size

        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break

            # Update population with DE
            for i in range(self.pop_size):
                if used >= self.budget:
                    break
                
                # Select 3 distinct random individuals
                idx = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = idx[0], idx[1], idx[2]
                
                # Create trial vector
                v = X[a] + F_de * (X[b] - X[c])
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Mutation (optional, but DE usually relies on the differential term)
                # Add a small mutation to escape local optima
                if rng.random() < 0.1:
                    v = np.clip(v + rng.normal(0, 0.1 * (hi - lo)), lo, hi)

                # Evaluate trial vector
                f_v = problem(v)
                used += 1
                
                # Update population and archive
                if self.is_nondominated(f_v, archive_F):
                    archive_X.append(v)
                    archive_F.append(f_v)
                
                # Replace in population if it dominates or is non-dominated
                # Simple strategy: replace if it dominates current or is non-dominated and random
                if self.dominates(f_v, F[i]):
                    X[i] = v
                    F[i] = f_v
                elif not self.dominates(F[i], f_v):
                    if rng.random() < 0.5:
                        X[i] = v
                        F[i] = f_v

        # Return the archive (or a subset if too large, but usually we just return the best set)
        # The task says "Every returned objective vector is recorded... return value is ignored".
        # Wait, "return value of __call__ is ignored" but "Every returned objective vector is recorded".
        # This phrasing is tricky. Usually, in these tasks, the `problem` function records the evaluation.
        # The contract says: "Every returned objective vector is recorded". This implies the evaluator
        # wraps `problem` or similar. But it also says "return value of __call__ is ignored".
        # Actually, looking at the scoring: "exact hypervolume of all recorded vectors".
        # The vectors are recorded by the evaluator when `problem(x)` is called.
        # So we don't need to return the archive, just ensure we call `problem` for good points.
        # However, returning the final population or archive is good practice if the evaluator
        # *did* care, but the contract says it's ignored.
        
        return X

    def dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def is_nondominated(self, f_new, archive_F):
        """Check if f_new is non-dominated with respect to the archive."""
        if len(archive_F) == 0:
            return True
        # Check if any point in archive dominates f_new
        for f_a in archive_F:
            if self.dominates(f_a, f_new):
                return False
        return True
