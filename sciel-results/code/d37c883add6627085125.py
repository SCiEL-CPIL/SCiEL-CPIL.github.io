"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 with Pareto-front-biased sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # WFG4 Pareto front: for 3 objectives, the front is characterized by x1.
        # For WFG4, the Pareto front has a specific structure. We'll use a strategy
        # that samples x1 uniformly and then performs local search around those points.
        
        # First, let's sample some points biased toward the front
        # For WFG4 with 12 dims and 3 objectives, x1 controls the first objective
        # Let's create initial population with x1 spread across range
        n_init = min(30, budget // 5)
        X = np.zeros((n_init, dim))
        # Spread x1 across its range
        x1_vals = np.linspace(lo[0], hi[0], n_init)
        X[:, 0] = x1_vals
        # Randomize other dimensions
        X[:, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random((n_init, dim - 1))
        
        F = np.array([problem(x) for x in X])
        used = n_init

        # Keep track of all evaluated points for final result
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Local search phase: perturb points and keep non-dominated improvements
        step_size = 0.1 * (hi - lo)
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Select a parent: prefer non-dominated points
            # Compute dominance
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            if np.any(~dominated):
                parents_idx = np.where(~dominated)[0]
            else:
                parents_idx = np.arange(len(F))
            
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[parent_idx]
            
            # Generate child with adaptive step size
            # Use smaller steps for more refined search
            current_step = step_size * (0.5 + 0.5 * rng.random())
            child = parent + current_step * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child is non-dominated with respect to current population
            is_dominated = False
            for j in range(len(F)):
                if np.all(F[j] <= fc) and np.any(F[j] < fc):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Add to population (replace worst if population is full)
                if len(F) < 50:
                    X = np.vstack([X, child.reshape(1, -1)])
                    F = np.vstack([F, fc.reshape(1, -1)])
                else:
                    # Replace worst point (highest sum of objectives)
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = child
                    F[worst] = fc
            else:
                # Even if dominated, sometimes keep for diversity if population is small
                if len(F) < 20:
                    X = np.vstack([X, child.reshape(1, -1)])
                    F = np.vstack([F, fc.reshape(1, -1)])

        # Collect all evaluated points
        # We need to return all objective vectors that were recorded
        # The problem says "Every returned objective vector is recorded"
        # So we need to make sure all points we evaluated are in the final result
        
        # Let's do a final cleanup: remove dominated points from the archive
        final_X = X
        final_F = F
        
        # Filter to keep only non-dominated points for better hypervolume
        if len(final_F) > 0:
            nondom_mask = np.zeros(len(final_F), dtype=bool)
            for i in range(len(final_F)):
                dominated = False
                for j in range(len(final_F)):
                    if i != j:
                        if np.all(final_F[j] <= final_F[i]) and np.any(final_F[j] < final_F[i]):
                            dominated = True
                            break
                nondom_mask[i] = not dominated
            
            if np.any(nondom_mask):
                final_X = final_X[nondom_mask]
                final_F = final_F[nondom_mask]
        
        return final_X
# EVOLVE-BLOCK-END
