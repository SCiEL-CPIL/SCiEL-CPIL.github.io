"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using LHS initialization and Pareto-based local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        # 1. Latin Hypercube Sampling for initial population
        n_init = 40
        # Generate LHS in [0, 1]^dim
        lhs_samples = np.zeros((n_init, dim))
        for d in range(dim):
            perm = np.random.permutation(n_init)
            lhs_samples[:, d] = (perm + np.random.uniform(size=n_init)) / n_init
        
        # Map to bounds
        X_init = lo + lhs_samples * (hi - lo)
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        
        X_archive = X_init
        F_archive = F_init
        eval_count = n_init

        # Helper functions for Pareto dominance
        def is_dominated(point, points):
            """Check if point is dominated by any point in points."""
            if len(points) == 0:
                return False
            # points <= point (all objectives)
            all_leq = np.all(points <= point, axis=1)
            # points < point (at least one objective)
            any_lt = np.any(points < point, axis=1)
            return np.any(all_leq & any_lt)

        def get_dominated_indices(point, points):
            """Get indices of points in points that are dominated by point."""
            if len(points) == 0:
                return np.array([], dtype=int)
            all_leq = np.all(point <= points, axis=1)
            any_lt = np.any(point < points, axis=1)
            return np.where(all_leq & any_lt)[0]

        # 2. Local Search Loop
        # Use remaining budget for local refinement
        while eval_count < budget:
            # Identify non-dominated points in archive
            nd_indices = []
            for i in range(len(F_archive)):
                # Check if F_archive[i] is dominated by any other point in archive
                others = np.delete(F_archive, i, axis=0)
                if not is_dominated(F_archive[i], others):
                    nd_indices.append(i)
            
            if not nd_indices:
                nd_indices = list(range(len(F_archive)))

            # Select a parent randomly from non-dominated set
            parent_idx = np.random.choice(nd_indices)
            parent_x = X_archive[parent_idx].copy()
            
            # Generate child via Gaussian mutation
            # Scale noise to be a fraction of the range
            # DTLZ7 variables are in [0,1]
            noise_scale = 0.05 * (hi - lo)
            noise = np.random.normal(0, noise_scale, size=dim)
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate child
            child_f = problem(child_x)
            eval_count += 1
            
            # Update archive
            # If child is not dominated by any point in archive, it is a candidate for addition
            if not is_dominated(child_f, F_archive):
                # Find points in archive dominated by child
                dom_idx = get_dominated_indices(child_f, F_archive)
                
                # Remove dominated points from archive
                mask = np.ones(len(F_archive), dtype=bool)
                mask[dom_idx] = False
                X_archive = X_archive[mask]
                F_archive = F_archive[mask]
                
                # Add child to archive
                X_archive = np.vstack([X_archive, child_x.reshape(1, -1)])
                F_archive = np.vstack([F_archive, child_f.reshape(1, -1)])
            # If dominated, we just discard it from our archive, but it's already recorded by evaluator

        return None
# EVOLVE-BLOCK-END
