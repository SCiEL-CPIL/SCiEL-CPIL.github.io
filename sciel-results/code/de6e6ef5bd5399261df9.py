# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3.0, 5.0, 7.0], dtype=np.float64)
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        budget_used = 0

        # Halton sequence for space-filling initialization
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            return result

        # Generate Halton points
        n_init = 20
        X_init = np.zeros((n_init, dim))
        for i in range(n_init):
            for j in range(dim):
                X_init[i, j] = halton(i + 1, j + 2)
        
        # Map to bounds
        X_init = lo + X_init * (hi - lo)
        
        # Evaluate initial points
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        budget_used += n_init

        # Keep only non-dominated points as initial archive
        def is_non_dominated(f, F):
            for j in range(len(F)):
                if np.all(F[j] <= f) and np.any(F[j] < f):
                    return False
            return True

        archive_X = []
        archive_F = []
        for i in range(n_init):
            if is_non_dominated(F_init[i], F_init):
                archive_X.append(X_init[i])
                archive_F.append(F_init[i])
        
        if not archive_X:
            archive_X.append(X_init[0])
            archive_F.append(F_init[0])

        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        def hypervolume_contribution(point, F, ref):
            """Compute the hypervolume contribution of a point to the non-dominated set."""
            # Point must be non-dominated and better than ref
            if np.any(point >= ref):
                return 0.0
            
            # Check if point is dominated by any other point
            for j in range(len(F)):
                if np.all(F[j] <= point) and np.any(F[j] < point):
                    return 0.0
            
            # Calculate contribution: volume of region where this point is the "best"
            # In 3D, this is complex. Use a simplified approach:
            # For each objective, find the next worse point in that dimension among non-dominated points
            # Actually, let's use a simpler heuristic: contribution proportional to how unique the point is
            
            # Better approach: compute the actual contribution using inclusion-exclusion or recursive method
            # For 3 objectives, we can compute it as:
            # Sort non-dominated points by first objective
            # For each point, the contribution is the volume of the box that is not covered by other points
            
            # Simplified: use the "slice" method
            # For each point p, its contribution is the volume of [p1, next1] x [p2, next2] x [p3, next3]
            # where next_k is the minimum value of objective k among points that dominate p in at least one other objective
            # This is complex. Let's use a simpler proxy:
            
            # Proxy: contribution = product of (ref_k - p_k) * uniqueness_factor
            # Uniqueness factor based on distance to other points
            
            # Even simpler: just use the volume of the box from p to ref
            # This overestimates but gives a reasonable ordering
            return np.prod(ref - point)

        def select_best_for_local_search(archive_F, archive_X):
            """Select a point from archive to use as base for local search."""
            # Select point with highest hypervolume contribution
            if len(archive_F) == 0:
                return None, None
            
            # Compute contributions
            contributions = []
            for i in range(len(archive_F)):
                # Check if point is non-dominated in the current archive
                dominated = False
                for j in range(len(archive_F)):
                    if i != j and np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated = True
                        break
                if dominated:
                    contributions.append(0.0)
                else:
                    contributions.append(hypervolume_contribution(archive_F[i], archive_F, ref))
            
            contributions = np.array(contributions)
            # Add some randomness to avoid getting stuck
            contributions += np.random.uniform(0, 0.1 * np.max(contributions) if np.max(contributions) > 0 else 0.1)
            best_idx = np.argmax(contributions)
            return archive_X[best_idx], best_idx

        # Local search loop
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        while budget_used < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Select base point
            base_x, base_idx = select_best_for_local_search(archive_F, archive_X)
            if base_x is None:
                break
            
            # Try local perturbations
            found_improvement = False
            for step_size in step_sizes:
                if budget_used >= budget or time.time() > deadline - 5.0:
                    break
                
                # Generate random direction
                direction = np.random.randn(dim)
                direction /= np.linalg.norm(direction)
                
                # Create trial point
                trial = base_x + step_size * direction
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                budget_used += 1
                
                # Check if trial is non-dominated
                if is_non_dominated(f_trial, archive_F):
                    # Add to archive
                    archive_X = np.vstack([archive_X, trial.reshape(1, -1)])
                    archive_F = np.vstack([archive_F, f_trial.reshape(1, -1)])
                    found_improvement = True
                    break  # Move to next iteration with new archive
            
            # If no improvement found, try a larger jump
            if not found_improvement and budget_used < budget:
                if time.time() > deadline - 5.0:
                    break
                
                # Random restart near a good point
                if len(archive_X) > 0:
                    # Pick a random point from archive
                    rand_idx = np.random.randint(len(archive_X))
                    base = archive_X[rand_idx]
                    trial = base + 0.2 * (hi - lo) * np.random.randn(dim)
                    trial = np.clip(trial, lo, hi)
                    
                    f_trial = problem(trial)
                    budget_used += 1
                    
                    if is_non_dominated(f_trial, archive_F):
                        archive_X = np.vstack([archive_X, trial.reshape(1, -1)])
                        archive_F = np.vstack([archive_F, f_trial.reshape(1, -1)])

        # Return the archive
        return archive_X
# EVOLVE-BLOCK-END
