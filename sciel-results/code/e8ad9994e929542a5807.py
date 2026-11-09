"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using PAES (Pareto Archived Evolutionary Strategy)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        d = self.dim
        n_obj = self.n_obj

        # Archive size: Larger is better for HV, but slower to update.
        # 100 is a good balance for 200 budget.
        archive_size = 100
        
        # Initialize archive
        # Start with a few random points to populate the archive
        init_pop_size = min(archive_size, self.budget // 2)
        X_init = lo + range_ * rng.random((init_pop_size, d))
        F_init = np.array([problem(x) for x in X_init])
        used = init_pop_size

        # Build initial archive
        archive_X = []
        archive_F = []
        
        # Simple insertion: keep non-dominated points
        # Since init is random, most will be non-dominated or few will be
        for i in range(init_pop_size):
            x = X_init[i]
            f = F_init[i]
            
            # Check if f is dominated by any in archive
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                # f is dominated if exists g in F_arr such that g <= f and g != f
                dominates = np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1)
                if np.any(dominates):
                    continue
            
            # Remove points in archive dominated by f
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                dominated_by_f = np.all(f <= F_arr, axis=1) & np.any(f < F_arr, axis=1)
                if np.any(dominated_by_f):
                    # Filter out dominated
                    mask = ~dominated_by_f
                    archive_X = [archive_X[i] for i in range(len(archive_X)) if mask[i]]
                    archive_F = [archive_F[i] for i in range(len(archive_F)) if mask[i]]
            
            # Add if space available
            if len(archive_X) < archive_size:
                archive_X.append(x)
                archive_F.append(f)
            else:
                # If full, we might need to replace the worst (lowest crowding)
                # But for init, we just skip if full and dominated check passed?
                # Actually, if it's non-dominated and archive is full, we should compare crowding.
                # For simplicity in init, just add if not dominated, even if it exceeds size slightly, 
                # or truncate later. Let's just add and truncate in main loop.
                archive_X.append(x)
                archive_F.append(f)

        # Truncate initial archive if too large
        if len(archive_X) > archive_size:
            # Keep the ones with highest crowding distance
            F_arr = np.array(archive_F)
            cd = self._crowding_distance(F_arr)
            order = np.argsort(-cd)
            archive_X = [archive_X[i] for i in order[:archive_size]]
            archive_F = [archive_F[i] for i in order[:archive_size]]

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break

            if not archive_X:
                break

            # Select parent
            # Use tournament selection on archive based on crowding distance
            # This biases towards diverse regions
            F_arr = np.array(archive_F)
            cd = self._crowding_distance(F_arr)
            
            # Binary tournament
            i1 = rng.integers(len(archive_X))
            i2 = rng.integers(len(archive_X))
            if cd[i1] >= cd[i2]:
                p_idx = i1
            else:
                p_idx = i2
            
            parent = archive_X[p_idx]
            
            # Mutation: Gaussian
            # Adaptive sigma: start larger, shrink later?
            # Or just fixed.
            sigma = 0.1
            # For DTLZ4, smaller sigma is often better for refinement
            # Let's use a slightly smaller sigma to refine
            sigma = 0.05 
            
            child = parent + sigma * range_ * rng.standard_normal(d)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update Archive
            # 1. Check if child is dominated by archive
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                dominates = np.all(F_arr <= fc, axis=1) & np.any(F_arr < fc, axis=1)
                if np.any(dominates):
                    # Child is dominated, discard
                    continue
            
            # 2. Remove points in archive dominated by child
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                dominated_by_child = np.all(fc <= F_arr, axis=1) & np.any(fc < F_arr, axis=1)
                if np.any(dominated_by_child):
                    mask = ~dominated_by_child
                    archive_X = [archive_X[i] for i in range(len(archive_X)) if mask[i]]
                    archive_F = [archive_F[i] for i in range(len(archive_F)) if mask[i]]
            
            # 3. Add child if space, else replace worst
            if len(archive_X) < archive_size:
                archive_X.append(child)
                archive_F.append(fc)
            else:
                # Calculate crowding for current archive + child
                # Append temporarily
                temp_X = archive_X + [child]
                temp_F = archive_F + [fc]
                F_temp = np.array(temp_F)
                cd_temp = self._crowding_distance(F_temp)
                
                # Find index of minimum crowding
                worst_idx = np.argmin(cd_temp)
                
                # Replace
                archive_X[worst_idx] = temp_X[worst_idx]
                archive_F[worst_idx] = temp_F[worst_idx]

        # Return the archive as the final population
        # The evaluator records all objective vectors returned.
        # The return value is ignored, but we should return something valid.
        if archive_X:
            return np.array(archive_X)
        return np.array([])

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 1:
            return np.ones(n)
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            rng_m = F[order[-1], m] - F[order[0], m]
            if rng_m > 1e-15:
                for i in range(1, n - 1):
                    cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / rng_m
        return cd
# EVOLVE-BLOCK-END
