"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Multi-Start Differential Evolution with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        rng = np.random.default_rng(0)

        # Structured initialization: corners, midpoints, random
        # For 5D with bounds [1,3], create diverse initial points
        init_points = []
        
        # Corners (2^5 = 32 points, but we'll use a subset)
        for i in range(32):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)
        
        # Midpoints
        init_points.append((lo + hi) / 2)
        
        # Add random points to fill up to population size
        pop_size = 30
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * rng.random(dim))
        
        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]

        def is_dominated(f, archive_F):
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f, archive_X, archive_F):
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            # Remove dominated solutions from archive
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            return new_archive_X, new_archive_F

        # Differential Evolution parameters
        F_factor = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        while used < budget:
            # Check deadline
            if time.time() - start_time > deadline:
                break

            # Select parents from non-dominated solutions in archive
            if len(archive_X) > 0:
                parent_idx = rng.integers(len(archive_X))
                parent = archive_X[parent_idx]
            else:
                parent = X[rng.integers(len(X))]

            # Differential Evolution: select 3 random distinct indices
            idx1, idx2, idx3 = rng.choice(len(X), size=3, replace=False)
            
            # Ensure they are distinct from each other
            while idx1 == idx2 or idx2 == idx3 or idx1 == idx3:
                idx1, idx2, idx3 = rng.choice(len(X), size=3, replace=False)

            # Create trial vector
            r = rng.random(dim)
            trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            trial = np.where(r < CR, trial, parent)
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)

            # Replace worst solution in population if trial is better
            # Use a simple dominance check or sum-based comparison
            if is_dominated(F[np.argmax(np.sum(F, axis=1))], [ft]):
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = trial
                F[worst_idx] = ft
            elif len(archive_F) > 0 and not is_dominated(ft, archive_F):
                # If trial is non-dominated, replace the worst non-archive solution
                # Find the worst solution not in archive
                in_archive = [False] * len(X)
                for i, xf in enumerate(X):
                    for j, af in enumerate(archive_F):
                        if np.allclose(xf, archive_X[j]):
                            in_archive[i] = True
                            break
                candidates = [i for i in range(len(X)) if not in_archive[i]]
                if candidates:
                    worst_idx = max(candidates, key=lambda i: np.sum(F[i]))
                    X[worst_idx] = trial
                    F[worst_idx] = ft

        return X
# EVOLVE-BLOCK-END
