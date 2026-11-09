"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential evolution based multi-objective optimizer with archive."""

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
        span = hi - lo

        # Use the global numpy random state (seeded by evaluator)
        # This ensures reproducibility with the evaluator's seed
        # But we also use a local rng for internal randomness
        rng = np.random

        # Structured initialization: corners, midpoints, and random
        # For 5D with bounds [1,3], create diverse initial points
        init_points = []

        # Corners (2^5 = 32, but we limit)
        for i in range(min(16, 2 ** dim)):
            point = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    point[j] = hi[j]
                else:
                    point[j] = lo[j]
            init_points.append(point)

        # Midpoints
        init_points.append((lo + hi) / 2.0)

        # Random points to fill up to population size
        pop_size = min(30, budget // 3)
        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive: non-dominated set
        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for a in archive:
                if np.all(a <= f) and np.any(a < f):
                    return True
            return False

        def update_archive(archive, x, f):
            """Add (x, f) to archive if non-dominated, remove dominated points."""
            if is_dominated(f, [a[1] for a in archive]):
                return archive

            # Remove points dominated by new point
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            return new_archive

        archive = []
        for i in range(len(X)):
            archive = update_archive(archive, X[i], F[i])

        # Differential evolution parameters
        F_factor = 0.8  # Scale factor
        CR = 0.5  # Crossover rate

        while used < budget:
            # Check time
            if time.time() - start_time > deadline:
                break

            # Select three distinct parents from archive if possible, else from population
            if len(archive) >= 3:
                idxs = rng.choice(len(archive), size=3, replace=False)
                x1, x2, x3 = archive[idxs[0]][0], archive[idxs[1]][0], archive[idxs[2]][0]
            else:
                idxs = rng.choice(len(X), size=3, replace=False)
                x1, x2, x3 = X[idxs[0]], X[idxs[1]], X[idxs[2]]

            # DE/rand/1 mutation
            v = x1 + F_factor * (x2 - x3)

            # Binomial crossover
            j_rand = rng.integers(dim)
            trial = np.copy(X[0])  # temp
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[rng.integers(len(X))][j]

            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            archive = update_archive(archive, trial, f_trial)

            # Replace worst in population if trial is better
            # Use a simple dominance check: if trial is not dominated by any in X, replace worst
            dominated_by_pop = False
            for i in range(len(X)):
                if np.all(F[i] <= f_trial) and np.any(F[i] < f_trial):
                    dominated_by_pop = True
                    break

            if not dominated_by_pop:
                # Find the worst point in X (dominated by most others)
                dominance_count = np.zeros(len(X))
                for i in range(len(X)):
                    for j in range(len(X)):
                        if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            dominance_count[j] += 1
                worst_idx = np.argmax(dominance_count)
                if dominance_count[worst_idx] == 0:
                    # All non-dominated, replace randomly
                    worst_idx = rng.integers(len(X))
                X[worst_idx] = trial
                F[worst_idx] = f_trial
            else:
                # Trial is dominated, but still might be useful for exploration
                # Replace the most dominated point with some probability
                if rng.random() < 0.3:
                    dominance_count = np.zeros(len(X))
                    for i in range(len(X)):
                        for j in range(len(X)):
                            if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                                dominance_count[j] += 1
                    worst_idx = np.argmax(dominance_count)
                    X[worst_idx] = trial
                    F[worst_idx] = f_trial

            # Periodically re-initialize some population members to maintain diversity
            if used % 50 == 0 and used < budget - 10:
                for k in range(3):
                    idx = rng.integers(len(X))
                    X[idx] = lo + span * rng.random(dim)
                    F[idx] = problem(X[idx])
                    used += 1
                    if used >= budget:
                        break
                    archive = update_archive(archive, X[idx], F[idx])

        return X
# EVOLVE-BLOCK-END
