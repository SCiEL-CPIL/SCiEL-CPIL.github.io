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
        span = hi - lo

        rng = np.random.default_rng(0)

        # Archive: list of (x, f) for non-dominated solutions
        archive_x = []
        archive_f = []
        used = 0

        def is_dominated(f, archive_f_list):
            """Check if f is dominated by any point in archive_f_list."""
            if not archive_f_list:
                return False
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            """Add to archive if non-dominated; remove dominated archive points."""
            nonlocal archive_x, archive_f
            # Check if new point is dominated by archive
            if is_dominated(f, archive_f):
                return
            # Remove archive points dominated by new point
            keep = []
            for i, af in enumerate(archive_f):
                if np.all(f <= af) and np.any(f < af):
                    continue  # archive point is dominated
                keep.append(i)
            archive_x = [archive_x[i] for i in keep]
            archive_f = [archive_f[i] for i in keep]
            archive_x.append(x)
            archive_f.append(f)

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x))
            used += 1
            update_archive(x, f)
            return f

        # Phase 1: Structured initialization
        # Use a mix of random and stratified samples
        n_init = min(30, budget // 3)
        # Random samples
        for i in range(n_init):
            x = lo + span * rng.random(dim)
            x = np.clip(x, lo, hi)
            evaluate(x)
            if used >= budget:
                break
            if time.time() > deadline:
                break

        # Phase 2: Multi-Start DE
        # Population size for DE
        pop_size = min(15, max(5, (budget - used) // 4))
        if pop_size < 3:
            pop_size = 3

        # Build initial DE population from archive + random
        de_x = []
        de_f = []
        for i in range(min(len(archive_x), pop_size)):
            de_x.append(archive_x[i].copy())
            de_f.append(archive_f[i].copy())
        while len(de_x) < pop_size:
            x = lo + span * rng.random(dim)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            de_x.append(x)
            de_f.append(f)
            if used >= budget:
                break
            if time.time() > deadline:
                break

        de_x = np.array(de_x)
        de_f = np.array(de_f)

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        n_restarts = 0
        max_restarts = 5

        while used < budget and time.time() < deadline:
            # Check if we need a restart (diversity check)
            if len(de_x) > 5:
                # Simple diversity check: if all points are very close, restart
                center = de_x.mean(axis=0)
                dists = np.linalg.norm(de_x - center, axis=1)
                if np.max(dists) < 0.05 * np.max(span):
                    n_restarts += 1
                    if n_restarts > max_restarts:
                        break
                    # Restart with random population
                    de_x = lo + span * rng.random((pop_size, dim))
                    de_x = np.clip(de_x, lo, hi)
                    de_f = np.array([evaluate(x) for x in de_x])
                    if used >= budget or time.time() > deadline:
                        break
                    continue

            # Generate offspring using DE/BI/2 strategy
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break

                # Select base and two distinct random parents
                idxs = rng.choice(pop_size, size=4, replace=False)
                base = idxs[0]
                p1, p2, p3 = idxs[1], idxs[2], idxs[3]

                # DE mutation
                v = de_x[base] + F * (de_x[p1] - de_x[p2]) + 0.3 * (de_x[p1] - de_x[p3])
                v = np.clip(v, lo, hi)

                # Binary crossover
                trial = de_x[base].copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)

                # Evaluate trial
                f_trial = evaluate(trial)

                # Selection: keep better of trial and base
                if np.all(f_trial <= de_f[base]) and np.any(f_trial < de_f[base]):
                    de_x[base] = trial
                    de_f[base] = f_trial
                elif not (np.all(de_f[base] <= f_trial) and np.any(de_f[base] < f_trial)):
                    # Incomparable: keep trial with some probability for diversity
                    if rng.random() < 0.3:
                        de_x[base] = trial
                        de_f[base] = f_trial

            # Periodically refresh population with archive points
            if len(archive_x) > pop_size and used < budget and time.time() < deadline:
                # Replace worst DE points with best archive points
                # Sort DE by sum of objectives
                sums = de_f.sum(axis=1)
                worst_idx = np.argsort(sums)[-pop_size//3:]
                # Pick random archive points not already in DE
                archive_indices = rng.choice(len(archive_x), size=len(worst_idx), replace=False)
                for wi, ai in zip(worst_idx, archive_indices):
                    de_x[wi] = archive_x[ai].copy()
                    de_f[wi] = archive_f[ai].copy()

        return archive_x
# EVOLVE-BLOCK-END
