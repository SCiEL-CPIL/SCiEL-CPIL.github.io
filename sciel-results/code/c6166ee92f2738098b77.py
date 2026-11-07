"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS initialization + archive-based differential evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = 290.0  # Safety margin before 300s wall-clock kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def check_time():
            return time.time() - t_start < deadline

        # --- Helper: non-dominated check ---
        def is_dominated(f, F_archive):
            # Returns True if any point in F_archive dominates f
            # f is (n_obj,), F_archive is (k, n_obj)
            if len(F_archive) == 0:
                return False
            # A point dominates f if all objectives <= and at least one <
            leq = np.all(F_archive <= f, axis=1)
            lt = np.any(F_archive < f, axis=1)
            return np.any(leq & lt)

        def remove_dominated(F_archive):
            """Remove dominated points from F_archive. Returns non-dominated subset."""
            if len(F_archive) <= 1:
                return F_archive
            keep = np.ones(len(F_archive), dtype=bool)
            for i in range(len(F_archive)):
                if not keep[i]:
                    continue
                # Check if F_archive[i] is dominated by any other kept point
                others = np.arange(len(F_archive)) != i
                others_keep = others & keep
                if not np.any(others_keep):
                    continue
                sub = F_archive[others_keep]
                leq = np.all(sub <= F_archive[i], axis=1)
                lt = np.any(sub < F_archive[i], axis=1)
                if np.any(leq & lt):
                    keep[i] = False
            return F_archive[keep]

        # --- Latin Hypercube Sampling for initialization ---
        def lhs_sample(n, dim, lo, hi):
            """Generate n Latin Hypercube samples in [lo, hi]^dim."""
            samples = np.zeros((n, dim))
            for j in range(dim):
                perm = np.random.permutation(n)
                # Place each sample in a unique stratum
                samples[np.arange(n), j] = (perm + np.random.random(n)) / n
            return lo + (hi - lo) * samples

        # --- Phase 1: LHS initialization ---
        n_init = min(40, budget // 5)
        X_init = lhs_sample(n_init, dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Build initial archive
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        # Remove dominated
        archive_F = remove_dominated(archive_F)
        archive_X = archive_X[np.arange(len(archive_X))][np.where(np.ones(len(archive_X), dtype=bool))[np.arange(len(archive_F))]]
        # Simpler: rebuild
        mask = np.ones(len(F_init), dtype=bool)
        for i in range(len(F_init)):
            for j in range(len(F_init)):
                if i == j:
                    continue
                leq = np.all(F_init[j] <= F_init[i], axis=0)
                lt = np.any(F_init[j] < F_init[i], axis=0)
                if leq and lt:
                    mask[i] = False
                    break
        archive_X = X_init[mask]
        archive_F = F_init[mask]

        # --- Phase 2: Archive-based Differential Evolution ---
        # We maintain a population that evolves, and an archive of non-dominated solutions
        pop_size = min(30, max(10, budget // 6))
        # Initialize population from archive + random
        if len(archive_X) >= pop_size:
            idx = np.random.choice(len(archive_X), pop_size, replace=False)
            X_pop = archive_X[idx].copy()
        else:
            X_pop = np.vstack([archive_X, lhs_sample(pop_size - len(archive_X), dim, lo, hi)])
            X_pop = X_pop[:pop_size]

        F_pop = np.array([problem(x) for x in X_pop])
        used += pop_size

        # DE parameters
        CR = 0.9  # Crossover rate
        F_mut = 0.8  # Mutation factor

        while used < budget and check_time():
            # Select parents from archive (or population if archive is small)
            if len(archive_X) >= 3:
                i1, i2, i3 = np.random.choice(len(archive_X), 3, replace=False)
                base = archive_X[i1]
                v = base + F_mut * (archive_X[i2] - archive_X[i3])
            elif len(archive_X) >= 2:
                i1, i2 = np.random.choice(len(archive_X), 2, replace=False)
                base = archive_X[i1]
                v = base + F_mut * (archive_X[i1] - archive_X[i2])
            else:
                # Fallback to random
                v = lo + (hi - lo) * np.random.random(dim)

            # Clip
            v = np.clip(v, lo, hi)

            # Evaluate
            fv = problem(v)
            used += 1

            if not check_time():
                break

            # Check if v is non-dominated w.r.t. archive
            dominated = False
            for af in archive_F:
                leq = np.all(af <= fv)
                lt = np.any(af < fv)
                if leq and lt:
                    dominated = True
                    break

            if not dominated:
                # Remove archive points dominated by v
                new_mask = np.ones(len(archive_F), dtype=bool)
                for idx, af in enumerate(archive_F):
                    leq = np.all(fv <= af)
                    lt = np.any(fv < af)
                    if leq and lt:
                        new_mask[idx] = False
                archive_X = np.vstack([archive_X[new_mask], v])
                archive_F = np.vstack([archive_F[new_mask], fv])
            else:
                # Replace worst point in population
                # Use sum of normalized objectives as a simple indicator
                F_norm = F_pop / (np.max(F_pop, axis=0) + 1e-10)
                scores = np.sum(F_norm, axis=1)
                worst_idx = np.argmax(scores)
                X_pop[worst_idx] = v
                F_pop[worst_idx] = fv

        return archive_X
# EVOLVE-BLOCK-END
