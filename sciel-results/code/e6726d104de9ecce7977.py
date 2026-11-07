# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using archive-based DE with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for diverse initial archive
        n_lhs = min(50, budget // 4)
        X_lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            X_lhs[:, d] = (perm + 0.5) / n_lhs
        X_lhs = lo + X_lhs * span
        F_lhs = np.array([problem(x) for x in X_lhs], dtype=float)
        used = n_lhs

        # Build initial archive (non-dominated points)
        archive_X = []
        archive_F = []

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Add LHS points to archive
        for i in range(n_lhs):
            if not is_dominated(F_lhs[i], [f for f in archive_F]):
                # Remove points in archive dominated by this new point
                new_archive_F = []
                new_archive_X = []
                for j in range(len(archive_F)):
                    if not dominates(F_lhs[i], archive_F[j]):
                        new_archive_F.append(archive_F[j])
                        new_archive_X.append(archive_X[j])
                new_archive_F.append(F_lhs[i])
                new_archive_X.append(X_lhs[i])
                archive_F = new_archive_F
                archive_X = new_archive_X

        # Also keep all evaluated points for reference
        all_X = list(X_lhs)
        all_F = list(F_lhs)

        # Phase 2: Differential Evolution with archive
        # Use a population that samples from the archive
        pop_size = min(30, max(10, budget // 6))
        
        # Initialize DE population from archive + random
        def sample_from_archive(archive_X, archive_F, lo, hi, span, dim):
            """Sample a point near archive points or randomly."""
            if len(archive_X) > 0:
                idx = np.random.randint(len(archive_X))
                base = archive_X[idx].copy()
                # Perturb
                noise = np.random.normal(0, 0.1, dim)
                base = base + noise * span
                base = np.clip(base, lo, hi)
                return base
            else:
                return lo + span * np.random.random(dim)

        # Create initial DE population
        X_pop = []
        F_pop = []
        for i in range(pop_size):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            if len(archive_X) > 0 and np.random.random() < 0.7:
                x = sample_from_archive(archive_X, archive_F, lo, hi, span, dim)
            else:
                x = lo + span * np.random.random(dim)
            f = problem(x)
            used += 1
            X_pop.append(x)
            F_pop.append(f)
            all_X.append(x)
            all_F.append(f)
            # Update archive
            if not is_dominated(f, archive_F):
                new_archive_F = []
                new_archive_X = []
                for j in range(len(archive_F)):
                    if not dominates(f, archive_F[j]):
                        new_archive_F.append(archive_F[j])
                        new_archive_X.append(archive_X[j])
                new_archive_F.append(f)
                new_archive_X.append(x)
                archive_F = new_archive_F
                archive_X = new_archive_X

        X_pop = np.array(X_pop)
        F_pop = np.array(F_pop)

        # DE parameters
        F_cr = 0.8  # Crossover rate
        F_scale = 0.5  # Scale factor

        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring via DE
            new_X = []
            new_F = []
            n_offspring = min(pop_size, budget - used)

            for i in range(n_offspring // 2):
                if time.time() > deadline:
                    break

                # Select three distinct parents
                idxs = np.random.choice(len(X_pop), size=3, replace=False)
                a, b, c = idxs
                x_a = X_pop[a]
                x_b = X_pop[b]
                x_c = X_pop[c]

                # DE mutation: v = x_a + F_scale * (x_b - x_c)
                v = x_a + F_scale * (x_b - x_c)
                v = np.clip(v, lo, hi)

                # Binary crossover
                trial = x_a.copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < F_cr or j == j_rand:
                        trial[j] = v[j]

                # Evaluate
                f_trial = problem(trial)
                used += 1
                all_X.append(trial)
                all_F.append(f_trial)

                # Update archive
                if not is_dominated(f_trial, archive_F):
                    new_archive_F = []
                    new_archive_X = []
                    for j in range(len(archive_F)):
                        if not dominates(f_trial, archive_F[j]):
                            new_archive_F.append(archive_F[j])
                            new_archive_X.append(archive_X[j])
                    new_archive_F.append(f_trial)
                    new_archive_X.append(trial)
                    archive_F = new_archive_F
                    archive_X = new_archive_X

                new_X.append(trial)
                new_F.append(f_trial)

                # Second offspring from another pair
                if used < budget and time.time() < deadline:
                    idxs2 = np.random.choice(len(X_pop), size=3, replace=False)
                    a2, b2, c2 = idxs2
                    x_a2 = X_pop[a2]
                    x_b2 = X_pop[b2]
                    x_c2 = X_pop[c2]

                    v2 = x_a2 + F_scale * (x_b2 - x_c2)
                    v2 = np.clip(v2, lo, hi)

                    trial2 = x_a2.copy()
                    j_rand2 = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.random() < F_cr or j == j_rand2:
                            trial2[j] = v2[j]

                    f_trial2 = problem(trial2)
                    used += 1
                    all_X.append(trial2)
                    all_F.append(f_trial2)

                    # Update archive
                    if not is_dominated(f_trial2, archive_F):
                        new_archive_F = []
                        new_archive_X = []
                        for j in range(len(archive_F)):
                            if not dominates(f_trial2, archive_F[j]):
                                new_archive_F.append(archive_F[j])
                                new_archive_X.append(archive_X[j])
                        new_archive_F.append(f_trial2)
                        new_archive_X.append(trial2)
                        archive_F = new_archive_F
                        archive_X = new_archive_X

                    new_X.append(trial2)
                    new_F.append(f_trial2)

            if len(new_X) > 0:
                X_pop = np.array(new_X)
                F_pop = np.array(new_F)
            else:
                break

        return all_X
# EVOLVE-BLOCK-END
