"""Multi-objective optimizer for RE37: Archive-based Differential Evolution with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based Multi-Objective Differential Evolution for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initial random population
        np.random.seed(42)
        pop_size = min(20, budget // 2)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive_X = X.copy()
        archive_F = F.copy()

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(new_x, new_f, archive_X, archive_F):
            """Add new point to archive if non-dominated, remove dominated archive points."""
            # Check if new point is dominated by archive
            if is_dominated(new_f, archive_F):
                return archive_X, archive_F

            # Remove archive points dominated by new point
            keep = []
            for i, af in enumerate(archive_F):
                if not (np.all(new_f <= af) and np.any(new_f < af)):
                    keep.append(i)

            new_archive_X = np.vstack([archive_X[keep], new_x])
            new_archive_F = np.vstack([archive_F[keep], new_f])
            return new_archive_X, new_archive_F

        # Differential Evolution loop
        Fp = 0.7  # mutation factor
        Cr = 0.9  # crossover rate

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select base vector
            base_idx = np.random.randint(pop_size)
            base = X[base_idx]

            # Generate trial vector using DE/rand/1
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            mutant = X[a] + Fp * (X[b] - X[c])
            mutant = np.clip(mutant, lo, hi)

            # Binary crossover
            trial = mutant.copy()
            jrand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < Cr or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = base[j]

            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(trial, f_trial, archive_X, archive_F)

            # Tournament selection: replace worst if trial is better
            # Simple strategy: replace the worst in population
            worst_idx = np.argmax(np.sum(F, axis=1))
            if np.sum(f_trial) < np.sum(F[worst_idx]):
                X[worst_idx] = trial
                F[worst_idx] = f_trial
            elif np.random.random() < 0.1:
                # Occasionally replace even if not strictly better (for diversity)
                X[worst_idx] = trial
                F[worst_idx] = f_trial

        # Final cleanup: ensure we return something valid
        if len(archive_F) > 0:
            return archive_X
        else:
            return X

# EVOLVE-BLOCK-END
