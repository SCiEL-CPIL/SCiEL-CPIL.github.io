"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution based multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _non_dominated_mask(F):
        """Return boolean mask of non-dominated points in F (n x m array)."""
        n = F.shape[0]
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use most of budget for exploration
        pop_size = max(20, min(60, self.budget // 3))
        F_scale = 0.8  # DE mutation scale factor
        CR = 0.9       # Crossover rate

        # Initial population
        X = lo + rng.random((pop_size, self.dim)) * span
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of best non-dominated points seen
        archive_X = [x.copy() for x in X]
        archive_F = [f.copy() for f in F]

        while used < self.budget:
            if time.time() > deadline:
                break

            # DE/rand/1/bin for each individual in population
            for i in range(pop_size):
                if used >= self.budget or time.time() > deadline:
                    break

                # Select three distinct random individuals different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = X_a + F * (X_b - X_c)
                v = X[a] + F_scale * (X[b] - X[c])

                # Binary crossover
                trial = X[i].copy()
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                trial_F = np.array(problem(trial))
                used += 1

                # Selection: keep better individual (for single-objective it's simple,
                # for multi-objective we use dominance)
                # If trial dominates i, replace
                if np.all(trial_F <= F[i]) and np.any(trial_F < F[i]):
                    X[i] = trial
                    F[i] = trial_F
                    # Add to archive if non-dominated
                    self._update_archive(trial, trial_F, archive_X, archive_F)
                elif np.all(F[i] <= trial_F) and np.any(F[i] < trial_F):
                    # i dominates trial, keep i
                    pass
                else:
                    # Neither dominates the other - keep both in archive, 
                    # but for population, keep the one with better normalized sum
                    if np.sum(trial_F) < np.sum(F[i]):
                        X[i] = trial
                        F[i] = trial_F
                    self._update_archive(trial, trial_F, archive_X, archive_F)

            # Periodically prune archive to maintain diversity
            if len(archive_X) > 200:
                archive_X, archive_F = self._prune_archive(archive_X, archive_F)

        return X

    def _update_archive(self, x, f, archive_X, archive_F):
        """Add point to archive if it's non-dominated with respect to archive."""
        # Check if any archive point dominates this one
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return  # This point is dominated, don't add
        # Remove archive points dominated by this one
        to_remove = []
        for idx, af in enumerate(archive_F):
            if np.all(f <= af) and np.any(f < af):
                to_remove.append(idx)
        for idx in reversed(to_remove):
            archive_X.pop(idx)
            archive_F.pop(idx)
        archive_X.append(x.copy())
        archive_F.append(f.copy())

    def _prune_archive(self, archive_X, archive_F):
        """Prune archive to keep at most 200 points, maintaining diversity."""
        if len(archive_X) <= 200:
            return archive_X, archive_F
        
        F = np.array(archive_F)
        # Keep non-dominated points
        mask = self._non_dominated_mask(F)
        if np.sum(mask) <= 200:
            return [archive_X[i] for i in range(len(archive_X)) if mask[i]], \
                   [archive_F[i] for i in range(len(archive_F)) if mask[i]]
        
        # If still too many, use crowding-distance-like pruning
        # Simple approach: keep evenly spaced points along the front
        n_keep = 200
        # Sort by first objective and take evenly spaced
        sorted_idx = np.argsort(F[:, 0])
        step = max(1, len(sorted_idx) // n_keep)
        keep_idx = sorted_idx[::step][:n_keep]
        return [archive_X[i] for i in keep_idx], \
               [archive_F[i] for i in keep_idx]
# EVOLVE-BLOCK-END
