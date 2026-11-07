import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use a small population for DE-style search
        pop_size = 30
        CR = 0.5  # Crossover rate
        F_mut = 0.8  # Mutation factor

        # Initialize population
        X = lo + span * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best non-dominated points found so far (archive)
        archive_X = X.copy()
        archive_F = F.copy()

        def non_dominated_sort(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
                    # Check if j dominates i
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Main loop: one evaluation per step
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated points in current population
            nd_idx = non_dominated_sort(F)
            
            # If all points are dominated (shouldn't happen often), use all
            if len(nd_idx) == 0:
                nd_idx = np.arange(pop_size)
            
            # Pick 3 distinct parents from non-dominated set (or all if too few)
            if len(nd_idx) >= 3:
                r1, r2, r3 = np.random.choice(nd_idx, 3, replace=False)
            else:
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)

            # DE/rand/1 mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            v = np.clip(v, lo, hi)

            # Crossover
            trial = np.copy(X[r1])
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate trial
            f_trial = problem(trial)
            used += 1

            # Selection: keep better of trial and target
            # Compare trial vs X[r1]
            if np.all(f_trial <= F[r1]) and np.any(f_trial < F[r1]):
                # Trial dominates target
                X[r1] = trial
                F[r1] = f_trial
            elif np.all(F[r1] <= f_trial) and np.any(F[r1] < f_trial):
                # Target dominates trial, keep target
                pass
            else:
                # Non-dominated: keep the one with better crowding or random
                # Simple: keep trial with 50% probability
                if np.random.rand() < 0.5:
                    X[r1] = trial
                    F[r1] = f_trial

            # Update archive: add trial if it doesn't dominate and isn't dominated
            # Check if trial is dominated by any archive point
            dominated_by_archive = False
            for i in range(len(archive_F)):
                if np.all(archive_F[i] <= f_trial) and np.any(archive_F[i] < f_trial):
                    dominated_by_archive = True
                    break
            
            if not dominated_by_archive:
                # Remove archive points dominated by trial
                new_archive_X = []
                new_archive_F = []
                for i in range(len(archive_F)):
                    if not (np.all(f_trial <= archive_F[i]) and np.any(f_trial < archive_F[i])):
                        new_archive_X.append(archive_X[i])
                        new_archive_F.append(archive_F[i])
                
                # Add trial to archive
                new_archive_X.append(trial.copy())
                new_archive_F.append(f_trial.copy())
                
                archive_X = np.array(new_archive_X)
                archive_F = np.array(new_archive_F)

                # Limit archive size to 50 points, keep most spread out
                if len(archive_X) > 50:
                    # Simple: keep the first 50 non-dominated
                    nd_arch = non_dominated_sort(archive_F)
                    if len(nd_arch) > 50:
                        # Keep 50 from non-dominated
                        archive_X = archive_X[nd_arch[:50]]
                        archive_F = archive_F[nd_arch[:50]]
                    else:
                        archive_X = archive_X[nd_arch]
                        archive_F = archive_F[nd_arch]

        return X
