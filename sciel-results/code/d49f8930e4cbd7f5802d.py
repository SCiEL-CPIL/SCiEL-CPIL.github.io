"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE/rand/1 based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 30

        # Initialize population uniformly in bounds
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        # Differential Evolution parameters
        CR = 0.9
        F_factor = 0.8

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F_arr):
            """Check if f_new is dominated by any point in archive_F_arr."""
            if len(archive_F_arr) == 0:
                return False
            # f_new is dominated if there exists a point that is <= in all objectives and < in at least one
            for f in archive_F_arr:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def update_archive(x_new, f_new):
            """Update archive with new point if non-dominated."""
            non_local = []
            # Check if new point is dominated by any in archive
            dominated = False
            for i, f_old in enumerate(archive_F):
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    dominated = True
                    break
            if dominated:
                return
            # Remove points dominated by new point
            for i, f_old in enumerate(archive_F):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue
                else:
                    non_local.append((archive_X[i], f_old))
            archive_X.extend([x_new])
            archive_F.extend([f_new])
            # Replace archive with non-dominated ones
            del archive_X[:]
            del archive_F[:]
            for ax, af in non_local:
                archive_X.append(ax)
                archive_F.append(af)
            archive_X.append(x_new)
            archive_F.append(f_new)

        # Initial archive
        for i in range(pop_size):
            update_archive(X[i].copy(), F[i].copy())

        # Main DE loop
        while evaluations < budget:
            if time.time() > deadline:
                break

            # Create offspring
            for i in range(pop_size):
                if evaluations >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices
                while True:
                    a, b, c = rng.integers(0, pop_size, size=3)
                    if a != b and b != c and a != c:
                        break

                # Create trial vector
                v = X[a] + F_factor * (X[b] - X[c])
                v = np.clip(v, lo, hi)

                # Evaluate trial
                f_v = problem(v)
                evaluations += 1

                # Selection: compare with current and with X[i]
                # Use dominance-based selection
                # If v dominates X[i], accept
                # If X[i] dominates v, reject
                # Otherwise, accept if v is non-dominated in the population or randomly
                if np.all(f_v <= F[i]) and np.any(f_v < F[i]):
                    # v dominates X[i]
                    X[i] = v
                    F[i] = f_v
                    update_archive(v.copy(), f_v.copy())
                elif np.all(F[i] <= f_v) and np.any(F[i] < f_v):
                    # X[i] dominates v, reject
                    pass
                else:
                    # Incomparable; accept with some probability to maintain diversity
                    if rng.random() < 0.5:
                        X[i] = v
                        F[i] = f_v
                        update_archive(v.copy(), f_v.copy())

        # Final: return archive points (or all non-dominated from population)
        # The return value is ignored, but we return the archive
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return X

# EVOLVE-BLOCK-END
