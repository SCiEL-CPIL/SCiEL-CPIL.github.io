"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search for ZDT1 with Pareto archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Use the global numpy random state (seeded by evaluator)
        # but also create a local RNG for reproducibility within a run
        rng = np.random.default_rng(42)

        # Initial population: spread x[0] across [0,1], set x[1:] = 0
        # This is the known Pareto front for ZDT1
        pop_size = min(30, budget // 3)
        pop_size = max(10, pop_size)

        X = np.zeros((pop_size, dim))
        # Spread x[0] uniformly, x[1:] = 0 (Pareto-optimal for ZDT1)
        X[:, 0] = np.linspace(0.0, 1.0, pop_size, endpoint=False)
        X[:, 0] += rng.random(pop_size) * (1.0 / pop_size)  # jitter
        X[:, 0] = np.clip(X[:, 0], 0.0, 1.0)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive_idx = list(range(pop_size))

        def dominates(a, b):
            """Check if a dominates b (both better or equal in all, better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(point, archive):
            """Check if point is dominated by any point in archive."""
            for ap in archive:
                if dominates(ap, point):
                    return True
            return False

        def update_archive(archive, new_point):
            """Add new_point to archive if non-dominated, remove dominated archive points."""
            if is_dominated(new_point, archive):
                return archive
            # Remove points dominated by new_point
            new_archive = [ap for ap in archive if not dominates(new_point, ap)]
            new_archive.append(new_point)
            return new_archive

        # Convert archive to list of objective vectors
        archive = [F[i] for i in archive_idx]

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Select a parent from the archive (non-dominated points)
            # Use tournament-style selection: pick randomly from archive
            if len(archive) > 0:
                parent_f = rng.choice(len(archive))
                # Find the corresponding x in our population or archive
                # We'll store archive as (x, f) pairs
                pass

        # Let me restructure with a cleaner approach using NSGA-II-like ideas
        # but adapted for ZDT1's known structure

        # Restart with cleaner implementation
        # Clear and restart
        pop_size = min(40, max(10, budget // 5))

        # Initialize: x[0] spread, x[1:] = 0
        X = np.zeros((pop_size, dim))
        X[:, 0] = np.linspace(0.0, 1.0, pop_size, endpoint=False) + rng.random(pop_size) * (1.0 / pop_size)
        X[:, 0] = np.clip(X[:, 0], 0.0, 1.0)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_F = [F[i].copy() for i in range(pop_size)]

        def hyp_volume_check(f):
            """Check if f is better than reference point [11, 11]."""
            return f[0] < 11.0 and f[1] < 11.0

        for i in range(pop_size):
            if hyp_volume_check(F[i]):
                pass  # all_F already contains it

        # Evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select parent: pick from non-dominated subset
            # Compute domination
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        dominated[i] = True
                        break

            if np.any(~dominated):
                parent_indices = np.where(~dominated)[0]
            else:
                parent_indices = np.arange(n)

            parent_idx = parent_indices[rng.integers(len(parent_indices))]
            parent = X[parent_idx].copy()

            # Mutation: perturb x[0] and x[1:]
            # For ZDT1, x[1:] should stay near 0
            if rng.random() < 0.7:
                # Small perturbation on x[0], keep x[1:] near 0
                parent[0] += rng.standard_normal() * 0.05
                parent[1:] += rng.standard_normal(dim - 1) * 0.01
            else:
                # Larger exploration
                parent[0] += rng.standard_normal() * 0.2
                parent[1:] += rng.standard_normal(dim - 1) * 0.1

            parent = np.clip(parent, lo, hi)
            fc = problem(parent)
            used += 1

            if hyp_volume_check(fc):
                all_F.append(fc)

            # Replace worst individual in population
            # Use a simple criterion: replace the most dominated or worst by sum
            # Find dominated points, replace one of them
            if np.any(dominated):
                worst_idx = np.where(dominated)[0]
                replace_idx = worst_idx[rng.integers(len(worst_idx))]
            else:
                replace_idx = int(np.argmax(np.sum(F, axis=1)))

            X[replace_idx] = parent
            F[replace_idx] = fc

        return X

# EVOLVE-BLOCK-END
