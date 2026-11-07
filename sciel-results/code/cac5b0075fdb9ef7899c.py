# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using hypervolume-guided
    evolutionary strategy with adaptive mutation and restarts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.RandomState(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (lo + hi) / 2

        # Population size
        pop_size = 30

        # Initialize with Latin Hypercube for good diversity
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            for j in range(self.dim):
                # LHS: stratified sampling in each dimension
                stratum = i / pop_size
                offset = stratum + rng.random() / pop_size
                X[i, j] = lo[j] + offset * span[j]

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all non-dominated points found
        archive_X = [X[i].copy() for i in range(pop_size)]
        archive_F = [F[i].copy() for i in range(pop_size)]

        def dominates(a, b):
            """Return True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(f, archive_F):
            """Check if f is non-dominated with respect to archive_F."""
            for af in archive_F:
                if dominates(af, f):
                    return False
            return True

        def update_archive(x, f, archive_X, archive_F):
            """Update archive with new point if non-dominated, remove dominated."""
            if is_non_dominated(f, archive_F):
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                # Remove dominated points
                new_idx = len(archive_X) - 1
                to_remove = []
                for i in range(len(archive_F)):
                    if i != new_idx and dominates(archive_F[new_idx], archive_F[i]):
                        to_remove.append(i)
                for i in reversed(to_remove):
                    archive_X.pop(i)
                    archive_F.pop(i)
                return True
            return False

        # Initial archive update
        archive_X = []
        archive_F = []
        for i in range(pop_size):
            update_archive(X[i], F[i], archive_X, archive_F)

        # Adaptive mutation parameters
        mutation_scale = 0.3  # initial scale relative to span
        best_hv = 0.0
        stagnation_count = 0
        stagnation_window = 10  # evaluations without improvement

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        def hypervolume_2d(points, ref):
            """Calculate hypervolume for 2D case (for quick estimation)."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref[1]
            for p in points:
                if p[0] < ref[0] and p[1] < ref[1]:
                    width = ref[0] - p[0]
                    height = prev_y - p[1]
                    if height > 0:
                        hv += width * height
                    prev_y = p[1]
            return hv

        def approximate_hv(points, ref):
            """Approximate hypervolume for 3D using sampling."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
            if len(valid) == 0:
                return 0.0
            # Use Monte Carlo approximation
            n_samples = 200
            min_bounds = np.array([min(p[i] for p in valid) for i in range(3)])
            max_bounds = ref.copy()
            volume = np.prod(max_bounds - min_bounds)
            count = 0
            for _ in range(n_samples):
                sample = min_bounds + rng.random(3) * (max_bounds - min_bounds)
                # Check if sample is dominated by any point
                dominated = False
                for p in valid:
                    if all(p[i] <= sample[i] for i in range(3)):
                        dominated = True
                        break
                if dominated:
                    count += 1
            return volume * count / n_samples

        # Main loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Select parent: use tournament selection based on hypervolume contribution
            # Simple approach: pick randomly from population
            parent_idx = rng.randint(pop_size)
            parent = X[parent_idx].copy()
            parent_f = F[parent_idx].copy()

            # Mutation: adaptive scaling
            # Reduce mutation scale over time for exploitation
            progress = used / self.budget
            current_scale = mutation_scale * (1.0 - 0.5 * progress)

            # DE-like mutation: use two other random individuals
            other_indices = list(range(pop_size))
            other_indices.remove(parent_idx)
            p1_idx = rng.choice(other_indices)
            other_indices.remove(p1_idx)
            p2_idx = rng.choice(other_indices)

            F_scale = 0.5 + 0.3 * rng.random()
            child = parent + F_scale * (X[p1_idx] - X[p2_idx])

            # Crossover
            crossover_rate = 0.5
            for j in range(self.dim):
                if rng.random() < crossover_rate:
                    child[j] = parent[j]

            # Add noise with adaptive scale
            noise_scale = current_scale * span
            child += noise_scale * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Update archive
            added = update_archive(child, fc, archive_X, archive_F)

            # Compute approximate hypervolume for progress tracking
            if used % 5 == 0 or added:
                current_hv = approximate_hv(archive_F, ref_point)
                if current_hv > best_hv + 1e-6:
                    best_hv = current_hv
                    stagnation_count = 0
                    # Increase mutation scale slightly on improvement
                    mutation_scale = min(0.5, mutation_scale * 1.05)
                else:
                    stagnation_count += 1
                    # Decrease mutation scale on stagnation
                    mutation_scale = max(0.1, mutation_scale * 0.95)

            # Replace parent if child is better (dominates or non-dominated)
            if dominates(fc, F[parent_idx]):
                X[parent_idx] = child
                F[parent_idx] = fc
            elif is_non_dominated(fc, [F[i] for i in range(pop_size)]):
                # Replace worst individual in population (highest max objective)
                worst_idx = np.argmax(np.max(F, axis=1))
                X[worst_idx] = child
                F[worst_idx] = fc

            # Periodic restart of stagnant individuals
            if stagnation_count > stagnation_window:
                # Restart some random individuals with fresh exploration
                n_restart = max(1, pop_size // 5)
                for _ in range(n_restart):
                    idx = rng.randint(pop_size)
                    X[idx] = center + 0.2 * span * rng.standard_normal(self.dim)
                    X[idx] = np.clip(X[idx], lo, hi)
                    F[idx] = problem(X[idx])
                    used += 1
                    if used >= self.budget:
                        break
                    update_archive(X[idx], F[idx], archive_X, archive_F)
                stagnation_count = 0
                mutation_scale = 0.3  # reset to moderate scale

            if used >= self.budget:
                break

        return X
# EVOLVE-BLOCK-END
