"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT4 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 270
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # ZDT4: x0 in [0,1], x1..x9 in [-5,5]
        # f0 = 1 + 10*(n-1) + sum((xi-0.2)^2 - 10*cos(4*pi*(xi-0.2))) for i=1..n
        # f1 = (sum(xi)^2/n) * (1 + sin(3*pi*f0))
        # Pareto front: f0 in [1, 10*(n-1)+1], f1 = (sum xi^2 / n) * (1+sin(3*pi*f0))
        # Minimum f1 is 0 when sum(xi)^2 = 0, i.e., all xi = 0
        # But we want to maximize hypervolume, so we want points on or near the Pareto front

        # Strategy:
        # 1. Generate points with x0 in [0,1] and xi in [-5,5]
        # 2. Use the structure: for each x0, the optimal xi values make sum(xi)^2 small
        #    while keeping f0 in a useful range
        # 3. Use local refinement around promising points

        # Key insight: f1 depends on (sum xi)^2/n * (1+sin(3*pi*f0))
        # To get f1 close to 0, we need sum(xi) close to 0
        # f0 = 1 + 10*9 + sum((xi-0.2)^2 - 10*cos(4*pi*(xi-0.2)))
        #    = 91 + sum((xi-0.2)^2 - 10*cos(4*pi*(xi-0.2)))
        # The term (xi-0.2)^2 - 10*cos(4*pi*(xi-0.2)) has minima at xi = 0.2 + k/2 for integer k
        # At xi = 0.2: (0)^2 - 10*cos(0) = -10
        # At xi = 0.7: (0.5)^2 - 10*cos(2*pi) = 0.25 - 10 = -9.75
        # At xi = -0.3: (-0.5)^2 - 10*cos(-2*pi) = 0.25 - 10 = -9.75
        # So minimum of each term is -10 at xi = 0.2 + k/2

        # For f0 to be small, we want xi near 0.2 (or other minima)
        # For f1 to be small, we want sum(xi) near 0

        # Let's use a hybrid approach:
        # - Generate diverse x0 values
        # - For each x0, try to find xi that balance f0 and f1

        # Budget allocation
        n_eval = 0
        archive = []  # list of (x, f) pairs
        archive_f = []  # just f values for quick comparison

        def is_dominated(f_new, archive_f_list):
            """Check if f_new is dominated by any point in archive_f_list."""
            for f_a in archive_f_list:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if not dominated."""
            nonlocal archive, archive_f
            if is_dominated(f, archive_f):
                return False
            # Remove dominated points from archive
            new_archive = []
            new_archive_f = []
            for i, (xa, fa) in enumerate(zip(archive, archive_f)):
                if not (np.all(f <= fa) and np.any(f < fa)):
                    new_archive.append(xa)
                    new_archive_f.append(fa)
            new_archive.append(x)
            new_archive_f.append(f)
            archive = new_archive
            archive_f = new_archive_f
            return True

        # Phase 1: Structured exploration
        # Sample x0 uniformly, and for each, try several xi configurations

        # For a given x0, f0 depends on xi values
        # We want to explore the trade-off between f0 and f1

        # Let's generate points by:
        # 1. Random x0 in [0,1]
        # 2. xi values: try to make sum(xi) small while keeping f0 reasonable

        # A good strategy: set xi = 0.2 + delta_i where delta_i are small
        # This keeps f0 low (near minimum) and sum(xi) = 9*0.2 + sum(delta_i) = 1.8 + sum(delta_i)
        # To make sum(xi) close to 0, we need sum(delta_i) = -1.8
        # But this increases f0 because (xi-0.2)^2 = delta_i^2

        # Alternative: mix some xi near 0.2 and some near -0.3 or other values
        # to balance sum(xi)

        # Let's use a population-based approach with structure

        pop_size = 50
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        # Initialize with structured points
        for i in range(pop_size):
            x = np.zeros(dim)
            x[0] = rng.random()  # x0 in [0,1]
            # For xi: use a mix of values that balance f0 and f1
            # Try different strategies for different points
            strategy = i % 5
            if strategy == 0:
                # All xi near 0.2 (minimizes f0, but sum(xi) ~ 1.8)
                x[1:] = 0.2 + 0.1 * rng.standard_normal(dim - 1)
            elif strategy == 1:
                # Mix of 0.2 and -0.3 to reduce sum
                for j in range(1, dim):
                    if j % 2 == 1:
                        x[j] = 0.2 + 0.1 * rng.standard_normal()
                    else:
                        x[j] = -0.3 + 0.1 * rng.standard_normal()
            elif strategy == 2:
                # Random in [-5, 5]
                x[1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random(dim - 1)
            elif strategy == 3:
                # Try to make sum(xi) = 0
                # Set xi = c + noise, choose c so sum is near 0
                noise = rng.standard_normal(dim - 1)
                c = -noise.sum() / (dim - 1)
                x[1:] = np.clip(c + 0.5 * noise, lo[1:], hi[1:])
            else:
                # Use the known minima pattern: xi = 0.2 + k/2
                for j in range(1, dim):
                    k = rng.integers(-4, 5)
                    x[j] = 0.2 + k / 2.0
                    x[j] = np.clip(x[j], lo[j], hi[j])

            x = np.clip(x, lo, hi)
            F[i] = problem(x)
            n_eval += 1
            X[i] = x

        # Add to archive
        for i in range(pop_size):
            add_to_archive(X[i].copy(), F[i].copy())

        # Phase 2: Local refinement around archive points
        while n_eval < self.budget and time.time() < deadline:
            # Pick a random archive point and perturb it
            idx = rng.integers(len(archive))
            x_parent = archive[idx].copy()
            f_parent = archive_f[idx].copy()

            # Multi-scale perturbation
            scale = 0.05 + 0.2 * rng.random()
            x_child = x_parent + scale * rng.standard_normal(dim)
            x_child = np.clip(x_child, lo, hi)

            f_child = problem(x_child)
            n_eval += 1

            if not is_dominated(f_child, archive_f):
                add_to_archive(x_child.copy(), f_child.copy())
            else:
                # Even if dominated, keep it in the population for further refinement
                # Replace the worst point in the current population
                # Simple replacement: replace a random point
                repl_idx = rng.integers(pop_size)
                X[repl_idx] = x_child
                F[repl_idx] = f_child

            # Periodically do a more aggressive exploration
            if n_eval % 20 == 0 and time.time() < deadline - 30:
                # Try a completely new structured point
                x_new = np.zeros(dim)
                x_new[0] = rng.random()
                # Use the sum-zero strategy with larger scale
                noise = rng.standard_normal(dim - 1)
                c = -noise.sum() / (dim - 1)
                x_new[1:] = np.clip(c + 1.0 * noise, lo[1:], hi[1:])
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                n_eval += 1
                if not is_dominated(f_new, archive_f):
                    add_to_archive(x_new.copy(), f_new.copy())

            # Check deadline
            if time.time() > deadline - 5:
                break

        return X
# EVOLVE-BLOCK-END
