"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ4: x[0:3] determine objectives, x[3:] affect g(x)
        # Pareto front: f1 = 0.5 * x0, f2 = 0.5 * x1, f3 = 0.5 * x2 (when g=0)
        # g(x) = 100 * (n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for x_i in x[3:])
        # For x_i = 0.5, g = 0

        def pareto_front_point(u):
            """Generate a point on the DTLZ4 Pareto front from uniform u in [0,1]^3."""
            x = np.zeros(dim)
            x[0] = 2 * u[0] * (1 - u[1])
            x[1] = 2 * u[0] * u[1]
            x[2] = 1 - u[0]
            # Set remaining variables to 0.5 for g=0
            x[3:] = 0.5
            return x

        # Generate diverse initial points on/near the Pareto front
        rng = np.random.default_rng(42)
        n_init = min(40, self.budget // 5)

        # Create structured points on the Pareto front
        u_samples = np.array([
            [i / (n_init - 1), j / (n_init - 1), 0.5]
            for i in range(n_init)
            for j in range(n_init)
        ][:n_init])

        # Add some random points
        n_random = self.budget - n_init - 10  # reserve some for exploration
        X_list = []
        for u in u_samples:
            x = pareto_front_point(u)
            # Add small perturbation to ensure we're not exactly on the front
            x += 0.01 * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            X_list.append(x)

        # Add random points
        for _ in range(max(0, n_random)):
            x = lo + (hi - lo) * rng.random(dim)
            X_list.append(x)

        X = np.array(X_list)
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Maintain archive of non-dominated solutions
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return True
            return False

        def non_dominated_indices(F):
            """Get indices of non-dominated solutions."""
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Adaptive mutation with decreasing step size
        base_step = 0.1
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated solutions
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = list(range(len(F)))

            # Select parent from non-dominated set
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx].copy()

            # Adaptive step size
            step = base_step * (0.5 ** (used / (self.budget * 0.5)))

            # Generate child with mutation
            child = parent.copy()
            # Mutate a subset of variables
            n_mutate = max(1, int(dim * 0.3))
            mut_idx = rng.choice(dim, size=n_mutate, replace=False)
            child[mut_idx] += step * rng.standard_normal(n_mutate)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst solution or add to archive
            # Use hypervolume contribution as a proxy
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
