"""Multi-objective optimizer on DTLZ7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for DTLZ7 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ7 structure: M-1 objective variables + K constraint variables
        # M = 3, so M-1 = 2 objective variables, K = dim - (M-1) = 20 constraint vars
        M = n_obj
        K = dim - (M - 1)  # 20

        rng = np.random.default_rng(0)

        # Initial population: mix of random and DTLZ7-aware points
        pop_size = min(100, budget // 2)
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            if i < pop_size // 2:
                # Random initialization
                X[i] = lo + (hi - lo) * rng.random(dim)
            else:
                # DTLZ7-aware: set constraint variables to 0.5 (g=0)
                # and sample objective variables
                x_obj = rng.random(M - 1)
                X[i, :M - 1] = x_obj
                X[i, M - 1:] = 0.5  # makes g=0 for DTLZ7

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best points found (non-dominated set)
        all_points = []
        all_F = []

        def is_better_than_ref(f, ref):
            return np.all(f < ref)

        ref = np.array([1.1, 1.1, 25.0])

        # Record initial points
        for i in range(pop_size):
            if is_better_than_ref(F[i], ref):
                all_points.append(X[i].copy())
                all_F.append(F[i].copy())

        # Evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select parents: pick from non-dominated subset or random
            # Find non-dominated points in current population
            non_dom_idx = []
            for i in range(pop_size):
                dominated = False
                for j in range(pop_size):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)

            if len(non_dom_idx) == 0:
                non_dom_idx = list(range(pop_size))

            # Pick a parent
            parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            parent = X[parent_idx].copy()

            # Mutation: blend of Gaussian and DTLZ7-aware mutation
            if rng.random() < 0.3:
                # DTLZ7-aware mutation: perturb objective variables, keep constraint vars near 0.5
                child = parent.copy()
                # Perturb objective variables
                for j in range(M - 1):
                    child[j] += 0.1 * (hi[j] - lo[j]) * rng.standard_normal()
                    child[j] = np.clip(child[j], lo[j], hi[j])
                # Perturb constraint variables slightly around 0.5
                for j in range(M - 1, dim):
                    child[j] = 0.5 + 0.05 * rng.standard_normal()
                    child[j] = np.clip(child[j], lo[j], hi[j])
            else:
                # Gaussian mutation
                sigma = 0.1 * (hi - lo)
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Record if better than reference
            if is_better_than_ref(fc, ref):
                all_points.append(child.copy())
                all_F.append(fc.copy())

            # Replace worst point in population
            # Use sum of objectives as a simple fitness
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        # Final polish: local search around best points
        if time.time() < deadline - 10 and len(all_points) > 0:
            # Convert to array
            all_points_arr = np.array(all_points)
            all_F_arr = np.array(all_F)

            # Find non-dominated points
            n = len(all_F_arr)
            non_dom = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(all_F_arr[j] <= all_F_arr[i]) and np.any(all_F_arr[j] < all_F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom.append(i)

            # Local search around non-dominated points
            if len(non_dom) > 0:
                # Sample up to 20 non-dominated points for local search
                n_search = min(20, len(non_dom))
                search_idx = rng.choice(non_dom, size=n_search, replace=False)

                for idx in search_idx:
                    if time.time() > deadline:
                        break
                    if used >= budget:
                        break

                    base = all_points_arr[idx].copy()
                    # Try several small perturbations
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        child = base + 0.01 * (hi - lo) * rng.standard_normal(dim)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        if is_better_than_ref(fc, ref):
                            all_points.append(child.copy())
                            all_F.append(fc.copy())

        return X
# EVOLVE-BLOCK-END
