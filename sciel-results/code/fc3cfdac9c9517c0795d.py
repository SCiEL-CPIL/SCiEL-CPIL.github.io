"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using an NSGA-II-inspired approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, budget // 4)
        if pop_size < 10:
            pop_size = 10

        # Initialize population with stratified sampling for better coverage
        rng = np.random.default_rng(42)

        # Use Latin hypercube sampling for initial population
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=n, seed=42)
            init_samples = sampler.random(pop_size)
            X = lo + (hi - lo) * init_samples
        except ImportError:
            X = lo + (hi - lo) * rng.random((pop_size, n))

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all non-dominated points found
        all_points = list(X)
        all_f = list(F)

        # Pareto front tracking
        def non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n_pts = len(F_arr)
            mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask

        # Crowding distance
        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices in F_arr."""
            if len(indices) <= 2:
                return np.ones(len(indices))
            cd = np.zeros(len(indices))
            f_sub = F_arr[indices]
            for k in range(m):
            # Sort by objective k
                order = np.argsort(f_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = f_sub[order[-1], k] - f_sub[order[0], k]
                if f_range == 0:
                    continue
                for i in range(1, len(order) - 1):
                    cd[order[i]] += (f_sub[order[i + 1], k] - f_sub[order[i - 1], k]) / f_range
            return cd

        # Crossover and mutation
        def crossover(parents):
            """Simulated binary crossover (SBX)."""
            if len(parents) < 2:
                # Just mutate
                p = parents[0]
                child = p + 0.1 * (hi - lo) * rng.standard_normal(n)
                return np.clip(child, lo, hi)
            
            # Pick two random parents
            idx1, idx2 = rng.choice(len(parents), size=2, replace=False)
            p1, p2 = parents[idx1], parents[idx2]
            
            # SBX crossover
            eta = 20.0
            child1 = np.empty(n)
            child2 = np.empty(n)
            for i in range(n):
                if rng.random() < 0.5:
                    if abs(p1[i] - p2[i]) < 1e-14:
                        child1[i] = child2[i] = p1[i]
                    else:
                        if rng.random() <= 0.5:
                            beta = (0.5 * (1 + (p1[i] - p2[i]) / (hi[i] - lo[i] + 1e-14))) ** (1.0 / (eta + 1))
                        else:
                            beta = (0.5 * (1 + (p2[i] - p1[i]) / (hi[i] - lo[i] + 1e-14))) ** (1.0 / (eta + 1))
                        beta_f = 1.0 / beta
                        child1[i] = 0.5 * ((1 + beta_f) * p1[i] + (1 - beta_f) * p2[i])
                        child2[i] = 0.5 * ((1 + 1.0 / beta_f) * p1[i] + (1 - 1.0 / beta_f) * p2[i])
                else:
                    child1[i] = p1[i]
                    child2[i] = p2[i]
            return np.clip(child1, lo, hi)

        def mutate(x):
            """Polynomial mutation."""
            eta = 20.0
            child = x.copy()
            for i in range(n):
                if rng.random() < 0.9:  # mutation probability 0.9 per gene
                    if rng.random() <= 0.5:
                        delta1 = (1.0 + 2.0 * (lo[i] - x[i]) / (hi[i] - lo[i] + 1e-14)) ** (1.0 / (eta + 1))
                        deltaq = 1.0 - delta1
                    else:
                        delta2 = (1.0 + 2.0 * (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)) ** (1.0 / (eta + 1))
                        deltaq = delta2 - 1.0
                    child[i] = x[i] + (hi[i] - lo[i]) * deltaq
            return np.clip(child, lo, hi)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated solutions in current population
            nd_mask = non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # All dominated, pick best by sum
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Select parents from non-dominated set
            if len(nd_indices) >= 2:
                # Use crowding distance to select parents
                cd = crowding_distance(F, nd_indices)
                # Tournament selection based on crowding distance
                parent_indices = []
                for _ in range(min(2, len(nd_indices))):
                    candidates = rng.choice(len(nd_indices), size=2, replace=False)
                    parent_indices.append(nd_indices[candidates[np.argmax(cd[candidates])]])
            else:
                parent_indices = nd_indices

            # Generate offspring
            if len(parent_indices) >= 2:
                child = crossover(X[parent_indices])
            else:
                child = mutate(X[parent_indices[0]])
            
            child = mutate(child)
            fc = problem(child)
            used += 1

            # Track all points
            all_points.append(child)
            all_f.append(fc)

            # Update population: replace worst dominated solution
            # Find dominated solutions
            dom_mask = ~nd_mask
            if np.any(dom_mask):
                # Among dominated, replace the one with worst crowding or worst sum
                dom_indices = np.where(dom_mask)[0]
                worst_idx = dom_indices[np.argmax(np.sum(F[dom_indices], axis=1))]
            else:
                # All non-dominated, replace the one with lowest crowding distance
                cd_all = crowding_distance(F, np.arange(len(F)))
                worst_idx = np.argmin(cd_all)

            X[worst_idx] = child
            F[worst_idx] = fc

        # Final evaluation of any remaining budget
        while used < budget:
            if time.time() > deadline:
                break
            # Random perturbation of best point
            best_idx = np.argmin(np.sum(F, axis=1))
            child = X[best_idx] + 0.05 * (hi - lo) * rng.standard_normal(n)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            all_points.append(child)
            all_f.append(fc)

        return X
# EVOLVE-BLOCK-END
