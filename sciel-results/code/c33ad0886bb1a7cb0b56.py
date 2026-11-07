"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 with origin-focused exploration."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initial population: mix of origin-focused and random
        pop_size = 20
        X = np.zeros((pop_size, dim))
        n_origin = max(4, pop_size // 3)
        for i in range(n_origin):
            X[i] = 0.05 * span * np.random.standard_normal(dim)
        for i in range(n_origin, pop_size):
            X[i] = lo + span * np.random.random(dim)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive: list of (x, f) tuples for efficient non-dominated maintenance
        archive = []

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F (2D case)."""
            for af in archive_F:
                if af[0] <= f[0] and af[1] <= f[1] and (af[0] < f[0] or af[1] < f[1]):
                    return True
            return False

        def update_archive(x, f):
            """Add point to archive if not dominated; remove dominated points."""
            f_tuple = (f[0], f[1])
            # Check if dominated by any archive point
            for (_, af) in archive:
                if af[0] <= f[0] and af[1] <= f[1] and (af[0] < f[0] or af[1] < f[1]):
                    return
            # Remove archive points dominated by new point
            new_archive = []
            for (ax, af) in archive:
                if not (f[0] <= af[0] and f[1] <= af[1] and (f[0] < af[0] or f[1] < af[1])):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive.clear()
            archive.extend(new_archive)

        # Initialize archive
        for i in range(pop_size):
            update_archive(X[i], F[i])

        # Get non-dominated indices from a set of F values
        def get_nd_indices(F_set):
            n = len(F_set)
            if n == 0:
                return np.array([], dtype=int)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if F_set[j][0] <= F_set[i][0] and F_set[j][1] <= F_set[i][1] and \
                       (F_set[j][0] < F_set[i][0] or F_set[j][1] < F_set[i][1]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # Adaptive mutation
        base_scale = 0.3
        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            scale = base_scale * (0.2 + 0.8 * (1.0 - progress))

            # Select parent: prefer non-dominated archive points
            if len(archive) > 0:
                archive_F_list = [af for (_, af) in archive]
                nd_idx = get_nd_indices(archive_F_list)
                if len(nd_idx) == 0:
                    nd_idx = np.array([np.random.randint(len(archive))])
                parent_idx = nd_idx[np.random.randint(len(nd_idx))]
                parent = archive[parent_idx][0]
            else:
                parent = np.random.uniform(lo, hi)

            # Mutation with adaptive scale
            child = parent + scale * span * np.random.standard_normal(dim)

            # Aggressive origin exploitation (RE21 structure)
            if np.random.random() < 0.2:
                origin_point = 0.01 * span * np.random.standard_normal(dim)
                child = 0.4 * child + 0.6 * origin_point

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Crowding-based replacement: replace the point with largest crowding distance
            # among dominated points, or the worst non-dominated point
            all_F = np.vstack([F, fc.reshape(1, -1)])
            all_X = np.vstack([X, child.reshape(1, -1)])
            
            # Find dominated points in population
            n_pop = len(all_F)
            dom_mask = np.zeros(n_pop, dtype=bool)
            for i in range(n_pop):
                for j in range(n_pop):
                    if i == j:
                        continue
                    if all_F[j][0] <= all_F[i][0] and all_F[j][1] <= all_F[i][1] and \
                       (all_F[j][0] < all_F[i][0] or all_F[j][1] < all_F[i][1]):
                        dom_mask[i] = True
                        break

            if np.any(dom_mask):
                # Replace a dominated point (random choice among dominated)
                dom_indices = np.where(dom_mask)[0]
                replace_idx = dom_indices[np.random.randint(len(dom_indices))]
            else:
                # All non-dominated: replace the one farthest from origin in objective space
                # (proxy for worst spread)
                replace_idx = int(np.argmax(np.sum(all_F, axis=1)))

            X = np.delete(all_X, replace_idx, axis=0)
            F = np.delete(all_F, replace_idx, axis=0)

            # Update archive
            update_archive(child, fc)

        return X
# EVOLVE-BLOCK-END
