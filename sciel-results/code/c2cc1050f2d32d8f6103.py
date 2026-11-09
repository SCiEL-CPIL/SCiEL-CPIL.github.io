"""Multi-objective optimizer for ZDT4 using a DE-based approach with niche preservation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with crowding distance and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        def is_deadline_exceeded():
            return (time.time() - start_time) > deadline

        def random_point():
            return lo + span * np.random.random(dim)

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_set(F):
            """Return indices of non-dominated points in F."""
            n = len(F)
            nd_idx = []
            for i in range(n):
                dom = False
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        dom = True
                        break
                if not dom:
                    nd_idx.append(i)
            return nd_idx

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            n_nd = len(nd_idx)
            if n_nd <= 2:
                return np.full(n_nd, np.inf)
            dist = np.zeros(n_nd)
            F_nd = F[nd_idx]
            for k in range(n_obj):
                order = np.argsort(F_nd[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_nd[order[-1], k] - F_nd[order[0], k]
                if f_range == 0:
                    continue
                for i in range(1, n_nd - 1):
                    dist[order[i]] += (F_nd[order[i + 1], k] - F_nd[order[i - 1], k]) / f_range
            return dist

        def update_archive(archive_F, archive_X, f_new, x_new):
            """Add new point to archive if not dominated. Remove dominated points."""
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return archive_F, archive_X
            archive_F.append(f_new)
            archive_X.append(x_new)
            new_archive_F = []
            new_archive_X = []
            for i in range(len(archive_F)):
                if not dominates(f_new, archive_F[i]):
                    new_archive_F.append(archive_F[i])
                    new_archive_X.append(archive_X[i])
            new_archive_F.append(f_new)
            new_archive_X.append(x_new)
            return new_archive_F, new_archive_X

        # Main search: DE with multi-objective selection
        pop_size = 20
        if budget < pop_size * 3:
            pop_size = max(8, budget // 4)

        # Initialize population
        X = np.array([random_point() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Initialize archive
        nd_idx = non_dominated_set(F)
        archive_F = [F[i] for i in nd_idx]
        archive_X = [X[i] for i in nd_idx]

        # DE parameters
        CR = 0.9
        F_mut = 0.5

        while used < budget:
            if is_deadline_exceeded():
                break

            # Select base from non-dominated
            nd_idx = non_dominated_set(F)
            if len(nd_idx) == 0:
                nd_idx = list(range(len(F)))
            base = nd_idx[np.random.randint(len(nd_idx))]

            r1 = np.random.randint(len(F))
            r2 = np.random.randint(len(F))
            while r1 == r2 or r1 == base:
                r1 = np.random.randint(len(F))
            while r2 == base or r2 == r1:
                r2 = np.random.randint(len(F))

            # Mutation
            v = X[base] + F_mut * (X[r1] - X[r2])

            # Crossover
            u = np.random.random(dim)
            j_rand = np.random.randint(dim)
            child = np.where(u < CR, v, X[base])
            child[j_rand] = v[j_rand]
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Selection
            if dominates(fc, F[base]):
                X[base] = child
                F[base] = fc
            elif not dominates(F[base], fc):
                dominated_idx = [i for i in range(len(F)) 
                                 if any(dominates(F[j], F[i]) for j in range(len(F)) if j != i)]
                if dominated_idx:
                    dom_counts = []
                    for i in dominated_idx:
                        count = sum(1 for j in range(len(F)) if j != i and dominates(F[j], F[i]))
                        dom_counts.append((count, i))
                    dom_counts.sort(reverse=True)
                    worst_idx = dom_counts[0][1]
                    X[worst_idx] = child
                    F[worst_idx] = fc
                else:
                    cd = crowding_distance(F, list(range(len(F))))
                    worst_idx = np.argmin(cd)
                    X[worst_idx] = child
                    F[worst_idx] = fc

            # Update archive
            archive_F, archive_X = update_archive(archive_F, archive_X, list(fc), child)

            # Periodic re-seeding for diversity
            if used % 25 == 0 and used < budget - 10:
                if is_deadline_exceeded():
                    break
                idx = np.random.randint(len(F))
                X[idx] = random_point()
                F[idx] = problem(X[idx])
                used += 1
                archive_F, archive_X = update_archive(archive_F, archive_X, list(F[idx]), X[idx])

        return X
# EVOLVE-BLOCK-END
