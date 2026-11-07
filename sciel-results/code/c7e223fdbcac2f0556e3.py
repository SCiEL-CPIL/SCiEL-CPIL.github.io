import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_front(F_mat, n_total):
            """Return indices of the non-dominated front (rank 0)."""
            front = []
            for i in range(n_total):
                dominated = False
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[j], F_mat[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            return front

        def crowding_distance(F_mat, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-10:
                    for i in range(1, m - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Store all evaluated points and their objective values
        all_X = []
        all_F = []
        evals_used = 0

        # Center and radius of the search space
        center = 0.5 * (lo + hi)
        radius = 0.5 * np.linalg.norm(hi - lo)

        # Initial random sampling to get a diverse starting population
        n_init = min(50, max_evals)
        for i in range(n_init):
            x = np.random.uniform(lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            evals_used += 1

        # Maintain an archive of non-dominated points
        def update_archive():
            if not all_F:
                return []
            F_arr = np.array(all_F)
            front_indices = fast_non_dominated_front(F_arr, len(all_F))
            if len(front_indices) > 30:
                dists = crowding_distance(F_arr, front_indices)
                keep_idx = np.argsort(dists)[::-1][:30]
                keep_set = set(front_indices[i] for i in keep_idx)
            else:
                keep_set = set(front_indices)
            
            new_X = [all_X[i] for i in sorted(keep_set)]
            new_F = [all_F[i] for i in sorted(keep_set)]
            return new_X, new_F

        # Differential Evolution style search
        # DE/rand/1
        cr = 0.9  # crossover rate
        F = 0.8   # scale factor

        while evals_used < max_evals:
            if time.time() > deadline:
                break

            progress = evals_used / max_evals
            
            # Update archive
            archive_X, archive_F = update_archive()
            if not archive_X:
                break

            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            n_archive = len(archive_X)

            # Select batch size
            batch_size = min(10, max_evals - evals_used)

            for _ in range(batch_size):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break

                # DE/rand/1: pick 3 distinct parents
                if n_archive >= 3:
                    p1, p2, p3 = np.random.choice(n_archive, 3, replace=False)
                    v = archive_X_arr[p1] + F * (archive_X_arr[p2] - archive_X_arr[p3])
                else:
                    # Fallback to random if archive is too small
                    v = np.random.uniform(lo, hi)

                # Crossover
                r = np.random.rand(self.dim)
                jrand = np.random.randint(self.dim)
                u = np.where(r < cr, v, archive_X_arr[p1])
                u[jrand] = v[jrand]

                # Bounds
                u = np.clip(u, lo, hi)

                # Evaluate
                f = problem(u)
                all_X.append(u)
                all_F.append(f)
                evals_used += 1

                # Simple replacement: if new point is non-dominated, add to consideration
                # We'll let the archive update handle this

            # Periodically re-seed with random points to maintain diversity
            if evals_used % 30 == 0 and evals_used < max_evals:
                n_reseed = min(5, max_evals - evals_used)
                for _ in range(n_reseed):
                    if evals_used >= max_evals:
                        break
                    x = np.random.uniform(lo, hi)
                    f = problem(x)
                    all_X.append(x)
                    all_F.append(f)
                    evals_used += 1

        return np.array(all_X) if all_X else np.zeros((0, self.dim))
