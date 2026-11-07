import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280

        # MOPSO parameters
        n_particles = 30
        w_max = 0.9
        w_min = 0.1
        c1 = 1.5
        c2 = 1.5
        archive_size = 50

        # Initialize particles
        X = lo + (hi - lo) * np.random.random((n_particles, dim))
        V = (hi - lo) * np.random.random((n_particles, dim)) * 0.1
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_particles

        # Personal bests
        pbest_X = X.copy()
        pbest_F = F.copy()

        # Archive: list of (x, f)
        archive = [(x.copy(), f.copy()) for x, f in zip(X, F)]

        # Velocity bounds
        v_max = 0.5 * (hi - lo)

        while used < budget:
            if time.time() > deadline:
                break

            # Update archive
            archive = self.update_archive(archive, X, F, archive_size)

            # Select leaders from archive for social component
            # Use a subset of archive for leader selection to save time
            n_leaders = min(len(archive), 10)
            if n_leaders > 0:
                # Randomly select leaders
                leader_indices = np.random.choice(len(archive), n_leaders, replace=False)
                leaders_X = np.array([archive[i][0] for i in leader_indices])
                leaders_F = np.array([archive[i][1] for i in leader_indices])
            else:
                leaders_X = pbest_X
                leaders_F = pbest_F

            # Update velocity and position
            w = w_max - (w_max - w_min) * (used / budget)

            for i in range(n_particles):
                if used >= budget:
                    break

                # Cognitive component
                r1 = np.random.random(dim)
                cognitive = c1 * r1 * (pbest_X[i] - X[i])

                # Social component: select a leader
                leader_idx = np.random.randint(n_leaders)
                r2 = np.random.random(dim)
                social = c2 * r2 * (leaders_X[leader_idx] - X[i])

                # Update velocity
                V[i] = w * V[i] + cognitive + social

                # Clip velocity
                V[i] = np.clip(V[i], -v_max, v_max)

                # Update position
                X[i] = X[i] + V[i]

                # Clip position to bounds
                X[i] = np.clip(X[i], lo, hi)

                # Evaluate
                F[i] = problem(X[i])
                used += 1

                # Update personal best
                if self.dominates(F[i], pbest_F[i]):
                    pbest_F[i] = F[i].copy()
                    pbest_X[i] = X[i].copy()

        return X

    def update_archive(self, archive, X, F, max_size):
        """
        Update archive with new solutions, maintaining non-dominance and diversity.
        """
        # Add new solutions
        for x, f in zip(X, F):
            archive.append((x.copy(), f.copy()))

        # Remove dominated solutions
        new_archive = []
        for i, (xi, fi) in enumerate(archive):
            dominated = False
            for j, (xj, fj) in enumerate(archive):
                if i == j:
                    continue
                if self.dominates(fj, fi):
                    dominated = True
                    break
            if not dominated:
                new_archive.append((xi, fi))

        # If archive is too large, remove the point with smallest hypervolume contribution
        while len(new_archive) > max_size:
            if len(new_archive) <= 1:
                break

            # Compute hypervolume contribution of each point
            contributions = []
            for i, (xi, fi) in enumerate(new_archive):
                # Check if fi is better than reference [3,5,7]
                ref = np.array([3.0, 5.0, 7.0])
                if np.any(fi >= ref):
                    contributions.append(0.0)
                    continue

                # Compute contribution: hypervolume of fi vs rest
                # Simplified: compute the volume of the box dominated by fi
                # that is not dominated by any other point in the archive
                # This is expensive, so use a simpler heuristic:
                # contribution = product of (ref_k - fi_k) minus overlaps
                # For simplicity, just use the raw volume as a proxy
                vol = 1.0
                for k in range(len(fi)):
                    vol *= (ref[k] - fi[k])
                contributions.append(vol)

            # Remove the point with smallest contribution
            min_idx = np.argmin(contributions)
            new_archive.pop(min_idx)

        return new_archive

    def dominates(self, a, b):
        """
        Returns True if a dominates b (a is better than b in all objectives).
        """
        return np.all(a <= b) and np.any(a < b)
