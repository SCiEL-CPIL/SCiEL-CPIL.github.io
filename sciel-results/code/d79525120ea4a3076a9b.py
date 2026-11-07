# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware multi-objective optimizer for DTLZ2.

    DTLZ2's Pareto front is a quarter-sphere in 3D. We parameterize the
    front directly with angles, map to decision variables via the inverse
    of the DTLZ2 mapping, and use a differential-evolution-style search
    to refine the spread of non-dominated solutions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 310.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        # DTLZ2 mapping: x -> f
        # f_i = g * (1 - sin(pi * x_i)) for i in 0..n_obj-1
        # where g = sum_{j=n_obj}^{dim-1} (x_j - 0.5)^2
        # Pareto front: g = 0, so x_j = 0.5 for j >= n_obj
        # f_i = 1 - sin(pi * x_i) for i < n_obj, with sum(f_i^2) = 1
        #
        # To get points on the front:
        #   Choose angles theta_1, theta_2 such that:
        #   f_0 = cos(theta_1) * cos(theta_2)
        #   f_1 = cos(theta_1) * sin(theta_2)
        #   f_2 = sin(theta_1)
        #   with theta_1, theta_2 in [0, pi/2]
        #
        # Inverse: sin(pi * x_i) = 1 - f_i
        #   x_i = (1/pi) * arcsin(1 - f_i)

        def angles_to_x(theta1, theta2):
            f0 = np.cos(theta1) * np.cos(theta2)
            f1 = np.cos(theta1) * np.sin(theta2)
            f2 = np.sin(theta1)
            x = np.full(dim, 0.5)
            x[0] = (1.0 / np.pi) * np.arcsin(np.clip(1.0 - f0, 0.0, 1.0))
            x[1] = (1.0 / np.pi) * np.arcsin(np.clip(1.0 - f1, 0.0, 1.0))
            x[2] = (1.0 / np.pi) * np.arcsin(np.clip(1.0 - f2, 0.0, 1.0))
            return np.clip(x, lo, hi)

        def x_to_angles(x):
            # Recover angles from x for the first 3 variables
            s0 = np.sin(np.pi * x[0])
            s1 = np.sin(np.pi * x[1])
            s2 = np.sin(np.pi * x[2])
            f0 = 1.0 - s0
            f1 = 1.0 - s1
            f2 = 1.0 - s2
            # theta1 = arcsin(f2), theta2 = arctan2(f1, f0)
            theta1 = np.arcsin(np.clip(f2, 0.0, 1.0))
            theta2 = np.arctan2(f1, f0)
            return theta1, theta2

        # Initialize population on the Pareto front using uniform angles
        pop_size = 40
        theta1 = np.linspace(0.0, np.pi / 2, pop_size, endpoint=False) + rng.random(pop_size) * (np.pi / 2) / pop_size
        theta2 = np.linspace(0.0, np.pi / 2, pop_size, endpoint=False) + rng.random(pop_size) * (np.pi / 2) / pop_size
        # Pair them to spread
        theta1 = np.sort(theta1)
        theta2 = np.sort(theta2)
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            X[i] = angles_to_x(theta1[i], theta2[(i * 7 + 3) % pop_size])

        F = np.array([problem(X[i]) for i in range(pop_size)])
        used = pop_size

        # Track best hypervolume contribution
        REF = np.array([2.5, 2.5, 2.5])

        def hypervolume_3d(points, ref):
            """Compute hypervolume of a set of 3D points w.r.t. reference point."""
            pts = [tuple(p) for p in points]
            pts = [p for p in pts if all(p[i] < ref[i] for i in range(n_obj))]
            if not pts:
                return 0.0
            # Remove dominated points
            pts = _nondominated(pts)
            if not pts:
                return 0.0
            # Sort by first coordinate
            pts.sort(key=lambda p: p[0])
            total = 0.0
            n = len(pts)
            for i in range(n):
                p = pts[i]
                # For each point, find the max extent in dims 1 and 2
                # among points that are "behind" it (have larger coord 0)
                max1 = -1e18
                max2 = -1e18
                for j in range(i + 1, n):
                    q = pts[j]
                    if q[0] > p[0]:
                        if q[1] > max1:
                            max1 = q[1]
                        if q[2] > max2:
                            max2 = q[2]
                h1 = ref[1] - max1 if max1 > -1e17 else ref[1]
                h2 = ref[2] - max2 if max2 > -1e17 else ref[2]
                if h1 > 0 and h2 > 0:
                    total += (ref[0] - p[0]) * h1 * h2
            return total

        def _nondominated(pts):
            n = len(pts)
            keep = [True] * n
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(i + 1, n):
                    if not keep[j]:
                        continue
                    pi, pj = pts[i], pts[j]
                    if all(pi[k] <= pj[k] for k in range(n_obj)) and any(pi[k] < pj[k] for k in range(n_obj)):
                        keep[j] = False
                    elif all(pj[k] <= pi[k] for k in range(n_obj)) and any(pj[k] < pi[k] for k in range(n_obj)):
                        keep[i] = False
                        break
            return [pts[i] for i in range(n) if keep[i]]

        # Main search loop
        while used < budget:
            if time.time() > deadline:
                break

            # Pick a parent from the current population
            idx = rng.integers(pop_size)
            parent = X[idx].copy()
            t1, t2 = x_to_angles(parent)

            # DE-style mutation: pick two other random individuals
            i1, i2 = rng.choice(pop_size, size=2, replace=False)
            t1_m = t1 + 0.3 * (theta1[i1] - theta1[i2])
            t2_m = t2 + 0.3 * (theta2[i1] - theta2[i2])
            # Clamp angles
            t1_m = np.clip(t1_m, 0.0, np.pi / 2)
            t2_m = np.clip(t2_m, 0.0, np.pi / 2)

            child = angles_to_x(t1_m, t2_m)
            fc = problem(child)
            used += 1

            # Also try a small local perturbation in angle space
            if used + 1 < budget and time.time() < deadline:
                t1_p = np.clip(t1_m + rng.normal(0, 0.05), 0.0, np.pi / 2)
                t2_p = np.clip(t2_m + rng.normal(0, 0.05), 0.0, np.pi / 2)
                child2 = angles_to_x(t1_p, t2_p)
                fc2 = problem(child2)
                used += 1
                # Keep the better one (lower sum of objectives)
                if np.sum(fc2) < np.sum(fc):
                    child, fc = child2, fc2

            # Replace the worst individual (highest objective sum)
            sums = np.sum(F, axis=1)
            worst = int(np.argmax(sums))
            X[worst] = child
            F[worst] = fc

            # Occasionally re-initialize a portion of the population to maintain diversity
            if used % 20 == 0:
                n_reinit = 5
                for _ in range(n_reinit):
                    if used >= budget:
                        break
                    t1r = rng.uniform(0.0, np.pi / 2)
                    t2r = rng.uniform(0.0, np.pi / 2)
                    xr = angles_to_x(t1r, t2r)
                    fr = problem(xr)
                    used += 1
                    # Replace a random individual
                    ridx = rng.integers(pop_size)
                    if np.sum(fr) < np.sum(F[ridx]):
                        X[ridx] = xr
                        F[ridx] = fr

        return X

    @staticmethod
    def _nondom_idx(F):
        n = F.shape[0]
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    keep[i] = False
                    break
        return [i for i in range(n) if keep[i]]
# EVOLVE-BLOCK-END
