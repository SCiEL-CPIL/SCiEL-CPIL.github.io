"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex with budget management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, max_iter, calls_ref, deadline, rng):
        """Run Nelder-Mead from x0, updating calls_ref[0] with count used."""
        dim = self.dim
        calls = 0

        # Initial simplex
        simplex = []
        f0 = func(x0)
        calls += 1
        simplex.append((x0.copy(), f0))

        # Create simplex points
        base = x0.copy()
        for i in range(dim):
            point = base.copy()
            point[i] = point[i] + 0.5 if point[i] == 0 else point[i] * 1.05
            point = np.clip(point, -5, 5)
            f = func(point)
            calls += 1
            simplex.append((point, f))

        # Sort by function value
        simplex.sort(key=lambda p: p[1])

        best_x = simplex[0][0].copy()
        best_f = simplex[0][1]

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        for _ in range(max_iter):
            if calls_ref[0] + calls >= self.budget:
                break
            if time.time() > deadline - 2.0:
                break

            # Sort simplex
            simplex.sort(key=lambda p: p[1])

            # Check convergence
            fvals = np.array([p[1] for p in simplex])
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

            x_worst = simplex[-1][0]
            x_best = simplex[0][0]

            # Centroid of all but worst
            centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < simplex[0][1]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = (x_e, f_e)
                else:
                    simplex[-1] = (x_r, f_r)
            elif f_r < simplex[-2][1]:
                simplex[-1] = (x_r, f_r)
            else:
                # Contraction
                if f_r < simplex[-1][1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1

                if f_c <= f_r:
                    simplex[-1] = (x_c, f_c)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i][0] = x_best + sigma * (simplex[i][0] - x_best)
                        simplex[i][0] = np.clip(simplex[i][0], -5, 5)
                        simplex[i][1] = func(simplex[i][0])
                        calls += 1
                    if calls_ref[0] + calls >= self.budget:
                        break

            # Update best
            simplex.sort(key=lambda p: p[1])
            if simplex[0][1] < best_f:
                best_f = simplex[0][1]
                best_x = simplex[0][0].copy()

        calls_ref[0] += calls
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        calls_ref = [0]

        # Initial evaluation at origin
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_ref[0] = 1

        # Multi-start Nelder-Mead
        # Generate diverse starting points
        num_starts = 15
        start_points = [np.zeros(dim)]  # Already evaluated

        # Add random points
        for _ in range(num_starts - 1):
            p = rng.uniform(-4, 4, size=dim)
            start_points.append(p)

        # Also add axis-aligned perturbations from origin
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 2.0
                start_points.append(p)

        # Limit total starts based on budget
        # Each NM run uses roughly (dim+2) initial + ~dim*max_iter evaluations
        # Budget is 500, dim=5, so ~7 initial per run
        # Let's do at most ~30-40 NM runs
        max_runs = min(len(start_points), 40)

        for i in range(max_runs):
            if calls_ref[0] >= budget:
                break
            if time.time() > deadline - 2.0:
                break

            x0 = start_points[i].copy()

            # If this is the origin, we already evaluated it
            if i == 0:
                # Already have f for origin, but NM needs the full simplex
                # So we'll just run NM and it will re-evaluate (that's fine, budget allows)
                pass

            # Determine max_iter for this run based on remaining budget
            remaining = budget - calls_ref[0]
            # Each NM iteration uses ~1-3 function calls
            # Initial simplex uses dim+1 calls
            # Be conservative
            max_iter = max(5, min(50, (remaining - dim - 1) // 2))

            if max_iter < 5:
                break

            bx, bf = self._nelder_mead(func, x0, max_iter, calls_ref, deadline, rng)

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
