"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() > deadline

        def try_eval(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start Nelder-Mead
        # Start with 10 diverse initial points
        n_starts = 10
        simplex_size = 1.5
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_shrink = 0.5

        # Generate diverse starting points
        start_points = []
        # Some in corners, some random
        for i in range(n_starts):
            if i < 4:
                # Corner-ish points
                x = rng.choice([-4, -2, 0, 2, 4], size=dim)
                x = x + rng.uniform(-0.5, 0.5, dim)
            elif i < 8:
                # Random in box
                x = rng.uniform(-4.5, 4.5, dim)
            else:
                # Near origin and far
                if i == 8:
                    x = rng.uniform(-1, 1, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
            start_points.append(np.clip(x, -5, 5))

        # Each simplex needs dim+1 points
        # We'll manage a pool of active simplexes
        # Each simplex: list of (x, f) pairs

        class Simplex:
            def __init__(self, x0):
                self.points = []  # list of (x, f)
                self.x0 = x0.copy()
                self.f0 = None
                self.total_evals = 0
                self.best_f = np.inf
                self.last_improvement_eval = 0
                self.stagnant = False
                self.converged = False

            def add_point(self, x, f):
                self.points.append((x.copy(), f))
                self.total_evals += 1
                if f < self.best_f:
                    self.best_f = f
                    self.last_improvement_eval = self.total_evals

            def is_full(self):
                return len(self.points) == self.dim + 1

        simplexes = []
        for sp in start_points:
            sx = Simplex(sp)
            simplexes.append(sx)

        # Initialize each simplex with dim+1 points
        for sx in simplexes:
            if check_time():
                break
            # First point
            f0 = try_eval(sx.x0)
            sx.add_point(sx.x0, f0)
            sx.f0 = f0

            # Additional dim points
            for d in range(dim):
                if check_time():
                    break
                x_new = sx.x0.copy()
                # Perturb in direction d
                step = simplex_size
                if x_new[d] + step > 5:
                    step = -step
                x_new[d] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = try_eval(x_new)
                sx.add_point(x_new, f_new)

            if check_time():
                break

        # Main loop: Nelder-Mead iterations on active simplexes
        # Allocate budget dynamically: give more to promising simplexes
        while not check_time() and eval_count < budget:
            # Sort simplexes by best_f (best first)
            active_simplexes = [sx for sx in simplexes if not sx.converged]
            if not active_simplexes:
                break

            # Mark stagnant simplexes
            for sx in active_simplexes:
                if sx.total_evals - sx.last_improvement_eval > 3 * (dim + 1):
                    sx.stagnant = True
                else:
                    sx.stagnant = False

            # Weight: non-stagnant get more weight
            weights = []
            for sx in active_simplexes:
                if sx.stagnant:
                    w = 0.1
                else:
                    w = 1.0 / (1 + sx.best_f / (abs(sx.best_f) + 1))
                weights.append(w)
            total_w = sum(weights)
            if total_w == 0:
                weights = [1.0] * len(active_simplexes)
                total_w = len(active_simplexes)

            # Pick next simplex based on weights (deterministic round-robin with priority)
            # Use a simple approach: iterate through sorted by weight, do one NM step each
            # But to be efficient, do one step on the best non-stagnant simplex

            # Sort by (stagnant, best_f) so best non-stagnant comes first
            active_simplexes.sort(key=lambda s: (s.stagnant, s.best_f))

            # Do one Nelder-Mead step on the first non-stagnant simplex
            target = None
            for sx in active_simplexes:
                if not sx.stagnant:
                    target = sx
                    break
            if target is None:
                target = active_simplexes[0]

            # Check if simplex is full
            if not target.is_full():
                # Fill it
                for d in range(dim - len(target.points) + 1):
                    if check_time() or eval_count >= budget:
                        break
                    x_new = target.x0.copy()
                    step = simplex_size
                    if x_new[d % dim] + step > 5:
                        step = -step
                    x_new[d % dim] += step
                    x_new = np.clip(x_new, -5, 5)
                    f_new = try_eval(x_new)
                    target.add_point(x_new, f_new)
                continue

            # Nelder-Mead step
            pts = target.points
            # Sort by f
            pts.sort(key=lambda p: p[1])
            target.points = pts

            worst = pts[-1]
            second_worst = pts[-2]
            centroid = np.mean([p[0] for p in pts[:-1]], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - worst[0])
            x_r = np.clip(x_r, -5, 5)
            f_r = try_eval(x_r)
            target.add_point(x_r, f_r)

            if f_r < pts[0][1]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst[0])
                x_e = np.clip(x_e, -5, 5)
                f_e = try_eval(x_e)
                target.add_point(x_e, f_e)
                if f_e < f_r:
                    # Replace worst with expansion
                    target.points = target.points[:-1]
                    target.points.append((x_e, f_e))
                else:
                    # Replace worst with reflection
                    target.points = target.points[:-1]
                    target.points.append((x_r, f_r))
            elif f_r < second_worst[1]:
                # Replace worst with reflection
                target.points = target.points[:-1]
                target.points.append((x_r, f_r))
            else:
                # Contraction
                if f_r < worst[1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst[0] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = try_eval(x_c)
                target.add_point(x_c, f_c)

                if f_c < min(f_r, worst[1]):
                    # Replace worst with contraction
                    target.points = target.points[:-1]
                    target.points.append((x_c, f_c))
                else:
                    # Shrink
                    new_pts = []
                    for i, (x, f) in enumerate(pts):
                        if i == 0:
                            new_pts.append((x, f))
                        else:
                            x_s = pts[0][0] + sigma_shrink * (x - pts[0][0])
                            x_s = np.clip(x_s, -5, 5)
                            f_s = try_eval(x_s)
                            target.add_point(x_s, f_s)
                            new_pts.append((x_s, f_s))
                    target.points = new_pts

            # Check convergence
            if len(target.points) >= 2:
                f_vals = [p[1] for p in target.points]
                f_range = max(f_vals) - min(f_vals)
                if f_range < 1e-6:
                    target.converged = True

            # Remove worst point to keep simplex size = dim+1
            if len(target.points) > dim + 1:
                target.points.sort(key=lambda p: p[1])
                target.points = target.points[:dim + 1]

        # Phase 2: Fine-tuning with local search around global best
        if not check_time() and best_x is not None and eval_count < budget:
            # Do a few local steps around best
            local_sigma = 0.5
            for _ in range(min(50, budget - eval_count)):
                if check_time():
                    break
                x_trial = best_x + local_sigma * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = try_eval(x_trial)
                if f_trial < best_f:
                    local_sigma = max(0.1, local_sigma * 0.9)
                else:
                    local_sigma = min(2.0, local_sigma * 1.2)

        return best_x, best_f
# EVOLVE-BLOCK-END
