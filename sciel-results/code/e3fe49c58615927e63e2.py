"""Multi-Start Nelder-Mead with time guard for BBOB f11."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(x)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Grid-based global exploration
        # Use a coarse grid to find promising basins
        grid_n = 4  # 4 points per dimension -> 4^5 = 1024 points, too many
        # Use fewer: 3 per dim -> 3^5 = 243 points
        grid_n = 3
        grid_points = []
        for i in range(grid_n):
            for j in range(grid_n):
                for k in range(grid_n):
                    for l in range(grid_n):
                        for m in range(grid_n):
                            x = np.array([
                                -4 + 8 * i / (grid_n - 1),
                                -4 + 8 * j / (grid_n - 1),
                                -4 + 8 * k / (grid_n - 1),
                                -4 + 8 * l / (grid_n - 1),
                                -4 + 8 * m / (grid_n - 1)
                            ])
                            grid_points.append(x)

        # Shuffle and evaluate until budget or time runs out
        rng.shuffle(grid_points)
        for x in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            evaluate(x)

        # Phase 2: Multi-start Nelder-Mead from top candidates
        # Sort grid points by function value
        if best_x is not None:
            # Evaluate all grid points that we haven't yet
            # Actually we already evaluated them in order; let's collect evaluated points
            # We need to track which points were evaluated and their values
            pass

        # Let's redo: collect evaluated points
        # Reset and do it properly
        calls_made = 0
        best_x = None
        best_f = np.inf
        evaluated = []  # list of (x, f)

        def evaluate2(x):
            nonlocal calls_made, best_x, best_f
            x = clip(x)
            f = func(x)
            calls_made += 1
            evaluated.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Evaluate grid
        for x in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break
            evaluate2(x)

        # Sort by function value, take top K for local refinement
        evaluated.sort(key=lambda t: t[1])
        top_k = min(10, len(evaluated))
        starts = [x for x, f in evaluated[:top_k]]

        # Phase 3: Nelder-Mead from each start
        for start_x in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break

            # Nelder-Mead
            x = start_x.copy()
            # Initialize simplex
            simplex = [x.copy()]
            for i in range(dim):
                dx = x.copy()
                # Step size: small perturbation
                step = 0.1
                dx[i] += step
                simplex.append(clip(dx))

            # Evaluate simplex vertices
            simplex_f = []
            for sv in simplex:
                if calls_made >= budget:
                    break
                if time.time() > deadline - 3:
                    break
                f = evaluate2(sv)
                simplex_f.append(f)

            if len(simplex_f) < dim + 1:
                continue

            # Nelder-Mead iterations
            max_iter = 50
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 3:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = clip(centroid + alpha * (centroid - simplex[-1]))
                fr = evaluate2(xr) if calls_made < budget and time.time() < deadline - 3 else None

                if fr is None:
                    break

                if fr < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    xe = clip(centroid + gamma * (centroid - simplex[-1]))
                    fe = evaluate2(xe) if calls_made < budget and time.time() < deadline - 3 else None
                    if fe is not None and fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    # Outside contraction if fr < worst, else inside
                    if fr < simplex_f[-1]:
                        xc = clip(centroid + rho * (xr - centroid))
                    else:
                        xc = clip(centroid - rho * (centroid - simplex[-1]))
                    fc = evaluate2(xc) if calls_made < budget and time.time() < deadline - 3 else None
                    if fc is None:
                        break
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            if calls_made >= budget or time.time() > deadline - 3:
                                break
                            simplex_f[i] = evaluate2(simplex[i])
                        if calls_made >= budget or time.time() > deadline - 3:
                            break

            # Check convergence
            if len(simplex_f) >= 2:
                spread = np.max(simplex_f) - np.min(simplex_f)
                if spread < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
