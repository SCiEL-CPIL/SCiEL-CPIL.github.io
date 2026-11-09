"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a fast (1+1)-ES with adaptive step size, multi-start
    exploration, and a Nelder-Mead polish phase. Designed for Rastrigin's
    many local minima: the (1+1)-ES quickly finds a good basin, then NM refines.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0

        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls = 0

        # Phase 1: Multi-start (1+1)-ES
        # Use 3 starts with different initial points
        # Allocate budget: 40% for first start, 30% for second, 30% for third
        starts = [
            (np.zeros(dim), int(budget * 0.40)),
            (np.array([2.0, -2.0, 3.0, -3.0, 1.5]), int(budget * 0.30)),
            (np.array([-2.5, 2.5, -1.5, 1.5, -2.0]), int(budget * 0.30)),
        ]

        for start_x, start_budget in starts:
            if calls >= budget or time.time() > deadline - 5:
                break

            # Try the starting point
            x = np.clip(start_x.copy(), -5, 5)
            fx = func(x)
            calls += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # (1+1)-ES from this start
            sigma = 2.0  # initial step size
            success_count = 0
            window = 8  # adapt every 8 steps
            step_count = 0
            max_steps = start_budget

            for _ in range(max_steps):
                if calls >= budget or time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                step_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = x.copy()

                # Adapt sigma every 'window' steps
                if step_count % window == 0:
                    success_rate = success_count / window
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                    success_count = 0

                sigma = np.clip(sigma, 0.005, 10.0)

        # Phase 2: Nelder-Mead local polish from best point
        if best_x is not None and calls < budget - 5 and time.time() < deadline - 5:
            x_nm = best_x.copy()

            # Build initial simplex
            simplex = [x_nm.copy()]
            fvals = [best_fx]
            for i in range(dim):
                pert = x_nm.copy()
                delta = 0.1 if x_nm[i] < 4.9 else -0.1
                pert[i] += delta
                pert = np.clip(pert, -5, 5)
                simplex.append(pert)
                fvals.append(func(pert))
                calls += 1

            # Nelder-Mead iterations
            max_nm_iters = min(80, budget - calls)
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
