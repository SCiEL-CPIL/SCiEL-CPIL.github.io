"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = np.inf
        calls = 0

        # Generate diverse starting points within [-4, 4]^5
        # Use a mix of deterministic and random points
        rng = np.random.default_rng(42)

        # Deterministic grid-like points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Axis-aligned points
        for d in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[d] = sign * 3.0
                starts.append(pt)
        # Corner-like points
        for sign1 in [1, -1]:
            for sign2 in [1, -1]:
                pt = np.zeros(dim)
                pt[0] = sign1 * 3.5
                pt[1] = sign2 * 3.5
                starts.append(pt)
        # Random points
        for _ in range(20):
            starts.append(rng.uniform(-4, 4, dim))

        # Shuffle starts to interleave deterministic and random
        rng.shuffle(starts)

        # Multi-start Nelder-Mead
        n_starts = min(10, len(starts))
        calls_per_start = max(1, (budget - 20) // n_starts)

        for start_idx in range(n_starts):
            if calls >= budget or time_left() < 10:
                break

            x0 = np.clip(starts[start_idx], -5, 5)
            fx0 = func(x0)
            calls += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [fx0]

            # Add simplex vertices with adaptive step size
            step = 0.5
            for d in range(dim):
                if calls >= budget or time_left() < 5:
                    break
                pt = x0.copy()
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                # Check if point is too close to existing simplex vertices
                too_close = False
                for existing in simplex:
                    if np.linalg.norm(pt - existing) < 1e-8:
                        too_close = True
                        break
                if too_close:
                    pt[d] -= 2 * step
                    pt = np.clip(pt, -5, 5)
                fv = func(pt)
                calls += 1
                simplex.append(pt)
                fvals.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = pt.copy()

            if len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations for this start
            for it in range(calls_per_start):
                if calls >= budget or time_left() < 5:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time_left() < 3:
                    break
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time_left() < 3:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                        continue
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget or time_left() < 3:
                        break
                    fc = func(xc)
                    calls += 1
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if calls >= budget or time_left() < 2:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
