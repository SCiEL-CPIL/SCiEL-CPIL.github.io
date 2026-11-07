"""Model-assisted optimizer for BBOB-style Katsuura function f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Katsuura-structure-informed sampling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        def nelder_mead(x0, max_iterations=200):
            nonlocal calls_made, best_f, best_x
            n = dim
            # Build initial simplex with adaptive steps based on Katsuura period
            simplex = [np.array(x0, dtype=float, copy=True)]
            for i in range(n):
                xi = np.array(x0, dtype=float, copy=True)
                # Katsuura has features on scale ~1/12 in transformed space,
                # but in original space with shift/rotation, use moderate steps
                step = 0.25
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    return

            if len(fvals) != n + 1:
                return

            alpha, gamma, rho = 1.0, 2.0, -0.5

            for _ in range(max_iterations):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if max(fvals) - min(fvals) < 1e-14:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                try:
                    fr = counted_func(xr)
                except RuntimeError:
                    return

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    try:
                        fe = counted_func(xe)
                    except RuntimeError:
                        return
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
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    try:
                        fc = counted_func(xc)
                    except RuntimeError:
                        return
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink: move all points toward best
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            try:
                                fvals[i] = counted_func(simplex[i])
                            except RuntimeError:
                                return

        # Phase 1: Generate structurally-informed starting points
        # Katsuura function: f(x) = (10/n) * sum_{i=1}^n [1 - sqrt(1 - cos(24*pi*x_i - 12*pi)) * cos(24*pi*x_i - 12*pi))]^2
        #                    + (10/n) * sum_{i=1}^{n-1} (x_i - 1)^2
        # The minima in the canonical (untransformed) space are near x_i = 1.
        # With hidden shift s and rotation R: f(x) = g(R*(x - s)) + c
        # So minima are near x = R^{-1} * 1_vec + s

        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Points near canonical minimum locations (before transformation)
        # Since we don't know R and s, sample around 0 and 1 in various directions
        for k in [-2, -1, 0, 1, 2]:
            pt = np.ones(dim) * (1.0 + k * 0.1)
            start_points.append(np.clip(pt, -5, 5))

        # Grid of points in [-4, 4]^5 with step 1.0 (coarse)
        # This gives us 9^5 = 59049 points, too many. Use a subset.
        # Instead, use a stratified random approach
        for _ in range(30):
            pt = rng.uniform(-4.0, 4.0, size=dim)
            start_points.append(pt)

        # Add points at the vertices of the box
        for d in range(dim):
            for val in [-4.0, 4.0]:
                pt = np.zeros(dim)
                pt[d] = val
                start_points.append(pt)

        # Deduplicate
        seen = set()
        unique_starts = []
        for p in start_points:
            key = tuple(np.round(p, 2))
            if key not in seen:
                seen.add(key)
                unique_starts.append(p)

        # Phase 2: Evaluate all starting points
        for x in unique_starts:
            if calls_made >= budget - 50 or time.time() > deadline - 20:
                break
            try:
                counted_func(x)
            except RuntimeError:
                break

        if best_x is None:
            # Fallback
            try:
                counted_func(np.zeros(dim))
            except RuntimeError:
                pass
            return best_x, best_f

        # Phase 3: Run Nelder-Mead from the best point
        nelder_mead(best_x, max_iterations=150)

        # Phase 4: If budget remains, try from other promising starts
        if calls_made < budget - 30 and time.time() < deadline - 20:
            # Find a few other good starting points by re-evaluating
            # We'll pick points that are far from the current best
            candidates = []
            for x in unique_starts[:20]:
                if np.linalg.norm(x - best_x) < 0.5:
                    continue
                try:
                    f = counted_func(x)
                    candidates.append((f, x))
                except RuntimeError:
                    break

            if candidates:
                candidates.sort(key=lambda t: t[0])
                for f, x in candidates[:2]:
                    if calls_made >= budget - 20 or time.time() > deadline - 15:
                        break
                    nelder_mead(x, max_iterations=100)

        # Phase 5: Final high-precision polish from the best point
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline - 10:
            # Use a smaller initial simplex for the polish
            n = dim
            simplex = [np.array(best_x, dtype=float, copy=True)]
            for i in range(n):
                xi = np.array(best_x, dtype=float, copy=True)
                step = 0.05
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    break

            if len(fvals) == n + 1:
                alpha, gamma, rho = 1.0, 2.0, -0.5
                for _ in range(50):
                    if calls_made >= budget - 3 or time.time() > deadline - 3:
                        break

                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if max(fvals) - min(fvals) < 1e-15:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)

                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    try:
                        fr = counted_func(xr)
                    except RuntimeError:
                        break

                    if fr < fvals[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        try:
                            fe = counted_func(xe)
                        except RuntimeError:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        if fr < fvals[-1]:
                            xc = centroid + 0.5 * (xr - centroid)
                        else:
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        try:
                            fc = counted_func(xc)
                        except RuntimeError:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                try:
                                    fvals[i] = counted_func(simplex[i])
                                except RuntimeError:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
