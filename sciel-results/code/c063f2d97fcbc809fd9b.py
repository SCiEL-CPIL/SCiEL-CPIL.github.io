"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Multi-restart Nelder-Mead with coarse grid initialization for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, safe_call, x0, max_iter, initial_step=0.5):
        """Nelder-Mead simplex optimization."""
        n = self.dim
        # Build initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            pert = x0.copy()
            # Alternate direction based on coordinate sign to explore both sides
            step = initial_step if x0[i] >= 0 else -initial_step
            pert[i] += step
            simplex.append(np.clip(pert, -5, 5))

        # Evaluate initial simplex
        fvals = []
        for i in range(n + 1):
            f = safe_call(simplex[i])
            if f is None:
                f = np.inf
            fvals.append(f)

        if any(f is None or f == np.inf for f in fvals):
            # If any point failed, just return the best finite one
            finite = [(f, i) for i, f in enumerate(fvals) if f < np.inf]
            if finite:
                finite.sort()
                return simplex[finite[0][1]].copy(), finite[0][0]
            return None, np.inf

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        best_f = min(fvals)
        best_x = simplex[np.argmin(fvals)].copy()

        for _ in range(max_iter):
            if self._calls_made >= self.budget:
                break
            if time.time() > self._deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                fr = np.inf

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                if fe is None:
                    fe = np.inf
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
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    fc = np.inf

                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f = safe_call(simplex[i])
                        if f is None:
                            f = np.inf
                        fvals[i] = f
                        if self._calls_made >= self.budget:
                            break

            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

            # Check convergence
            f_range = max(fvals) - min(fvals)
            if f_range < 1e-12:
                break

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        self._deadline = start_time + 290.0  # Safety margin before 330s kill
        self._calls_made = 0
        self._budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal best_x, best_f
            if self._calls_made >= self._budget:
                return None
            if time.time() > self._deadline:
                return None
            self._calls_made += 1
            try:
                val = func(x)
                val = float(val)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse Grid Search to find promising basins
        # The minimum is in [-4, 4]^5.
        # Use a 3-point grid per dimension: -4, 0, 4.
        # Total points: 3^5 = 243. This fits within the 500 budget.
        
        if self.dim == 5:
            grid_coords = [-4.0, 0.0, 4.0]
            grid_points = list(itertools.product(grid_coords, repeat=self.dim))
        else:
            # Fallback for other dimensions
            grid_coords = np.linspace(-4.0, 4.0, 3)
            grid_points = []
            for i in range(3 ** self.dim):
                p = []
                for d in range(self.dim):
                    p.append(grid_coords[(i // (3 ** (self.dim - 1 - d))) % 3])
                grid_points.append(p)

        # Evaluate grid points
        for p in grid_points:
            if time.time() > self._deadline or self._calls_made >= self._budget:
                break
            x = np.array(p, dtype=float)
            f = safe_call(x)

        # Phase 2: Local Refinement with Nelder-Mead from the best grid point
        if best_x is not None and self._calls_made < self._budget:
            remaining = self._budget - self._calls_made
            # Allocate most of the remaining budget to Nelder-Mead
            max_iter = min(remaining - 2, 200)
            # Initial step size: Since we are in a coarse grid, the valley might be nearby.
            # A step of 1.0 is reasonable.
            bx, bf = self._nelder_mead(safe_call, best_x, max_iter, initial_step=1.0)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: If budget remains, try a few random restarts or finer local search
        if self._calls_made < self._budget and time.time() < self._deadline:
            # Try a few random restarts to see if we missed a better basin
            for attempt in range(5):
                if self._calls_made >= self._budget or time.time() > self._deadline:
                    break
                x = rng.uniform(-4, 4, self.dim)
                f = safe_call(x)
                if f is None:
                    continue
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    # If we found a new best, run a short NM
                    if self._calls_made < self._budget - 10:
                        remaining = self._budget - self._calls_made
                        max_iter = min(remaining - 2, 50)
                        bx, bf = self._nelder_mead(safe_call, best_x, max_iter, initial_step=0.5)
                        if bf < best_f:
                            best_f = bf
                            best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
