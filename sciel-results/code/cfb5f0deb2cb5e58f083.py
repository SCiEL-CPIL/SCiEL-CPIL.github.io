"""Black-box continuous optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic global-scan + Nelder-Mead + coordinate-descent local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lo = np.full(dim, -5.0)
        self.hi = np.full(dim, 5.0)

    def _clip(self, x):
        return np.clip(x, self.lo, self.hi)

    def _nm(self, func, x0, f0, maxit, tol, deadline, state):
        """Nelder-Mead with adaptive step size and best tracking."""
        d = self.dim
        simplex = np.empty((d + 1, d))
        simplex[0] = x0
        for i in range(d):
            step = 0.25
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += step
            simplex[i + 1] = self._clip(simplex[i + 1])
        fvals = np.empty(d + 1)
        fvals[0] = f0
        for i in range(1, d + 1):
            fvals[i] = func(simplex[i])
            state["calls"] += 1
        alpha, gamma, rho = 1.0, 2.0, 0.5
        for _ in range(maxit):
            if time.time() > deadline:
                break
            if state["calls"] >= self.budget:
                break
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            if fvals[0] < state["best_f"] - 1e-15:
                state["best_f"] = fvals[0]
                state["best_x"] = simplex[0].copy()
            if (fvals[-1] - fvals[0]) < tol * (1.0 + abs(fvals[0])):
                break
            centroid = simplex[:-1].mean(axis=0)
            xr = self._clip(centroid + alpha * (centroid - simplex[-1]))
            fr = func(xr)
            state["calls"] += 1
            if fr < fvals[0]:
                xe = self._clip(centroid + gamma * (xr - centroid))
                fe = func(xe)
                state["calls"] += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = self._clip(centroid + rho * (simplex[-1] - centroid))
                fc = func(xc)
                state["calls"] += 1
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, d + 1):
                        simplex[i] = self._clip(0.5 * (simplex[i] + simplex[0]))
                        fvals[i] = func(simplex[i])
                        state["calls"] += 1
                        if state["calls"] >= self.budget:
                            return
        if fvals[0] < state["best_f"]:
            state["best_f"] = fvals[0]
            state["best_x"] = simplex[0].copy()

    def _cd(self, func, x, f, maxit, deadline, state):
        """Coordinate descent with line search; adaptive step."""
        d = self.dim
        step = 0.5
        for _ in range(maxit):
            if time.time() > deadline:
                break
            if state["calls"] >= self.budget:
                break
            improved = False
            for i in range(d):
                if state["calls"] >= self.budget or time.time() > deadline:
                    break
                best_xi, best_fi = x[i], f
                s = step
                for _ in range(12):
                    for sign in (1, -1):
                        x[i] = self._clip(x[i] + sign * s)
                        f_try = func(x)
                        state["calls"] += 1
                        if f_try < best_fi:
                            best_fi, best_xi = f_try, x[i]
                        if state["calls"] >= self.budget:
                            break
                    if state["calls"] >= self.budget:
                        break
                    if best_xi != x[i]:
                        x[i] = best_xi
                    s *= 0.5
                    if s < 1e-6:
                        break
                if best_fi < f:
                    x[i] = best_xi
                    f = best_fi
                    improved = True
            if improved:
                if f < state["best_f"]:
                    state["best_f"] = f
                    state["best_x"] = x.copy()
                step = min(step * 1.2, 2.0)
            else:
                step *= 0.5
                if step < 1e-7:
                    break

    def __call__(self, func):
        start = time.time()
        deadline = start + 300.0
        state = {"calls": 0, "best_f": np.inf, "best_x": None}

        def wrapped(x):
            x = self._clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            state["calls"] += 1
            if fx < state["best_f"]:
                state["best_f"] = fx
                state["best_x"] = x.copy()
            return fx

        d = self.dim
        # 1) Coarse global scan (deterministic grid) to find a good basin.
        n = 4
        grid = np.linspace(self.lo, self.hi, n)
        for i in range(n):
            for j in range(n):
                if state["calls"] >= self.budget or time.time() > deadline:
                    break
                x0 = np.zeros(d)
                x0[0] = grid[i]
                x0[1] = grid[j]
                wrapped(x0)
        if state["best_x"] is None:
            x0 = np.zeros(d)
            wrapped(x0)

        # 2) Local polish from the best scan point.
        self._nm(func, state["best_x"], state["best_f"], maxit=120,
                 tol=1e-12, deadline=deadline, state=state)
        self._cd(func, state["best_x"].copy(), state["best_f"], maxit=60,
                 deadline=deadline, state=state)
        self._nm(func, state["best_x"], state["best_f"], maxit=200,
                 tol=1e-14, deadline=deadline, state=state)
        self._cd(func, state["best_x"].copy(), state["best_f"], maxit=80,
                 deadline=deadline, state=state)

        # 3) Use any remaining budget for extra refinement passes.
        while state["calls"] < self.budget and time.time() < deadline:
            self._nm(func, state["best_x"], state["best_f"], maxit=50,
                     tol=1e-14, deadline=deadline, state=state)
            self._cd(func, state["best_x"].copy(), state["best_f"], maxit=30,
                     deadline=deadline, state=state)

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
