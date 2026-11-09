"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Custom Nelder-Mead with adaptive simplex for anisotropic functions.
    Combines multi-start initialization with a tightly-controlled simplex method
    that adapts its shape to the local landscape, exploiting the narrow valley
    structure of the Bent Cigar function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        try:
            # Multi-start initialization
            rng = np.random.default_rng(42)
            n_starts = 8
            starts = []
            for i in range(n_starts):
                if i == 0:
                    x0 = np.zeros(dim)
                elif i == 1:
                    x0 = rng.uniform(-4.5, 4.5, size=dim)
                else:
                    x0 = rng.uniform(-4.5, 4.5, size=dim)
                starts.append(x0)

            # Evaluate starts
            f_starts = []
            for x0 in starts:
                try:
                    fx = wrapped_func(x0)
                    f_starts.append(fx)
                except (TimeoutError, ValueError):
                    break

            if not f_starts:
                return np.zeros(dim), float('inf')

            # Sort starts by function value
            order = np.argsort(f_starts)

            # Run Nelder-Mead from the best starts
            # Parameters for Nelder-Mead
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            # We'll run Nelder-Mead on the best 3 starts, splitting remaining budget
            n_refine = min(3, len(order))
            remaining_budget = budget - len(f_starts)
            if remaining_budget <= 0:
                return self.best_x, self.best_f

            budget_per_start = remaining_budget // n_refine

            for idx in range(n_refine):
                if time.time() > deadline or self.calls >= budget:
                    break

                start_idx = order[idx]
                x0 = starts[start_idx].copy()

                # Initialize simplex
                simplex = [x0.copy()]
                fvals = [f_starts[start_idx]]

                # Create initial simplex around x0
                for i in range(dim):
                    xi = x0.copy()
                    # Adaptive initial step: larger for first few dimensions
                    step = 1.0 if i == 0 else 0.5
                    xi[i] += step
                    simplex.append(xi)
                    try:
                        fvals.append(wrapped_func(xi))
                    except (TimeoutError, ValueError):
                        break

                if len(simplex) < dim + 1:
                    continue

                # Nelder-Mead iterations
                max_iter = min(budget_per_start, 200)
                for _ in range(max_iter):
                    if time.time() > deadline or self.calls >= budget:
                        break

                    # Sort simplex by function value
                    idx_sorted = np.argsort(fvals)
                    simplex = [simplex[i] for i in idx_sorted]
                    fvals = [fvals[i] for i in idx_sorted]

                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    try:
                        fr = wrapped_func(xr)
                    except (TimeoutError, ValueError):
                        break

                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        try:
                            fe = wrapped_func(xe)
                        except (TimeoutError, ValueError):
                            break

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
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        try:
                            fc = wrapped_func(xc)
                        except (TimeoutError, ValueError):
                            break

                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                try:
                                    fvals[i] = wrapped_func(simplex[i])
                                except (TimeoutError, ValueError):
                                    break
                            if self.calls >= budget or time.time() > deadline:
                                break

                    # Check for convergence
                    if len(fvals) > 0:
                        fspread = max(fvals) - min(fvals)
                        if fspread < 1e-12:
                            break

            return self.best_x, self.best_f

        except Exception:
            if self.best_x is None:
                return np.zeros(dim), float('inf')
            return self.best_x, self.best_f
# EVOLVE-BLOCK-END
