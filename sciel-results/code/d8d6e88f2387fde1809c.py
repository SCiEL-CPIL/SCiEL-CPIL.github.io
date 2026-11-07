"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start restarts for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def nelder_mead(x0, max_evals):
            """Run Nelder-Mead simplex method."""
            nonlocal best_x, best_f
            n = dim
            # Initialize simplex: x0 and n points offset along each axis
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                if xi[i] > 0:
                    xi[i] -= 0.5
                else:
                    xi[i] += 0.5
                simplex.append(np.clip(xi, -5, 5))

            fvals = np.array([func(simplex[i]) for i in range(n + 1)])

            # Track best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = simplex[idx_best].copy()

            eval_count = n + 1
            alpha, gamma, rho = 1.0, 2.0, 0.5  # reflection, expansion, contraction
            max_iter = max(1, (max_evals - n - 1) // (n + 1))

            for _ in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                if eval_count >= max_evals:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                eval_count += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    eval_count += 1
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
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = func(xc)
                        eval_count += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                eval_count += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = func(xc)
                        eval_count += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                eval_count += 1

                # Update global best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < best_f:
                    best_f = fvals[idx_best]
                    best_x = simplex[idx_best].copy()

            return eval_count

        # Multi-start strategy: try several initial points
        # First: center, then random points, then near boundaries
        starts = [np.zeros(dim)]

        # Add some structured starts
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))

        # Add a few more if budget allows
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, dim))

        # Evaluate starts and run Nelder-Mead from best ones
        start_fvals = []
        for s in starts:
            s = np.clip(s, -5, 5)
            fs = func(s)
            start_fvals.append(fs)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # Sort starts by function value
        order = np.argsort(start_fvals)

        # Allocate remaining budget among top starts
        used_evals = len(starts)
        remaining = budget - used_evals

        if remaining <= 0:
            return best_x, best_f

        # Run NM from the best 3 starts
        top_k = min(3, len(starts))
        for i in range(top_k):
            if time.time() > deadline - 10.0:
                break
            if remaining <= 0:
                break
            # Give more budget to better starts
            if i == 0:
                alloc = min(remaining, max(50, remaining // 2))
            elif i == 1:
                alloc = min(remaining, max(30, remaining // 3))
            else:
                alloc = remaining

            alloc = max(alloc, 2 * dim + 2)  # minimum for one NM iteration
            if alloc > remaining:
                alloc = remaining

            x0 = np.clip(starts[order[i]], -5, 5)
            used = nelder_mead(x0, alloc)
            remaining -= used

        return best_x, best_f
# EVOLVE-BLOCK-END
