import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300  # safety margin

        call_count = 0
        def call_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        def time_ok():
            return time.time() < deadline and call_count < budget

        # Halton sequence
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        def halton_sequence(n, dim):
            pts = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    pts[i - 1, d] = halton(i, d + 2)
            return pts

        def to_box(u):
            return -5.0 + 10.0 * u

        # Phase 1: Halton exploration
        n_halton = min(120, budget - 30)
        halton_pts = to_box(halton_sequence(n_halton, dim))

        best_x = None
        best_f = np.inf
        all_points = []

        for i in range(n_halton):
            if not time_ok():
                break
            x = halton_pts[i]
            fx = call_func(x)
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-start Nelder-Mead
        all_points.sort(key=lambda t: t[0])
        top_k = min(6, len(all_points))
        starts = []
        for fx, x in all_points[:top_k]:
            diverse = True
            for sx in starts:
                if np.linalg.norm(x - sx) < 0.3:
                    diverse = False
                    break
            if diverse:
                starts.append(x)

        def nelder_mead(x0, max_iter):
            nonlocal best_x, best_f
            simplex = [x0.copy() + 0.05 * np.ones(dim)]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.05
                simplex.append(xi)

            fvals = [call_func(s) for s in simplex]
            local_best_x = simplex[np.argmin(fvals)]
            local_best_f = min(fvals)

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for it in range(max_iter):
                if not time_ok():
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)

                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
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
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = call_func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = call_func(simplex[i])
                    else:
                        xc = centroid - rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = call_func(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = call_func(simplex[i])

                if np.max(fvals) - np.min(fvals) < 1e-12:
                    break

            return local_best_x, local_best_f

        remaining = budget - call_count
        nm_per_start = max(10, remaining // max(1, len(starts)))

        for sx in starts:
            if not time_ok():
                break
            nelder_mead(sx, nm_per_start)

        # Phase 3: Polish
        if time_ok():
            remaining2 = budget - call_count
            if remaining2 > 10:
                nelder_mead(best_x, remaining2)

        return best_x, best_f
