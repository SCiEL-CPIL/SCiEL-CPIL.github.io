"""Optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
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
        deadline = start_time + 280  # safety margin before 300s limit

        # Phase 1: Multi-start Nelder-Mead with Rastrigin-aware local refinement
        # Rastrigin has minima at integer points; we use this structure

        best_x = None
        best_f = np.inf

        def try_candidate(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Generate diverse starting points
        # Include integer lattice points (Rastrigin minima are at integers)
        # and random points
        starts = []

        # Integer lattice points in [-4, 4]^5 (subset to keep manageable)
        # Too many for full grid, so use a structured subset
        for _ in range(20):
            starts.append(np.round(np.random.uniform(-4, 4, dim)))

        # Random points
        for _ in range(30):
            starts.append(np.random.uniform(-4.5, 4.5, dim))

        # Some points near origin and corners
        starts.append(np.zeros(dim))
        for sign in [1, -1]:
            starts.append(sign * 4.0 * np.ones(dim))
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                starts.append(p)

        # Shuffle starts
        np.random.shuffle(starts)

        # Nelder-Mead implementation (vectorized where possible)
        def nelder_mead(func, x0, max_iter=200, tol=1e-10):
            """Nelder-Mead simplex method."""
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = func(x0)
            simplex.append(f0)

            # Add perturbed points
            for i in range(dim):
                x_i = x0.copy()
                if abs(x0[i]) > 1e-10:
                    x_i[i] *= 1.05
                else:
                    x_i[i] += 0.05
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                simplex.append(func(x_i))

            # Convert to arrays
            simplex = [np.array(simplex[i]) for i in range(0, len(simplex), 2)]
            fvals = np.array(simplex[1::2])

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            n_iter = 0

            for _ in range(max_iter):
                n_iter += 1
                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = fvals[order]

                if fvals[-1] - fvals[0] < tol:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])

            # Return best
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 2: Rastrigin-aware coordinate descent
        def rastrigin_cd(func, x0, max_iter=500):
            """Coordinate descent exploiting Rastrigin structure."""
            x = x0.copy()
            f = func(x)
            step = 0.1

            for _ in range(max_iter):
                improved = False
                for i in range(dim):
                    # Try snapping to nearest integer (Rastrigin minima at integers)
                    x_int = x.copy()
                    x_int[i] = round(x_int[i])
                    x_int = np.clip(x_int, -5, 5)
                    f_int = func(x_int)
                    if f_int < f - 1e-12:
                        x = x_int
                        f = f_int
                        improved = True

                    # Local perturbation
                    for delta in [step, -step]:
                        x_p = x.copy()
                        x_p[i] += delta
                        x_p = np.clip(x_p, -5, 5)
                        f_p = func(x_p)
                        if f_p < f - 1e-12:
                            x = x_p
                            f = f_p
                            improved = True

                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            return x, f

        # Run Nelder-Mead from multiple starts
        nm_starts = starts[:40]  # Use first 40 starts for NM
        for x0 in nm_starts:
            if time.time() > deadline:
                break
            if budget <= 0:
                break
            try:
                x_nm, f_nm = nelder_mead(func, x0, max_iter=150)
                try_candidate(x_nm, f_nm)
            except Exception:
                pass

        # Run Rastrigin CD from multiple starts
        cd_starts = starts[:30]
        for x0 in cd_starts:
            if time.time() > deadline:
                break
            if budget <= 0:
                break
            try:
                x_cd, f_cd = rastrigin_cd(func, x0, max_iter=300)
                try_candidate(x_cd, f_cd)
            except Exception:
                pass

        # Phase 3: Fine-tune best solution with small perturbations
        if best_x is not None and time.time() < deadline:
            # Try small perturbations around best
            for _ in range(50):
                if time.time() > deadline:
                    break
                x_pert = best_x + 0.01 * np.random.randn(dim)
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = func(x_pert)
                try_candidate(x_pert, f_pert)

            # Try integer snapping of best
            x_int = np.round(best_x)
            x_int = np.clip(x_int, -5, 5)
            f_int = func(x_int)
            try_candidate(x_int, f_int)

            # Coordinate-wise integer snapping
            for i in range(dim):
                if time.time() > deadline:
                    break
                x_snap = best_x.copy()
                x_snap[i] = round(x_snap[i])
                x_snap = np.clip(x_snap, -5, 5)
                f_snap = func(x_snap)
                try_candidate(x_snap, f_snap)

        # Phase 4: If still have budget, do a CMA-ES-like local search
        if best_x is not None and time.time() < deadline:
            # Simple adaptive local search
            sigma = 0.1
            x = best_x.copy()
            f = best_f
            for _ in range(100):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                try_candidate(x, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
