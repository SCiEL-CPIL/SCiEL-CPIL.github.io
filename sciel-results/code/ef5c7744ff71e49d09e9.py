"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: coarse exploration + (1+1)-ES portfolio + Nelder-Mead polish.

    Strategy:
      1. Quick coarse sampling (center, axis points, random) for early AOCC gain.
      2. Four (1+1)-ES restarts with diverse initializations and 1/5-rule adaptation.
      3. Final Nelder-Mead simplex refinement around the best point found,
         exploiting deterministic local convergence for high-precision polish.
      4. Explicit wall-clock deadline guard with safety margin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Coarse exploration for early AOCC gains ---
        # Sample a small set of strategic points: center, axis midpoints, random
        exploration_points = []
        exploration_points.append(np.zeros(dim))
        for d in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[d] = sign * 2.5
                exploration_points.append(p)
        for _ in range(8):
            exploration_points.append(rng.uniform(-5, 5, dim))

        # Reserve budget for ES + NM
        reserve_es = max(20, budget // 7)
        reserve_nm = max(15, budget // 10)
        exploration_budget = budget - reserve_es - reserve_nm

        for gp in exploration_points:
            if call_count[0] >= exploration_budget:
                break
            if time.time() > deadline - 40:
                break
            gp = np.clip(gp, -5.0, 5.0)
            f = make_call(gp)
            if f is None:
                break
            update_best(gp, f)

        # --- Phase 2: Portfolio of (1+1)-ES restarts ---
        n_restarts = 4
        remaining_for_es = budget - reserve_nm - call_count[0]
        alloc_per_restart = max(10, remaining_for_es // n_restarts)

        # Restart initializations: best found + random
        restart_inits = []
        if best_x is not None:
            restart_inits.append(best_x.copy())
        while len(restart_inits) < n_restarts:
            restart_inits.append(rng.uniform(-5, 5, dim))

        initial_sigmas = [1.5, 2.0, 1.0, 3.0]

        for r in range(n_restarts):
            if call_count[0] >= budget - reserve_nm:
                break
            if time.time() > deadline - 30:
                break

            x = restart_inits[r].copy()
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = initial_sigmas[r]

            for _ in range(alloc_per_restart):
                if call_count[0] >= budget - reserve_nm:
                    break
                if time.time() > deadline - 30:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.5
                else:
                    sigma *= 0.87

                sigma = float(np.clip(sigma, 0.01, 10.0))

        # --- Phase 3: Nelder-Mead simplex refinement ---
        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 10:
            x0 = best_x.copy()
            f0 = best_f

            # Build initial simplex: regular simplex centered at x0
            # Size: start with moderate step size
            step = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                v = x0.copy()
                v[i] += step
                v = np.clip(v, -5.0, 5.0)
                simplex.append(v)

            # Evaluate initial simplex (first point already known)
            simplex_f = [f0]
            for i in range(1, dim + 1):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline - 10:
                    break
                fi = make_call(simplex[i])
                if fi is None:
                    break
                simplex_f.append(fi)
                update_best(simplex[i], fi)
            else:
                # Full simplex evaluated, run NM iterations
                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma_nm = 0.5  # shrink

                max_nm_iters = budget - call_count[0] - 5
                for _ in range(max_nm_iters):
                    if call_count[0] >= budget - 5:
                        break
                    if time.time() > deadline - 10:
                        break

                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:dim], axis=0)

                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[dim])
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = make_call(xr)
                    if fr is None:
                        break
                    update_best(xr, fr)

                    if fr < simplex_f[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[dim])
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = make_call(xe)
                        if fe is None:
                            break
                        update_best(xe, fe)
                        if fe < fr:
                            simplex[dim] = xe
                            simplex_f[dim] = fe
                        else:
                            simplex[dim] = xr
                            simplex_f[dim] = fr
                    elif fr < simplex_f[dim - 1]:
                        # Accept reflection
                        simplex[dim] = xr
                        simplex_f[dim] = fr
                    else:
                        # Contraction
                        if fr < simplex_f[dim]:
                            # Outside contraction
                            xc = centroid + rho * (xr - centroid)
                        else:
                            # Inside contraction
                            xc = centroid - rho * (simplex[dim] - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = make_call(xc)
                        if fc is None:
                            break
                        update_best(xc, fc)

                        if fc <= fr:
                            simplex[dim] = xc
                            simplex_f[dim] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = (simplex[0] + simplex[i]) / 2.0
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fi = make_call(simplex[i])
                                if fi is None:
                                    break
                                update_best(simplex[i], fi)
                                simplex_f[i] = fi
                            else:
                                continue
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
