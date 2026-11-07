"""Hybrid optimizer: Multi-start Nelder-Mead + Coordinate Descent for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Multi-start Nelder-Mead + Coordinate Descent for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != dim:
                x = np.resize(x, dim)
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_calls):
            """Nelder-Mead local refinement with stall detection."""
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() > deadline:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]
            alpha = 0.5  # initial step size for simplex expansion

            for i in range(dim):
                if calls >= max_calls or time.time() > deadline:
                    return
                xi = x0.copy()
                xi[i] += alpha
                simplex.append(xi)
                fvals.append(safe_func(xi))

            # Nelder-Mead parameters
            alpha_n = 1.0   # reflection
            gamma = 2.0     # expansion
            rho = 0.5       # contraction
            sigma_s = 0.5   # shrink

            max_iter = max_calls - calls
            stall_count = 0
            stall_limit = 15

            for _ in range(max_iter):
                if calls >= max_calls or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check for convergence
                if fvals[-1] - fvals[0] < 1e-12:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha_n * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    stall_count = 0
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                    stall_count = 0
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = safe_func(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        stall_count = 0
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= max_calls or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_func(simplex[i])
                        stall_count += 1

                # Update best
                new_best_idx = int(np.argmin(fvals))
                if fvals[new_best_idx] < best_f:
                    best_x = simplex[new_best_idx].copy()
                    best_f = fvals[new_best_idx]

                # Stall detection
                if stall_count >= stall_limit:
                    # Random restart within current region
                    x_new = best_x + 1.0 * np.random.default_rng(42 + calls).standard_normal(dim)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = safe_func(x_new)
                    if f_new < best_f:
                        best_x = x_new.copy()
                        best_f = f_new
                    # Reinitialize simplex around new best
                    simplex = [best_x.copy()]
                    fvals = [best_f]
                    for i in range(dim):
                        if calls >= max_calls or time.time() > deadline:
                            break
                        xi = best_x.copy()
                        xi[i] += 0.5
                        simplex.append(xi)
                        fvals.append(safe_func(xi))
                    stall_count = 0

        def coordinate_descent(x0, max_calls):
            """Coarse-to-fine coordinate descent polish."""
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() > deadline:
                return

            x = x0.copy()
            fx = best_f
            best_x_local = x.copy()
            best_fx_local = fx

            # Step sizes: coarse to fine
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]

            for step in step_sizes:
                if calls >= max_calls or time.time() > deadline:
                    break

                improved = True
                while improved and calls < max_calls and time.time() <= deadline:
                    improved = False
                    for i in range(dim):
                        if calls >= max_calls or time.time() > deadline:
                            break

                        # Try positive step
                        xp = x.copy()
                        xp[i] += step
                        xp = np.clip(xp, -5, 5)
                        fp = safe_func(xp)
                        if fp < fx:
                            x = xp
                            fx = fp
                            improved = True
                            if fx < best_fx_local:
                                best_fx_local = fx
                                best_x_local = x.copy()

                        # Try negative step
                        if calls >= max_calls or time.time() > deadline:
                            break
                        xm = x.copy()
                        xm[i] -= step
                        xm = np.clip(xm, -5, 5)
                        fm = safe_func(xm)
                        if fm < fx:
                            x = xm
                            fx = fm
                            improved = True
                            if fx < best_fx_local:
                                best_fx_local = fx
                                best_x_local = x.copy()
            
            if best_fx_local < best_f:
                best_x = best_x_local.copy()
                best_f = best_fx_local

        # Phase 1: Multi-start exploration
        rng = np.random.default_rng(42)
        n_initial = 15
        candidates = []
        
        # Strategy: Mix of global random, semi-local, and very local to find basins
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            if i < 5:
                # Broad random
                x = rng.uniform(-5.0, 5.0, dim)
            elif i < 10:
                # Medium random
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                # Local random around origin
                x = rng.uniform(-1.5, 1.5, dim)
            
            f = safe_func(x)
            candidates.append((f, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead on top candidates
        # We allocate budget based on remaining calls.
        # Let's try NM on the best 3 candidates, splitting remaining budget.
        # Reserve some budget for final CD polish.
        
        remaining_budget = budget - calls
        # Reserve 20% for CD polish, or at least 20 calls
        cd_reserve = max(20, int(0.2 * remaining_budget))
        nm_budget_total = remaining_budget - cd_reserve
        
        if nm_budget_total > 10:
            # Split NM budget among top 3 candidates
            # Give more to the best one
            weights = [0.5, 0.3, 0.2]
            for i in range(min(3, len(candidates))):
                if calls >= budget - cd_reserve or time.time() > deadline:
                    break
                alloc = int(nm_budget_total * weights[i])
                if alloc < 5:
                    break
                # Run NM on this candidate
                # We need to ensure we don't exceed the total budget allocated for NM phase
                max_calls_for_this_nm = min(calls + alloc, budget - cd_reserve)
                nelder_mead(candidates[i][1].copy(), max_calls_for_this_nm)
                
                # Update candidate list with new best if improved? 
                # Actually, safe_func updates global best. 
                # We just proceed to next candidate if budget allows.

        # Phase 3: Coordinate Descent Polish on the global best
        if calls < budget and time.time() < deadline:
            coordinate_descent(best_x.copy(), budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
