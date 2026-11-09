"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel Nelder-Mead ensemble with adaptive restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Brief random sampling to identify promising regions
        n_starts = 10
        for i in range(n_starts):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx is None:
                break

        # Phase 2: Parallel Nelder-Mead ensemble
        # Initialize 5 simplexes at strategic locations
        # For Rastrigin, local minima are at integer coordinates, so we sample near integers
        
        # Strategic base points: center, best found, and points near integer lattice
        base_points = [best_x.copy(), np.zeros(dim)]
        # Add points near integer lattice (where Rastrigin minima tend to be)
        for coord in [-3, -2, -1, 0, 1, 2, 3]:
            pt = np.full(dim, float(coord))
            base_points.append(pt)
        # Keep only 5 diverse base points
        # Sort by function value and pick top 5 with diversity
        base_points_f = []
        for bp in base_points:
            if call_count >= budget or time.time() > deadline:
                break
            fp = call_func(bp)
            if fp is None:
                break
            base_points_f.append((bp, fp))
        
        # Select 5 diverse points: best, center, and 3 most different from best
        base_points_f.sort(key=lambda t: t[1])
        selected_bases = [base_points_f[0][0].copy()]
        if len(base_points_f) > 1:
            selected_bases.append(base_points_f[1][0].copy())
        
        # Add diverse points
        for bp, fp in base_points_f[2:]:
            if len(selected_bases) >= 5:
                break
            # Check distance from already selected
            min_dist = min(np.linalg.norm(bp - sb) for sb in selected_bases)
            if min_dist > 2.0:
                selected_bases.append(bp.copy())
        
        # Ensure we have exactly 5 (pad with random if needed)
        while len(selected_bases) < 5:
            pt = rng.uniform(-4, 4, dim)
            selected_bases.append(pt)

        # Nelder-Mead parameters
        alpha_init = 0.8  # initial simplex size
        alpha_nm = 1.0    # reflection
        gamma_nm = 2.0    # expansion
        rho_nm = 0.5      # contraction
        sigma_nm = 0.5    # shrink
        max_iter_per_simplex = 40
        restart_threshold = 8  # iterations without improvement before restart

        # Run Nelder-Mead in parallel (sequential but independent)
        for base_pt in selected_bases:
            if call_count >= budget or time.time() > deadline:
                break

            # Initialize Nelder-Mead simplex
            simplex = [base_pt.copy()]
            simplex_f = [call_func(base_pt)]
            if simplex_f[0] is None:
                break

            for d in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                pt = base_pt.copy()
                pt[d] += alpha_init
                pt = np.clip(pt, -5, 5)
                fpt = call_func(pt)
                if fpt is None:
                    break
                simplex.append(pt)
                simplex_f.append(fpt)

            if len(simplex) < dim + 1:
                continue

            # Track best for this simplex
            local_best_f = min(simplex_f)
            local_best_x = simplex[np.argmin(simplex_f)].copy()
            no_improve_count = 0

            # Nelder-Mead iterations with adaptive restart
            for nm_iter in range(max_iter_per_simplex):
                if call_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Update local best
                if simplex_f[0] < local_best_f - 1e-10:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # If stagnating, restart with smaller simplex around local best
                if no_improve_count >= restart_threshold:
                    # Shrink and reinitialize around local best
                    simplex = [local_best_x.copy()]
                    simplex_f = [call_func(local_best_x)]
                    if simplex_f[0] is None:
                        break
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        pt = local_best_x.copy()
                        pt[d] += alpha_init * 0.5
                        pt = np.clip(pt, -5, 5)
                        fpt = call_func(pt)
                        if fpt is None:
                            break
                        simplex.append(pt)
                        simplex_f.append(fpt)
                    if len(simplex) < dim + 1:
                        break
                    no_improve_count = 0
                    continue

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = call_func(x_ref)
                if f_ref is None:
                    break

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma_nm * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = call_func(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho_nm * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = call_func(x_con)
                    if f_con is None:
                        break
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shr = call_func(simplex[i])
                            if f_shr is None:
                                break
                            simplex_f[i] = f_shr
                        if call_count >= budget or time.time() > deadline:
                            break
                        break  # exit NM loop after shrink

        # Phase 3: Final fine-tuning with coordinate descent around best point
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            for step in [0.3, 0.15, 0.07, 0.03, 0.01]:
                if call_count >= budget or time.time() > deadline:
                    break

                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-10:
                                x = x_new
                                fx = fx_new
                                improved = True
                                break  # restart coordinate loop

        return best_x, best_f
# EVOLVE-BLOCK-END
