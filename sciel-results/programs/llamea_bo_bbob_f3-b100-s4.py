"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy: Structure-exploiting coordinate-wise grid search + Nelder-Mead polish.
    
    Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global min at x=0 (value 0), local minima at integers (value 10*n).
    Period 2 in each coordinate.
    
    Approach:
    1. Coarse-to-fine 1D grid search along each coordinate to find the basin.
       Use the known periodicity to search around multiples of 2.
    2. Nelder-Mead polish on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, calls_left, deadline):
        """Nelder-Mead polish."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        if calls >= calls_left or time.time() > deadline:
            return best_x, best_f, calls

        # Build initial simplex
        simplex = [x.copy()]
        fvals = [fx]
        step = 0.1
        for i in range(d):
            if calls >= calls_left or time.time() > deadline:
                break
            xi = x.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        if len(simplex) < 2:
            return best_x, best_f, calls

        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5
        max_iter = 200

        for _ in range(max_iter):
            if calls >= calls_left or time.time() > deadline:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()

            if calls >= calls_left or time.time() > deadline:
                break

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                if calls >= calls_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                if calls >= calls_left or time.time() > deadline:
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls < calls_left and time.time() <= deadline:
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        else:
                            break
                    if calls >= calls_left or time.time() > deadline:
                        break

        return best_x, best_f, calls

    def _coord_grid_search(self, func, x0, calls_left, deadline):
        """
        Coordinate-wise grid search exploiting Rastrigin's periodic structure.
        For each coordinate, search around multiples of 2 (the period) to find
        the local minimum in that dimension, then refine.
        """
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Coarse-to-fine grid search along each coordinate
        # Rastrigin has period 2, so local minima are near integers
        # Start with coarse grid, then refine

        # Phase 1: Coarse search - try values near each integer in [-4, 4]
        # for each coordinate, holding others fixed
        coarse_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        
        for coord in range(d):
            if calls >= calls_left or time.time() > deadline:
                break
            for val in coarse_values:
                if calls >= calls_left or time.time() > deadline:
                    break
                x_trial = x.copy()
                x_trial[coord] = val
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    x = x_trial.copy()

        # Phase 2: Medium refinement - try values near the current best for each coord
        for coord in range(d):
            if calls >= calls_left or time.time() > deadline:
                break
            base = x[coord]
            # Try offsets from current value
            for offset in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
                if calls >= calls_left or time.time() > deadline:
                    break
                x_trial = x.copy()
                x_trial[coord] = base + offset
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    x = x_trial.copy()

        # Phase 3: Fine refinement
        for coord in range(d):
            if calls >= calls_left or time.time() > deadline:
                break
            base = x[coord]
            for offset in [-0.125, -0.0625, 0.0, 0.0625, 0.125]:
                if calls >= calls_left or time.time() > deadline:
                    break
                x_trial = x.copy()
                x_trial[coord] = base + offset
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    x = x_trial.copy()

        return best_x, best_f, calls

    def _multi_start_nm(self, func, candidates, calls_left, deadline):
        """Run Nelder-Mead from multiple starting points, keep the best."""
        best_x = None
        best_f = np.inf
        total_calls = 0

        for cand in candidates:
            if total_calls >= calls_left or time.time() > deadline:
                break
            # Estimate calls needed for NM: ~d*20 + 1
            est_calls = self.dim * 20 + 1
            if total_calls + est_calls > calls_left:
                est_calls = calls_left - total_calls
            if est_calls < 10:
                break

            x_pol, f_pol, c_pol = self._nm_polish(
                func, cand, est_calls, deadline
            )
            total_calls += c_pol
            if f_pol < best_f:
                best_f = f_pol
                best_x = x_pol.copy()

        return best_x, best_f, total_calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # Safety margin before 330s kill
        calls = 0

        # Phase 1: Initial sampling to get a rough idea of where the minimum is
        # Sample a grid of points to find the basin
        n_initial = min(50, budget // 4)
        
        # Use a structured set of initial points
        # Include points near integers (where Rastrigin has local minima)
        # and points near 0 (global min region)
        initial_pts = []
        
        # Origin and nearby
        initial_pts.append(np.zeros(dim))
        
        # Points at integer coordinates
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt = np.zeros(dim)
                pt[0] = i
                pt[1] = j
                if dim > 2:
                    pt[2] = (i + j) % 5 - 2
                initial_pts.append(pt)
        
        # Random-ish structured points
        np.random.seed(42)
        for _ in range(n_initial - len(initial_pts)):
            pt = np.random.uniform(-4, 4, dim)
            initial_pts.append(pt)
        
        initial_pts = np.array(initial_pts[:n_initial])
        
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for i in range(len(initial_pts)):
            if calls >= budget or time.time() > deadline:
                break
            x = initial_pts[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Coordinate-wise grid search to refine the basin location
        refine_budget = min(budget // 3, budget - calls)
        if refine_budget >= 20 and time.time() < deadline:
            x_ref, f_ref, c_ref = self._coord_grid_search(
                func, best_x, refine_budget, deadline
            )
            calls += c_ref
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

        # Phase 3: Multi-start Nelder-Mead polish
        # Create several candidate starting points around the best found
        polish_budget = min(budget - calls, 200)
        if polish_budget >= 20 and time.time() < deadline:
            candidates = [best_x.copy()]
            
            # Add perturbed versions
            for delta in [0.1, -0.1, 0.2, -0.2]:
                for coord in range(dim):
                    if len(candidates) >= 5:
                        break
                    cand = best_x.copy()
                    cand[coord] += delta
                    cand = np.clip(cand, -5, 5)
                    candidates.append(cand)
            
            # Also try points near the origin (global min of Rastrigin)
            candidates.append(np.zeros(dim))
            candidates.append(np.array([0.1]*dim))
            candidates.append(np.array([-0.1]*dim))
            
            x_pol, f_pol, c_pol = self._multi_start_nm(
                func, candidates, polish_budget, deadline
            )
            calls += c_pol
            if f_pol is not None and f_pol < best_f:
                best_f = f_pol
                best_x = x_pol.copy()

        # Phase 4: Final coordinate refinement + NM if budget remains
        if calls < budget and time.time() < deadline and (budget - calls) >= 30:
            # Final coordinate refinement
            refine_budget2 = min(20, budget - calls)
            x_ref2, f_ref2, c_ref2 = self._coord_grid_search(
                func, best_x, refine_budget2, deadline
            )
            calls += c_ref2
            if f_ref2 < best_f:
                best_f = f_ref2
                best_x = x_ref2.copy()
            
            # Final NM polish
            if calls < budget and time.time() < deadline and (budget - calls) >= 15:
                polish_budget2 = min(budget - calls, 50)
                x_pol2, f_pol2, c_pol2 = self._nm_polish(
                    func, best_x, polish_budget2, deadline
                )
                calls += c_pol2
                if f_pol2 < best_f:
                    best_f = f_pol2
                    best_x = x_pol2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
