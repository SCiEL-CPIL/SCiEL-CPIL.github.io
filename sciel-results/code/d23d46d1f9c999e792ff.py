"""Circle packing in perimeter-4 rectangle: Global Nelder-Mead + Deterministic Polish."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import Delaunay

N = 21
_NUM_PAIRS = N * (N - 1) // 2
_IU = np.triu_indices(N, k=1)


def construct():
    start_time = time.time()
    # 145s safety margin for 150s limit
    deadline = start_time + 145.0
    
    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        # Boundary checks (vectorized)
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False, 0.0
        if np.any(x + r > w + tol) or np.any(y + r > h + tol):
            return False, 0.0

        # Pairwise checks (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]
        if np.any(d_sq_pairs < r_sum_pairs ** 2 - tol):
            return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Smoothed objective for global search
    # Uses log-sum-exp for overlaps to allow gradient-free methods to navigate
    def _smooth_objective(z, beta=100.0, penalty=1000.0):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        # Heavy penalty for out-of-bounds or invalid dims to guide search
        if w <= 0.1 or h <= 0.1 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Pairwise overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        overlap_vals = r_sum_pairs - d_pairs
        # Log-sum-exp smoothing
        shifted = np.clip(beta * overlap_vals, -500, 500)
        max_shifted = np.max(shifted)
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        smooth_overlap = lse / beta
        obj += penalty * smooth_overlap
        
        # Add hard quadratic penalty for large violations to help convergence
        hard_overlap = np.maximum(overlap_vals, 0.0)
        obj += 50 * np.sum(hard_overlap * hard_overlap)

        # Boundary violations
        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h
        all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
        
        shifted_b = np.clip(beta * all_viol, -500, 500)
        max_b = np.max(shifted_b)
        lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
        smooth_viol = lse_b / beta
        obj += penalty * smooth_viol
        
        hard_viol = np.maximum(all_viol, 0.0)
        obj += 50 * np.sum(hard_viol * hard_viol)
        
        return obj

    def generate_initial_guesses():
        guesses = []
        
        # 1. Grid Start
        side = int(np.ceil(np.sqrt(N)))
        w0, h0 = 1.0, 1.0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.4
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)
        
        # 2. Hexagonal Start
        w0, h0 = 1.0, 1.0
        rows, cols = 5, 5
        z = np.zeros(3 * N + 1)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) / cols * w0
            cy = (row + 0.5) / rows * h0
            r0 = 0.4 / np.sqrt(cols)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)
        
        # 3. Delaunay Start (Random points -> Inradii)
        # This creates a structurally valid packing immediately, which NM can refine
        rng = np.random.default_rng(42)
        for trial in range(3):
            w0 = rng.uniform(0.8, 1.2)
            h0 = 2.0 - w0
            points = np.vstack([rng.uniform(0, w0, N), rng.uniform(0, h0, N)]).T
            try:
                tri = Delaunay(points)
                # Calculate circumradii as initial radii (shrunk)
                radii = []
                for t in tri.simplices:
                    p1, p2, p3 = points[t]
                    a = np.linalg.norm(p2 - p3)
                    b = np.linalg.norm(p1 - p3)
                    c = np.linalg.norm(p1 - p2)
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    if area > 1e-6 and a*b*c > 1e-6:
                        r_circ = (a*b*c)/(4*area)
                    else:
                        r_circ = 0.0
                    radii.extend([r_circ]*3)
                
                # Map radii to points (approximation)
                z = np.zeros(3 * N + 1)
                for i in range(N):
                    # Find closest triangle centroid
                    dists = np.linalg.norm(tri.points - points[i], axis=1)
                    idx = np.argmin(dists)
                    # Scale radius to avoid overlap
                    r0 = max(0.02, radii[idx] * 0.8)
                    # Clamp to walls
                    r0 = min(r0, points[i, 0], w0 - points[i, 0], points[i, 1], h0 - points[i, 1])
                    z[3 * i] = points[i, 0]
                    z[3 * i + 1] = points[i, 1]
                    z[3 * i + 2] = r0
                z[-1] = w0
                guesses.append(z)
            except Exception:
                continue
                
        # 4. Random Starts
        for trial in range(5):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.15)
                # Ensure validity against walls
                r0 = min(r0, w0/2, h0/2)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    guesses = generate_initial_guesses()
    
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    # Main Search Loop: Global Nelder-Mead
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 40: # Save time for polish
            break
            
        # Nelder-Mead is derivative-free and handles the smoothed landscape well
        # maxfev=2000 is a good balance for a "quick" global search per start
        try:
            res = minimize(_smooth_objective, z0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if not valid:
                # Simple repair for NM output
                r_idx = np.arange(2, len(z_opt) - 1, 3)
                # Iteratively shrink until valid (crude)
                z_repair = z_opt.copy()
                for _ in range(100):
                    z_repair[r_idx] *= 0.95
                    valid, val = check_validity(z_repair)
                    if valid:
                        break
                z_opt = z_repair
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
                
        except Exception:
            continue

    # Polish Phase: If we found a candidate, squeeze it.
    if best_result is not None and time.time() < deadline - 20:
        # 1. Force Relaxation to ensure local validity and "settled" state
        polished = _force_relax_unified(best_result, max_iter=2000)
        
        # 2. Individual Radius Expansion (Binary Search)
        # This is the key step to extract maximum volume from a local optimum
        final_result = _final_polish(polished, deadline - (time.time() - start_time))
        
        if final_result is not None:
            valid, val = check_validity(final_result)
            if valid and val > best_value:
                best_value = val
                best_result = final_result

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return np.column_stack((best_result[0::3][:N], best_result[1::3][:N], best_result[2::3][:N]))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation to resolve overlaps and settle into a local minimum of energy."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    # Simple iterative relaxation
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Walls
        fx += np.where(x - r < 0, -(x - r), 0) * 50
        fx += np.where(x + r - w > 0, (x + r - w), 0) * 50
        fy += np.where(y - r < 0, -(y - r), 0) * 50
        fy += np.where(y + r - h > 0, (y + r - h), 0) * 50
        
        # Pairs
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        d_sq_pairs = d_sq[_IU]
        r_sum_pairs = r_sum[_IU]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) < 1e-9:
            break
            
        f_pairs = overlap * 10.0
        dx_pairs = dx[_IU]
        dy_pairs = dy[_IU]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        
        fx_contrib = np.zeros(n)
        fy_contrib = np.zeros(n)
        np.add.at(fx_contrib, _IU[0], ux * f_pairs)
        np.add.at(fx_contrib, _IU[1], -ux * f_pairs)
        np.add.at(fy_contrib, _IU[0], uy * f_pairs)
        np.add.at(fy_contrib, _IU[1], -uy * f_pairs)
        
        fx += fx_contrib
        fy += fy_contrib
        
        # Adaptive step
        max_f = np.max(np.sqrt(fx**2 + fy**2))
        step = min(0.05, 0.1 / (1.0 + max_f))
        
        x += step * fx
        y += step * fy
        
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    z[0::3][:n] = x
    z[1::3][:n] = y
    return z


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]: return False
    if new_y < r[i] or new_y > h - r[i]: return False
    
    dx = new_x - x
    dy = new_y - y
    d2 = dx * dx + dy * dy
    d2[i] = np.inf
    r_sum = r[i] + r
    if np.any(d2 < r_sum * r_sum - 1e-9):
        return False
    return True


def _final_polish(z, time_budget):
    """
    Aggressive expansion polish.
    1. Repeatedly tries to increase each radius to its maximum possible value
       (limited by neighbors and walls).
    2. If a radius is blocked, tries to move the circle away from blockers.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    # We iterate until time runs out or no improvement is made
    while time.time() < time_budget:
        improved = False
        # Iterate circles in order of radius (smallest first might be easier to expand? or largest?)
        # Random or fixed order is fine, but let's sort by current radius for stability
        indices = np.argsort(r)
        
        for i in indices:
            if time.time() > time_budget: break
            
            # Calculate max possible radius for i
            # 1. Walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            
            # 2. Neighbors
            dx_i = x[i] - x
            dy_i = y[i] - y
            d_i = np.sqrt(dx_i * dx_i + dy_i * dy_i)
            d_i[i] = np.inf
            r_allowed = d_i - r
            r_max = min(r_wall, np.min(r_allowed))
            r_max = max(0.0, r_max)
            
            if r_max > r[i] + 1e-8:
                # Binary search for exact fit
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    # Check validity against neighbors and walls
                    # (Walls are handled by r_max calc, but check neighbors rigorously)
                    # Actually, just use the vectorized check for safety
                    valid, _ = _check_validity_simple(x, y, r_test, w, h)
                    if valid:
                        lo = mid
                    else:
                        hi = mid
                
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True
            else:
                # Blocked. Try to move away from blocker to allow expansion
                # Find closest neighbor
                dx_i = x[i] - x
                dy_i = y[i] - y
                d_i = np.sqrt(dx_i * dx_i + dy_i * dy_i)
                d_i[i] = np.inf
                closest_j = np.argmin(d_i)
                
                # Direction from closest_j to i
                dx_move = x[i] - x[closest_j]
                dy_move = y[i] - y[closest_j]
                norm = np.sqrt(dx_move**2 + dy_move**2)
                if norm > 1e-9:
                    dx_move /= norm
                    dy_move /= norm
                    # Try moving in that direction
                    for step in [0.02, 0.01, 0.005, 0.001]:
                        new_x = x[i] + dx_move * step
                        new_y = y[i] + dy_move * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            break
                    
        if not improved:
            break
            
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    return z


def _check_validity_simple(x, y, r, w, h, tol=1e-9):
    if np.any(x - r < -tol) or np.any(y - r < -tol): return False, 0
    if np.any(x + r > w + tol) or np.any(y + r > h + tol): return False, 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    if np.any(d_sq[_IU] < r_sum[_IU] ** 2 - tol): return False, 0
    return True, 0
