import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-12
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
    
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    # Precompute rotation matrices for each inner hexagon index (will be updated)
    # For speed, we precompute the vertex offsets from center for each angle
    # Actually, let's precompute the 6 base vertex angles
    base_angles = np.deg2rad(np.arange(6) * 60)
    base_cos = np.cos(base_angles)  # (6,)
    base_sin = np.sin(base_angles)  # (6,)

    def compute_outer_side(theta_out, phis):
        """
        Vectorized computation of minimum outer hexagon side length.
        """
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        n_cos = np.cos(angles)  # (6,)
        n_sin = np.sin(angles)  # (6,)

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            # Rotated vertex offsets from center: (6, 2)
            vx = base_cos * c - base_sin * s
            vy = base_cos * s + base_sin * c
            # Add center
            vx += centers[i, 0]
            vy += centers[i, 1]
            # Dot products with normals: (6, 6)
            dots = vx[:, None] * n_cos[None, :] + vy[:, None] * n_sin[None, :]
            local_max = np.max(np.abs(dots))
            if local_max > max_proj:
                max_proj = local_max

        return 2.0 * max_proj / np.sqrt(3)

    def smooth_objective(params, beta=100.0):
        """
        Vectorized smooth approximation using log-sum-exp.
        """
        theta_out = params[0]
        phis = params[1:]

        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        n_cos = np.cos(angles)
        n_sin = np.sin(angles)

        all_abs_dots = np.empty(n_hex * 36)
        idx = 0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            vx = base_cos * c - base_sin * s + centers[i, 0]
            vy = base_cos * s + base_sin * c + centers[i, 1]
            dots = vx[:, None] * n_cos[None, :] + vy[:, None] * n_sin[None, :]
            np.abs(dots, out=dots)
            all_abs_dots[idx:idx+36] = dots.ravel()
            idx += 36

        max_val = np.max(all_abs_dots)
        if max_val == 0:
            return 0.0
        shifted = (all_abs_dots - max_val) * beta
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)) / len(all_abs_dots)) / beta

        return 2.0 * log_sum_exp / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Analytical initial guess based on known optimal configuration
    # All inner hexagons at 30 degrees (flat-top), outer at 30 degrees
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Phase 1: Multi-start Powell optimization with various starting points
    n_starts = 12
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),  # Analytical
        np.array([0.0] + [0.0] * n_hex),  # All zero
        np.array([np.pi/6] + [0.0] * n_hex),  # Outer at 30, inner at 0
        np.array([0.0] + [np.pi/6] * n_hex),  # Outer at 0, inner at 30
        np.array([np.deg2rad(0)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(60)] + [np.deg2rad(30)] * n_hex),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex),
        np.array([np.deg2rad(20)] + [np.deg2rad(25)] * n_hex),
        np.array([np.deg2rad(25)] + [np.deg2rad(20)] * n_hex),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score - 1e-14:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Coordinate descent exploiting hexagon symmetry (60° periodicity)
    if time.time() < deadline - 20:
        # For a regular hexagon, rotating by 60° gives the same shape.
        # So we can reduce the search space: theta_out mod 60°, phi_i mod 60°.
        # Do a fine grid search over theta_out in [0, 60°) with fixed phis,
        # then refine.
        theta_grid = np.linspace(0, np.pi/3, 121)  # 121 points over 60°
        for t_val in theta_grid:
            if time.time() > deadline - 20:
                break
            test_params = np.concatenate(([t_val], best_params[1:]))
            val = objective(test_params)
            if val < best_score - 1e-14:
                best_score = val
                best_params = test_params.copy()
        
        # Now do coordinate descent on each phi_i
        for sweep in range(5):
            if time.time() > deadline - 15:
                break
            improved = False
            for j in range(n_hex):
                if time.time() > deadline - 15:
                    break
                # Search phi_j in [0, 60°) with fine grid
                phi_grid = np.linspace(0, np.pi/3, 61)
                best_local = best_score
                best_local_phi = best_params[j+1]
                for p_val in phi_grid:
                    test_params = best_params.copy()
                    test_params[j+1] = p_val
                    val = objective(test_params)
                    if val < best_local - 1e-14:
                        best_local = val
                        best_local_phi = p_val
                if best_local < best_score - 1e-14:
                    best_score = best_local
                    best_params[j+1] = best_local_phi
                    improved = True
            if not improved:
                break

    # Phase 3: Smooth surrogate optimization with adaptive beta
    if time.time() < deadline - 15:
        for beta_val in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 12:
                break
            try:
                res = minimize(lambda p: smooth_objective(p, beta=beta_val), best_params,
                              method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
                true_score = objective(res.x)
                if true_score < best_score - 1e-14:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 4: Binary search epigraph polish
    # For a given side length S, check if all inner hexagons fit.
    # The outer hexagon with center (0,0), side S, rotation theta has
    # support function h(n) = S * |n · u| where u are the edge normals.
    # Actually for a regular hexagon of side S centered at origin with rotation theta:
    # The 6 edge normals are at angles 30° + k*60° + theta.
    # The support in direction n is S * |cos(angle between n and nearest normal)|... 
    # More precisely, for a regular hexagon of side length S, the inradius is S*sqrt(3)/2.
    # The support function in direction making angle alpha with a normal is S*sqrt(3)/2 / cos(alpha)
    # for |alpha| <= 30°.
    # A point (x,y) is inside iff for all 6 normals n_k: x*n_kx + y*n_ky <= S*sqrt(3)/2 / cos(delta_k)
    # where delta_k is the angle from the point's projection to the nearest normal.
    # Simpler: a point p is inside the hexagon iff max_k |p · n_k| <= S*sqrt(3)/2 / cos(30°) = S
    # Wait, let me think again. For a regular hexagon of side S:
    # The vertices are at distance S from center (circumradius = S).
    # The edge midpoints are at distance S*sqrt(3)/2 from center (inradius).
    # The hexagon is the intersection of 6 half-planes: p · n_k <= inradius = S*sqrt(3)/2
    # where n_k are unit normals to the edges.
    # So a point p is inside iff max_k |p · n_k| <= S*sqrt(3)/2.
    # For a rotated hexagon, the normals rotate too.
    
    def fits_in_hexagon(theta_out, phis, S):
        """Check if all inner hexagon vertices are inside the outer hexagon of side S."""
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        n_cos = np.cos(angles)
        n_sin = np.sin(angles)
        inradius = S * np.sqrt(3) / 2.0
        
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            vx = base_cos * c - base_sin * s + centers[i, 0]
            vy = base_cos * s + base_sin * c + centers[i, 1]
            dots = vx * n_cos + vy * n_sin  # (6,)
            if np.max(np.abs(dots)) > inradius + 1e-12:
                return False
        return True

    if time.time() < deadline - 8:
        # Binary search for the tightest S
        lo = best_score - 0.5
        hi = best_score + 0.01
        # First, find a valid upper bound
        while not fits_in_hexagon(best_params[0], best_params[1:], hi):
            hi *= 1.01
            if hi > best_score + 1.0:
                break
        
        for _ in range(40):
            if time.time() > deadline - 8:
                break
            mid = (lo + hi) / 2.0
            if fits_in_hexagon(best_params[0], best_params[1:], mid):
                hi = mid
            else:
                lo = mid
        
        # Use the binary search result as a tighter estimate
        if hi < best_score - 1e-12:
            best_score = hi
            # Also try to optimize angles with this tighter constraint
            # (skip for now to save time)

    # Phase 5: Final high-precision polish with tight tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_params, method='Powell',
                          options={'maxiter': 1000, 'disp': False, 'ftol': 1e-15, 'xtol': 1e-15})
            true_score = objective(res.x)
            if true_score < best_score - 1e-15:
                best_score = true_score
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    side = best_score + 1e-9  # Tiny margin for floating point safety
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    This is a known good configuration.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # The side length for the optimal flower packing is exactly 4.0
    # Add a tiny margin for floating point safety
    side = 4.0 + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
