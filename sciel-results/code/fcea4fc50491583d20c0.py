"""Packing 12 unit hexagons in a hexagonal container; minimize outer side length."""
import numpy as np
import time
from scipy.optimize import minimize
import math

N = 12
_START = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon (side s, center, rotation rad)."""
    ang = theta + np.arange(6) * (np.pi / 3.0)
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True


def _inner_fits_outer(inner_verts, outer_verts, tol=1e-9):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_convex(v[0], v[1], outer_verts, tol):
            return False
    return True


def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = np.min(v1 @ normal), np.max(v1 @ normal)
            r2min, r2max = np.min(v2 @ normal), np.max(v2 @ normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True


def _feasible(outer_side, placements, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if given placements fit inside outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_verts = _hex_vertices(ox, oy, outer_side, math.radians(outer_angle))
    
    # Precompute inner vertices
    inner_verts_list = []
    for (ix, iy, iangle_deg) in placements:
        inner_verts_list.append(_hex_vertices(ix, iy, 1.0, math.radians(iangle_deg)))
    
    # Check each inner hexagon fits inside outer
    for iv in inner_verts_list:
        if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
            return False
    
    # Check pairwise non-overlap
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                return False
    
    return True


def _smooth_collision_penalty(A, B, smooth=0.05):
    """Smooth penalty: 0 if separated, grows as overlap.
    Uses separating axis to compute gap/penetration depth."""
    def edge_axes(P):
        axes = []
        n = len(P)
        for i in range(n):
            p1, p2 = P[i], P[(i + 1) % n]
            e = p2 - p1
            ax = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-12:
                axes.append(ax / nrm)
        return axes

    axes = edge_axes(A) + edge_axes(B)
    gaps = []
    for ax in axes:
        a_max = np.max(A @ ax)
        b_min = np.min(B @ ax)
        gaps.append(b_min - a_max)
    gaps = np.array(gaps)
    sep = np.max(gaps)
    if sep > 0:
        return 0.0
    depth = -np.min(gaps)
    return depth


def _objective(params, S):
    """params: [S, x0,y0,th0, x1,y1,th1, ...]. Minimize S + penalties."""
    S = params[0]
    if S <= 0:
        return 1e6
    outer = _hex_vertices(0.0, 0.0, S, 0.0)
    penalty = 0.0
    hexes = []
    for k in range(N):
        cx, cy, th = params[1 + 3 * k], params[2 + 3 * k], params[3 + 3 * k]
        H = _hex_vertices(cx, cy, 1.0, th)
        hexes.append(H)
        # Containment: all vertices inside outer hexagon.
        if not _inner_fits_outer(H, outer, tol=1e-9):
            # Compute violation magnitude using point-in-convex distance approximation
            # Use max violation over vertices
            for v in H:
                # Check each edge of outer
                for i in range(6):
                    j = (i + 1) % 6
                    ex = outer[j, 0] - outer[i, 0]
                    ey = outer[j, 1] - outer[i, 1]
                    px_rel = v[0] - outer[i, 0]
                    py_rel = v[1] - outer[i, 1]
                    cross = ex * py_rel - ey * px_rel
                    if cross < 0:
                        # Distance to edge line
                        edge_len = math.sqrt(ex*ex + ey*ey)
                        dist = abs(cross) / (edge_len + 1e-12)
                        penalty += dist ** 2
    # inner-inner collisions
    for i in range(N):
        for j in range(i + 1, N):
            d = _smooth_collision_penalty(hexes[i], hexes[j])
            penalty += d ** 2
    return S + 10.0 * penalty


def _make_initial(S0, pattern, seed=42):
    """Create initial params."""
    params = np.zeros(1 + 3 * N)
    params[0] = S0
    if pattern == 'cluster':
        sp = math.sqrt(3.0) + 0.05
        positions = []
        for i in range(3):
            positions.append((i * sp, 0.0))
        y1 = sp * math.sin(math.pi / 3.0)
        for i in range(4):
            positions.append((i * sp - sp / 2.0, y1))
        y2 = 2 * y1
        for i in range(3):
            positions.append((i * sp, y2))
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        mx = (max(xs) + min(xs)) / 2.0
        my = (max(ys) + min(ys)) / 2.0
        for k, (px, py) in enumerate(positions):
            params[1 + 3 * k] = px - mx
            params[2 + 3 * k] = py - my
            params[3 + 3 * k] = 0.0
    elif pattern == 'ring':
        # Place 12 hexagons in a ring around center
        for k in range(N):
            angle = 2 * math.pi * k / N
            r = S0 * 0.6
            params[1 + 3 * k] = r * math.cos(angle)
            params[2 + 3 * k] = r * math.sin(angle)
            params[3 + 3 * k] = angle  # rotate to align with radial direction
    else:
        rng = np.random.default_rng(seed)
        for k in range(N):
            params[1 + 3 * k] = rng.uniform(-2, 2)
            params[2 + 3 * k] = rng.uniform(-2, 2)
            params[3 + 3 * k] = rng.uniform(0, np.pi)
    return params


def _try_optimize(S0, pattern, seed=42, max_iter=400):
    params0 = _make_initial(S0, pattern, seed)
    try:
        res = minimize(_objective, params0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6})
        return res
    except Exception:
        return None


def _bisection_search(lo, hi, max_iters=20, seed=42):
    """Bisection search on outer side length using greedy placement."""
    best_side = hi
    best_placements = None
    
    for it in range(max_iters):
        mid = (lo + hi) / 2.0
        
        # Try to place 12 hexagons in a hexagon of side mid
        # Use a simple grid-based greedy approach
        inradius = mid * math.sqrt(3.0) / 2.0
        
        # Generate candidate positions on a grid
        num_grid = 15
        candidates = []
        for i in range(num_grid):
            for j in range(num_grid):
                x = -mid + 2.0 * mid * i / (num_grid - 1)
                y = -mid * math.sqrt(3) / 2.0 + mid * math.sqrt(3) * j / (num_grid - 1)
                for angle in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
                    candidates.append((x, y, angle))
        
        # Sort by distance from center
        candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
        
        # Greedy placement
        placed = []
        placed_verts = []
        outer_verts = _hex_vertices(0.0, 0.0, mid, 0.0)
        
        for (x, y, angle) in candidates:
            if len(placed) >= N:
                break
            iv = _hex_vertices(x, y, 1.0, math.radians(angle))
            
            # Check fits inside outer
            if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
                continue
            
            # Check no overlap with already placed
            overlaps = False
            for pv in placed_verts:
                if _sat_overlap(iv, pv, tol=1e-9):
                    overlaps = True
                    break
            if overlaps:
                continue
            
            placed.append((x, y, angle))
            placed_verts.append(iv)
        
        if len(placed) >= N:
            best_side = mid
            best_placements = placed[:N]
            hi = mid
        else:
            lo = mid
    
    return best_side, best_placements


def construct():
    global _START
    best_S = None
    best_params = None
    best_obj = np.inf
    
    # Try optimization with different initializations
    patterns = ['cluster', 'ring', 'random']
    S0_list = [5.0, 6.0, 7.0]
    
    for S0 in S0_list:
        for pat in patterns:
            if time.time() - _START > _DEADLINE:
                break
            try:
                res = _try_optimize(S0, pat, max_iter=300)
                if res is not None:
                    S_val = res.x[0]
                    obj = _objective(res.x, S_val)
                    if obj < best_obj and S_val > 0:
                        best_obj = obj
                        best_S = S_val
                        best_params = res.x.copy()
            except Exception:
                continue
        if time.time() - _START > _DEADLINE:
            break
    
    # Also try bisection search
    if time.time() - _START < _DEADLINE - 30.0:
        try:
            b_side, b_placements = _bisection_search(2.5, 15.0, max_iters=15)
            if b_placements is not None and (best_S is None or b_side < best_S):
                best_S = b_side
                # Convert placements to params format
                best_params = np.zeros(1 + 3 * N)
                best_params[0] = b_side
                for k, (px, py, pang) in enumerate(b_placements):
                    best_params[1 + 3 * k] = px
                    best_params[2 + 3 * k] = py
                    best_params[3 + 3 * k] = math.radians(pang)
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: return something valid-ish (large container)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Build inner list
    inner = []
    for k in range(N):
        cx = best_params[1 + 3 * k]
        cy = best_params[2 + 3 * k]
        th = best_params[3 + 3 * k]
        inner.append([float(cx), float(cy), float(math.degrees(th))])
    
    return inner, (0.0, 0.0), float(best_S), 0.0
