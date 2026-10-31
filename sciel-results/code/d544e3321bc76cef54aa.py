"""
Hexagon packing via polyhex enumeration + smooth outer optimization.

Strategy:
1. Enumerate all 11-cell polyhexes (connected sets of 11 cells on the hex
   lattice) via BFS / backtracking with canonicalization to avoid duplicates.
2. For each polyhex, compute the tight enclosing regular hexagon (aligned,
   i.e. same orientation as the lattice) side length.
3. Keep the top-K polyhexes by side length as starting points.
4. Refine each with a smooth (log-sum-exp) outer-side objective using
   L-BFGS-B, allowing small rotations and translations of each inner hexagon.
5. Final validity check with SAT overlap and point-in-polygon containment.
"""
import numpy as np
import math
import time
import sys
from itertools import product
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
EPS = 1e-9

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def hex_vertices(cx, cy, s, angle_rad):
    """6 vertices of a regular hexagon (CCW), side s, center (cx,cy)."""
    a = angle_rad
    out = np.empty((6, 2))
    for i in range(6):
        th = a + i * math.pi / 3.0
        out[i, 0] = cx + s * math.cos(th)
        out[i, 1] = cy + s * math.sin(th)
    return out

def convex_hull_2d(pts):
    """Andrew's monotone chain. pts: (n,2). Returns hull in CCW order."""
    pts = np.array(pts, dtype=float)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.vstack([lower[:-1], upper[:-1]])

def outer_side_from_hull(hull, outer_angle_deg=0.0):
    """
    Minimum side length of a regular hexagon (given rotation) that contains
    the convex hull.  Uses support-function projection on 12 normals.
    """
    oa = math.radians(outer_angle_deg)
    # 6 outward face-normals of a hexagon with vertex-angle 0
    normals = np.array([[math.cos(math.radians(30 + 60*k)),
                         math.sin(math.radians(30 + 60*k))] for k in range(6)])
    all_normals = np.vstack([normals, -normals])
    # Rotate hull by -outer_angle so normals align with standard axes
    c, s = math.cos(-oa), math.sin(-oa)
    R = np.array([[c, -s], [s, c]])
    hull_rot = hull @ R.T
    supports = np.max(hull_rot @ all_normals.T, axis=1)
    return np.max(supports) / (SQRT3 / 2.0)

# ---------------------------------------------------------------------------
# Polyhex enumeration
# ---------------------------------------------------------------------------

# Axial coordinate neighbours (pointy-top hex grid)
_NEIGH = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]

def _axial_to_cart(q, r, scale=1.0):
    x = scale * (1.5 * q)
    y = scale * (SQRT3 * (r + q / 2.0))
    return x, y

def _canonical_key(cells):
    """Canonical representation of a set of axial coords for dedup."""
    return frozenset(cells)

def _normalize(cells):
    """Translate so min-q, min-r is at origin; return frozenset."""
    cells = list(cells)
    min_q = min(c[0] for c in cells)
    min_r = min(c[1] for c in cells)
    norm = frozenset((c[0]-min_q, c[1]-min_r) for c in cells)
    return norm

def generate_polyhexes(n_cells, max_radius=4):
    """
    Generate all distinct polyhexes of size n_cells via backtracking.
    Returns list of frozensets of (q,r) in canonical (normalized) form.
    """
    seen = set()
    results = []
    
    def backtrack(cells, available, count):
        if count == n_cells:
            key = _normalize(cells)
            if key not in seen:
                seen.add(key)
                results.append(key)
            return
        # Try adding each neighbour of each existing cell
        added = set()
        for cell in cells:
            for dq, dr in _NEIGH:
                nc = (cell[0]+dq, cell[1]+dr)
                if nc in available or nc in added:
                    continue
                added.add(nc)
                backtrack(cells | {nc}, available | {nc}, count + 1)
    
    # Start from origin
    backtrack({(0,0)}, {(0,0)}, 1)
    return results

# ---------------------------------------------------------------------------
# Smooth objective for refinement
# ---------------------------------------------------------------------------

def _smooth_outer_side(params, beta=80.0):
    """
    params layout: [x1,y1,a1, ..., xN,yN,aN, ox,oy,oa]  (angles in radians)
    Returns smooth approximation of the outer side length.
    """
    n = N
    ox = params[3*n]
    oy = params[3*n+1]
    oa = params[3*n+2]
    
    # 12 normals in the outer's local frame
    normals = np.array([[math.cos(math.radians(30+60*k)),
                         math.sin(math.radians(30+60*k))] for k in range(6)])
    all_normals = np.vstack([normals, -normals])
    
    cos_a, sin_a = math.cos(-oa), math.sin(-oa)
    
    projections = []
    for i in range(n):
        cx, cy, ca = params[3*i], params[3*i+1], params[3*i+2]
        # 6 vertices of unit hexagon
        for j in range(6):
            th = ca + j * math.pi / 3.0
            vx = cx + math.cos(th)
            vy = cy + math.sin(th)
            # Translate to outer center, rotate by -oa
            dx, dy = vx - ox, vy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in all_normals:
                projections.append(lx*nx + ly*ny)
    
    if not projections:
        return 0.0
    arr = np.array(projections)
    mx = arr.max()
    # log-sum-exp
    shifted = beta * (arr - mx)
    # clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    s = np.exp(shifted).sum()
    smooth_max = mx + math.log(s) / beta
    return 2.0 * smooth_max / SQRT3

def _overlap_penalty(params, strength=200.0):
    """Penalty if any pair of inner hexagons are too close (center distance)."""
    n = N
    min_d = 2.0 * SQRT3 - 0.05  # conservative
    pen = 0.0
    for i in range(n):
        x1, y1 = params[3*i], params[3*i+1]
        for j in range(i+1, n):
            x2, y2 = params[3*j], params[3*j+1]
            d = math.hypot(x1-x2, y1-y2)
            if d < min_d:
                pen += strength * (min_d - d) ** 2
    return pen

def _objective(params, beta=80.0):
    return _smooth_outer_side(params, beta) + _overlap_penalty(params)

# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

def _sat_overlap(v1, v2, eps=1e-7):
    """SAT test for two convex polygons (numpy arrays (6,2))."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i+1)%6] - v[i]
            nrm = np.array([-e[1], e[0]])
            nn = np.linalg.norm(nrm)
            if nn > 1e-12:
                axes.append(nrm / nn)
    for ax in axes:
        p1 = v1 @ ax
        p2 = v2 @ ax
        if p1.max() < p2.min() - eps or p2.max() < p1.min() - eps:
            return False
    return True

def _check_valid(inner, ox, oy, S, oa_deg, eps=1e-6):
    """inner: list of (x,y,angle_deg)."""
    # pairwise overlap
    verts = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in inner]
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            if _sat_overlap(verts[i], verts[j], eps):
                return False
    # containment
    outer_v = hex_vertices(ox, oy, S, math.radians(oa_deg))
    for v in verts:
        for pt in v:
            inside = True
            for i in range(6):
                p1 = outer_v[i]
                p2 = outer_v[(i+1)%6]
                e = p2 - p1
                nrm = np.array([-e[1], e[0]])
                d1 = np.dot(nrm, pt - p1)
                d2 = np.dot(nrm, outer_v[(i+2)%6] - p1)
                if d1 * d2 < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10 s safety margin
    
    # ---- Phase 1: enumerate polyhexes -----------------------------------
    print("Enumerating polyhexes...", file=sys.stderr)
    polyhexes = generate_polyhexes(N)
    print(f"Found {len(polyhexes)} polyhexes", file=sys.stderr)
    
    # Score each polyhex: compute enclosing side (aligned, angle=0)
    scored = []
    for ph in polyhexes:
        cells = list(ph)
        # Centres in Cartesian (scale=1 → adjacent centres 2*sqrt(3)/... 
        # actually for unit hexagons on a hex lattice the centre distance is
        # 2*sqrt(3)/sqrt(3) = 2?  No: for pointy-top, adjacent centre dist =
        # sqrt(3) * scale.  With scale=1, distance = sqrt(3) ≈ 1.732.
        # But unit hexagons have inradius sqrt(3)/2, so adjacent centres at
        # distance sqrt(3) just touch.  Good.
        pts = np.array([_axial_to_cart(q, r, scale=1.0) for q, r in cells])
        hull = convex_hull_2d(pts)
        S = outer_side_from_hull(hull, 0.0)
        scored.append((S, ph, pts))
    
    scored.sort(key=lambda x: x[0])
    print(f"Best polyhex side: {scored[0][0]:.6f}", file=sys.stderr)
    
    # Keep top-K distinct starting points
    K = min(20, len(scored))
    top = scored[:K]
    
    best_S = float('inf')
    best_result = None
    
    def _try(params):
        nonlocal best_S, best_result
        n = N
        inner = []
        for i in range(n):
            inner.append((params[3*i], params[3*i+1],
                          math.degrees(params[3*i+2]) % 360.0))
        ox, oy = params[3*n], params[3*n+1]
        oa = math.degrees(params[3*n+2]) % 360.0
        # Compute exact (non-smooth) side
        all_pts = []
        for cx, cy, a in inner:
            all_pts.append(hex_vertices(cx, cy, 1.0, math.radians(a)))
        all_pts = np.vstack(all_pts)
        hull = convex_hull_2d(all_pts)
        S = outer_side_from_hull(hull, oa)
        if _check_valid(inner, ox, oy, S, oa):
            if S < best_S:
                best_S = S
                best_result = (inner, (ox, oy), S, oa)
                print(f"  New best: {S:.8f}", file=sys.stderr)
    
    # ---- Phase 2: refine each top polyhex --------------------------------
    for rank, (S0, ph, pts) in enumerate(top):
        if time.time() > deadline - 30:
            break
        
        # Build initial params
        params = np.zeros(3*N + 3)
        # Center the polyhex at origin
        cx_mean = pts[:, 0].mean()
        cy_mean = pts[:, 1].mean()
        for i in range(N):
            params[3*i]   = pts[i, 0] - cx_mean
            params[3*i+1] = pts[i, 1] - cy_mean
            params[3*i+2] = 0.0  # aligned
        params[3*N]   = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        
        # Quick eval of initial
        _try(params)
        
        # L-BFGS-B refinement with smooth objective
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14,
                                    'gtol': 1e-12, 'maxls': 50})
            if res.fun < float('inf'):
                _try(res.x)
        except Exception as e:
            print(f"  L-BFGS-B failed for rank {rank}: {e}", file=sys.stderr)
        
        # Nelder-Mead polish on outer params only (if time)
        if best_result is not None and time.time() < deadline - 15:
            try:
                inner_fixed = best_result[0]
                def outer_obj(p):
                    return _smooth_outer_side(np.concatenate([
                        np.array([x, y, math.radians(a)] for x, y, a in inner_fixed),
                        p
                    ]))
                p0 = np.array([best_result[1][0], best_result[1][1],
                               math.radians(best_result[3])])
                res2 = minimize(outer_obj, p0, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-10,
                                         'fatol': 1e-12})
                if res2.fun < float('inf'):
                    _try(np.concatenate([
                        np.array([x, y, math.radians(a)] for x, y, a in inner_fixed),
                        res2.x
                    ]))
            except Exception:
                pass
    
    # ---- Phase 3: global DE fallback if time remains ---------------------
    if time.time() < deadline - 25:
        try:
            from scipy.optimize import differential_evolution
            bounds = []
            for i in range(N):
                bounds.append((-6, 6))
                bounds.append((-6, 6))
                bounds.append((0, 2*math.pi))
            bounds.append((-3, 3))
            bounds.append((-3, 3))
            bounds.append((0, 2*math.pi))
            
            def de_obj(p):
                if time.time() > deadline:
                    return 1e10
                return _objective(p)
            
            de_res = differential_evolution(
                de_obj, bounds, maxiter=40, popsize=12,
                mutation=(0.4, 1.0), recombination=0.8,
                seed=42, polish=False, tol=1e-5,
                update_interval=3,
                callback=lambda x, conv: time.time() > deadline - 10)
            if de_res.fun < float('inf'):
                _try(de_res.x)
        except Exception as e:
            print(f"  DE failed: {e}", file=sys.stderr)
    
    # ---- Fallback ---------------------------------------------------------
    if best_result is None:
        # Simple hex lattice
        cells = [(0,0)]
        for k in range(6):
            cells.append((1,0) if k==0 else (0,1) if k==1 else (-1,1) if k==2
                         else (-1,0) if k==3 else (0,-1) if k==4 else (1,-1))
        for k in range(6):
            cells.append((2,-1) if k==0 else (1,1) if k==1 else (-2,1) if k==2
                         else (-1,-1) if k==3 else (-2,0) if k==4 else (0,-2))
        cells = cells[:N]
        pts = np.array([_axial_to_cart(q,r) for q,r in cells])
        inner = [(x, y, 0.0) for x, y in pts]
        hull = convex_hull_2d(pts)
        S = outer_side_from_hull(hull, 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s, best S = {best_S:.8f}", file=sys.stderr)
    return best_result
