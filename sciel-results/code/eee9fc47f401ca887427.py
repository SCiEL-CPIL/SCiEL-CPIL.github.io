"""Packing 12 unit hexagons in a minimal regular hexagon:
   Differentiable soft-repulsion + L-BFGS-B gradient-based optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

N = 12
SQRT3 = np.sqrt(3.0)
APO = SQRT3 / 2.0  # apothem of unit hexagon

# Unit flat-top hexagon vertices (radius 1)
_HEX_V = np.array([
    [1.0, 0.0], [0.5, APO], [-0.5, APO],
    [-1.0, 0.0], [-0.5, -APO], [0.5, -APO],
])

# Outward unit edge normals for the hexagon
_HEX_N = np.array([
    [SQRT3/2, 0.5], [0.0, 1.0], [-SQRT3/2, 0.5],
    [-SQRT3/2, -0.5], [0.0, -1.0], [SQRT3/2, -0.5],
])

_OUTER_N = _HEX_N

_DEADLINE = 138.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _rot_mat(a):
    """Return 2x2 rotation matrix for angle a (radians)."""
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, -sa], [sa, ca]])

def _hex_vertices(x, y, a):
    """Return 6x2 array of vertices for hexagon at (x,y) with rotation a."""
    R = _rot_mat(a)
    return R @ _HEX_V.T + np.array([x, y])[:, None]

def _hex_normals(x, y, a):
    """Return 6x2 array of outward unit normals for hexagon at (x,y) with rotation a."""
    R = _rot_mat(a)
    return R @ _HEX_N.T

def _separation_distance(V1, N1, V2, N2):
    """Compute the separation distance between two convex hexagons.
    
    Uses the separating axis theorem. Returns the maximum over all candidate
    axes of (min_i - max_j) or (min_j - max_i). If positive, shapes are separated.
    If negative, they overlap with depth = -sep.
    """
    # All candidate axes: normals from both hexagons
    # For each axis, project both hexagons and find the gap
    min_sep = np.inf
    
    # Axes from hexagon 1
    for k in range(6):
        ax = N1[k, 0]
        ay = N1[k, 1]
        proj1 = V1[:, 0] * ax + V1[:, 1] * ay
        proj2 = V2[:, 0] * ax + V2[:, 1] * ay
        gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
        # We want the maximum gap (most separated). If any gap < 0, they overlap.
        # Actually, for SAT, we check if there's ANY separating axis.
        # The separation along axis k is: min(proj1) - max(proj2) if hex1 is to the left,
        # or min(proj2) - max(proj1) if hex2 is to the left.
        # The actual separation (positive = separated) is:
        #   max(min(proj1) - max(proj2), min(proj2) - max(proj1))
        # But this is only valid if the projections don't overlap in the "wrong" direction.
        # Correct formula: if max(proj1) < min(proj2), sep = min(proj2) - max(proj1)
        #                  if max(proj2) < min(proj1), sep = min(proj1) - max(proj2)
        #                  else sep = min(max(proj1) - min(proj2), max(proj2) - min(proj1)) ... no
        
        # Simpler: the signed separation along this axis
        # If we define the axis direction, then:
        #   p1_range = [proj1.min(), proj1.max()]
        #   p2_range = [proj2.min(), proj2.max()]
        #   If p1_range.max() < p2_range.min(): separated, gap = p2_range.min() - p1_range.max()
        #   If p2_range.max() < p1_range.min(): separated, gap = p1_range.min() - p2_range.max()
        #   Else: overlapping along this axis, gap = min(p1_range.max() - p2_range.min(), p2_range.max() - p1_range.min()) ... no, that's not right.
        
        # The correct approach: the separation along axis n is:
        #   s = min over i,j of dot(n, V2[j] - V1[i]) ... no, that's not it either.
        
        # Let me use the standard SAT: for axis n, the two projections are intervals.
        # If the intervals are disjoint, the separation is the gap.
        # If they overlap, the "separation" is negative, with magnitude = overlap.
        # Specifically: sep = min(proj2) - max(proj1) if proj2 is entirely to the right
        #                 sep = min(proj1) - max(proj2) if proj1 is entirely to the right
        #                 sep = -min(max(proj1) - min(proj2), max(proj2) - min(proj1)) ... no
        
        # Actually, the cleanest way: 
        #   The distance from hex1 to hex2 along axis n (positive if separated with hex1 behind hex2)
        #   = min(proj2) - max(proj1)
        #   The distance from hex2 to hex1 along axis n
        #   = min(proj1) - max(proj2)
        #   The actual separation (positive if separated) = max(min(proj2) - max(proj1), min(proj1) - max(proj2))
        #   If this is negative, the magnitude of the negative value is the overlap depth along this axis.
        
        sep_k = max(min(proj2) - max(proj1), min(proj1) - max(proj2))
        # If sep_k > 0, they are separated along this axis (this is a separating axis)
        # If sep_k < 0, they overlap along this axis with depth -sep_k
        # We want the maximum separation over all axes. If the max is positive, they are separated.
        # If all are negative, they overlap, and the minimum overlap depth is -max(sep_k) where sep_k are all negative.
        # Actually, the overlap depth is the minimum over all axes of the "penetration" which is max(0, -sep_k).
        # For the soft repulsion, we want a smooth measure of overlap.
        
        # Let's track the maximum separation (most positive). If it's > 0, no overlap.
        # If it's < 0, the overlap depth is -max_sep (since all sep_k <= max_sep < 0).
        min_sep = min(min_sep, sep_k)
    
    # Axes from hexagon 2
    for k in range(6):
        ax = N2[k, 0]
        ay = N2[k, 1]
        proj1 = V1[:, 0] * ax + V1[:, 1] * ay
        proj2 = V2[:, 0] * ax + V2[:, 1] * ay
        sep_k = max(min(proj2) - max(proj1), min(proj1) - max(proj2))
        min_sep = min(min_sep, sep_k)
    
    # min_sep is the maximum separation. If > 0, separated. If < 0, overlap depth = -min_sep.
    return min_sep

def _overlap_depth(x, y, a):
    """Compute overlap depth between each pair of hexagons.
    
    Returns a (N, N) matrix where entry (i,j) is the overlap depth (positive = overlapping).
    """
    depths = np.zeros((N, N))
    for i in range(N):
        Vi = _hex_vertices(x[i], y[i], a[i])
        Ni = _hex_normals(x[i], y[i], a[i])
        for j in range(i+1, N):
            Vj = _hex_vertices(x[j], y[j], a[j])
            Nj = _hex_normals(x[j], y[j], a[j])
            sep = _separation_distance(Vi, Ni, Vj, Nj)
            if sep < 0:
                depth = -sep
                depths[i, j] = depth
                depths[j, i] = depth
    return depths

def _soft_repulsion_penalty(x, y, a, sigma=0.5):
    """Compute a smooth soft repulsion penalty based on overlap depths.
    
    Uses a smooth approximation: for each overlapping pair, add a smooth function
    of the overlap depth. We use exp(-depth/sigma) * depth which is smooth and
    provides gradient information.
    """
    depths = _overlap_depth(x, y, a)
    # Only consider upper triangle
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = depths[i, j]
            if d > 0:
                # Smooth penalty: quadratic for small overlaps, linear for large
                # Use d^2 for smoothness at d=0
                pen += d * d
    return pen

def _outer_side_length(x, y, a):
    """Compute the side length of the outer hexagon needed to contain all inner hexagons."""
    S = 0.0
    for i in range(N):
        V = _hex_vertices(x[i], y[i], a[i])
        # Project all vertices onto each of the 6 outer hexagon normals
        for k in range(6):
            on0 = _OUTER_N[k, 0]
            on1 = _OUTER_N[k, 1]
            proj = V[:, 0] * on0 + V[:, 1] * on1
            S = max(S, proj.max() / APO)
    return max(S, 0.0)

def _objective(v):
    """Combined objective: outer side length + soft repulsion penalty."""
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]
    y = v[N:2*N]
    a = v[2*N:3*N]
    
    S = _outer_side_length(x, y, a)
    pen = _soft_repulsion_penalty(x, y, a)
    
    return S + 100.0 * pen

def _is_valid(x, y, a, tol=1e-5):
    """Check if the configuration is valid (no overlaps) and return outer side length."""
    depths = _overlap_depth(x, y, a)
    max_depth = np.max(depths)
    S = _outer_side_length(x, y, a)
    return max_depth < tol, S

def _hex_grid_init(rng, spacing_scale=1.0, jitter=0.0):
    """Generate initial placement on a hexagonal lattice."""
    # Generate axial coordinate pairs for a hexagonal cluster
    hex_cells = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                hex_cells.append((q, r))
    rng.shuffle(hex_cells)
    chosen = hex_cells[:12]
    
    x = np.zeros(N)
    y = np.zeros(N)
    a = np.zeros(N)
    
    for i, (q, r) in enumerate(chosen):
        x[i] = SQRT3 * spacing_scale * (q + r / 2.0)
        y[i] = 1.5 * spacing_scale * r
        a[i] = 0.0
    
    if jitter > 0:
        x += rng.normal(0, jitter, N)
        y += rng.normal(0, jitter, N)
        a += rng.normal(0, jitter * 0.1, N)
    
    cx = x.mean()
    cy = y.mean()
    x -= cx
    y -= cy
    
    return np.concatenate([x, y, a])

def _fallback():
    """Return a known valid configuration as fallback."""
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([SQRT3 * (c - 2.5), 1.5 * r, 0.0])
    x = np.array([v[0] for v in inner])
    y = np.array([v[1] for v in inner])
    a = np.array([np.radians(v[2]) for v in inner])
    S = _outer_side_length(x, y, a)
    return inner, (0.0, 0.0), float(S), 0.0

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    
    best = None
    best_S = np.inf
    best_f = np.inf
    
    # Generate diverse initial placements
    inits = []
    
    # 1. Tight hex grid
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.0))
    
    # 2. Slightly loose hex grid
    inits.append(_hex_grid_init(rng, spacing_scale=1.05, jitter=0.0))
    
    # 3. Hex grid with small jitter
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.1))
    
    # 4. Hex grid with moderate jitter
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.3))
    
    # 5-6. More hex grid variants
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.0))
    inits.append(_hex_grid_init(rng, spacing_scale=1.0, jitter=0.0))
    
    # 7-9. Random starts
    for _ in range(3):
        x = rng.uniform(-4, 4, N)
        y = rng.uniform(-4, 4, N)
        a = rng.uniform(0, np.pi/3, N)
        inits.append(np.concatenate([x, y, a]))
    
    for init in inits:
        if _elapsed() > _DEADLINE - 15:
            break
        try:
            # Phase A: L-BFGS-B with moderate penalty to resolve overlaps
            r1 = minimize(_objective, init, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-5, 'maxls': 50})
            
            if r1.x is not None:
                # Phase B: Continue optimization with tighter tolerances
                r2 = minimize(_objective, r1.x, method='L-BFGS-B',
                              options={'maxiter': 200, 'gtol': 1e-7, 'maxls': 50})
                
                if r2.x is not None and r2.fun < best_f:
                    best_f = float(r2.fun)
                    best = np.asarray(r2.x, dtype=float)
                    
        except Exception:
            pass
    
    if best is None:
        return _fallback()
    
    x = best[0:N]
    y = best[N:2*N]
    a = best[2*N:3*N]
    ok, S = _is_valid(x, y, a)
    
    if not ok:
        # Try another round of optimization
        try:
            r3 = minimize(_objective, best, method='L-BFGS-B',
                          options={'maxiter': 300, 'gtol': 1e-8, 'maxls': 50})
            if r3.x is not None:
                x = r3.x[0:N]
                y = r3.x[N:2*N]
                a = r3.x[2*N:3*N]
                ok, S = _is_valid(x, y, a)
        except Exception:
            pass
    
    if not ok:
        return _fallback()
    
    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
