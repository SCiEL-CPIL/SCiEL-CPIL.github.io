"""Combined program: Smoothed penalty optimization for hexagon packing."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _softplus(x, tau=0.1):
    """Smooth approximation of max(0, x)."""
    # For numerical stability
    x = x / tau
    # Clip x to prevent overflow in exp
    x = np.clip(x, -50, 50)
    return tau * np.log1p(np.exp(x))

def _containment_penalty_smooth(inner_verts, outer_verts, tau=0.1):
    """
    Smooth penalty for inner hexagon not being contained in outer hexagon.
    For each vertex of inner, check signed distance to each edge of outer.
    If vertex is outside (signed dist < 0), add softplus(-dist).
    """
    n_out = len(outer_verts)
    penalty = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            # Inward normal for CCW polygon: (-dy, dx) / len
            nrm = np.array([-edge[1], edge[0]]) / el
            # Signed distance from iv to line passing through v1
            sd = np.dot(iv - v1, nrm)
            # If sd < 0, iv is outside. Violation is -sd.
            # We want to penalize if sd < 0.
            # So we compute softplus(-sd)
            penalty += _softplus(-sd, tau)
    return penalty

def _sat_overlap_depth(A, B):
    """
    Calculate minimum penetration depth between two convex polygons A and B.
    Returns 0 if disjoint, > 0 if overlapping.
    """
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _disjointness_penalty_smooth(inner_verts_list, tau=0.1):
    """
    Smooth penalty for overlapping inner hexagons.
    Sum of softplus(overlap_depth) for all pairs.
    """
    penalty = 0.0
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                penalty += _softplus(depth, tau)
    return penalty

def _objective(params, S, center, outer_angle, tau=0.1):
    """
    Total smooth penalty: containment + disjointness.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    """
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    pen_cont = 0.0
    for iv in inner_list:
        pen_cont += _containment_penalty_smooth(iv, outer, tau)
        
    pen_disj = _disjointness_penalty_smooth(inner_list, tau)
    
    return pen_cont + pen_disj

def _make_grid(nrows, ncols):
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    xs = np.linspace(-S*0.7, S*0.7, N)
    return np.column_stack([xs, np.zeros(N)])

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin for 120s limit

    best_S = float('inf')
    best_params = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Initial configurations
    inits = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_linear_init(4.0))
    
    rng = np.random.default_rng(42)
    for _ in range(3):
        r = rng.uniform(0, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    outer_angles = [0.0, 30.0]
    
    # Strategy: Binary search / Shrink
    # Start with a large S to ensure feasibility
    S_start = 5.0
    S_min = 3.0
    S_step = 0.1
    
    # First, find a valid solution for a large S to get a good starting point
    # Then shrink S iteratively
    
    current_S = S_start
    current_params = None
    
    # Try to find a valid packing for S_start with different inits
    valid_init_found = False
    for init_pts in inits:
        if time.time() > DEADLINE - 20:
            break
        init_params = np.zeros(3 * N)
        for i in range(N):
            init_params[3*i] = init_pts[i, 0]
            init_params[3*i+1] = init_pts[i, 1]
            init_params[3*i+2] = 0.0
            
        def obj(p, S=current_S, ctr=(0,0), oa=0.0):
            return _objective(p, S, ctr, oa, tau=0.1)
            
        try:
            res = minimize(obj, init_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e-5:
                current_params = res.x
                valid_init_found = True
                break
        except Exception:
            pass
            
    if not valid_init_found:
        # Fallback if we can't even find a solution for S=5
        # Return a safe, non-optimal but valid grid in a large hex
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    # Now shrink S
    # We try to shrink S by small amounts. If optimization fails to find a valid solution, we stop.
    # We use the previous best params as initial guess for the next smaller S.
    
    # To improve search, we can try multiple outer angles and centers in the shrinking phase
    # But for simplicity and speed, we'll stick to center (0,0) and try angles 0 and 30.
    
    # Let's do a more aggressive search:
    # For a range of S, try to optimize.
    
    S_current = current_S
    # We found a valid solution for S_current.
    best_S = S_current
    best_params = current_params.copy()
    
    # Shrink loop
    while S_current > S_min and time.time() < DEADLINE - 10:
        S_try = S_current - 0.05 # Small step
        
        # Try to optimize for S_try using current_params as init
        success = False
        for oa in outer_angles:
            if time.time() > DEADLINE - 5:
                break
                
            def obj_shrink(p, S=S_try, ctr=(0,0), oa=oa):
                return _objective(p, S, ctr, oa, tau=0.05) # Tighter tau for final convergence
                
            try:
                res = minimize(obj_shrink, current_params, method='L-BFGS-B',
                               options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-9})
                if res.fun < 1e-6:
                    # Valid solution found for smaller S
                    S_current = S_try
                    current_params = res.x
                    success = True
                    best_S = S_try
                    best_params = res.x.copy()
                    best_angle = oa
                    break
            except Exception:
                pass
                
        if not success:
            # Try a slightly larger step? Or just stop?
            # Let's try to restart from a different init for this S_try to see if we missed a local min
            # This is expensive, so only do it if we have time
            if time.time() < DEADLINE - 20:
                for init_pts in inits[:2]: # Try a couple of inits
                    if time.time() > DEADLINE - 5:
                        break
                    init_params = np.zeros(3 * N)
                    for i in range(N):
                        init_params[3*i] = init_pts[i, 0]
                        init_params[3*i+1] = init_pts[i, 1]
                        init_params[3*i+2] = 0.0
                        
                    def obj_restart(p, S=S_try, ctr=(0,0), oa=0.0):
                        return _objective(p, S, ctr, oa, tau=0.1)
                        
                    try:
                        res = minimize(obj_restart, init_params, method='L-BFGS-B',
                                       options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
                        if res.fun < 1e-5:
                            S_current = S_try
                            current_params = res.x
                            best_S = S_try
                            best_params = res.x.copy()
                            best_angle = 0.0
                            # Continue shrinking
                            break
                    except Exception:
                        pass
            # If still no success, stop shrinking
            break

    # Final polish: Try to optimize the best_S found with very tight tolerances
    if time.time() < DEADLINE - 2:
        def obj_polish(p, S=best_S, ctr=(0,0), oa=best_angle):
            return _objective(p, S, ctr, oa, tau=0.01)
            
        try:
            res = minimize(obj_polish, best_params, method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-16, 'gtol': 1e-10})
            if res.fun < 1e-8:
                best_params = res.x
        except Exception:
            pass

    inner = [[float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), float(best_angle)

# EVOLVE-BLOCK-END
