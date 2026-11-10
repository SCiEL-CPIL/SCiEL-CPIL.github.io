# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0
_SAFETY = 30.0

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _validate(inner, outer, side):
    # inner: (11,3) (x,y,angle_deg); outer: (3,) (x,y,angle_deg)
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
        
    vs = []
    for (x, y, ang) in inner:
        vs.append(_hex_vertices(x, y, 1.0, ang))
        
    # Pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            # Check gap
            # For each of 12 normals, check if projected intervals are disjoint
            # v1, v2 are (6,2)
            v1 = vs[i]
            v2 = vs[j]
            
            # Normals of v1
            e1 = np.roll(v1, -1, axis=0) - v1
            n1 = np.column_stack([e1[:, 1], -e1[:, 0]])
            n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-30)
            
            # Normals of v2
            e2 = np.roll(v2, -1, axis=0) - v2
            n2 = np.column_stack([e2[:, 1], -e2[:, 0]])
            n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-30)
            
            found_separation = False
            for n in np.vstack([n1, n2]):
                p1 = v1 @ n
                p2 = v2 @ n
                gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                if gap > 1e-6:
                    found_separation = True
                    break
            
            if not found_separation:
                return False
                
    # Containment
    ox, oy, oang = outer
    ov = _hex_vertices(ox, oy, side, oang)
    
    # Check all inner vertices inside outer
    # Outer edges
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        # Cross product for all inner vertices
        for v in vs:
            cross = (p2[0] - p1[0]) * (v[:, 1] - p1[1]) - (p2[1] - p1[1]) * (v[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
                
    # Edge lengths
    for v in vs:
        e = np.roll(v, -1, axis=0) - v
        L = np.linalg.norm(e, axis=1)
        if np.any(np.abs(L - 1.0) > 1e-8):
            return False
            
    return True

def _get_side_and_center(positions, angles, theta_deg):
    """
    positions: (11, 2)
    angles: (11,)
    theta_deg: outer hex orientation
    Returns: side, center (2,)
    """
    # Get all vertices
    all_verts = []
    for i in range(11):
        all_verts.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    all_verts = np.vstack(all_verts)
    
    # Normals for outer hex orientation
    a = np.deg2rad(theta_deg)
    n_angles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    # Project all vertices onto normals
    projs = all_verts @ N.T # (M, 6)
    
    M = projs.max(axis=0)
    m = projs.min(axis=0)
    
    widths = M - m
    # s * sqrt(3) >= width_k
    s = np.max(widths) / np.sqrt(3.0)
    
    # Find center
    # c . n_k must be in [m_k + s*sqrt(3)/2, M_k - s*sqrt(3)/2]
    # Let mid_k = (M_k + m_k) / 2
    # We want c such that c . n_k = mid_k for the tightest constraints.
    # A good approximation is the centroid of the vertices, but we need to ensure validity.
    # If we just return the centroid, it might be invalid if the cluster is skewed.
    # However, for the score, only 'side' matters.
    # But we must return a valid (c, s).
    # Let's try to find a valid c.
    # If we set c to the mean of the midpoints in the dual space?
    # Simple approach: Use the centroid of all vertices.
    c = np.mean(all_verts, axis=0)
    
    # Verify if this c works. If not, we might need to adjust.
    # Given the symmetry of the problem, the optimal c is likely close to the centroid.
    # If the returned hexagon is invalid, the validator will fail.
    # To be safe, we can solve for c using linear programming or iterative projection.
    # For now, centroid is a good bet for symmetric clusters.
    
    return s, c

def hexagon_packing_11():
    start_time = time.time()
    
    # Initial positions: Flower + 4
    d = np.sqrt(3.0)
    pos = np.zeros((11, 2))
    pos[0] = [0.0, 0.0]
    for k in range(6):
        ang = np.deg2rad(k * 60.0)
        pos[k + 1] = [d * np.cos(ang), d * np.sin(ang)]
    
    # 4 extra: Place at 2nd ring
    # Directions 30, 150, 270, 30? No.
    # Let's place them at angles 0, 120, 240, 300 at distance 2*d?
    # Or 60, 180, 300, ?
    # Let's try 4 spots at 120 deg intervals?
    # 0, 120, 240, 300 is 4 spots.
    extra_angles = [0, 120, 240, 300]
    for k, ang_deg in enumerate(extra_angles):
        ang = np.deg2rad(ang_deg)
        pos[7 + k] = [2 * d * np.cos(ang), 2 * d * np.sin(ang)]
        
    angles = np.zeros(11)
    
    best_side = np.inf
    best_inner = None
    best_outer = None
    
    # Try a few orientations
    for theta in [0.0, 30.0, 15.0, 45.0]:
        if time.time() - start_time > _DEADLINE - _SAFETY:
            break
            
        s, c = _get_side_and_center(pos, angles, theta)
        inner = np.column_stack([pos, angles])
        outer = np.array([c[0], c[1], theta])
        
        if _validate(inner, outer, s):
            if s < best_side:
                best_side = s
                best_inner = inner.copy()
                best_outer = outer.copy()
                
    # Optimization
    if time.time() - start_time < _DEADLINE - _SAFETY:
        def objective(x):
            p = x[:22].reshape(11, 2)
            a = x[22:33]
            # We fix theta to 0 for optimization simplicity, or include it?
            # Let's include theta as a variable? No, 34 vars is too many for NM.
            # Let's fix theta=0.
            s, c = _get_side_and_center(p, a, 0.0)
            
            # Penalty for overlaps
            pen = 0.0
            all_verts = []
            for i in range(11):
                all_verts.append(_hex_vertices(p[i, 0], p[i, 1], 1.0, a[i]))
            all_verts = np.vstack(all_verts)
            
            for i in range(11):
                for j in range(i + 1, 11):
                    v1 = all_verts[i*6:(i+1)*6]
                    v2 = all_verts[j*6:(j+1)*6]
                    
                    e1 = np.roll(v1, -1, axis=0) - v1
                    n1 = np.column_stack([e1[:, 1], -e1[:, 0]])
                    n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-30)
                    
                    e2 = np.roll(v2, -1, axis=0) - v2
                    n2 = np.column_stack([e2[:, 1], -e2[:, 0]])
                    n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-30)
                    
                    sep = False
                    for n in np.vstack([n1, n2]):
                        p1 = v1 @ n
                        p2 = v2 @ n
                        gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                        if gap > 1e-6:
                            sep = True
                            break
                    if not sep:
                        # Calculate min gap to make penalty smooth
                        min_gap = np.inf
                        for n in np.vstack([n1, n2]):
                            p1 = v1 @ n
                            p2 = v2 @ n
                            gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                            if gap < min_gap:
                                min_gap = gap
                        pen += (1e-6 - min_gap) ** 2 * 1000
            
            return s + pen
        
        x0 = np.concatenate([pos.flatten(), angles])
        
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_side:
            p = res.x[:22].reshape(11, 2)
            a = res.x[22:33]
            s, c = _get_side_and_center(p, a, 0.0)
            inner = np.column_stack([p, a])
            outer = np.array([c[0], c[1], 0.0])
            
            if _validate(inner, outer, s):
                if s < best_side:
                    best_side = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
                    
    if best_inner is None:
        # Fallback
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ])
        outer = np.array([0, 0, 0])
        return inner, outer, 8.0
        
    return best_inner, best_outer, float(best_side)

# EVOLVE-BLOCK-END
