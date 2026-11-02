import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3
_PI = math.pi

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * _PI / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _hex_vertices_all(positions, angle_deg):
    """Get all vertices for multiple hexagons."""
    verts = []
    for px, py in positions:
        verts.append(_hex_vertices(px, py, angle_deg))
    return np.vstack(verts)

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (flat-top offset by outer_angle_deg)
    that contains all points in verts. Uses LP.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * _PI / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _min_enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast version: just compute the required side length for a given center (centroid)."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * _PI / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _compute_side_for_positions(positions, inner_angle, outer_angle):
    """Compute the enclosing hex side for given positions and angles."""
    verts = _hex_vertices_all(positions, inner_angle)
    _, _, side = _min_enclosing_hex_side_fast(verts, outer_angle)
    return side

def _overlap_penalty(positions, inner_angle, threshold=0.01):
    """
    Calculate a penalty for overlapping hexagons.
    Uses center distance as a proxy for overlap.
    Minimum distance between centers of touching unit hexagons is sqrt(3).
    If dist < sqrt(3) - threshold, penalty increases.
    """
    n = len(positions)
    penalty = 0.0
    min_dist = _SQRT3 - threshold
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                # Quadratic penalty
                diff = min_dist - dist
                penalty += 100.0 * diff * diff
    return penalty

def _generate_valid_layouts():
    """Generate a few valid, non-overlapping initial layouts."""
    layouts = []
    s3 = _SQRT3
    
    # Layout 1: Hexagonal cluster (Center + 6 neighbors + 4 from ring 2)
    pos = []
    pos.append((0, 0))
    for i in range(6):
        angle = i * _PI / 3
        x = s3 * math.cos(angle)
        y = 1.5 * math.sin(angle)
        pos.append((x, y))
    # Ring 2: Pick 4 non-adjacent to minimize extent? Or just pick 4 specific ones.
    # In a hex lattice, ring 2 has 12 positions. We need 4.
    # To keep it compact, pick positions that are "corners" of the hexagon shape.
    # Indices in ring 2 relative to center:
    # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) ... 
    # Let's use axial coordinates.
    # Center (0,0).
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # We have 6 in Ring 1. We need 4 more.
    # Let's add (2,0), (0,2), (-2,0), (0,-2) -> These are opposite pairs.
    # Convert axial (q,r) to Cartesian (x,y):
    # x = s3 * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cartesian(q, r):
        x = s3 * (q + r/2.0)
        y = 1.5 * r
        return (x, y)
        
    ring2_positions = [
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    for q, r in ring2_positions:
        pos.append(axial_to_cartesian(q, r))
    layouts.append(pos)
    
    # Layout 2: Linear row (11 hexagons in a line) - Very elongated, but valid.
    # Not good for minimizing outer side, but valid.
    pos = []
    for i in range(11):
        pos.append((i * s3, 0))
    layouts.append(pos)
    
    # Layout 3: 3x4 Grid (offset)
    # Row 1: 4
    # Row 2: 3
    # Row 3: 4
    # Total 11.
    pos = []
    # Row 1 y = -1.5
    for i in range(-1, 3):
        pos.append(axial_to_cartesian(i, -1))
    # Row 2 y = 0
    for i in range(-1, 2):
        pos.append(axial_to_cartesian(i, 0))
    # Row 3 y = 1.5
    for i in range(-1, 3):
        pos.append(axial_to_cartesian(i, 1))
    layouts.append(pos)
    
    return layouts

def _optimize_positions(initial_positions, inner_angle, outer_angle, time_limit=10.0):
    """
    Optimize positions to minimize enclosing hex side length + overlap penalty.
    Uses Nelder-Mead.
    """
    start_time = time.time()
    n = len(initial_positions)
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return float('inf')
        positions = [(x[2*i], x[2*i+1]) for i in range(n)]
        verts = _hex_vertices_all(positions, inner_angle)
        _, _, side = _min_enclosing_hex_side_fast(verts, outer_angle)
        pen = _overlap_penalty(positions, inner_angle)
        return side + pen
    
    x0 = np.array([initial_positions[i][0] for i in range(n)] + 
                  [initial_positions[i][1] for i in range(n)])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.success or res.fun < float('inf'):
            positions = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            # Verify no overlap
            if _overlap_penalty(positions, inner_angle) < 1e-4:
                return positions
    except Exception:
        pass
    
    return initial_positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 30.0
    
    # Try multiple inner angles and outer angles
    inner_angles = [0.0, 30.0, 60.0] # Keep it simple
    outer_angles = np.arange(0, 30.0, 2.0)
    
    # Initial layouts
    layouts = _generate_valid_layouts()
    
    # Try each layout with different angles
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 15.0:
                break
            
            best_local_side = float('inf')
            best_local_center = (0.0, 0.0)
            best_local_outer_angle = 0.0
            
            for outer_angle in outer_angles:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                side = _compute_side_for_positions(positions, inner_angle, outer_angle)
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = _min_enclosing_hex_side_lp(_hex_vertices_all(positions, inner_angle), outer_angle)[:2]
                    best_local_outer_angle = outer_angle
            
            if best_local_side < best_side:
                best_side = best_local_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_local_center
                best_outer_angle = best_local_outer_angle
                best_inner_angle = inner_angle
        
        # Try to optimize the best configuration for this layout
        if time.time() < _DEADLINE - 20.0:
            # Find best angles for this layout
            best_layout_side = float('inf')
            best_layout_inner_angle = 30.0
            best_layout_outer_angle = 0.0
            
            for inner_angle in inner_angles:
                for outer_angle in outer_angles:
                    side = _compute_side_for_positions(positions, inner_angle, outer_angle)
                    if side < best_layout_side:
                        best_layout_side = side
                        best_layout_inner_angle = inner_angle
                        best_layout_outer_angle = outer_angle
            
            # Optimize positions
            optimized = _optimize_positions(positions, best_layout_inner_angle, best_layout_outer_angle, time_limit=8.0)
            side = _compute_side_for_positions(optimized, best_layout_inner_angle, best_layout_outer_angle)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], best_layout_inner_angle] for pos in optimized]
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_layout_inner_angle), best_layout_outer_angle)[:2]
                best_outer_angle = best_layout_outer_angle
                best_inner_angle = best_layout_inner_angle
    
    # Refine with finer angle search
    if best_inner is not None and time.time() < _DEADLINE - 25.0:
        best_positions = [(p[0], p[1]) for p in best_inner]
        
        # Finer outer angle search
        angle_start = max(0, best_outer_angle - 2.0)
        angle_end = min(30.0, best_outer_angle + 2.0)
        
        for outer_angle in np.arange(angle_start, angle_end, 0.1):
            if time.time() > _DEADLINE - 15.0:
                break
            
            side = _compute_side_for_positions(best_positions, best_inner_angle, outer_angle)
            if side < best_side:
                best_side = side
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(best_positions, best_inner_angle), outer_angle)[:2]
                best_outer_angle = outer_angle
        
        # Try optimizing positions with the refined angles
        if time.time() < _DEADLINE - 20.0:
            optimized = _optimize_positions(best_positions, best_inner_angle, best_outer_angle, time_limit=10.0)
            side = _compute_side_for_positions(optimized, best_inner_angle, best_outer_angle)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], best_inner_angle] for pos in optimized]
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_inner_angle), best_outer_angle)[:2]
    
    # Final high-precision polish
    if best_inner is not None and time.time() < _DEADLINE - 10.0:
        best_positions = [(p[0], p[1]) for p in best_inner]
        
        # Very fine angle search
        angle_start = max(0, best_outer_angle - 0.2)
        angle_end = min(30.0, best_outer_angle + 0.2)
        
        for outer_angle in np.arange(angle_start, angle_end, 0.01):
            if time.time() > _DEADLINE - 5.0:
                break
            
            side = _compute_side_for_positions(best_positions, best_inner_angle, outer_angle)
            if side < best_side:
                best_side = side
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(best_positions, best_inner_angle), outer_angle)[:2]
                best_outer_angle = outer_angle
        
        # Final position optimization
        if time.time() < _DEADLINE - 5.0:
            optimized = _optimize_positions(best_positions, best_inner_angle, best_outer_angle, time_limit=5.0)
            side = _compute_side_for_positions(optimized, best_inner_angle, best_outer_angle)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], best_inner_angle] for pos in optimized]
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_inner_angle), best_outer_angle)[:2]
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
