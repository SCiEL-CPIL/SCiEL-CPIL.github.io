"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _hexagon_containment_penalty(inner_verts, outer_verts):
    """Check if inner hexagon is contained in outer hexagon.
    Returns penalty > 0 if violated, 0 if contained."""
    # For each vertex of inner, check it's inside outer
    # Outer hexagon: check each edge
    n_out = len(outer_verts)
    penalty = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            # inward normal (for CCW polygon, left of edge is inside)
            # cross product: if point is to the right of edge, it's outside
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                # distance violation
                dist = np.linalg.norm(edge)
                if dist > 1e-12:
                    penalty += (-cross) / dist
    return penalty

def _hex_sat_overlap_depth(h1, h2):
    """
    Calculate the minimum penetration depth between two convex polygons h1 and h2.
    Returns 0 if disjoint, > 0 if overlapping.
    """
    min_depth = float('inf')
    
    # Get all edge normals (axes) from both polygons
    axes = []
    for poly in [h1, h2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            # Normal to the edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
    
    # Check each axis
    for axis in axes:
        # Project vertices of h1 onto axis
        proj1 = np.dot(h1, axis)
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project vertices of h2 onto axis
        proj2 = np.dot(h2, axis)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0 # Disjoint
        
        # Calculate overlap depth on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def _disjointness_penalty(inner_verts_list):
    """Penalty for overlapping inner hexagons using SAT."""
    n = len(inner_verts_list)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            depth = _hex_sat_overlap_depth(vi, vj)
            if depth > 0:
                penalty += depth
    return penalty

def _compute_objective(params, outer_side, outer_center, outer_angle):
    """Compute total penalty: containment + disjointness."""
    # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment penalty
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _hexagon_containment_penalty(iv, outer_verts)
    
    # Disjointness penalty
    disj_pen = _disjointness_penalty(inner_verts_list)
    
    return cont_pen + disj_pen

def _optimize_from_init(init_params, outer_side, outer_center, outer_angle, deadline):
    """Run L-BFGS-B optimization."""
    def objective(params):
        return _compute_objective(params, outer_side, outer_center, outer_angle)
    
    try:
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x, result.fun
    except Exception:
        return init_params, objective(init_params)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s limit
    
    best_score = float('inf')
    best_result = None
    
    def make_hex_grid(nrows, ncols):
        """Create hexagonal grid positions."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    # Try several initial configurations
    configs = []
    
    # 3x4 hex grid
    pos34 = make_hex_grid(3, 4)
    # 4x3 hex grid
    pos43 = make_hex_grid(4, 3)
    # Ring + center
    ring = []
    center_pos = np.array([0.0, 0.0])
    for i in range(6):
        angle = np.radians(60 * i)
        ring.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    for i in range(5):
        angle = np.radians(60 * i + 30)
        ring.append((3.0 * np.cos(angle), 3.0 * np.sin(angle)))
    all_pos = np.array(ring + [center_pos])
    centroid = all_pos.mean(axis=0)
    centered_ring = (all_pos - centroid).tolist()
    configs.append(centered_ring)
    
    for pos in [pos34, pos43]:
        if len(pos) == N:
            arr = np.array(pos)
            centroid = arr.mean(axis=0)
            centered = arr - centroid
            configs.append(centered.tolist())
    
    # Try different outer side lengths
    # Start with a slightly larger side to ensure fit, then shrink
    outer_sides = [4.5, 4.0, 3.8]
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        for outer_side in outer_sides:
            if time.time() > deadline - 10.0:
                break
            
            outer_center = (0.0, 0.0)
            
            for config in configs:
                if time.time() > deadline - 10.0:
                    break
                
                # Initialize params: x, y, angle for each hex
                init_params = []
                for (x, y) in config:
                    init_params.extend([x, y, 0.0])
                init_params = np.array(init_params)
                
                # Run optimization
                opt_params, opt_val = _optimize_from_init(
                    init_params, outer_side, outer_center, outer_angle, deadline
                )
                
                # Check if solution is valid (penalty ≈ 0)
                if opt_val < 1e-5:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
                    
                    # Try smaller outer side
                    for smaller_side in [outer_side - 0.2, outer_side - 0.1]:
                        if smaller_side < 3.5:
                            continue
                        if time.time() > deadline - 10.0:
                            break
                        opt_params2, opt_val2 = _optimize_from_init(
                            opt_params, smaller_side, outer_center, outer_angle, deadline
                        )
                        if opt_val2 < 1e-5:
                            if smaller_side < best_score:
                                best_score = smaller_side
                                best_result = (opt_params2, outer_center, smaller_side, outer_angle)
    
    # Fallback: safe default
    if best_result is None:
        inner = []
        pos34 = make_hex_grid(3, 4)
        arr = np.array(pos34)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation: ensure no overlaps
    # Recompute penalty for the best result
    inner_verts_list = []
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        ang = opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _hexagon_containment_penalty(iv, outer_verts)
    
    disj_pen = _disjointness_penalty(inner_verts_list)
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        # Invalid, return fallback
        inner = []
        pos34 = make_hex_grid(3, 4)
        arr = np.array(pos34)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0

    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
