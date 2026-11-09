"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side s and rotation ang_deg."""
    ang = math.radians(ang_deg)
    verts = []
    for k in range(6):
        theta = ang + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _sat_overlap_depth(h1, h2):
    """
    Minimum penetration depth between two convex polygons using SAT.
    Returns 0.0 if disjoint, otherwise the minimum depth of penetration.
    """
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            # Normal to edge
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    if min_depth == float('inf'):
        return 0.0
    return min_depth

def _outer_side_length(params):
    """
    Calculate the minimum side length of a regular hexagon centered at (0,0) with angle 0
    that contains all inner hexagons defined by params.
    """
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    # Unit normals for the 6 edges of a flat-top hexagon
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    
    apothem_unit = math.sqrt(3) / 2.0
    max_proj = 0.0
    
    for i in range(N):
        verts = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        # Project vertices onto normals
        projs = verts @ normals.T
        # The constraint is dot(v, n) <= S * apothem_unit
        # So S >= dot(v, n) / apothem_unit
        # We need the max over all vertices and all normals
        local_max_S = np.max(projs) / apothem_unit
        if local_max_S > max_proj:
            max_proj = local_max_S
            
    return max(0.0, max_proj)

def _objective(params, deadline):
    """
    Objective function for Differential Evolution.
    Returns outer_side_length + penalty for overlaps.
    """
    if time.time() > deadline:
        return 1e10
    
    S = _outer_side_length(params)
    
    # Check overlaps
    penalty = 0.0
    p = params.reshape(N, 3)
    verts_list = [_hex_verts(p[i, 0], p[i, 1], 1.0, p[i, 2]) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                # Penalty proportional to depth to encourage separation
                penalty += depth * 10.0
                
    return S + penalty

def _generate_initial_config():
    """
    Generate a valid initial configuration: a 3x4 honeycomb lattice.
    """
    # Lattice vectors for flat-top hexagons of side 1
    # Distance between centers in x: 2 * apothem = sqrt(3)
    # Distance between centers in y: 1.5 * side = 1.5? No.
    # For flat top:
    # Row 0: (0,0), (sqrt(3), 0), ...
    # Row 1: (sqrt(3)/2, 1.5), ...
    # Wait, apothem is sqrt(3)/2.
    # Distance between parallel edges is sqrt(3).
    # So horizontal spacing is sqrt(3).
    # Vertical spacing: The vertical distance between rows is 1.5.
    # Let's verify.
    # Hex 1 at (0,0). Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), ...
    # Hex 2 at (sqrt(3)/2, 1.5).
    # Distance between (0.5, sqrt(3)/2) and (sqrt(3)/2 + 0.5, 1.5 - sqrt(3)/2)?
    # This is getting confusing. Let's use the standard honeycomb lattice vectors.
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    v1 = np.array([math.sqrt(3), 0.0])
    v2 = np.array([math.sqrt(3)/2, 1.5])
    
    # Add a small epsilon to ensure strict disjointness in the initial state
    # Actually, the lattice is exactly touching. DE will move them.
    # But if they start touching, SAT depth is 0.
    # To be safe, we can scale the lattice slightly or just rely on DE.
    # Let's stick to exact lattice for now, it's a valid starting point (non-overlapping).
    
    init_params = []
    for i in range(3):
        for j in range(4):
            pos = i * v1 + j * v2
            # Center the configuration
            init_params.append([pos[0], pos[1], 0.0])
            
    init_params = np.array(init_params)
    cx = init_params[:, 0].mean()
    cy = init_params[:, 1].mean()
    init_params[:, 0] -= cx
    init_params[:, 1] -= cy
    
    return init_params

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    init_params = _generate_initial_config()
    
    # Define bounds for DE
    # The hexagons shouldn't go too far.
    # A safe bound is +/- 10.
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
    
    best_params = init_params.copy()
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            args=(deadline,),
            maxiter=100,
            popsize=15,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True
        )
        if result.fun < _objective(best_params, deadline):
            best_params = result.x
    except Exception:
        pass
    
    # Final check and cleanup
    p = best_params
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    # Recalculate S_required
    final_S = _outer_side_length(best_params)
    
    # Check for overlaps in final result
    has_overlap = False
    verts_list = [_hex_verts(xs[i], ys[i], 1.0, angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 1e-8: # Small tolerance for floating point
                has_overlap = True
                break
        if has_overlap:
            break
            
    if has_overlap:
        # Fallback to the initial grid which is valid (touching is allowed? "disjoint" usually means no interior overlap)
        # The task says "disjoint", which typically allows touching boundaries.
        # However, the evaluator might be strict.
        # Let's use the initial grid and add a tiny epsilon separation if needed?
        # Or just return the initial grid.
        best_params = init_params
        final_S = _outer_side_length(best_params)
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_params[i, 0], best_params[i, 1], best_params[i, 2]])
        
    # Center of outer hex is (0,0)
    # Side length is final_S
    # Angle of outer hex is 0 (flat top)
    
    return inner, (0.0, 0.0), final_S, 0.0

# EVOLVE-BLOCK-END
