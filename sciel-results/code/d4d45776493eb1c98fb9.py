"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate candidate axial coordinates
    # We look for compact sets of 11 hexagons.
    # Base: Hexagon of side 2 (7 hexes).
    # Add 4 from the next ring.
    
    # Axial coords for side 2 hexagon
    base_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # Next ring coords (distance 2 from center in grid steps, but physically further)
    # Actually, let's just generate all coords in a larger radius and pick subsets.
    # Radius 3 in axial coords covers all candidates for tight packing.
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Hex distance from origin
            dist = (abs(q) + abs(r) + abs(-q-r)) / 2
            if dist <= 3:
                candidates.append((q, r))
    
    # We want subsets of size 11.
    # To ensure compactness, we require that the set contains the center (0,0)
    # and is "connected" in the hex grid.
    # For speed, we just check bounding box in axial coords.
    
    best_side = float('inf')
    best_inner = []
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    # Pre-compute cartesian coords for all candidates
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    cart_coords = []
    for q, r in candidates:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        cart_coords.append((x, y))
    
    # Map axial to index
    axial_to_idx = { (q,r): i for i, (q,r) in enumerate(candidates) }
    
    # Generate combinations
    # This might be too slow if we do all C(37, 11).
    # 37 choose 11 is huge.
    # We must prune.
    
    # Strategy: Start with the 7 center ones. Add 4 from the ring of 12 (dist=2).
    # The 7 center ones are dist <= 1.
    # The ring of 12 are dist == 2.
    # There are 12 hexes at dist 2.
    # C(12, 4) = 495 combinations.
    # This is very fast.
    
    center_set = set([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)])
    ring2_coords = [c for c in candidates if (abs(c[0]) + abs(c[1]) + abs(-c[0]-c[1])) // 2 == 2]
    
    from itertools import combinations
    
    best_candidate_indices = None
    min_side_est = float('inf')
    
    # Evaluate all 495 combinations
    for combo in combinations(range(len(ring2_coords)), 4):
        if time.time() > deadline - 10:
            break
            
        selected_axial = list(center_set) + [ring2_coords[i] for i in combo]
        
        # Check if all are in candidates (they are)
        # Calculate bounding box in cartesian to estimate side length
        xs = [cart_coords[axial_to_idx[ax]][0] for ax in selected_axial]
        ys = [cart_coords[axial_to_idx[ax]][1] for ax in selected_axial]
        
        # Rough estimate: max range
        # A hexagon of side S has width 2*S (flat-to-flat) and 2*S*sqrt(3)/2?
        # Width (vertex to vertex) = 2S.
        # Inradius R = S * sqrt(3) / 2.
        # So S = 2R / sqrt(3).
        # Max distance from center to any vertex of any inner hex.
        
        # For a quick estimate, use max projection on 6 axes.
        # We can just compute the exact side length for this fixed layout (angle 0)
        # and optimize the outer center/angle later.
        
        # For now, just compute min side for angle 0, center 0,0.
        # This is a lower bound for the optimal side for this shape.
        
        inner_temp = [[cart_coords[axial_to_idx[ax]][0], cart_coords[axial_to_idx[ax]][1], 0.0] for ax in selected_axial]
        
        # Center of mass
        cx = sum(p[0] for p in inner_temp) / N
        cy = sum(p[1] for p in inner_temp) / N
        
        side_est = compute_required_side(inner_temp, (cx, cy), 0.0)
        
        if side_est < min_side_est:
            min_side_est = side_est
            best_candidate_indices = [axial_to_idx[ax] for ax in selected_axial]
    
    if best_candidate_indices is None:
        # Fallback to simple grid
        best_candidate_indices = [axial_to_idx[ax] for ax in list(center_set)[:7]]
        # Add 4 more arbitrarily
        for c in ring2_coords[:4]:
            best_candidate_indices.append(axial_to_idx[c])
            
    # Get the best candidate layout
    selected_axial = [candidates[i] for i in best_candidate_indices]
    initial_inner = [[cart_coords[axial_to_idx[ax]][0], cart_coords[axial_to_idx[ax]][1], 0.0] for ax in selected_axial]
    
    # Optimize outer container for this layout
    cx_avg = sum(c[0] for c in initial_inner) / len(initial_inner)
    cy_avg = sum(c[1] for c in initial_inner) / len(initial_inner)
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
    
    best_val = float('inf')
    best_params = None
    
    # Try a few angles
    for init_angle in [0, 15, 30, 45, 60, 75, 90]:
        if time.time() > deadline - 5:
            break
        init_params = [cx_avg, cy_avg, float(init_angle)]
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is not None:
        best_outer_center = (best_params[0], best_params[1])
        best_outer_angle = best_params[2]
        best_side = best_val
        best_inner = initial_inner
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
        best_inner = initial_inner

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
