import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, angle_rad, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    # Pre-computed offsets for unit hexagon (side=1)
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # Radius (center to vertex) = side
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + angle_rad
    return np.column_stack((cx + side * np.cos(angles), cy + side * np.sin(angles)))

def _get_normals(verts):
    """Get edge normals for a convex polygon (verts in CCW order)."""
    edges = np.roll(verts, -1, axis=0) - verts
    # Normal pointing outward (for CCW polygon, right normal is outward)
    # Edge vector (dx, dy) -> Normal (dy, -dx)
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    return normals / norms

def _check_overlap_fast(v1, v2, n1, n2):
    """
    Fast SAT check. v1, v2 are (6,2) arrays.
    n1, n2 are precomputed normals (6,2) arrays.
    Returns True if overlap.
    """
    # Project v1 onto n1 axes
    proj1_n1 = v1 @ n1.T
    min1_n1 = proj1_n1.min(axis=0)
    max1_n1 = proj1_n1.max(axis=0)
    
    # Project v2 onto n1 axes
    proj2_n1 = v2 @ n1.T
    min2_n1 = proj2_n1.min(axis=0)
    max2_n1 = proj2_n1.max(axis=0)
    
    # Check separation on n1 axes
    if np.any(max1_n1 < min2_n1 - 1e-9) or np.any(max2_n1 < min1_n1 - 1e-9):
        return False
        
    # Project v1 onto n2 axes
    proj1_n2 = v1 @ n2.T
    min1_n2 = proj1_n2.min(axis=0)
    max1_n2 = proj1_n2.max(axis=0)
    
    # Project v2 onto n2 axes
    proj2_n2 = v2 @ n2.T
    min2_n2 = proj2_n2.min(axis=0)
    max2_n2 = proj2_n2.max(axis=0)
    
    # Check separation on n2 axes
    if np.any(max1_n2 < min2_n2 - 1e-9) or np.any(max2_n2 < min1_n2 - 1e-9):
        return False
        
    return True

def _calculate_outer_side(centers, angles, outer_angle_rad):
    """
    Calculate the minimum side length of an outer hexagon centered at (0,0)
    with rotation outer_angle_rad that contains all inner hexagons.
    Inner hexagons have side 1.
    """
    # Outer hexagon face normals
    # For a regular hexagon centered at origin, the face normals are at angles:
    # outer_angle + 0, 60, 120, 180, 240, 300 degrees?
    # Let's define the outer hexagon vertices.
    # If side is S, inradius is R = S * sqrt(3) / 2.
    # The condition for a point P to be inside is P . n_i <= R for all face normals n_i.
    # Face normals for a hexagon with "flat top" (vertex at 0 deg) are at 30, 90, 150...
    # Wait, standard hexagon vertices at 0, 60... -> Edges are at 30, 90... -> Normals at 30, 90...
    # Let's just use the 6 directions separated by 60 degrees.
    # The specific orientation matters for the "tightness".
    # We will optimize outer_angle_rad as well.
    
    outer_normals_angles = outer_angle_rad + np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    # Note: The normals of the outer hexagon are perpendicular to its edges.
    # If we define outer hexagon vertices at angles theta + k*pi/3, 
    # then edges are perpendicular to normals at theta + pi/6 + k*pi/3?
    # Let's stick to: A hexagon defined by 6 half-planes n_i . x <= R.
    # The normal vectors n_i are separated by 60 degrees.
    # The shape is a regular hexagon.
    # Side length S = 2 * R / sqrt(3).
    
    n_outer = np.column_stack((np.cos(outer_normals_angles), np.sin(outer_normals_angles)))
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        # Project all 6 vertices of inner hex i onto 6 outer normals
        projs = verts @ n_outer.T
        # We need max over all vertices and all outer normals
        # projs is (6, 6)
        max_val = projs.max()
        if max_val > max_proj:
            max_proj = max_val
            
    inradius = max_proj
    side = 2.0 * inradius / np.sqrt(3)
    return side

def _evaluate(params):
    """
    params: (N, 3) array of [x, y, angle] for each inner hex.
    plus outer_angle.
    Returns (outer_side, is_valid).
    """
    # Extract params
    # We will assume centers are centered around (0,0) for the outer hex.
    # Actually, we should allow outer center to move? 
    # Symmetry suggests centering is optimal, but let's allow it for robustness?
    # For now, fix outer center at (0,0) and optimize outer angle.
    # This reduces dimensionality. If it's not optimal, we can add 2 more dims.
    
    centers = params[:, :2]
    angles = params[:, 2]
    outer_angle = params[-1] # Last element is outer angle
    
    # 1. Check Overlaps
    verts_list = []
    normals_list = []
    for i in range(N):
        v = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        verts_list.append(v)
        normals_list.append(_get_normals(v))
        
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap_fast(verts_list[i], verts_list[j], normals_list[i], normals_list[j]):
                return np.inf, False
                
    # 2. Calculate Outer Side
    side = _calculate_outer_side(centers, angles, outer_angle)
    return side, True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial Configuration: Hexagonal Lattice Cluster
    # 12 hexagons. 
    # A compact shape is roughly 4x3 or similar.
    # Let's use a "flower" shape: 1 center + 6 ring + 5 more?
    # Or just a rectangular-ish grid on hex lattice.
    # Hex lattice spacing: dx = 1.5, dy = sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Generate a grid of candidates and pick 12 closest to center?
    # Or hardcode a good shape.
    # Shape:
    #   . X X X .
    #   X X X X X
    #   X X X X X
    #   . X X X .
    # Total 14? No.
    # Let's try:
    # Row 0: 3
    # Row 1: 4
    # Row 2: 4
    # Row 3: 3
    # Total 14. Too many.
    
    # How about:
    # Row 0: 2
    # Row 1: 4
    # Row 2: 4
    # Row 3: 2
    # Total 12.
    
    axial_coords = []
    # Row r=0 (y=0): q=-1, 0, 1, 2 -> 4 hexes?
    # Let's use axial coordinates (q, r)
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Target 12 hexes.
    # Center at (0,0) in axial?
    # Let's just generate a grid of 5x5 axial coords and take the 12 with smallest radius.
    axial_candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            dist = x**2 + y**2
            axial_candidates.append((dist, q, r, x, y))
            
    axial_candidates.sort(key=lambda x: x[0])
    chosen = axial_candidates[:12]
    
    init_params = np.zeros((N+1, 3)) # N rows for inner, 1 row for outer angle (x,y unused for outer, just angle in 3rd col? No, let's make it (N,3) + [outer_angle])
    # Let's use a flat array for optimization: [x0, y0, a0, ..., x11, y11, a11, outer_angle]
    
    init_flat = []
    for _, q, r, x, y in chosen:
        init_flat.extend([x, y, 0.0])
    init_flat.append(0.0) # outer angle
    
    current_params = np.array(init_flat)
    current_side, current_valid = _evaluate(current_params)
    
    # If initial is invalid (overlapping), which it shouldn't be on a lattice, but check.
    # Lattice hexes with side 1 do not overlap if centers are distance > 2*inradius?
    # Inradius = sqrt(3)/2 ~ 0.866.
    # Distance between adjacent centers in hex lattice (side 1):
    # dx=1.5, dy=sqrt(3)/2? No.
    # Axial (1,0) -> x=1.5, y=sqrt(3)/2. Dist = sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732.
    # Sum of inradii = 2 * 0.866 = 1.732.
    # So they touch exactly. This is valid (disjoint interiors).
    
    if not current_valid:
        # Perturb slightly to break exact touching if SAT is too strict?
        # Or just accept inf and let SA fix it?
        # Better to ensure initial is valid.
        # Add tiny noise
        current_params[2::3] += np.random.normal(0, 0.01, N) # Angles
        current_side, current_valid = _evaluate(current_params)
        if not current_valid:
             # Fallback: spread them out
             for i in range(N):
                 current_params[3*i] += np.random.normal(0, 0.1)
                 current_params[3*i+1] += np.random.normal(0, 0.1)
             current_side, current_valid = _evaluate(current_params)

    best_params = current_params.copy()
    best_side = current_side
    if current_valid:
         best_side = current_side
    else:
         best_side = np.inf
         
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    
    # Step size
    step_size = 0.5
    
    while T > min_T and time.time() < deadline:
        # Generate neighbor
        new_params = current_params.copy()
        
        # Move a random inner hex
        i = np.random.randint(0, N)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        da = np.random.normal(0, step_size * np.pi / 180) # Small angle change
        
        new_params[3*i] += dx
        new_params[3*i+1] += dy
        new_params[3*i+2] += da
        
        # Occasionally move outer angle
        if np.random.random() < 0.1:
            new_params[-1] += np.random.normal(0, step_size * np.pi / 180)
            
        new_side, new_valid = _evaluate(new_params)
        
        if not new_valid:
            new_side = np.inf
            
        # Calculate acceptance
        if new_side < best_side:
            best_side = new_side
            best_params = new_params.copy()
            
        if new_side < current_side:
            current_params = new_params
            current_side = new_side
        else:
            # Accept with probability exp(-delta / T)
            delta = new_side - current_side
            if delta < 0:
                accept = True
            else:
                # If current_side is inf, we must accept anything finite
                if np.isinf(current_side):
                    accept = True
                else:
                    accept_prob = np.exp(-delta / T)
                    accept = np.random.random() < accept_prob
                    
            if accept:
                current_params = new_params
                current_side = new_side
            # else reject, keep current_params
        
        T *= cooling
        
        # Adaptive step size?
        # If we accept a lot, increase step. If we reject a lot, decrease?
        # Keep it simple for now.
        
    # Final polish with local search?
    # The SA should be good enough.
    
    # Extract result
    bx = best_params
    inner = []
    for i in range(N):
        inner.append([float(bx[3*i]), float(bx[3*i+1]), float(np.degrees(bx[3*i+2]))])
    
    side = float(best_side)
    outer_angle = float(np.degrees(bx[-1]))
    
    # If best_side is inf (invalid), return something reasonable
    if np.isinf(side):
         # Return the last valid one if we stored it? 
         # We didn't store last_valid separately from best.
         # If best is inf, it means we never found a valid one.
         # This should not happen with lattice start.
         side = 10.0 # Fallback large value
        
    return inner, (0.0, 0.0), side, outer_angle

# EVOLVE-BLOCK-END
