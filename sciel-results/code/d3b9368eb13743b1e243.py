# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon (center, side, angle in deg)."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _separated(v1, v2, eps=1e-6):
    """Check if two convex hexagons (6 vertices each) are separated on at least one of the 12 edge normals."""
    for i in range(6):
        # Edge of v1
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Project v1 and v2 onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
            return True
        # Edge of v2
        q1 = v2[i]
        q2 = v2[(i + 1) % 6]
        edge2 = q2 - q1
        normal2 = np.array([-edge2[1], edge2[0]])
        proj1b = v1 @ normal2
        proj2b = v2 @ normal2
        if proj1b.max() < proj2b.min() - eps or proj2b.max() < proj1b.min() - eps:
            return True
    return False


def _all_separated(vertices_list, eps=1e-6):
    """Check all pairs of hexagons are separated."""
    n = len(vertices_list)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(vertices_list[i], vertices_list[j], eps):
                return False
    return True


def _minimal_enclosing_hexagon(inner_vertices_list, n_inner=11):
    """
    Find the minimal side length s and angle a such that all inner hexagon vertices
    are inside the outer regular hexagon centered at origin with side s and angle a.
    
    Uses 1D optimization over the angle a in [0, 60).
    For a fixed angle a, the minimal side length s is determined by the maximum
    projection of any inner vertex onto any of the 6 outward normals of the outer hexagon.
    """
    # Collect all inner vertices
    all_pts = np.vstack(inner_vertices_list)
    
    def min_side_for_angle(a_deg):
        a = np.radians(a_deg)
        min_s = 0.0
        for k in range(6):
            n_k = np.array([np.cos(a + np.radians(60 * k + 30)), np.sin(a + np.radians(60 * k + 30))])
            proj = all_pts @ n_k
            max_proj = proj.max()
            s_needed = max_proj / (np.sqrt(3) / 2)
            if s_needed > min_s:
                min_s = s_needed
        return max(min_s, 0.0)
    
    # Use scipy.optimize.minimize_scalar for high precision
    # The function is periodic with 60 degrees, so we search in [0, 60]
    # We might have multiple local minima, but for convex-ish packings, it's often unique or symmetric.
    # To be safe, we do a coarse grid search first to identify the best basin, then refine.
    
    best_s = np.inf
    best_a = 0.0
    
    # Coarse search to find the approximate minimum
    coarse_angles = np.linspace(0, 60, 121, endpoint=False)
    for a_deg in coarse_angles:
        s = min_side_for_angle(a_deg)
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    # Refine around the best coarse angle
    # Define a local optimization problem
    # We optimize in a window around best_a
    # Since the function is periodic, we need to handle boundaries if best_a is near 0 or 60
    # But min_side_for_angle handles any a_deg.
    
    from scipy.optimize import minimize_scalar
    
    def objective(a_deg):
        return min_side_for_angle(a_deg)
    
    # We search in a small interval around the coarse best
    # We must be careful with the bounds if best_a is close to 0 or 60
    # However, since the function is continuous and periodic, we can just search [best_a - 1, best_a + 1]
    # and take mod 60 if needed, or just let it be.
    # Actually, minimize_scalar requires bounds.
    
    # Let's just do a very fine grid search around the best coarse point, which is robust and fast enough.
    # 121 coarse points -> 1 degree resolution.
    # Fine search: 0.01 degree resolution in a 2 degree window?
    # Or use minimize_scalar with bounded method.
    
    # Let's use minimize_scalar for the final step.
    # We need to ensure the function is well-behaved.
    # We'll search in a window of +/- 1 degree around the coarse best.
    # If the coarse best is 0, the window is [-1, 1].
    
    a_lower = best_a - 1.0
    a_upper = best_a + 1.0
    
    res = minimize_scalar(objective, bounds=(a_lower, a_upper), method='bounded', options={'xatol': 1e-8})
    
    if res.success:
        best_a = res.x
        best_s = res.fun
    
    # Ensure angle is in [0, 60)
    best_a = best_a % 60
    
    return best_s, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length.
    """
    sqrt3 = np.sqrt(3)
    # Neighbor offsets for flat-top unit hexagons
    offsets = np.array([
        [1.5, sqrt3/2],
        [0, sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [0, -sqrt3],
        [1.5, -sqrt3/2]
    ])
    
    basis1 = offsets[0]
    basis2 = offsets[1]
    
    # Define several cluster shapes in axial coordinates
    # We want compact shapes.
    # Axial coordinates (q, r) where pos = q*basis1 + r*basis2
    # basis1 = (1.5, sqrt(3)/2), basis2 = (0, sqrt(3))
    
    # Standard neighbors of (0,0):
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    
    shapes = [
        # Shape 1: Center + 6 neighbors + 4 more
        # This shape is not fully symmetric but compact.
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,2), (-2,0)],
        
        # Shape 2: Another arrangement of the 4 extras
        # Place extras at (2,0), (-2,0), (0,2), (0,-2)? No, (0,2) is far.
        # Let's try placing extras to fill "corners" of the hexagonal bounding box.
        # The convex hull of the 7 central hexes is a hexagon.
        # Adding hexes at the vertices of this hull might be good.
        # Vertices of hull are near (1,0), (0,1), etc.
        # Let's try:
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)],
        
        # Shape 3: 
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (-2,1), (-2,0)],
        
        # Shape 4: 
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,-1), (1,-2), (-1,2)],
        
        # Shape 5: 
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,2), (2,-2), (-2,0)],
        
        # Shape 6: 
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,1), (1,-1), (-1,-1)] # This is invalid, duplicates
    ]
    # Fix Shape 6 to be valid and unique
    shapes[5] = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,-1), (2,0), (-2,0)]
    
    # Add a few more variations
    shapes.append([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,2), (0,-2)])
    shapes.append([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,2), (2,-2), (-2,0)])
    
    # Remove duplicates and invalid shapes (len != 11 or duplicates in coords)
    unique_shapes = []
    seen = set()
    for s in shapes:
        if len(s) != 11:
            continue
        # Check for duplicates
        if len(set(s)) != 11:
            continue
        # Canonical form for comparison (sorted tuple of tuples)
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_shapes.append(s)
    
    shapes = unique_shapes
    
    best_result = None
    best_side = np.inf
    
    # Try different expansion factors to ensure separation
    # The separation constraint requires gap > 1e-6.
    # On a hex grid, gap is 0.
    # We scale the positions by lambda.
    # If lambda = 1 + delta, the gap becomes approx delta * sqrt(3) / 2?
    # Actually, the distance between centers increases by factor lambda.
    # The "gap" in terms of non-overlap is (lambda - 1) * (distance between centers).
    # For neighbors, distance is sqrt(3). Gap is (lambda-1)*sqrt(3).
    # We need (lambda-1)*sqrt(3) > 1e-6.
    # So lambda > 1 + 1e-6/sqrt(3).
    
    for shape in shapes:
        # Generate positions for this shape
        base_positions = []
        for (q, r) in shape:
            pos = q * basis1 + r * basis2
            base_positions.append(pos)
        base_positions = np.array(base_positions)
        
        # Try a few lambda values
        for lambda_val in [1.0 + 1e-5, 1.0 + 1e-4, 1.0 + 1e-3]:
            positions = base_positions * lambda_val
            
            # Compute vertices
            verts_list = []
            for pos in positions:
                verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
                verts_list.append(verts)
            
            # Check separation
            if not _all_separated(verts_list):
                continue
            
            # Compute minimal enclosing hexagon
            s, a = _minimal_enclosing_hexagon(verts_list)
            
            if s < best_side:
                best_side = s
                best_result = (positions, s, a)
                
    if best_result is None:
        # Fallback: use a simple grid that's guaranteed to work with larger gaps
        # We can just use the fallback from the original code but ensure it's valid
        # The original fallback had gaps, so it should be valid.
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    positions, best_s, best_a = best_result
    
    # Build inner hex data
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        inner_hex_data[i] = [pos[0], pos[1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
