"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def _hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    cx, cy = center
    theta = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = theta + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _convex_hull(points):
    """Compute convex hull of a set of 2D points. Returns list of hull vertices in CCW order."""
    pts = np.array(points)
    if len(pts) <= 2:
        return pts
    # Sort by x, then y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    
    # Remove duplicate points
    unique_pts = []
    for p in pts:
        if not unique_pts or np.linalg.norm(p - unique_pts[-1]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    if len(pts) <= 2:
        return pts
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-10:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-10:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    return hull

def _hexagon_contains_hexagon(outer_center, outer_side, outer_angle, inner_center, inner_side, inner_angle):
    """Check if outer hexagon contains inner hexagon.
    Uses support function: for each edge normal of outer, check max dot product of inner vertices."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = _hex_vertices(inner_center, inner_side, inner_angle)
    
    # For each edge of outer hexagon, check that all inner vertices are on the inside
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        # Edge vector
        edge = v2 - v1
        # Normal pointing inward (for CCW polygon, left normal is inward)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            continue
        normal /= norm
        
        # The outer hexagon interior is on the side where dot(normal, p - v1) >= 0
        # Actually for CCW polygon, the interior is to the left of each edge, so normal = (-edge_y, edge_x) points inward
        # Check all inner vertices
        for iv in inner_verts:
            dist = np.dot(iv - v1, normal)
            if dist < -1e-9:
                return False
    return True

def _min_bounding_hexagon_side(centers, angles, outer_angle):
    """Find the minimum side length of a regular hexagon with given outer_angle 
    that contains all unit hexagons at given centers and angles.
    Returns the minimum side length."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for c, a in zip(centers, angles):
        verts = _hex_vertices(c, 1.0, a)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a hexagon with center (cx, cy) and angle theta, the constraint is:
    # For each of the 6 directions (edge normals), the max projection of all points 
    # onto that normal from the center must be <= side * cos(pi/6) = side * sqrt(3)/2
    # Wait, let me think more carefully.
    
    # A regular hexagon with side s and angle theta has 6 edges. The distance from center 
    # to each edge is s * cos(pi/6) = s * sqrt(3)/2 (the apothem).
    # The 6 outward normals are at angles theta + i*pi/3 + pi/6 for i=0..5? 
    # Let me reconsider.
    
    # Vertices are at angles theta + i*pi/3.
    # Edge between vertex i and i+1: midpoint direction is theta + (i+0.5)*pi/3 = theta + i*pi/3 + pi/6
    # The outward normal of that edge points in direction theta + i*pi/3 + pi/6
    # The apothem (distance from center to edge) is s * cos(pi/6) = s * sqrt(3)/2
    
    # So for containment, for each of the 6 outward normals n_i:
    # max_{p in all_verts} dot(p - center, n_i) <= s * sqrt(3)/2
    
    # This means: s >= max_i [ max_p dot(p - center, n_i) ] / (sqrt(3)/2)
    #             = (2/sqrt(3)) * max_i [ max_p dot(p - center, n_i) ]
    
    # But the center is also variable! We need to find center that minimizes the max over i.
    
    # Let's use a different approach: for a fixed outer angle, the minimum bounding hexagon 
    # is determined by the convex hull of all points. We can compute the minimum enclosing hexagon
    # with fixed orientation.
    
    # For a fixed orientation, the minimum enclosing hexagon can be found by considering
    # the support function in 6 directions. The optimal center minimizes the maximum of
    # (support_i + support_{i+3}) / 2 where support_i is the max projection in direction i.
    
    # Actually, for a centrally symmetric convex body (which a hexagon is), the minimum enclosing
    # hexagon with fixed orientation has its center at the center of the convex hull's 
    # "diameter" in each of the 3 principal axes.
    
    # The 3 principal axes are the directions of the 3 pairs of opposite edges.
    # For each axis, the width of the point set is support_pos + support_neg.
    # The hexagon side needed is max over 3 axes of width / sqrt(3).
    # Wait: apothem = s * sqrt(3)/2, so width across opposite edges = 2 * apothem = s * sqrt(3).
    # So s = width / sqrt(3).
    
    # The optimal center for a given axis is the midpoint of the min and max projections.
    # For 3 axes, we need to find a center that works for all 3. Since the hexagon is centrally
    # symmetric, the optimal center is the intersection of the midlines.
    
    # Let's compute:
    theta = np.radians(outer_angle)
    # 6 outward normal directions
    normals = []
    for i in range(6):
        a = theta + i * np.pi / 3 + np.pi / 6
        normals.append(np.array([np.cos(a), np.sin(a)]))
    normals = np.array(normals)
    
    # Compute projections
    projections = all_verts @ normals.T  # shape (n_points, 6)
    
    # For each of the 6 directions, get min and max projection
    mins = projections.min(axis=0)
    maxs = projections.max(axis=0)
    
    # The 3 axes are pairs (0,3), (1,4), (2,5)
    # For axis k (k=0,1,2), the width is maxs[k] - mins[k]
    # The center projection on axis k should be (mins[k] + maxs[k]) / 2
    
    # The center (cx, cy) satisfies: dot((cx,cy), normals[k]) = (mins[k] + maxs[k]) / 2 for k=0,1,2
    # This is a 2x2 system (using any 2 of the 3 equations, the 3rd should be consistent 
    # if the point set fits in a hexagon of that orientation, but generally we need to check)
    
    # Actually, the 3 equations are overdetermined. The optimal center minimizes the maximum
    # deviation. But for a regular hexagon with fixed orientation, the minimum enclosing hexagon
    # is well-defined. Let me just compute the center from 2 equations and verify with the 3rd.
    
    # Use equations 0 and 1 to solve for center
    A = normals[:2]  # 2x2
    b = np.array([(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2])
    
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: use least squares
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Verify with 3rd equation
    # dot(center, normals[2]) should be approximately (mins[2] + maxs[2]) / 2
    
    # Compute the required side length
    # For each axis k, the required apothem is (maxs[k] - mins[k]) / 2
    # But we also need to account for the center offset
    # Actually, let's just compute the max projection from the computed center
    centered = all_verts - center
    proj = centered @ normals.T
    max_proj = proj.max(axis=0)
    min_proj = proj.min(axis=0)
    
    # The required apothem in direction i is max(max_proj[i], -min_proj[i])
    # Since the hexagon is symmetric, the apothem must be >= max(max_proj[i], -min_proj[i]) for all i
    apothems = np.maximum(max_proj, -min_proj)
    max_apothem = apothems.max()
    
    # Side length = apothem / cos(pi/6) = apothem / (sqrt(3)/2) = 2 * apothem / sqrt(3)
    side = 2 * max_apothem / np.sqrt(3)
    
    return side

def construct():
    """Find 12 disjoint unit hexagons inside a minimal regular hexagon.
    Uses triangular lattice with global rotation."""
    
    # Triangular lattice basis
    # For a triangular lattice with spacing d, the basis vectors are:
    # v1 = (d, 0)
    # v2 = (d/2, d*sqrt(3)/2)
    
    # We need 12 hexagons on a triangular lattice.
    # Possible configurations: 3x4 grid, or other arrangements.
    # Let's use a 3x4 rectangular grid on the triangular lattice.
    
    # For unit hexagons (side 1), the minimum distance between centers for disjointness
    # is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732 (distance between centers of adjacent hexagons 
    # in a hexagonal packing).
    
    # Actually, for two unit regular hexagons to be disjoint (non-overlapping), the distance
    # between centers must be at least the sum of their inradii if aligned, but it's more complex
    # with rotation. For safety, let's use a spacing that guarantees disjointness.
    
    # The inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866.
    # For two hexagons to be disjoint, the distance between centers must be at least 
    # 2 * inradius = sqrt(3) ≈ 1.732 if they are aligned. With rotation, it could be less,
    # but let's be conservative.
    
    # Let's use a triangular lattice with spacing s = sqrt(3) + epsilon
    # Actually, let's parameterize the lattice spacing and check disjointness.
    
    # For now, let's fix the lattice structure: 3 rows x 4 columns on a triangular lattice.
    # Row offsets: rows 0, 2 are aligned; row 1 is offset by half a lattice spacing.
    
    # Lattice points for 3x4:
    # Row 0: (0,0), (1,0), (2,0), (3,0)
    # Row 1: (0.5, sqrt(3)/2), (1.5, sqrt(3)/2), (2.5, sqrt(3)/2), (3.5, sqrt(3)/2)
    # Row 2: (0, sqrt(3)), (1, sqrt(3)), (2, sqrt(3)), (3, sqrt(3))
    
    # Scale by lattice spacing d
    lattice_points = []
    for i in range(4):
        for j in range(3):
            x = i + (j % 2) * 0.5
            y = j * np.sqrt(3) / 2
            lattice_points.append((x, y))
    
    lattice_points = np.array(lattice_points)
    
    def objective(rotation_deg):
        """Compute the bounding hexagon side length for a given rotation."""
        theta = np.radians(rotation_deg)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        
        # Rotate lattice points
        centers = (lattice_points @ rot_matrix.T) * d_global[0]
        
        # All inner hexagons have the same rotation
        angles = [rotation_deg] * 12
        
        # Check disjointness: for each pair, check if hexagons overlap
        # For efficiency, only check nearby pairs
        for i in range(12):
            for j in range(i + 1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                # Minimum distance for disjointness depends on relative rotation
                # For same rotation, min distance is 2 * inradius = sqrt(3)
                # For different rotations, it's at least the sum of inradii
                # Conservative bound: sqrt(3)
                if dist < np.sqrt(3) - 1e-6:
                    return 1e6  # Penalize overlap
        
        # Compute bounding hexagon
        side = _min_bounding_hexagon_side(centers.tolist(), angles, rotation_deg)
        return side
    
    # Try different lattice spacings
    best_side = 1e10
    best_rotation = 0
    best_d = 0
    
    # The lattice spacing must be at least sqrt(3) for disjointness with same rotation
    # Let's try a range of spacings and rotations
    
    for d in [np.sqrt(3) + 0.01, np.sqrt(3) + 0.05, np.sqrt(3) + 0.1, np.sqrt(3) + 0.2]:
        d_global = [d]
        # Try rotations from 0 to 30 degrees (hexagon symmetry)
        for rot in np.linspace(0, 30, 31):
            side = objective(rot)
            if side < best_side:
                best_side = side
                best_rotation = rot
                best_d = d
    
    # Refine with optimization
    d_global = [best_d]
    res = minimize_scalar(objective, bounds=(0, 30), method='bounded', 
                          options={'xatol': 1e-4})
    if res.success and res.fun < best_side:
        best_side = res.fun
        best_rotation = res.x
    
    # Compute final centers
    theta = np.radians(best_rotation)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    centers = (lattice_points @ rot_matrix.T) * best_d
    
    # Compute bounding hexagon center
    all_verts = []
    for c in centers:
        verts = _hex_vertices(c, 1.0, best_rotation)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find center of bounding hexagon
    # Use the same logic as _min_bounding_hexagon_side
    normals = []
    for i in range(6):
        a = theta + i * np.pi / 3 + np.pi / 6
        normals.append(np.array([np.cos(a), np.sin(a)]))
    normals = np.array(normals)
    
    projections = all_verts @ normals.T
    mins = projections.min(axis=0)
    maxs = projections.max(axis=0)
    
    A = normals[:2]
    b = np.array([(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2])
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Build inner list
    inner = []
    for c in centers:
        inner.append([c[0], c[1], best_rotation])
    
    return inner, (float(center[0]), float(center[1])), float(best_side), float(best_rotation)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
