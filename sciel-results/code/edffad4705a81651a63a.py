# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _support_function_hexagon(d, angle_diff_deg):
    """
    Calculate the support function of a regular hexagon with side d 
    in a direction given by angle_diff_deg relative to the hexagon's angle.
    h(u) = d * max(cos(theta_i)) where theta_i are angles between u and vertices.
    Actually, for a regular polygon, h(u) = d * cos(pi/6) * ... no, simpler:
    The apothem is d * sqrt(3) / 2.
    The support function of a regular hexagon with side d in direction phi 
    (relative to hexagon orientation) is:
    d * cos( (pi/6) - (phi mod pi/3) ) if we define orientation such that vertex is at 0?
    Let's derive:
    Vertices at angles 0, 60, 120...
    Edges normals at 30, 90, 150...
    Distance from center to edge normal direction is d*sqrt(3)/2.
    Distance from center to vertex is d.
    The support function h(u) for a regular hexagon with side d is:
    d * cos( delta ) where delta is the angle between u and the closest vertex? No.
    Let's use the property that h(u) is the max dot product with vertices.
    Vertices v_i = d * (cos(a + i*pi/3), sin(a + i*pi/3)).
    u = (cos(phi), sin(phi)).
    v_i . u = d * cos(a + i*pi/3 - phi).
    Max over i of cos(a + i*pi/3 - phi).
    Let delta = a - phi. We want max_i cos(delta + i*pi/3).
    The maximum value of cos(x + k*pi/3) is cos( (delta mod pi/3) - pi/6 )? 
    No, the function is periodic with period pi/3.
    Within [0, pi/3], the max of cos(x) is at x=0.
    So if we map the relative angle to [-pi/6, pi/6], the max is d * cos(mapped_angle).
    """
    # Map angle_diff to [-pi/6, pi/6]
    angle_rad = np.radians(angle_diff_deg)
    # Reduce modulo pi/3
    period = np.pi / 3.0
    # We want the angle closest to a multiple of pi/3 (vertex direction)
    # Vertices are at 0, pi/3, etc.
    # So we map angle to distance from nearest multiple of pi/3.
    # Let k = round(angle / period)
    # diff = angle - k*period
    # This diff is in [-pi/6, pi/6]
    k = np.round(angle_rad / period)
    diff = angle_rad - k * period
    return d * np.cos(diff)

def _min_enclosing_hex_side(centers, inner_angle_deg, outer_angle_deg=0.0):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, given angle)
    that contains all inner hexagons.
    centers: (N, 2) array
    inner_angle_deg: angle of inner hexagons
    outer_angle_deg: angle of outer hexagon
    """
    outer_normals = _hex_edge_normals(outer_angle_deg)
    
    max_required_s = 0.0
    # Inner hexagon support function in direction n
    # We need to evaluate h_inner(n - inner_angle) for each outer normal n
    
    for i in range(len(centers)):
        c = centers[i]
        for n in outer_normals:
            # Angle of normal n
            n_angle = np.degrees(np.arctan2(n[1], n[0]))
            # Relative angle for support function
            rel_angle = n_angle - inner_angle_deg
            h_val = _support_function_hexagon(1.0, rel_angle)
            
            # Projection of center onto normal
            proj_c = c @ n
            
            # Required distance from center to edge
            # s * sqrt(3)/2 >= proj_c + h_val
            required_s = (proj_c + h_val) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _get_lattice_positions(radius=3):
    """
    Generate a set of hexagonal lattice points within a hexagonal region of given radius.
    The lattice is defined by basis vectors e1 = (1.5, sqrt(3)/2) and e2 = (0, sqrt(3)).
    Actually, for unit hexagons (side 1), the center-to-center distance is sqrt(3).
    Wait.
    Unit hexagon side d=1.
    Distance between centers of adjacent hexagons in a tiling is 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3).
    
    Lattice vectors:
    v1 = (sqrt(3), 0) ? No.
    Let's align hexagons with flat tops or pointy tops?
    Standard: Pointy top?
    If angle=0, vertices are at 0, 60, 120...
    Vertex 0 is at (1, 0).
    Edge 0 is between (1,0) and (1/2, sqrt(3)/2).
    Normal to edge 0 is at 30 deg?
    Let's check _hex_edge_normals(0).
    angles = 30 + 60*i.
    Normal 0 is (cos(30), sin(30)) = (sqrt(3)/2, 1/2).
    
    For a tiling, the neighbors are in directions of the normals.
    Distance to neighbor is 2 * apothem = sqrt(3).
    So neighbor centers are at distance sqrt(3) in directions of normals.
    
    Lattice basis:
    a1 = sqrt(3) * (cos(30), sin(30)) = sqrt(3) * (sqrt(3)/2, 1/2) = (3/2, sqrt(3)/2)
    a2 = sqrt(3) * (cos(90), sin(90)) = (0, sqrt(3))
    
    Any center is i*a1 + j*a2.
    """
    positions = []
    sqrt3 = np.sqrt(3)
    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([0, sqrt3])
    
    # Generate points in a hexagonal shape of "radius" R in lattice coordinates
    # A hexagon of side R (in terms of number of steps) contains 3*R*(R-1)+1 points?
    # For R=2, we have 19 points.
    # We want 12 points.
    # We can just generate a large chunk and pick the most compact 12.
    # Or define specific compact shapes.
    
    # Let's generate all points with |i|<=R, |j|<=R, |i+j|<=R? No, that's not the metric.
    # The metric is the hexagonal distance.
    # Simpler: generate i, j in range [-R, R] and filter by distance from origin.
    # Distance from origin in Cartesian coords.
    
    pts = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            p = i * a1 + j * a2
            # Check if it's within a hexagonal boundary of radius R*sqrt(3)?
            # Just keep them for now, we will pick subsets later.
            pts.append(p)
            
    pts = np.array(pts)
    # Sort by distance from origin to prefer compact sets
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices]
    
    return pts

def _find_best_12_subset(all_pts, target_count=12):
    """
    Heuristic to find a compact subset of 12 points from a sorted list of lattice points.
    Greedy approach: Start with the closest point, then add the next closest point 
    that is not too far from the current centroid? 
    Actually, for a fixed topology, the shape matters.
    Since we have a sorted list of all lattice points by distance from origin,
    the first 12 points *are* the most compact set centered at origin.
    Is it always optimal to take the 12 closest points to the origin?
    Yes, for minimizing the bounding box of the set, taking the points closest to the center 
    is generally optimal because it minimizes the extent in all directions.
    """
    # The input all_pts is already sorted by distance from origin.
    return all_pts[:target_count]

def _optimize_rotation(centers, inner_angle_base=0.0):
    """
    Optimize the rotation of the cluster to minimize the enclosing hexagon side length.
    We fix the outer hexagon angle to 0.
    We vary the inner cluster angle.
    """
    def obj(angle_deg):
        # The centers are fixed in the "cluster frame".
        # If we rotate the cluster by angle_deg, the new centers are R * centers.
        # However, _min_enclosing_hex_side takes centers and inner_angle.
        # If we rotate the centers, we must also rotate the inner hexagon orientation.
        # Let's say the cluster is rigid.
        # Original centers: C. Inner angle: 0.
        # Rotate by theta: New centers C' = R_theta C. New inner angle: theta.
        # We want to minimize s(C', theta).
        
        theta_rad = np.radians(angle_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        new_centers = centers @ R.T
        new_inner_angle = angle_deg
        
        return _min_enclosing_hex_side(new_centers, new_inner_angle, 0.0)
        
    # The function is periodic with period 60 degrees.
    # We can search in [0, 60).
    res = minimize_scalar(obj, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
    return res.x, res.fun

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # 1. Generate lattice points
    # We need enough points to pick 12.
    # Radius 2 gives 19 points. Radius 3 gives more.
    all_pts = _get_lattice_positions(radius=3)
    
    # 2. Select the 12 closest points to the origin as the cluster topology
    # This forms a compact cluster.
    centers = _find_best_12_subset(all_pts, 12)
    
    # Center the cluster at the origin to ensure symmetry
    # The 12 closest points in a hex lattice are symmetric.
    # Let's verify the centroid is close to 0.
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # 3. Optimize rotation
    best_angle = 0.0
    best_s = np.inf
    
    try:
        angle, s_val = _optimize_rotation(centers)
        best_angle = angle
        best_s = s_val
    except Exception as e:
        # Fallback to angle 0
        best_s = _min_enclosing_hex_side(centers, 0.0, 0.0)
    
    # 4. Construct the result
    # Inner hexagons
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = best_angle  # All inner hexagons have the same angle
    
    # Outer hexagon
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    # 5. Validation and Refinement (Optional but good for safety)
    # The analytical calculation of s should be exact for the support function.
    # However, numerical errors might make s slightly too small.
    # Add a tiny epsilon to s to ensure validity.
    outer_hex_side_length *= 1.000001
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
