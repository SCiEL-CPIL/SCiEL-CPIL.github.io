"""Program for packing 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 11 unit regular hexagons packed in a line (1D cluster).
    The cluster is a row of 11 hexagons sharing edges.
    We compute the convex hull of this rigid cluster, then find the minimal
    enclosing regular hexagon by minimizing over rotation angle.
    
    For a set of points, the minimal enclosing regular hexagon of side length s
    centered at c with orientation theta can be found by checking that all points
    lie within the hexagon. The hexagon with center (cx, cy), side length s, and 
    rotation angle phi has vertices at:
      v_k = (cx, cy) + s * (cos(phi + k*pi/3), sin(phi + k*pi/3)) for k=0..5
    
    A point p is inside if for all 6 face normals n_k, dot(p - c, n_k) <= s * cos(pi/3) = s/2... 
    Actually, for a regular hexagon with side length s, the apothem (inradius) is s * sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes. The distance from center to each side is the apothem a = s * sqrt(3)/2.
    
    So for a given orientation phi, the constraint for each point p_i is:
      max_k |dot(p_i - c, u_k)| <= a  where u_k are the 6 unit normals to the sides.
    
    But we can choose c freely. The optimal c for a given phi is the center of the minimal enclosing hexagon,
    which for a convex polygon can be found, but for simplicity we can note that the cluster is symmetric.
    
    Actually, let's think differently. The cluster of 11 hexagons in a row:
    Each hexagon has side length 1. When placed in a row sharing edges, the center-to-center distance
    is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
    
    Wait, for regular hexagons of side 1, the distance between centers of two edge-adjacent hexagons is 2 * apothem = sqrt(3).
    
    So the 11 hexagons have centers at:
      x_k = k * sqrt(3) for k = -5, -4, ..., 4, 5 (11 positions)
      y_k = 0
    
    The cluster is elongated along x-axis. Its convex hull is a long thin shape.
    
    For the minimal enclosing regular hexagon, we need to consider the orientation. Due to symmetry,
    the optimal orientation will likely have one of the hexagon's sides parallel to the cluster axis,
    or at some specific angle.
    
    Let me compute the convex hull of the 11 hexagons. Each hexagon has 6 vertices. For a hexagon centered at (cx, cy) with orientation 0 (flat top or pointy top?), let's define:
    
    A regular hexagon with side length 1, centered at origin, with one vertex pointing right (pointy-right):
      vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius = 1 (circumradius = side length for regular hexagon)
      So vertices: (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    
    Actually, circumradius of regular hexagon = side length = 1.
    Apothem = sqrt(3)/2.
    
    For hexagon centered at (cx, cy), vertices are:
      (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
    
    The 11 hexagons are centered at (k*sqrt(3), 0) for k=-5..5.
    
    The overall shape is symmetric about both x and y axes. The convex hull will be determined by the extreme points.
    
    Let me just compute all 66 vertices and find the convex hull, then for each candidate rotation angle of the enclosing hexagon, compute the minimal side length needed.
    
    For a given enclosing hexagon orientation phi, the 6 outward normals are at angles phi, phi+60°, ..., phi+300°.
    The half-width in direction of normal n_k is the apothem a = s * sqrt(3)/2.
    
    The constraint is that for each point p, and each normal direction, the projection must be within [-a, a] relative to the center.
    
    For a fixed phi, the minimal enclosing hexagon centered optimally: the center c should be chosen so that the max over points and normals of dot(p, n_k) - dot(c, n_k) is minimized, and similarly for the negative side.
    
    This is equivalent to: for each normal direction n_k, we need dot(c, n_k) to be between min_p dot(p, n_k) + a and max_p dot(p, n_k) - a.
    
    For feasibility, we need: max_p dot(p, n_k) - min_p dot(p, n_k) <= 2a for all k.
    So a >= max_k (max_p dot(p, n_k) - min_p dot(p, n_k)) / 2.
    
    The minimal a for orientation phi is: a(phi) = max_k (range_k(phi)) / 2, where range_k(phi) = max_p dot(p, n_k(phi)) - min_p dot(p, n_k(phi)).
    
    Then s = 2a / sqrt(3).
    
    We minimize s(phi) over phi in [0, pi/3) due to hexagonal symmetry.
    """
    sqrt3 = np.sqrt(3.0)
    
    # Generate all vertices of the 11 hexagons
    centers = [(k * sqrt3, 0.0) for k in range(-5, 6)]
    
    all_points = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * np.pi / 3.0
            px = cx + np.cos(angle)
            py = cy + np.sin(angle)
            all_points.append((px, py))
    
    points = np.array(all_points)
    
    # Function to compute minimal side length for a given enclosing hexagon orientation
    def min_side_length(phi):
        """phi is the angle of the first normal direction."""
        max_range = 0.0
        for k in range(6):
            angle = phi + k * np.pi / 3.0
            nx = np.cos(angle)
            ny = np.sin(angle)
            projections = points[:, 0] * nx + points[:, 1] * ny
            range_val = projections.max() - projections.min()
            if range_val > max_range:
                max_range = range_val
        apothem = max_range / 2.0
        side_length = 2.0 * apothem / sqrt3
        return side_length
    
    # Minimize over phi in [0, pi/3]
    result = minimize_scalar(min_side_length, bounds=(0, np.pi / 3), method='bounded', 
                             options={'xatol': 1e-12})
    
    optimal_phi = result.x
    optimal_side = result.fun
    
    # The center of the enclosing hexagon: midpoint of the cluster's bounding box in each normal direction
    # For the optimal phi, compute center
    cx = 0.0
    cy = 0.0
    
    # Place the 11 hexagons at their centers with orientation 0
    inner = []
    for k in range(-5, 6):
        inner.append([k * sqrt3, 0.0, 0.0])
    
    return inner, (cx, cy), optimal_side, np.degrees(optimal_phi)
# EVOLVE-BLOCK-END
