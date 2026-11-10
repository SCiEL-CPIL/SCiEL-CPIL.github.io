import numpy as np

def hexagon_packing_12():
    """
    Constructs a tight packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    The packing uses a central hexagon, a first ring of 6 hexagons, and a second ring of 5 hexagons.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Small epsilon to ensure separation (gap > 1e-6)
    eps = 1e-4
    
    # Side length of inner hexagons is 1.
    # Distance between centers of touching unit hexagons is sqrt(3).
    d_touch = np.sqrt(3)
    
    # Radii for the rings
    r1 = d_touch + eps
    r2 = 2 * d_touch + 2 * eps
    
    # Inner hexagon centers
    centers = []
    # 1. Center
    centers.append([0.0, 0.0, 0.0])
    
    # 2. Ring 1 (6 hexagons)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append([x, y, 0.0])
        
    # 3. Ring 2 (5 hexagons, skipping 180 degrees)
    # Angles: 0, 60, 120, 240, 300
    angles_r2 = [0, 60, 120, 240, 300]
    for deg in angles_r2:
        angle = np.deg2rad(deg)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append([x, y, 0.0])
        
    inner_hex_data = np.array(centers)
    
    # Outer hexagon
    # Minimum side length s such that s * sqrt(3)/2 >= max projection
    # Max projection is from the outermost vertices of Ring 2 hexagons.
    # For hexagon at angle 0 in Ring 2, center (r2, 0).
    # Farthest vertex in direction 30 deg (normal to edge) is at x = r2 + 1.
    # Projection = (r2 + 1) * cos(30) = (r2 + 1) * sqrt(3)/2
    # We need s * sqrt(3)/2 >= (r2 + 1) * sqrt(3)/2
    # So s >= r2 + 1
    # However, we also need to check the "corner" constraints.
    # As derived earlier, s = r2 + 1 is sufficient for the radial vertices.
    # Let's verify the constraint for the hexagon at 60 deg.
    # Center (r2/2, r2*sqrt(3)/2).
    # Vertex pointing "up" (90 deg relative to center? No, 60 deg radial).
    # The vertex at 90 deg from center (local) is at:
    # x = r2/2 + 0, y = r2*sqrt(3)/2 + 1.
    # Projection on 90 deg normal (y-axis): y = r2*sqrt(3)/2 + 1.
    # Constraint: s*sqrt(3)/2 >= r2*sqrt(3)/2 + 1.
    # s >= r2 + 2/sqrt(3) = r2 + 2*d_touch/3? No.
    # s * sqrt(3)/2 >= r2 * sqrt(3)/2 + 1
    # s >= r2 + 2/sqrt(3).
    # 2/sqrt(3) = 2 * sqrt(3)/3 = 2/3 * sqrt(3) = 2/3 * d_touch.
    # So s >= r2 + 2/3 * d_touch.
    
    # Let's compare r2 + 1 vs r2 + 2/3 * d_touch.
    # d_touch = sqrt(3) ~ 1.732.
    # 2/3 * 1.732 = 1.1547.
    # 1 < 1.1547.
    # So the constraint from the "side" hexagons is tighter!
    # We need s >= r2 + 2/sqrt(3).
    
    s_min = r2 + 2.0 / np.sqrt(3.0)
    
    # Add a small margin to be safe
    s = s_min + 1e-5
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
