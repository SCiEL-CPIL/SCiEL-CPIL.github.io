import numpy as np
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Lattice basis vectors for unit hexagons touching side-to-side
    # u1 is along x-axis, u2 is at 60 degrees
    # Distance between centers is sqrt(3)
    u1 = np.array([np.sqrt(3), 0])
    u2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate the 7-hex core (radius 1 in lattice)
    # Lattice coords (i, j)
    core_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # Generate 5 additional hexagons from the radius 2 shell
    # The 6 corners of the radius 2 hexagon are at:
    # (2, 0), (1, 1), (-1, 2), (-2, 1), (-2, 0), (-1, -1) ... wait
    # Let's list all points with lattice distance <= 2
    # Distance in lattice steps is max(|i|, |j|, |i-j|) ? No, it's the hexagonal norm.
    # Points at distance 2:
    # (2,0), (2,-1), (1,1), (0,2), (-1,1) is dist 1? No.
    # Let's just generate all points with i,j in [-2,2] and filter by Euclidean distance or lattice distance.
    # Lattice distance d(i,j) = max(|i|, |j|, |i-j|) is for axial coords?
    # Let's use Euclidean distance approximation or just pick the 6 "corners" of the 19-point hex.
    # The 19-point hex has corners at:
    # (2,0), (1,1), (-1,2), (-2,1), (-2,0), (-1,-1) ... this is getting messy.
    
    # Simpler: The 6 neighbors of the 6 corners of the core.
    # Core corners are at (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # The "next" layer points adjacent to these corners are:
    # Adjacent to (1,0): (2,0) and (1,1).
    # Adjacent to (0,1): (1,1) and (0,2).
    # Adjacent to (-1,1): (0,2) and (-1,2).
    # Adjacent to (-1,0): (-1,2) and (-2,1).
    # Adjacent to (0,-1): (-2,1) and (-2,0).
    # Adjacent to (1,-1): (-2,0) and (2,0)?? No.
    
    # Let's just take the 6 points at maximum distance from origin in the 19-point set.
    # These are the "corners" of the radius-2 hexagon.
    # They are at lattice coords:
    corners = [
        (2, 0),
        (1, 1),
        (-1, 2),
        (-2, 1),
        (-2, 0),
        (-1, -1)
    ]
    # Wait, (-1, -1) is adjacent to (0,-1) and (1,-1)?
    # (0,-1) + (-1,0) = (-1,-1).
    # (1,-1) + (-1,0) = (0,-1).
    # Let's verify the 6 corners of the 19-point hex.
    # They should be symmetric.
    # (2,0) -> dist sqrt(3)*2 = 2sqrt(3).
    # (1,1) -> pos u1+u2 = (1.5sqrt(3), 1.5). dist sqrt(6.75+2.25)=sqrt(9)=3? No.
    # u1=(sqrt(3),0), u2=(sqrt(3)/2, 1.5).
    # u1+u2 = (1.5sqrt(3), 1.5).
    # |u1+u2|^2 = 6.75 + 2.25 = 9. Dist = 3.
    # |2u1|^2 = (2sqrt(3))^2 = 12. Dist = 2sqrt(3) approx 3.46.
    # So (2,0) is further than (1,1).
    # The "corners" of the hexagonal shape in Euclidean space are the points with max distance.
    # The 19-point hexagon has 6 "corner" points that are further out?
    # No, the 19-point hexagon is a regular hexagon in lattice space.
    # The Euclidean shape is also a hexagon, but the vertices are not all at the same distance?
    # Actually, the 19-point hexagon has 6 vertices at distance 2sqrt(3) and 12 at distance 3?
    # Let's check (1,1). Dist 3.
    # (2,0). Dist 2sqrt(3) approx 3.46.
    # So the "corners" of the Euclidean bounding box are (2,0), (0,2), etc?
    # (0,2) -> 2u2 = (sqrt(3), 3). Dist sqrt(3+9)=sqrt(12)=2sqrt(3).
    # So the 6 points at distance 2sqrt(3) are:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # Let's verify (-2,2). -2u1+2u2 = -2(sqrt(3),0) + 2(sqrt(3)/2, 1.5) = (-2sqrt(3)+sqrt(3), 3) = (-sqrt(3), 3).
    # Dist sqrt(3+9)=sqrt(12)=2sqrt(3).
    # Yes.
    
    # So the 6 "corner" points of the radius-2 shell are:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    
    # We need 5 of these.
    # To maintain symmetry as much as possible, we can drop one.
    # Dropping one breaks symmetry, but the bounding box is determined by the max projection.
    # If we drop (2,0), the max projection on the right side might decrease?
    # The projection of (2,0) on normal 30 deg is 3.
    # The projection of (0,2) on normal 30 deg is?
    # (0,2) pos = (sqrt(3), 3).
    # n1 = (sqrt(3)/2, 0.5).
    # Dot = 3/2 + 1.5 = 3.
    # So all 6 corners have projection 3 on the "adjacent" normals.
    # If we drop one, the max projection remains 3 (from the others).
    # So R_centers = 3.
    
    selected_corners = corners[:5]
    
    all_coords = core_coords + selected_corners
    
    # Calculate positions
    centers = []
    for i, j in all_coords:
        p = i * u1 + j * u2
        centers.append(p)
    
    # Center the configuration
    center_of_mass = np.mean(centers, axis=0)
    centers = [c - center_of_mass for c in centers]
    
    # Construct inner list
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Calculate side length
    # R_centers = max projection of centers on normals
    # Normals at 30, 90, 150, 210, 270, 330
    normals = []
    for k in range(6):
        ang = np.radians(30 + k * 60)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
        
    R_centers = 0
    for n in normals:
        max_proj = 0
        for c in centers:
            proj = np.dot(c, n)
            if proj > max_proj:
                max_proj = proj
        if max_proj > R_centers:
            R_centers = max_proj
            
    side = (2 * R_centers) / np.sqrt(3) + 1
    
    return inner, outer_center, side, outer_angle
