# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


def _hexagon_vertices(cx, cy, d, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with side d and angle angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Lattice basis for unit hexagons (side 1)
    # Distance between centers is sqrt(3)
    # Basis vectors at 60 degrees
    e1 = np.array([np.sqrt(3), 0.0])
    e2 = np.array([np.sqrt(3) / 2.0, 1.5]) # sqrt(3)/2 * sqrt(3) = 3/2 = 1.5
    
    # Define 12 lattice points: 3x4 rhombus
    # i in 0..2, j in 0..3
    lattice_points = []
    for i in range(3):
        for j in range(4):
            pos = i * e1 + j * e2
            lattice_points.append(pos)
            
    lattice_points = np.array(lattice_points)
    
    # Center the cluster
    centroid = np.mean(lattice_points, axis=0)
    lattice_points -= centroid
    
    # Generate vertices for all 12 hexagons at angle 0
    all_vertices = []
    for i in range(12):
        verts = _hexagon_vertices(lattice_points[i, 0], lattice_points[i, 1], 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices) # Shape (72, 2)
    
    # Normals for the outer hexagon (fixed at 0, 60, 120, 180, 240, 300 degrees)
    # The outer hexagon has angle 0, so vertices are at 0, 60...
    # Normals are perpendicular to edges.
    # Edge from v0 to v1.
    # v0 = (s, 0), v1 = (s/2, s*sqrt(3)/2).
    # Edge vector: (-s/2, s*sqrt(3)/2).
    # Normal: (s*sqrt(3)/2, s/2) -> normalized (sqrt(3)/2, 1/2) -> angle 30 deg.
    # Wait, the normal to the edge connecting (s,0) and (s/2, s*sqrt(3)/2) points outwards.
    # The angle of the normal is 30 degrees?
    # Let's check.
    # Vertex 0: (s, 0). Vertex 1: (s/2, s*sqrt(3)/2).
    # Midpoint: (3s/4, s*sqrt(3)/4).
    # Normal direction from origin to midpoint is (3, sqrt(3)) -> angle 30 deg.
    # Yes, the normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def compute_s(phi):
        """
        Compute the minimum side length s for the outer hexagon 
        given a rotation phi of the inner cluster.
        """
        c = np.cos(phi)
        s_ = np.sin(phi)
        rot = np.array([[c, -s_], [s_, c]])
        
        # Rotate all vertices
        rot_verts = all_vertices @ rot.T
        
        # Project onto all 6 normals
        # projections shape: (72, 6)
        projections = rot_verts @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(projections, axis=0)
        
        # The required apothem A is the max of these max_projs
        # s_outer = A / (sqrt(3)/2) = 2 * A / sqrt(3)
        A = np.max(max_projs)
        
        # s must be at least sqrt(12)
        s_val = 2.0 * A / np.sqrt(3)
        return max(s_val, np.sqrt(12))
    
    # Optimize phi in [0, pi/6] due to symmetry of the container and cluster (2-fold symmetry of cluster, 6-fold of container -> 3-fold effective? No, 2 and 6 -> 2? 
    # The cluster is a 3x4 rhombus. It has 180 deg rotational symmetry.
    # The container has 60 deg rotational symmetry.
    # The function f(phi) has period pi/3?
    # If we rotate cluster by pi/3, the container effectively rotates by -pi/3 which is same as 0.
    # So period is pi/3.
    # Due to 180 deg symmetry of cluster, f(phi) = f(phi + pi).
    # So we can search [0, pi/6] or [0, pi/3]. Let's do [0, pi/3].
    
    res = minimize_scalar(compute_s, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-10})
    
    phi_opt = res.x
    s_opt = res.fun
    
    # Reconstruct the final positions
    c = np.cos(phi_opt)
    s_ = np.sin(phi_opt)
    rot = np.array([[c, -s_], [s_, c]])
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        pos = lattice_points[i] @ rot.T
        inner_hex_data[i, 0] = pos[0]
        inner_hex_data[i, 1] = pos[1]
        inner_hex_data[i, 2] = np.rad2deg(phi_opt) # All inner hexagons rotate together
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s_opt)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
