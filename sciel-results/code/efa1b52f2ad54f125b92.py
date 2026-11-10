# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define 12 unit hexagons using hex grid axial coordinates (q, r)
    # Arrangement: 3 columns x 4 rows in hex grid
    axial_coords = []
    for r in range(-2, 2):  # r = -2, -1, 0, 1
        for q in range(3):  # q = 0, 1, 2
            axial_coords.append((q, r))
    
    axial_coords = np.array(axial_coords, dtype=float)
    
    # Convert axial to Cartesian for pointy-top hexagons
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    sqrt3 = np.sqrt(3)
    x = sqrt3 * (axial_coords[:, 0] + axial_coords[:, 1] / 2)
    y = 1.5 * axial_coords[:, 1]
    
    centers = np.column_stack([x, y])
    
    # Center the configuration
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # For a regular hexagon of side s centered at origin with angle theta,
    # the vertices are at (s*cos(theta + 60*i), s*sin(theta + 60*i)) for i=0..5
    # A point (px, py) is inside if for all edges, the cross product test passes.
    # Equivalently, the maximum over the 6 normals of (n_i · p) must be <= s * (sqrt(3)/2)
    # where n_i are unit normals to the edges.
    
    # The apothem (distance from center to edge midpoint) of a regular hexagon with side s is s * sqrt(3)/2
    # So for a point p to be inside, max_i |n_i · p| <= s * sqrt(3)/2 for the 6 edge normals
    
    # Actually, for a regular hexagon with vertices at angles theta + 60*i, the edge normals
    # are at angles theta + 60*i + 30 (perpendicular to edges)
    # The constraint is: for each of 6 directions, n_k · p <= apothem = s * sqrt(3)/2
    
    def compute_required_s(theta, centers):
        """Compute minimum side length s for outer hexagon with angle theta to contain all centers plus hexagon extents."""
        # For each inner hexagon, we need to ensure all its vertices are inside the outer hexagon.
        # Inner hexagon i has center c_i and side 1, angle 0 (we'll keep all inner hexagons axis-aligned for simplicity)
        # Vertices of inner hex i: c_i + cos(60*j), sin(60*j) for j=0..5
        
        # For the outer hexagon with angle theta, the constraint for a point p is:
        # For k=0..5: n_k · p <= s * sqrt(3)/2
        # where n_k = (cos(theta + 60*k + 30), sin(theta + 60*k + 30)) are unit edge normals
        
        theta_rad = np.radians(theta)
        # Edge normal angles
        normal_angles = theta_rad + np.pi/6 + 2*np.pi*np.arange(6)/6
        
        # For each inner hexagon vertex, compute max over normals of n·p
        # Then s must satisfy: s * sqrt(3)/2 >= max over all vertices and all normals of (n·p)
        
        # Vertices of all inner hexagons
        # Inner hexagons have angle 0, so vertices at 0, 60, 120, 180, 240, 300 degrees
        vertex_angles = 2*np.pi*np.arange(6)/6
        vertex_offsets = np.column_stack([np.cos(vertex_angles), np.sin(vertex_angles)])
        
        # All vertices: for each center, add 6 vertex offsets
        # centers shape (12,2), vertex_offsets shape (6,2)
        all_vertices = centers[:, np.newaxis, :] + vertex_offsets[np.newaxis, :, :]
        all_vertices = all_vertices.reshape(-1, 2)  # (72, 2)
        
        # Compute n·p for all normals and all vertices
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        # dots shape (6, 72)
        dots = normals @ all_vertices.T
        
        # Max over all normals and vertices
        max_dot = np.max(dots)
        
        # s * sqrt(3)/2 >= max_dot => s >= max_dot * 2/sqrt(3)
        s_required = max_dot * 2 / sqrt3
        
        # Also need to check the negative side? No, for a centered hexagon, the constraint is one-sided for each normal
        # Actually, the hexagon is defined by 6 half-planes: n_k · p <= apothem
        # So we need max over k of (n_k · p) <= apothem for all p
        # Which gives s >= (2/sqrt(3)) * max_{k,p} (n_k · p)
        
        return s_required
    
    # Optimize over theta
    def objective(theta_arr):
        theta = theta_arr[0]
        return compute_required_s(theta, centers)
    
    # Try multiple starting angles
    best_s = float('inf')
    best_theta = 0
    
    for theta_init in np.linspace(0, 60, 13):
        result = minimize(objective, [theta_init], method='Nelder-Mead', 
                         options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000})
        if result.fun < best_s:
            best_s = result.fun
            best_theta = result.x[0]
    
    # Build the result
    # Inner hexagons: centers with angle 0
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Outer hexagon: centered at origin, angle best_theta, side length best_s
    outer_hex_data = np.array([0.0, 0.0, best_theta])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
