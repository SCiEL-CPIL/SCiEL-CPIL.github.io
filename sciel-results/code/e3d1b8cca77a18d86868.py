# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Define 12 unit hexagons in a hexagonal lattice arrangement
    # Using axial coordinates for a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (2, 0),
        (-2, 1),
        (2, -1),
        (0, 2),
        (0, -2),
    ]
    
    # Convert axial to pixel coordinates for flat-top unit hexagons
    # Center-to-center distance = sqrt(3)
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
        return np.array([x, y])
    
    centers = np.array([axial_to_pixel(q, r) for q, r in axial_coords])
    
    # Shift so centroid is at origin
    centroid = np.mean(centers, axis=0)
    centers_shifted = centers - centroid
    
    # Generate all 72 vertices (12 hexagons * 6 vertices each)
    all_vertices = []
    for c in centers_shifted:
        for i in range(6):
            angle = i * np.pi / 3
            v = c + np.array([np.cos(angle), np.sin(angle)])
            all_vertices.append(v)
    all_vertices = np.array(all_vertices)
    
    # For a given outer hexagon rotation theta, find min s
    # Outer hexagon vertices at: s * (cos(theta + 60i), sin(theta + 60i))
    # Inradius (center to edge) = s * cos(30) = s * sqrt(3)/2
    # Normal directions to edges: theta + 30 + 60i degrees
    
    def compute_min_s(theta):
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        cos_na = np.cos(normal_angles)
        sin_na = np.sin(normal_angles)
        
        # Project all vertices onto each normal direction
        # Shape: (72, 6)
        projs = all_vertices[:, 0] * cos_na[np.newaxis, :] + all_vertices[:, 1] * sin_na[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (np.sqrt(3) / 2)
    
    # Optimize theta
    best_s = float('inf')
    best_theta = 0
    for init_theta in np.linspace(0, np.pi/3, 13):
        res = minimize(lambda t: compute_min_s(t[0]), [init_theta], method='Nelder-Mead',
                      options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000})
        if res.fun < best_s:
            best_s = res.fun
            best_theta = res.x[0]
    
    s = best_s
    theta = best_theta
    
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, np.degrees(theta)])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
