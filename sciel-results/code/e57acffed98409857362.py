# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hexagon_vertices(cx, cy, d, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with side d and angle angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _is_point_inside_hexagon(v, outer_verts):
    """
    Check if point v is inside the outer hexagon (CCW vertices).
    For each edge (p1, p2), cross(p2-p1, v-p1) must be >= -1e-6.
    """
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        vec = v - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-6:
            return False
    return True


def _compute_objective(params):
    """
    params = [a, b, theta, phi, s]
    a, b: lattice basis vector lengths
    theta: angle between basis vectors (radians)
    phi: global rotation of the lattice (radians)
    s: outer hexagon side length
    """
    a, b, theta, phi, s = params
    
    if s < np.sqrt(12) or s <= 0 or a < 2.0 - 1e-5 or b < 2.0 - 1e-5:
        return 1e10
        
    # Lattice basis vectors
    e1 = a * np.array([np.cos(phi), np.sin(phi)])
    e2 = b * np.array([np.cos(phi + theta), np.sin(phi + theta)])
    
    # Check distance between e1 and e2 (nearest neighbor distance for diagonal)
    d_diag = np.linalg.norm(e1 - e2)
    if d_diag < 2.0 - 1e-5:
        return 1e10
        
    # Define the 12 lattice points for a 3x4 parallelogram
    # This shape is more compact than 2x6
    lattice_points = []
    for i in range(3):
        for j in range(4):
            lattice_points.append((i, j))
            
    # Compute actual positions
    positions = np.array([i * e1 + j * e2 for i, j in lattice_points])
    
    # Outer hexagon vertices
    outer_verts = _hexagon_vertices(0, 0, s, 0)
    
    # Check if all inner hexagons are inside the outer hexagon
    # We check all 6 vertices of each inner hexagon
    for pos in positions:
        inner_verts = _hexagon_vertices(pos[0], pos[1], 1.0, 0)
        for v in inner_verts:
            if not _is_point_inside_hexagon(v, outer_verts):
                return 1e10
                
    # If valid, return s (minimize this)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Initial guess
    # Triangular lattice with spacing 2
    x0 = np.array([2.0, 2.0, np.pi / 3, 0.0, 6.0])
    
    # Bounds
    # a, b must be >= 2
    # theta between 0 and pi
    # phi between -pi and pi
    # s between sqrt(12) and 15
    bounds = [(2.0, 4.0), (2.0, 4.0), (np.pi/6, 5*np.pi/6), (-np.pi, np.pi), (np.sqrt(12), 15.0)]
    
    # Optimize
    # Use multiple starting points to avoid local minima
    best_result = None
    best_fun = 1e10
    
    starts = [
        np.array([2.0, 2.0, np.pi / 3, 0.0, 6.0]),
        np.array([2.0, 2.0, np.pi / 3, np.pi / 6, 6.0]),
        np.array([2.0, 2.0, np.pi / 3, -np.pi / 6, 6.0]),
        np.array([2.0, 2.0, np.pi / 3, np.pi / 12, 6.0]),
        np.array([2.0, 2.0, np.pi / 3, -np.pi / 12, 6.0]),
    ]
    
    for start in starts:
        try:
            result = minimize(_compute_objective, start, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
        except Exception:
            continue
            
    if best_result is None or best_fun > 10.0:
        # Fallback to a known valid configuration if optimization fails
        # Use a simple 2x6 grid with spacing 2
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [4, 0, 0],
                [6, 0, 0],
                [8, 0, 0],
                [10, 0, 0],
                [1, np.sqrt(3), 0],
                [3, np.sqrt(3), 0],
                [5, np.sqrt(3), 0],
                [7, np.sqrt(3), 0],
                [9, np.sqrt(3), 0],
                [11, np.sqrt(3), 0],
            ]
        )
        # Center the inner hexagons
        center = np.mean(inner_hex_data[:, :2], axis=0)
        inner_hex_data[:, :2] -= center
        outer_hex_data = np.array([0, 0, 0])
        # Estimate s for 2x6 grid
        # Width is approx 12, height approx 2*sqrt(3)
        # s needs to be large enough.
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    a, b, theta, phi, s = best_result.x
    
    # Compute final positions
    e1 = a * np.array([np.cos(phi), np.sin(phi)])
    e2 = b * np.array([np.cos(phi + theta), np.sin(phi + theta)])
    
    lattice_points = []
    for i in range(3):
        for j in range(4):
            lattice_points.append((i, j))
            
    inner_hex_data = np.zeros((12, 3))
    for idx, (i, j) in enumerate(lattice_points):
        pos = i * e1 + j * e2
        inner_hex_data[idx, 0] = pos[0]
        inner_hex_data[idx, 1] = pos[1]
        inner_hex_data[idx, 2] = 0.0
        
    # Center the entire configuration
    center = np.mean(inner_hex_data[:, :2], axis=0)
    inner_hex_data[:, :2] -= center
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
