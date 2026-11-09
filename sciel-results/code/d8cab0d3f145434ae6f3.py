# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side(inner_vertices, outer_angle_deg):
    """
    Given inner hexagon vertices (12x6x2) and outer hexagon angle,
    compute the minimum side length s of an outer regular hexagon (centered at origin)
    that contains all inner vertices.
    """
    a_outer = np.radians(outer_angle_deg)
    # Outer hexagon edges: for each edge, we need all points to be on the correct side
    # Edge i goes from vertex i to vertex i+1
    # For a regular hexagon centered at origin with side s and angle a_outer:
    # vertices are at (s*cos(a+60k), s*sin(a+60k))
    # The constraint is that for each edge, the cross product test must pass
    
    # For edge from v_i to v_{i+1}, the inward normal direction determines the constraint
    # For a regular hexagon, the distance from center to each edge (apothem) is s * cos(30°) = s * sqrt(3)/2
    # But we need to account for rotation
    
    # The constraint: for each edge, all points must satisfy the half-plane inequality
    # Edge i: from vertex i to vertex i+1
    # The outward normal of edge i points in direction (a_outer + 30° + 60°*i)
    # Actually, let's think differently.
    
    # For a regular hexagon with center at origin, side s, angle a:
    # The supporting line for edge i has normal direction n_i = (cos(a + 30° + 60°*i), sin(a + 30° + 60°*i))
    # Wait, let me reconsider.
    
    # Vertices: v_k = s * (cos(a + 60k), sin(a + 60k)) for k=0..5
    # Edge from v_k to v_{k+1}
    # The inward normal of this edge points toward the center
    # The direction from midpoint of edge to center is the inward normal
    
    # Midpoint of edge k: m_k = (v_k + v_{k+1})/2
    # Inward normal direction: -m_k / |m_k|
    
    # For a regular hexagon, |m_k| = s * cos(30°) = s * sqrt(3)/2
    # And the direction of m_k is at angle a + 30° + 60k
    
    # So the constraint is: for all points p, dot(p, n_k) <= s * sqrt(3)/2
    # where n_k is the unit vector in direction (a + 30° + 60k)
    
    # Actually, dot(p, n_k) <= apothem = s * sqrt(3)/2
    # So s >= 2/sqrt(3) * max over all edges k and all points p of dot(p, n_k)
    
    min_s = 0.0
    for k in range(6):
        angle_k = a_outer + np.radians(30 + 60 * k)
        n_k = np.array([np.cos(angle_k), np.sin(angle_k)])
        # Project all inner vertices onto this normal
        projections = inner_vertices @ n_k  # shape (72,)
        max_proj = np.max(projections)
        # s >= 2/sqrt(3) * max_proj
        s_needed = 2.0 / np.sqrt(3.0) * max_proj
        if s_needed > min_s:
            min_s = s_needed
    return min_s


def _compute_inner_vertices(lattice_params, inner_angle_deg):
    """
    Compute inner hexagon vertices given lattice parameters.
    lattice_params: (a, b, theta, rotation)
    - a, b: lattice basis vector lengths
    - theta: angle between basis vectors (degrees)
    - rotation: global rotation of the lattice (degrees)
    
    Returns: (12, 6, 2) array of vertices
    """
    a, b, theta_deg, rot_deg = lattice_params
    theta = np.radians(theta_deg)
    rot = np.radians(rot_deg)
    
    # Lattice basis vectors
    e1 = np.array([a, 0])
    e2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    
    # Apply global rotation
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    e1 = R @ e1
    e2 = R @ e2
    
    # 12 lattice points in a hexagonal arrangement
    # Using coordinates (i, j) where position = i*e1 + j*e2
    # For a compact hexagonal shape with 12 points:
    # Center + 6 neighbors + 5 more to make 12
    # Let's use a specific pattern:
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
    # (2,0), (-2,0), (0,2), (0,-2), (1,1)
    # Wait, that's not quite right for hexagonal symmetry.
    
    # Better: use the standard hexagonal packing positions
    # For 12 hexagons in a hexagonal arrangement:
    # A hexagonal number pattern: 1 + 6 + 5 = 12? No.
    # Let's use: center (0,0), 6 at distance 1, 5 at distance 2 (partial ring)
    
    # Actually, let's define specific (i,j) coordinates for 12 points
    # that form a compact cluster
    lattice_indices = [
        (0, 0),   # center
        (1, 0),   # right
        (-1, 0),  # left
        (0, 1),   # up
        (0, -1),  # down
        (1, -1),  # down-right
        (-1, 1),  # up-left
        (2, 0),   # far right
        (-2, 0),  # far left
        (0, 2),   # far up
        (0, -2),  # far down
        (1, 1),   # up-right
    ]
    
    centers = np.zeros((12, 2))
    for idx, (i, j) in enumerate(lattice_indices):
        centers[idx] = i * e1 + j * e2
    
    # Compute vertices for each inner hexagon
    vertices = np.zeros((12, 6, 2))
    for idx in range(12):
        cx, cy = centers[idx]
        verts = _hex_vertices(cx, cy, 1.0, inner_angle_deg)
        vertices[idx] = verts
    
    return vertices, centers


def _objective(params):
    """
    Objective function for optimization.
    params: (a, b, theta, rotation, inner_angle, outer_angle)
    """
    a, b, theta, rotation, inner_angle, outer_angle = params
    
    # Ensure valid lattice parameters
    if a < 1.5 or b < 1.5 or theta < 60 or theta > 120:
        return 1e10
    
    try:
        vertices, centers = _compute_inner_vertices([a, b, theta, rotation], inner_angle)
    except:
        return 1e10
    
    # Check for overlaps between inner hexagons using SAT
    for i in range(12):
        for j in range(i + 1, 12):
            vi = vertices[i]
            vj = vertices[j]
            
            # Check all edges of both hexagons as separating axes
            overlap = False
            for vset in [vi, vj]:
                for k in range(6):
                    p1 = vset[k]
                    p2 = vset[(k + 1) % 6]
                    edge = p2 - p1
                    # Normal to edge
                    normal = np.array([-edge[1], edge[0]])
                    norm_len = np.linalg.norm(normal)
                    if norm_len < 1e-10:
                        continue
                    normal = normal / norm_len
                    
                    # Project both hexagons onto this normal
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    # Check if projections overlap
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        # Separated, no overlap
                        overlap = False
                        break
                    else:
                        overlap = True
                if not overlap:
                    break
            
            if overlap:
                return 1e10
    
    # Compute minimum outer side length
    s = _min_outer_side(vertices.reshape(72, 2), outer_angle)
    
    # Ensure s is valid
    if s < np.sqrt(12):
        return 1e10
    
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 620  # Safety margin before 630s limit
    
    # Initial guess based on standard hexagonal packing
    # For unit hexagons in hexagonal lattice, nearest neighbor distance is 2*sqrt(3) ≈ 3.464
    # Lattice basis vectors for hexagonal packing:
    # e1 = (2*sqrt(3), 0), e2 = (sqrt(3), 3)
    a0 = 2 * np.sqrt(3)
    b0 = 2 * np.sqrt(3)
    theta0 = 60  # degrees
    rot0 = 0
    inner_angle0 = 0
    outer_angle0 = 0
    
    # Bounds for optimization
    bounds = [
        (1.5, 5.0),      # a: lattice basis vector 1 length
        (1.5, 5.0),      # b: lattice basis vector 2 length
        (50, 110),       # theta: angle between basis vectors
        (0, 60),         # rotation: global rotation
        (0, 60),         # inner_angle: inner hexagon rotation
        (0, 60),         # outer_angle: outer hexagon rotation
    ]
    
    best_params = [a0, b0, theta0, rot0, inner_angle0, outer_angle0]
    best_s = _objective(best_params)
    
    # If initial guess is invalid, try a few random starts
    if best_s >= 1e9:
        np.random.seed(42)
        for _ in range(20):
            params = [
                np.random.uniform(1.5, 4.0),
                np.random.uniform(1.5, 4.0),
                np.random.uniform(55, 105),
                np.random.uniform(0, 60),
                np.random.uniform(0, 60),
                np.random.uniform(0, 60),
            ]
            s = _objective(params)
            if s < best_s:
                best_s = s
                best_params = params
            if time.time() > deadline:
                break
    
    # Run differential evolution if we have a valid starting point
    if best_s < 1e9 and time.time() < deadline - 60:
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=False,
                callback=lambda x, convergence: None,
            )
            
            if result.fun < best_s:
                best_s = result.fun
                best_params = list(result.x)
        except Exception as e:
            pass
    
    # Use the best parameters found
    a, b, theta, rotation, inner_angle, outer_angle = best_params
    
    # Compute final positions
    vertices, centers = _compute_inner_vertices([a, b, theta, rotation], inner_angle)
    
    # Compute final outer side length
    s = _min_outer_side(vertices.reshape(72, 2), outer_angle)
    
    # If the result is invalid, fall back to the known valid grid
    if s < np.sqrt(12) - 1e-6:
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
        return inner_hex_data, outer_hex_data, s
    
    # Construct return values
    inner_hex_data = np.column_stack([centers, np.full(12, inner_angle)])
    outer_hex_data = np.array([0, 0, outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
