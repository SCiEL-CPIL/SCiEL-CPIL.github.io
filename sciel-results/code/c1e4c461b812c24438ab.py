import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute all C(11,3) = 165 triangle index combinations
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

# Centroid of the equilateral triangle
CENTROID = np.array([0.5, H / 3.0])

# Symmetry axes angles (3 axes at 60 degrees apart)
# The triangle has 3-fold rotational symmetry about the centroid
AXIS_ANGLES = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])

def _min_triangle_area(points):
    """Calculate the minimum area of all triangles formed by the points."""
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_point(p):
    """Project a single point into the equilateral triangle."""
    x, y = p
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    a = max(0.0, a)
    b = max(0.0, b)
    c = max(0.0, c)
    s = a + b + c
    if s < 1e-12:
        return CENTROID.copy()
    a /= s
    b /= s
    c /= s
    
    px = b * 1.0 + c * 0.5
    py = c * H
    return np.array([px, py])

def _project_points(points):
    """Project all points into the triangle."""
    return np.array([_project_point(p) for p in points])

def _polar_to_cartesian(polar_params):
    """Convert polar-like parameters to Cartesian coordinates.
    
    Each point is parameterized by:
    - r: distance from centroid (scaled to [0, 1])
    - angle_dev: angular deviation from nearest symmetry axis
    
    This enforces rotational structure and reduces effective dimensionality.
    """
    points = np.zeros((N, 2))
    for i in range(N):
        r = polar_params[2 * i]
        angle_dev = polar_params[2 * i + 1]
        
        # Find nearest symmetry axis
        # The angle of the point relative to centroid
        # We use the angle_dev to determine which sector and how far from axis
        # angle_dev is in [-pi/3, pi/3] relative to nearest axis
        
        # Determine which axis based on angle_dev
        # angle_dev in [-pi/3, pi/3] maps to axis 0
        # We need to map the full range properly
        
        # Actually, let's use a simpler approach:
        # angle_dev is the actual angle offset from the nearest axis
        # The base angle is determined by which sector
        # For simplicity, we assign each point to a sector based on index
        # and angle_dev gives the fine adjustment
        
        # Better approach: use angle_dev as absolute angle from centroid
        # but constrain it to be near symmetry axes
        # Actually, let's just use angle_dev as the polar angle directly
        # and let DE explore the space
        
        # Convert polar to cartesian relative to centroid
        # r is scaled: max distance from centroid to vertex is H/sqrt(3) * 2/3 * 3/2... 
        # Let me compute: centroid is at (0.5, H/3)
        # Distance to vertex (0,0): sqrt(0.25 + H^2/9) = sqrt(0.25 + 1/12) = sqrt(1/3) = 1/sqrt(3)
        # So max r in cartesian is 1/sqrt(3)
        
        r_cart = r / np.sqrt(3)  # Scale r to actual distance
        
        # Use angle_dev as the polar angle
        # But we want to encourage alignment with symmetry axes
        # Let's just use it directly
        angle = angle_dev
        
        px = CENTROID[0] + r_cart * np.cos(angle)
        py = CENTROID[1] + r_cart * np.sin(angle)
        
        points[i] = [px, py]
    
    return points

def _cartesian_to_polar(points):
    """Convert Cartesian coordinates to polar-like parameters."""
    polar_params = np.zeros(2 * N)
    for i in range(N):
        dx = points[i, 0] - CENTROID[0]
        dy = points[i, 1] - CENTROID[1]
        
        r_cart = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx)
        
        # Scale r back
        r = r_cart * np.sqrt(3)
        
        polar_params[2 * i] = r
        polar_params[2 * i + 1] = angle
    
    return polar_params

def _objective_polar(polar_params):
    """Objective function in polar coordinates: negative of min triangle area."""
    points = _polar_to_cartesian(polar_params)
    points = _project_points(points)
    return -_min_triangle_area(points)

def _objective_cartesian(flat):
    """Objective function in Cartesian coordinates: negative of min triangle area."""
    points = flat.reshape(N, 2)
    points = _project_points(points)
    return -_min_triangle_area(points)

def _structured_initial(seed):
    """Generate a structured initial configuration using symmetry."""
    rng = random.Random(seed)
    points = np.zeros((N, 2))
    
    # Place points in a pattern that respects the 3-fold symmetry
    # 3 vertices, 3 edge midpoints, 1 center, 4 more
    
    # Vertices (slightly inset)
    inset = 0.03
    vertices = [
        [inset, inset * H],
        [1 - inset, inset * H],
        [0.5, H - inset * H]
    ]
    points[0] = vertices[0]
    points[1] = vertices[1]
    points[2] = vertices[2]
    
    # Edge midpoints (slightly inset from edges)
    edge_mid = [
        [0.5, 0.02],
        [0.25 + 0.02 * 0.5, H * 0.5 - 0.02 * H],
        [0.75 - 0.02 * 0.5, H * 0.5 - 0.02 * H]
    ]
    points[3] = edge_mid[0]
    points[4] = edge_mid[1]
    points[5] = edge_mid[2]
    
    # Center
    points[6] = CENTROID.copy()
    
    # Remaining 4 points: distribute in the three sub-triangles
    # Each sub-triangle is formed by centroid and two adjacent vertices
    sub_triangles = [
        [CENTROID, vertices[0], vertices[1]],  # Bottom
        [CENTROID, vertices[1], vertices[2]],  # Right
        [CENTROID, vertices[2], vertices[0]],  # Left
    ]
    
    for i in range(4):
        tri_idx = i % 3
        tri = sub_triangles[tri_idx]
        # Random point in triangle
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1 - u, 1 - v
        # Barycentric interpolation
        px = u * tri[0][0] + v * tri[1][0] + (1 - u - v) * tri[2][0]
        py = u * tri[0][1] + v * tri[1][1] + (1 - u - v) * tri[2][1]
        points[6 + i] = [px, py]
    
    points = _project_points(points)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1e10
    
    def _update_best(points):
        nonlocal best_points, best_score
        score = _min_triangle_area(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score
    
    # Phase 1: Differential Evolution in polar coordinates
    # Bounds for polar parameters:
    # r: [0, 1] (scaled)
    # angle: [-pi, pi]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # r
        bounds.append((-np.pi, np.pi))  # angle
    
    try:
        result_de = differential_evolution(
            _objective_polar,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            atol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol',
            updating='immediate'
        )
        
        if result_de.x is not None:
            opt_points = _polar_to_cartesian(result_de.x)
            opt_points = _project_points(opt_points)
            _update_best(opt_points)
    except Exception:
        pass
    
    # Phase 2: Local optimization from best DE result
    if time.time() < deadline - 10.0 and best_points is not None:
        flat = best_points.flatten()
        try:
            result_local = minimize(
                _objective_cartesian,
                flat,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            opt_points = result_local.x.reshape(N, 2)
            opt_points = _project_points(opt_points)
            _update_best(opt_points)
        except Exception:
            pass
    
    # Phase 3: Multi-start local optimization from diverse initial points
    if time.time() < deadline - 5.0:
        for r in range(15):
            if time.time() > deadline - 3.0:
                break
            
            if r < 5:
                points = _structured_initial(42 + r)
            else:
                # Random points in triangle
                rng = random.Random(100 + r)
                points = []
                for _ in range(N):
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        px = u + 0.5 * v
                        py = H * v
                        points.append([px, py])
                points = _project_points(np.array(points))
            
            flat = points.flatten()
            try:
                result_local = minimize(
                    _objective_cartesian,
                    flat,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                opt_points = result_local.x.reshape(N, 2)
                opt_points = _project_points(opt_points)
                _update_best(opt_points)
            except Exception:
                pass
    
    # If no good result, use structured initial
    if best_points is None:
        best_points = _structured_initial(7)
    
    return best_points
