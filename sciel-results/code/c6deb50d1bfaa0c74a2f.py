"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place N=12 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy:
    1. Define a parameterization for inner hexagon positions and rotations.
    2. Use a smooth surrogate for the collision constraint (minimum distance).
    3. Optimize using L-BFGS-B with a penalty method or constrained optimization.
    4. The outer hexagon is determined by the convex hull of the inner hexagon vertices.
    
    We minimize the outer side length. The outer hexagon is circumscribed around the
    inner hexagons. For a regular hexagon, the side length is related to the circumradius.
    Actually, the "outer hexagon" in the task is a regular hexagon that contains all inner hexagons.
    We need to find the minimum side length of such a regular hexagon (with variable center and rotation).
    
    This is a non-convex problem. We will use a local optimization approach:
    - Parameters: positions (x_i, y_i) and rotations (theta_i) for each inner hexagon.
    - Objective: the minimum side length of a regular hexagon that contains all inner hexagons.
    
    To compute the minimum side length of a regular hexagon containing a set of points (vertices of inner hexagons):
    - The outer hexagon has center (cx, cy) and rotation phi.
    - For a given direction, the support function of the outer hexagon must be >= the support function of the inner set.
    
    A simpler approach: 
    - The outer hexagon is defined by its center (cx, cy), side length s, and rotation phi.
    - The inner hexagons must be inside it.
    - We can fix the outer hexagon's rotation to 0 (WLOG, since the whole configuration can be rotated) and optimize cx, cy, s, and the inner hexagon parameters.
    - But the task allows the outer hexagon to rotate, so we should optimize phi as well.
    
    Let's use the following approach:
    - Parameters: 
        - inner positions: 2N
        - inner rotations: N
        - outer center: 2
        - outer side length: 1
        - outer rotation: 1
    - Objective: outer side length
    - Constraints: all inner hexagon vertices must be inside the outer hexagon.
    
    This is a large-scale non-convex problem. We will use a penalty method with L-BFGS-B.
    
    For a regular hexagon with side length s, center (cx, cy), and rotation phi:
    The six vertices are at:
        (cx + s * cos(phi + k*pi/3), cy + s * sin(phi + k*pi/3)) for k=0..5
    The six edges are lines. A point (x, y) is inside if it satisfies all six half-plane inequalities.
    
    For a regular hexagon, the distance from center to each edge is s * sqrt(3) / 2 (the inradius).
    The normal to the k-th edge is at angle phi + k*pi/3 + pi/2.
    The inequality for the k-th edge is:
        (x - cx) * cos(phi + k*pi/3 + pi/2) + (y - cy) * sin(phi + k*pi/3 + pi/2) <= s * sqrt(3) / 2
    
    Wait, let's verify. The edge opposite to the vertex at angle phi + k*pi/3 is at distance s*sqrt(3)/2 from the center in the direction phi + k*pi/3.
    Actually, the support function of a regular hexagon with side s in direction theta is:
        h(theta) = s * sqrt(3) / 2 * (1 + cos(3*(theta - phi)))  ??? No.
    
    Let's derive it properly. A regular hexagon with side s can be seen as the intersection of 6 half-planes.
    The distance from the center to each edge (inradius) is r = s * sqrt(3) / 2.
    The outward normals of the 6 edges are at angles: phi + k*pi/3 + pi/6 for k=0..5? Or phi + k*pi/3?
    
    Let's use a standard form. A regular hexagon with side s, center at origin, and one vertex at angle 0:
    Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    Wait, that's not right. A regular hexagon with side s has circumradius s.
    Vertices: (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5.
    The edges connect adjacent vertices. The edge between vertex 0 and vertex 1 is a line.
    The distance from the origin to this edge is s * sqrt(3) / 2.
    The normal to this edge points in the direction of the midpoint of the edge, which is at angle pi/6.
    
    So the 6 edge normals are at angles: pi/6 + k*pi/3 for k=0..5.
    The inequality for the k-th edge is:
        x * cos(pi/6 + k*pi/3) + y * sin(pi/6 + k*pi/3) <= s * sqrt(3) / 2
    
    For a hexagon with center (cx, cy) and rotation phi, the inequality becomes:
        (x - cx) * cos(phi + pi/6 + k*pi/3) + (y - cy) * sin(phi + pi/6 + k*pi/3) <= s * sqrt(3) / 2
    
    So for a set of points P (the vertices of all inner hexagons), the minimum side length s of a regular hexagon (with center (cx, cy) and rotation phi) that contains all points is:
        s = max_{p in P, k in 0..5} [ (p_x - cx) * cos(phi + pi/6 + k*pi/3) + (p_y - cy) * sin(phi + pi/6 + k*pi/3) ] / (sqrt(3) / 2)
    
    We can optimize over cx, cy, phi, and the inner hexagon parameters to minimize this s.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Parameters:
    # x[0:N], y[0:N]: inner hexagon centers
    # theta[N:2N]: inner hexagon rotations
    # cx, cy: outer hexagon center
    # phi: outer hexagon rotation
    # s: outer hexagon side length (we'll optimize this too, but it's determined by the constraints)
    
    # Actually, s is determined by the maximum violation. So we can just minimize s subject to the constraints.
    # Or we can minimize the maximum violation and then compute s.
    
    # Let's use a penalty method:
    # Objective: s + lambda * sum(max(0, violation_i)^2)
    # But s is not a variable; it's computed from the positions.
    
    # Better approach: 
    # Variables: x, y, theta, cx, cy, phi
    # Objective: compute the minimum s required to contain all inner hexagons.
    
    def get_inner_hexagon_vertices(cx_i, cy_i, theta_i):
        """Return the 6 vertices of a unit regular hexagon."""
        # Unit hexagon: side length 1, circumradius 1
        angles = theta_i + np.pi / 3 * np.arange(6)
        return np.stack([cx_i + np.cos(angles), cy_i + np.sin(angles)], axis=1)
    
    def compute_outer_side(x, y, theta, cx, cy, phi):
        """Compute the minimum side length of a regular hexagon containing all inner hexagons."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for i in range(N):
            verts = get_inner_hexagon_vertices(x[i], y[i], theta[i])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # Compute the maximum support function value over all vertices and all 6 edge directions
        # Edge normal angles: phi + pi/6 + k*pi/3 for k=0..5
        edge_angles = phi + np.pi / 6 + np.pi / 3 * np.arange(6)
        cos_edges = np.cos(edge_angles)
        sin_edges = np.sin(edge_angles)
        
        # For each vertex, compute the projection onto each edge normal
        # (all_vertices - center) @ edge_normal
        rel_vertices = all_vertices - np.array([cx, cy])
        projections = rel_vertices @ np.stack([cos_edges, sin_edges], axis=1)
        
        # The maximum projection over all vertices and all edges
        max_projection = np.max(projections)
        
        # The side length s is related to the inradius r = s * sqrt(3) / 2
        # So s = max_projection / (sqrt(3) / 2)
        s = max_projection / (np.sqrt(3) / 2)
        
        return s
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        x = params[0:N]
        y = params[N:2*N]
        theta = params[2*N:3*N]
        cx = params[3*N]
        cy = params[3*N + 1]
        phi = params[3*N + 2]
        
        s = compute_outer_side(x, y, theta, cx, cy, phi)
        
        # Add a small penalty for large positions to keep the solution bounded
        penalty = 0.01 * np.sum(x**2 + y**2)
        
        return s + penalty
    
    # Initial guess: place hexagons in a ring or grid
    np.random.seed(42)
    
    # Initial positions: a hexagonal arrangement
    # For 12 hexagons, a 3x4 grid or a ring of 12
    # Let's try a ring of 12 hexagons around the center
    initial_params = np.zeros(3*N + 3)
    for i in range(N):
        angle = 2 * np.pi * i / N
        radius = 2.0  # Initial radius for the ring
        initial_params[i] = radius * np.cos(angle)
        initial_params[N + i] = radius * np.sin(angle)
        initial_params[2*N + i] = 0.0  # Initial rotation
    initial_params[3*N] = 0.0  # cx
    initial_params[3*N + 1] = 0.0  # cy
    initial_params[3*N + 2] = 0.0  # phi
    
    # Optimize
    best_params = initial_params
    best_score = objective(best_params)
    
    # Use L-BFGS-B
    options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    result = minimize(objective, best_params, method='L-BFGS-B', options=options)
    
    if result.success or result.fun < best_score:
        best_params = result.x
        best_score = result.fun
    
    # Extract the solution
    x = best_params[0:N]
    y = best_params[N:2*N]
    theta = best_params[2*N:3*N]
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    phi = best_params[3*N + 2]
    
    # Compute the final side length
    s = compute_outer_side(x, y, theta, cx, cy, phi)
    
    # Convert theta from radians to degrees
    theta_deg = np.degrees(theta)
    
    # Format the output
    inner = [[x[i], y[i], theta_deg[i]] for i in range(N)]
    center = [cx, cy]
    side = s
    angle_deg = np.degrees(phi)
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
