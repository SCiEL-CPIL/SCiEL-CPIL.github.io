"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _min_distance_hexagons(v1, v2):
    """Compute minimum distance between two convex hexagons (vertex sets).
    Returns negative if overlapping, positive if separated."""
    n1, n2 = len(v1), len(v2)
    min_dist = float('inf')
    # Check all pairs of edges from both hexagons
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            # Distance between two line segments
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < min_dist:
                min_dist = d
    # Also check vertex-to-vertex distances
    for i in range(n1):
        for j in range(n2):
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_dist:
                min_dist = d
    # Check if one hexagon contains the other (approximate by checking if all vertices of one are inside the other)
    # For simplicity, if min_dist is 0 or negative, they overlap
    return min_dist

def _seg_seg_dist(p1, p2, q1, q2):
    """Minimum distance between two line segments."""
    # Project p1 onto segment q1-q2
    d = q2 - q1
    if np.dot(d, d) < 1e-12:
        return np.linalg.norm(p1 - q1)
    t = np.dot(p1 - q1, d) / np.dot(d, d)
    t = max(0, min(1, t))
    closest = q1 + t * d
    dist = np.linalg.norm(p1 - closest)
    
    # Project p2 onto segment q1-q2
    t = np.dot(p2 - q1, d) / np.dot(d, d)
    t = max(0, min(1, t))
    closest = q1 + t * d
    dist = min(dist, np.linalg.norm(p2 - closest))
    
    # Project q1 onto segment p1-p2
    d2 = p2 - p1
    if np.dot(d2, d2) < 1e-12:
        return min(dist, np.linalg.norm(q1 - p1))
    t = np.dot(q1 - p1, d2) / np.dot(d2, d2)
    t = max(0, min(1, t))
    closest = p1 + t * d2
    dist = min(dist, np.linalg.norm(q1 - closest))
    
    # Project q2 onto segment p1-p2
    t = np.dot(q2 - p1, d2) / np.dot(d2, d2)
    t = max(0, min(1, t))
    closest = p1 + t * d2
    dist = min(dist, np.linalg.norm(q2 - closest))
    
    return dist

def _is_point_in_hexagon(point, hex_verts):
    """Check if a point is inside a convex hexagon."""
    n = len(hex_verts)
    for i in range(n):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % n]
        # Cross product to check if point is on the same side
        cross = np.cross(p2 - p1, point - p1)
        if cross < -1e-9:  # Allow small tolerance
            return False
    return True

def _enclosing_side_length(inner_positions, outer_center, outer_angle_deg):
    """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
    ox, oy = outer_center
    # For each inner hexagon, compute the maximum support in each of the 6 directions
    # of the outer hexagon
    max_supports = [0.0] * 6
    
    for (ix, iy, iangle) in inner_positions:
        # Get vertices of inner hexagon
        inner_verts = _hex_vertices(ix, iy, 1.0, iangle)
        
        # For each of the 6 normal directions of the outer hexagon
        for k in range(6):
            theta = np.radians(outer_angle_deg) + k * np.pi / 3
            # Normal direction
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            # Support function: max of dot product of vertices with normal
            supports = inner_verts @ np.array([nx, ny])
            max_support = np.max(supports)
            
            # The distance from outer center to this support point projected onto normal
            # The outer hexagon needs to extend at least this far in this direction
            # from its center
            center_support = ox * nx + oy * ny
            required = max_support - center_support
            if required > max_supports[k]:
                max_supports[k] = required
    
    # The side length of outer hexagon must satisfy:
    # For a regular hexagon with side s, the apothem (distance from center to edge) is s * sqrt(3) / 2
    # So s = 2 * max_supports / sqrt(3)
    max_required = max(max_supports)
    side = 2.0 * max_required / np.sqrt(3)
    return side

def _objective(params):
    """Objective function: outer side length + penalty for overlapping inner hexagons."""
    n = N
    # Unpack parameters: for each hexagon, (x, y, angle_deg)
    positions = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2]
        positions.append((x, y, angle))
    
    # Compute outer hexagon center as centroid of inner hexagons
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    outer_center = (cx, cy)
    outer_angle = 0.0  # Keep outer hexagon axis-aligned for simplicity
    
    # Compute enclosing side length
    side = _enclosing_side_length(positions, outer_center, outer_angle)
    
    # Penalty for overlapping inner hexagons
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
            vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
            min_dist = _min_distance_hexagons(vi, vj)
            if min_dist < 0.01:  # Overlapping or too close
                penalty += (0.01 - min_dist) ** 2
    
    return side + 100.0 * penalty

def _gradient(params):
    """Numerical gradient of the objective function."""
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial positions: hexagonal lattice arrangement
    positions = []
    for k in range(N):
        x = 2.0 * k
        y = 0.0
        angle = 0.0
        positions.append((x, y, angle))
    
    # Initial parameters
    params = np.array([p for pos in positions for p in pos])
    
    # Best solution found so far
    best_params = params.copy()
    best_score = _objective(params)
    
    # Try optimization with L-BFGS-B
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            jac=_gradient,
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if time.time() < deadline:
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
    except Exception as e:
        pass
    
    # Try a few more restarts if time permits
    restarts = 3
    for r in range(restarts):
        if time.time() >= deadline - 20.0:
            break
        
        # Perturb initial positions
        np.random.seed(42 + r)
        perturbed = best_params.copy()
        for i in range(len(perturbed)):
            perturbed[i] += np.random.normal(0, 0.5)
        
        try:
            result = minimize(
                _objective,
                perturbed,
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            
            if time.time() < deadline:
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
        except Exception as e:
            pass
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([x, y, angle])
    
    # Compute outer hexagon center and side
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    outer_center = (float(cx), float(cy))
    outer_angle = 0.0
    side = _enclosing_side_length(inner, outer_center, outer_angle)
    
    return inner, outer_center, float(side), outer_angle
# EVOLVE-BLOCK-END
