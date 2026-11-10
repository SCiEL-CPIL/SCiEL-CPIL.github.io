# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    n = 12
    # Initial grid arrangement of inner hexagons
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
    
    # Variables: [x1,y1,a1, ..., x12,y12,a12, ox,oy,oa, s]
    # Total: 12*3 + 3 + 1 = 40 variables
    x0 = np.concatenate([inner_hex_data.flatten(), [0, 0, 0, 8.0]])
    
    def compute_vertices(cx, cy, d, angle_deg):
        """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
        angles = np.radians(angle_deg) + np.radians(60 * np.arange(6))
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])
    
    def get_separation_violation(v1, v2):
        """
        Check if two hexagons (given as vertex arrays) overlap.
        Returns the minimum separation violation (positive if overlapping, 0 if touching or separated).
        """
        max_overlap = 0
        # Check all edges from both hexagons as separating axes
        for vertices in [v1, v2]:
            for i in range(6):
                p1 = vertices[i]
                p2 = vertices[(i+1) % 6]
                edge = p2 - p1
                # Normal perpendicular to edge
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                
                # Project both hexagons onto this normal
                proj1 = vertices @ normal
                proj2 = v2 @ normal if vertices is v1 else v1 @ normal
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Overlap on this axis
                overlap = min(max1, max2) - max(min1, min2)
                if overlap > max_overlap:
                    max_overlap = overlap
        
        return max_overlap
    
    def objective_and_constraints(x):
        """
        Objective: minimize s (outer hexagon side length)
        Constraints: 
        1. All inner hexagons must be inside the outer hexagon
        2. No two inner hexagons can overlap
        """
        if time.time() > deadline:
            return 1e10, []
        
        # Unpack variables
        inner_params = x[:36].reshape(12, 3)
        ox, oy, oa = x[36:39]
        s = x[39]
        
        # Compute outer hexagon vertices
        outer_vertices = compute_vertices(ox, oy, s, oa)
        
        # Compute inner hexagon vertices
        inner_vertices = []
        for i in range(12):
            cx, cy, angle = inner_params[i]
            v = compute_vertices(cx, cy, 1.0, angle)
            inner_vertices.append(v)
        
        # Objective: s
        obj = s
        
        # Constraints
        constraints = []
        
        # Constraint 1: All inner hexagon vertices must be inside outer hexagon
        for i in range(12):
            for j in range(6):
                v = inner_vertices[i][j]
                # Check if v is inside outer hexagon using cross product test
                for k in range(6):
                    p1 = outer_vertices[k]
                    p2 = outer_vertices[(k+1) % 6]
                    edge = p2 - p1
                    cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                    # cross must be >= -1e-6 (inside or on boundary)
                    constraints.append(cross + 1e-6)
        
        # Constraint 2: No two inner hexagons overlap
        for i in range(12):
            for j in range(i+1, 12):
                sep = get_separation_violation(inner_vertices[i], inner_vertices[j])
                # sep must be <= 0 (no overlap, touching is ok)
                constraints.append(-sep + 1e-6)
        
        return obj, constraints
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        obj, _ = objective_and_constraints(x)
        return obj
    
    def constraint_func(x):
        if time.time() > deadline:
            return np.zeros(1) * 1e10
        _, constraints = objective_and_constraints(x)
        return np.array(constraints)
    
    # Use SLSQP with constraints
    constraints_slsqp = [{'type': 'ineq', 'fun': constraint_func}]
    
    best_x = x0.copy()
    best_s = x0[39]
    best_result = None
    
    try:
        # First, try to find a feasible solution with reasonable s
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints_slsqp,
            bounds=[(None, None)] * 39 + [(2.0, 20.0)],
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success and result.fun < best_s:
            best_x = result.x
            best_s = result.x[39]
            best_result = result
            
        # Try to further optimize
        if time.time() < deadline:
            result2 = minimize(
                objective,
                best_x,
                method='SLSQP',
                constraints=constraints_slsqp,
                bounds=[(None, None)] * 39 + [(2.0, 20.0)],
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            if result2.success and result2.fun < best_s:
                best_x = result2.x
                best_s = result2.x[39]
                best_result = result2
    
    except Exception as e:
        pass
    
    # If optimization didn't work well, try a simpler approach: just shrink the outer hexagon
    if best_s > 8.0 or best_result is None:
        # Try to find the minimum s that works with the original positions
        inner_params = x0[:36].reshape(12, 3)
        
        def min_s_for_positions(inner_params):
            # Find minimum s such that all inner hexagons fit in outer hexagon centered at origin
            # Try different angles for outer hexagon
            best_s = 100.0
            for oa in range(0, 60, 5):
                # For a given outer angle, find minimum s
                # The constraint is that all inner vertices must be inside
                # For a regular hexagon, the distance from center to any point inside
                # is bounded by the inradius for points in certain directions
                
                # Simple approach: try increasing s until it works
                s = 5.0
                while s < 20.0:
                    outer_vertices = compute_vertices(0, 0, s, oa)
                    valid = True
                    for i in range(12):
                        cx, cy, angle = inner_params[i]
                        v = compute_vertices(cx, cy, 1.0, angle)
                        for j in range(6):
                            for k in range(6):
                                p1 = outer_vertices[k]
                                p2 = outer_vertices[(k+1) % 6]
                                edge = p2 - p1
                                cross = edge[0] * (v[j, 1] - p1[1]) - edge[1] * (v[j, 0] - p1[0])
                                if cross < -1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        best_s = min(best_s, s)
                        break
                    s += 0.1
            return best_s
        
        s_simple = min_s_for_positions(inner_params)
        if s_simple < best_s:
            best_s = s_simple
            best_x = np.concatenate([inner_params.flatten(), [0, 0, 0, s_simple]])
    
    # Extract final result
    inner_hex_data = best_x[:36].reshape(12, 3)
    outer_hex_data = best_x[36:39]
    outer_hex_side_length = best_x[39]
    
    # Ensure angles are in [0, 360)
    inner_hex_data[:, 2] = inner_hex_data[:, 2] % 360
    outer_hex_data[2] = outer_hex_data[2] % 360
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
