# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    n = 11
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 540  # 9 minutes, safety margin for 630s limit
    
    def get_hex_vertices(cx, cy, theta_rad):
        """Returns vertices of a unit-side regular hexagon at (cx,cy) rotated by theta_rad."""
        angles = theta_rad + np.arange(6) * np.pi / 3
        return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))
    
    def get_hex_edge_normals(cx, cy, theta_rad):
        """Returns 6 outward edge normals of a unit-side regular hexagon."""
        # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees + theta
        normals = theta_rad + np.pi/6 + np.arange(6) * np.pi / 3
        return np.column_stack((np.cos(normals), np.sin(normals)))
    
    def hexagon_separation(v1, n1, v2, n2):
        """
        Check if two hexagons are separated.
        v1, n1: vertices and normals of hex 1
        v2, n2: vertices and normals of hex 2
        Returns True if separated (non-overlapping), False if overlapping.
        """
        # Project both hexagons onto all 12 edge normals (6 from each)
        for normals, v_other in [(n1, v2), (n2, v1)]:
            for k in range(6):
                n = normals[k]
                proj_self = np.dot(v1, n)
                proj_other = np.dot(v_other, n)
                
                min_self = np.min(proj_self)
                max_self = np.max(proj_self)
                min_other = np.min(proj_other)
                max_other = np.max(proj_other)
                
                # Check if intervals are apart by more than 1e-6
                # Either max_self < min_other - 1e-6 OR max_other < min_self - 1e-6
                if max_self < min_other - 1e-6 or max_other < min_self - 1e-6:
                    return True  # Separated on this normal
        return False  # Not separated on any normal -> overlapping
    
    def outer_hex_vertices(S):
        """Returns vertices of outer hexagon centered at origin with side length S."""
        angles = np.arange(6) * np.pi / 3
        return S * np.column_stack((np.cos(angles), np.sin(angles)))
    
    def outer_hex_edge_normals(S):
        """Returns outward edge normals of outer hexagon."""
        normals = np.pi/6 + np.arange(6) * np.pi / 3
        return np.column_stack((np.cos(normals), np.sin(normals)))
    
    def is_inside_outer(vertices, S, tol=1e-6):
        """Check if all vertices are inside or on the outer hexagon."""
        normals = outer_hex_edge_normals(S)
        apothem = S * np.sqrt(3) / 2.0
        
        for n in normals:
            proj = np.dot(vertices, n)
            if np.max(proj) > apothem + tol:
                return False
        return True
    
    def objective(params):
        """
        Objective: minimize outer side length S.
        params: [x1, y1, th1, x2, y2, th2, ..., x11, y11, th11, S]
        Returns S + penalty for overlaps and containment violations.
        """
        if len(params) != n * 3 + 1:
            return 1e10
        
        xs = params[0:n*3:3]
        ys = params[1:n*3:3]
        ths = params[2:n*3:3]
        S = params[n*3]
        
        if S < np.sqrt(11) - 1e-9:
            return 1e10
        if S <= 0:
            return 1e10
        
        # Compute all vertices
        verts = []
        for i in range(n):
            verts.append(get_hex_vertices(xs[i], ys[i], ths[i]))
        
        penalty = 0.0
        
        # Check containment
        for i in range(n):
            if not is_inside_outer(verts[i], S, tol=-1e-4):
                # Compute how far outside
                normals = outer_hex_edge_normals(S)
                apothem = S * np.sqrt(3) / 2.0
                for nrm in normals:
                    proj = np.dot(verts[i], nrm)
                    violation = np.max(proj) - apothem
                    if violation > 0:
                        penalty += violation * 10.0
        
        # Check separation between all pairs
        for i in range(n):
            for j in range(i+1, n):
                n1 = get_hex_edge_normals(xs[i], ys[i], ths[i])
                n2 = get_hex_edge_normals(xs[j], ys[j], ths[j])
                
                # Check if overlapping by computing minimum gap
                # We need to find the minimum gap over all 12 normals
                min_gap = np.inf
                for normals, v_other in [(n1, verts[j]), (n2, verts[i])]:
                    for k in range(6):
                        n = normals[k]
                        proj_self = np.dot(verts[i], n)
                        proj_other = np.dot(v_other, n)
                        
                        max_self = np.max(proj_self)
                        min_self = np.min(proj_self)
                        max_other = np.max(proj_other)
                        min_other = np.min(proj_other)
                        
                        # Gap is the distance between intervals
                        # If max_self < min_other, gap = min_other - max_self
                        # If max_other < min_self, gap = min_self - max_other
                        # Otherwise, they overlap on this normal
                        if max_self < min_other:
                            gap = min_other - max_self
                        elif max_other < min_self:
                            gap = min_self - max_other
                        else:
                            gap = 0.0  # Overlap on this normal
                        
                        min_gap = min(min_gap, gap)
                
                # We need min_gap > 1e-6 for validity
                if min_gap < 1e-4:
                    # Penalty for being too close or overlapping
                    if min_gap < 1e-6:
                        penalty += (1e-4 - min_gap) * 50.0
                    else:
                        penalty += (1e-4 - min_gap) * 5.0
        
        return S + penalty
    
    # Initial guess: hexagonal arrangement
    # Center at origin, 6 around it, 4 in the gaps
    init_xs = np.zeros(n)
    init_ys = np.zeros(n)
    init_ths = np.zeros(n)
    
    # Center
    init_xs[0] = 0
    init_ys[0] = 0
    
    # 6 surrounding
    for i in range(6):
        angle = i * np.pi / 3
        init_xs[i+1] = 2.5 * np.cos(angle)
        init_ys[i+1] = 2.5 * np.sin(angle)
    
    # 4 more - placed further out
    init_xs[7] = -3.75
    init_ys[7] = 2.17
    init_xs[8] = 3.75
    init_ys[8] = 2.17
    init_xs[9] = -3.75
    init_ys[9] = -2.17
    init_xs[10] = 3.75
    init_ys[10] = -2.17
    
    init_params = np.concatenate([init_xs, init_ys, init_ths, [6.0]])
    
    best_params = init_params.copy()
    best_score = 1e10
    
    # Try multiple random restarts
    np.random.seed(42)
    
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Add noise for restarts > 0
        if restart == 0:
            params0 = init_params.copy()
        else:
            params0 = init_params.copy()
            params0[:-1] += np.random.randn(len(params0)-1) * 0.5
            params0[-1] = max(np.sqrt(11) + 0.5, init_params[-1] + np.random.randn() * 0.5)
        
        # Optimize
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception as e:
            pass
    
    # Extract best solution
    xs = best_params[0:n*3:3]
    ys = best_params[1:n*3:3]
    ths = best_params[2:n*3:3]
    S = best_params[n*3]
    
    # Ensure S is at least sqrt(11)
    S = max(S, np.sqrt(11) + 1e-6)
    
    # Final validation and adjustment
    # Check if solution is valid, if not, increase S
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        verts = [get_hex_vertices(xs[i], ys[i], ths[i]) for i in range(n)]
        
        valid = True
        
        # Check containment
        if not all(is_inside_outer(verts[i], S) for i in range(n)):
            valid = False
        
        # Check separation
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    n1 = get_hex_edge_normals(xs[i], ys[i], ths[i])
                    n2 = get_hex_edge_normals(xs[j], ys[j], ths[j])
                    
                    min_gap = np.inf
                    for normals, v_other in [(n1, verts[j]), (n2, verts[i])]:
                        for k in range(6):
                            n = normals[k]
                            proj_self = np.dot(verts[i], n)
                            proj_other = np.dot(v_other, n)
                            
                            max_self = np.max(proj_self)
                            min_self = np.min(proj_self)
                            max_other = np.max(proj_other)
                            min_other = np.min(proj_other)
                            
                            if max_self < min_other:
                                gap = min_other - max_self
                            elif max_other < min_self:
                                gap = min_self - max_other
                            else:
                                gap = 0.0
                            
                            min_gap = min(min_gap, gap)
                    
                    if min_gap <= 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            break
        
        # If invalid, increase S
        S += 0.05
    
    # Convert angles to degrees
    ths_deg = np.degrees(ths)
    
    inner_hex_data = np.column_stack((xs, ys, ths_deg))
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
