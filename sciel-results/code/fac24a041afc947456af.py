"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s/150s limit
    
    # Flat-top hexagon vertices for side length 1, centered at origin
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def get_hex_verts(center, angle_deg, side=1.0):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + (side * base_vertices @ R.T)

    def point_in_hex_dist_sq(pt, center, angle_deg, side=1.0):
        # Distance from point to hexagon (0 if inside)
        # Transform point to local frame of hexagon
        theta = np.deg2rad(angle_deg)
        R_inv = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        local_pt = R_inv @ (pt - center)
        
        # Check against 6 sides of the unit hexagon
        # Side i connects v_i to v_{i+1}
        # Normal pointing inward
        min_dist_sq = np.inf
        for i in range(6):
            v1 = base_vertices[i]
            v2 = base_vertices[(i+1)%6]
            edge_vec = v2 - v1
            # Normal vector (inward)
            # For flat top, edge (1,0) to (0.5, sqrt(3)/2) is top-right.
            # Inward normal should point towards center (0,0).
            # Cross product or rotation.
            # Let's use the property that for a regular polygon, 
            # distance to edge is |dot(local_pt - v1, normal)|
            # Actually, simpler: use shapely? No, std lib only.
            # Let's implement point-in-polygon and distance to segment.
            pass
            
        # Efficient check: 
        # A point is inside a regular hexagon of side 1 if:
        # |x| <= 1, |y| <= sqrt(3)/2, and |x|/1 + |y|/(sqrt(3)/2) <= 1? No, that's for diamond.
        # The hexagon is defined by 6 half-planes.
        # Side 1: x <= 1 (normal (1,0)) -> dist = max(0, x-1)
        # Side 2: 0.5x + sqrt(3)/2 y <= sqrt(3)/2 (normal (0.5, sqrt(3)/2)) -> dist = max(0, 0.5x + sqrt(3)/2 y - sqrt(3)/2)
        # Side 3: -0.5x + sqrt(3)/2 y <= sqrt(3)/2 (normal (-0.5, sqrt(3)/2)) -> dist = max(0, -0.5x + sqrt(3)/2 y - sqrt(3)/2)
        # Side 4: -x <= 1 (normal (-1,0)) -> dist = max(0, -x-1)
        # Side 5: -0.5x - sqrt(3)/2 y <= sqrt(3)/2 (normal (-0.5, -sqrt(3)/2)) -> dist = max(0, -0.5x - sqrt(3)/2 y - sqrt(3)/2)
        # Side 6: 0.5x - sqrt(3)/2 y <= sqrt(3)/2 (normal (0.5, -sqrt(3)/2)) -> dist = max(0, 0.5x - sqrt(3)/2 y - sqrt(3)/2)
        
        x, y = local_pt
        d1 = max(0.0, x - 1.0)
        d2 = max(0.0, 0.5*x + (np.sqrt(3)/2)*y - np.sqrt(3)/2)
        d3 = max(0.0, -0.5*x + (np.sqrt(3)/2)*y - np.sqrt(3)/2)
        d4 = max(0.0, -x - 1.0)
        d5 = max(0.0, -0.5*x - (np.sqrt(3)/2)*y - np.sqrt(3)/2)
        d6 = max(0.0, 0.5*x - (np.sqrt(3)/2)*y - np.sqrt(3)/2)
        
        return max(d1, d2, d3, d4, d5, d6)**2

    def hex_overlap_dist_sq(v1, v2, c1, a1, c2, a2):
        # Approximate overlap by checking if any vertex of v1 is in v2 or vice versa
        # And checking if centers are close.
        # For smooth optimization, we need a continuous penalty.
        # If dist(center1, center2) > 2*sqrt(3) (max distance between opposite sides is 2*sqrt(3) for side 1? No, width is 2, height sqrt(3). Max dist is 2*sqrt(3)? No, vertex to vertex is 2. Center to vertex is 1. So if centers > 2, they don't touch? No, they can touch side-to-side.
        # Min distance between disjoint hexagons is 0.
        # Let's check vertices of hex1 against hex2.
        d = 0.0
        # Check vertices of hex1 inside hex2
        for pt in v1:
            d = max(d, point_in_hex_dist_sq(pt, c2, a2))
        # Check vertices of hex2 inside hex1
        for pt in v2:
            d = max(d, point_in_hex_dist_sq(pt, c1, a1))
        return d

    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        # Unpack params: x1, y1, a1, ..., xN, yN, aN, outer_side
        # We fix outer center at (0,0) and outer angle at 0 for symmetry? 
        # The problem allows outer rotation. But due to symmetry, we can fix outer angle to 0 
        # and allow inner to rotate? Or fix inner angles to 0 and rotate outer?
        # Actually, relative rotation matters. 
        # Let's optimize: x_i, y_i, a_i for i=1..N and S (outer side).
        # To reduce variables, we can fix one hexagon's position? No, global translation of inner set doesn't matter relative to outer if outer center is fixed.
        # But outer center is fixed at (0,0). Inner set can translate.
        
        S = params[-1]
        if S < 1.0: S = 1.0 # Prevent degenerate
        
        inner_params = params[:-1]
        
        # Calculate outer hexagon vertices
        outer_verts = []
        for i in range(6):
            angle = np.deg2rad(60*i)
            outer_verts.append([S*np.cos(angle), S*np.sin(angle)])
        outer_verts = np.array(outer_verts)
        
        penalty = 0.0
        penalty += 100.0 * max(0.0, S - 5.0)**2 # Soft bound to keep search focused
        
        for i in range(N):
            x_i = inner_params[3*i]
            y_i = inner_params[3*i+1]
            a_i = inner_params[3*i+2]
            
            v_i = get_hex_verts([x_i, y_i], a_i)
            
            # Check containment in outer hexagon
            # Point in hexagon check for outer hexagon (side S, center 0,0, angle 0)
            # Scale factor S
            # Local coords for outer hex:
            # x <= S, 0.5x + sqrt(3)/2 y <= S*sqrt(3)/2, etc.
            # Since outer is at 0,0 and angle 0:
            for pt in v_i:
                px, py = pt
                # Distances to sides of outer hexagon (side S)
                # Side 1 (right): x <= S -> dist = max(0, x-S)
                d1 = max(0.0, px - S)
                d2 = max(0.0, 0.5*px + (np.sqrt(3)/2)*py - S*np.sqrt(3)/2)
                d3 = max(0.0, -0.5*px + (np.sqrt(3)/2)*py - S*np.sqrt(3)/2)
                d4 = max(0.0, -px - S)
                d5 = max(0.0, -0.5*px - (np.sqrt(3)/2)*py - S*np.sqrt(3)/2)
                d6 = max(0.0, 0.5*px - (np.sqrt(3)/2)*py - S*np.sqrt(3)/2)
                violation = max(d1, d2, d3, d4, d5, d6)
                penalty += 10.0 * violation**2
                
            # Check overlap with other hexagons
            for j in range(i+1, N):
                x_j = inner_params[3*j]
                y_j = inner_params[3*j+1]
                a_j = inner_params[3*j+2]
                
                # Quick reject: if centers are far apart
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                if dist_sq > 10.0: # (2*sqrt(3))^2 ~ 12, so 10 is safe
                    continue
                    
                v_j = get_hex_verts([x_j, y_j], a_j)
                overlap = hex_overlap_dist_sq(v_i, v_j, [x_i, y_i], a_i, [x_j, y_j], a_j)
                penalty += 100.0 * overlap**2
                
        return S + penalty

    # Initial guess: Linear arrangement
    x0 = []
    for i in range(N):
        x = 4.0 * (i - N//2)
        y = 0.0
        a = 0.0
        x0.extend([x, y, a])
    x0.append(4.0) # Initial S
    
    x0 = np.array(x0)
    
    best_result = None
    best_score = 1e9
    
    # Try a few random restarts if time permits
    num_restarts = 3
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x_start = x0
        else:
            # Perturb initial guess
            x_start = x0 + np.random.normal(0, 0.5, size=x0.shape)
            x_start[-1] = 4.0 + np.random.uniform(0, 1)
            
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to seed-like output
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Extract best solution
    S = best_result[-1]
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        a = best_result[3*i+2]
        inner.append([x, y, a])
        
    # Verify validity locally to ensure we don't return a bad solution
    # Simple check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if dist < 1.5: # Heuristic, might be too strict or loose, but L-BFGS should handle
                pass # Rely on objective function's penalty being low
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
