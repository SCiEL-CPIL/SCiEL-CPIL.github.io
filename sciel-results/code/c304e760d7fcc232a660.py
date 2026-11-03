import numpy as np
from scipy.optimize import minimize
import time

N = 12

# Precompute hexagon vertices for angle 0
# Unit hexagon side length 1
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

# Normals for Separating Axis Theorem (SAT) for aligned hexagons
# The axes are the normals to the sides.
# Side 1: (1,0) to (0.5, sqrt(3)/2). Normal is perpendicular to edge vector.
# Edge vector: (-0.5, sqrt(3)/2). Normal: (sqrt(3)/2, 0.5) or (1/sqrt(3), 0.5)?
# Let's use the 6 normals pointing outwards.
NORMALS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

def get_hex_vertices(center):
    return center + HEX_VERTS

def check_overlap_sat(c1, c2):
    """
    Check if two unit hexagons centered at c1 and c2 overlap.
    Uses Separating Axis Theorem.
    """
    v1 = get_hex_vertices(c1)
    v2 = get_hex_vertices(c2)
    
    # Project onto all 6 normals
    for normal in NORMALS:
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If intervals are disjoint, no overlap
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def calculate_min_S(positions):
    """
    Calculate the minimum side length S of a regular hexagon (centered at 0,0, angle 0)
    that contains all hexagons at given positions.
    """
    max_s = 0.0
    for i in range(N):
        center = positions[i]
        verts = get_hex_vertices(center)
        for v in verts:
            x, y = v
            # Constraints for hexagon containing (x,y):
            # 1. |x| <= S
            # 2. |y| <= sqrt(3)/2 * S  => S >= 2/sqrt(3) * |y|
            # 3. |sqrt(3)*x + y| <= sqrt(3)*S => S >= |x + y/sqrt(3)|
            # 4. |sqrt(3)*x - y| <= sqrt(3)*S => S >= |x - y/sqrt(3)|
            
            s1 = abs(x)
            s2 = (2.0 / np.sqrt(3)) * abs(y)
            s3 = abs(x + y / np.sqrt(3))
            s4 = abs(x - y / np.sqrt(3))
            
            s_req = max(s1, s2, s3, s4)
            if s_req > max_s:
                max_s = s_req
    return max_s

def penalty_function(positions_flat):
    """
    Objective function for optimizer.
    positions_flat: 1D array of length 2*N.
    """
    positions = positions_flat.reshape(N, 2)
    
    # Calculate S
    S = calculate_min_S(positions)
    
    # Check overlaps
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap_sat(positions[i], positions[j]):
                # Calculate overlap depth for penalty? 
                # Simple penalty: add a large constant
                overlap_penalty += 100.0
                
    return S + overlap_penalty

def generate_initial_positions(method='grid'):
    if method == 'grid_3x4':
        # 3 rows, 4 cols
        positions = []
        for r in range(3):
            for c in range(4):
                x = c * np.sqrt(3)
                y = r * 1.5
                # Center the block
                positions.append([x, y])
        positions = np.array(positions)
        # Center the block
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        positions -= [cx, cy]
        return positions.flatten()
        
    elif method == 'grid_7_plus_5':
        # 7 cluster
        pos = [[0,0]]
        for i in range(6):
            angle = i * np.pi / 3
            pos.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
        # Add 5 more randomly nearby
        for i in range(5):
            angle = (i * 2 * np.pi / 5)
            pos.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
        positions = np.array(pos)
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        positions -= [cx, cy]
        return positions.flatten()
        
    else: # random
        return np.random.randn(2*N) * 2.0

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_positions = None
    best_S = float('inf')
    
    methods = ['grid_3x4', 'grid_7_plus_5', 'random', 'random']
    
    for method in methods:
        if time.time() - start_time > timeout:
            break
            
        init_pos = generate_initial_positions(method)
        
        # Run Nelder-Mead
        try:
            res = minimize(penalty_function, init_pos, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 5000, 'maxfev': 5000})
        except Exception:
            continue
            
        final_pos = res.x.reshape(N, 2)
        S_val = calculate_min_S(final_pos)
        
        # Verify no overlap
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap_sat(final_pos[i], final_pos[j]):
                    valid = False
                    break
            if not valid: break
        
        if valid and S_val < best_S:
            best_S = S_val
            best_positions = final_pos
            
        if time.time() - start_time > timeout:
            break

    if best_positions is None:
        # Fallback: Return a valid but suboptimal packing
        # Use the 7+5 grid
        best_positions = generate_initial_positions('grid_7_plus_5').reshape(N, 2)
        best_S = calculate_min_S(best_positions)
        
    # Format output
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        inner.append([float(x), float(y), 0.0])
        
    return inner, (0.0, 0.0), float(best_S), 0.0

