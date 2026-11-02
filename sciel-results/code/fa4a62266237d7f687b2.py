import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave 10s buffer

def get_hex_vertices(center, angle):
    """
    Returns vertices of a unit regular hexagon.
    center: (x, y)
    angle: rotation in radians
    """
    cx, cy = center
    # Vertices at angles angle + k*pi/3
    angles = angle + np.arange(6) * (np.pi / 3)
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def get_hex_normals(center, angle):
    """
    Returns outward normals of a unit regular hexagon.
    """
    # Normals are at angle + pi/6 + k*pi/3
    angles = angle + np.pi/6 + np.arange(6) * (np.pi / 3)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def sat_overlap_penalty(vertices_list, normals_list):
    """
    Calculate overlap penalty for a list of hexagons.
    vertices_list: list of Nx6x2 arrays
    normals_list: list of Nx6x2 arrays
    """
    penalty = 0.0
    n_hex = len(vertices_list)
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            n1 = normals_list[i]
            n2 = normals_list[j]
            
            # Combine normals
            # We check all 12 axes
            axes = np.vstack((n1, n2))
            
            # Project v1 and v2 onto all axes
            # proj1: shape (12, 6) -> min/max along axis 1
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            min1 = np.min(proj1, axis=1)
            max1 = np.max(proj1, axis=1)
            min2 = np.min(proj2, axis=1)
            max2 = np.max(proj2, axis=1)
            
            # Separation for each axis
            # If max1 < min2 or max2 < min1, separated.
            # Overlap if max1 >= min2 and max2 >= min1.
            # Penetration depth along axis k is min(max1, max2) - max(min1, min2)
            # If this is negative, they are separated by that amount.
            sep = np.minimum(max1, max2) - np.maximum(min1, min2)
            
            # If sep < 0, they overlap.
            # We want to push them apart.
            # Penalty is sum of squared negative separations.
            neg_sep = np.minimum(sep, 0)
            penalty += np.sum(neg_sep ** 2)
            
    return penalty

def objective(x):
    """
    x: array of length 34.
    x[0:22]: x, y for 11 hexagons.
    x[22:33]: angle for 11 hexagons.
    x[33]: angle for outer hexagon.
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return 1e6 # Timeout guard
    
    # Reshape
    centers = x[0:22].reshape(11, 2)
    angles = x[22:33]
    outer_angle = x[33]
    
    # Calculate vertices and normals for inner hexagons
    vertices_list = []
    normals_list = []
    for i in range(11):
        v = get_hex_vertices(centers[i], angles[i])
        n = get_hex_normals(centers[i], angles[i])
        vertices_list.append(v)
        normals_list.append(n)
        
    # Calculate L (outer side length)
    # Outer hexagon centered at (0,0), rotation outer_angle.
    # Normals of outer hexagon
    outer_normals = get_hex_normals((0,0), outer_angle)
    
    # Max projection of any inner vertex onto any outer normal
    max_proj = 0.0
    for i in range(11):
        # Project vertices of inner hex i onto outer normals
        # v_i: 6x2, outer_normals: 6x2
        # We need max over k (outer normal) and j (vertex) of dot(v_ij, n_k)
        # This is equivalent to max over k of (max_j dot(v_ij, n_k))
        # Or max over k of (v_i @ n_k.T) max? No.
        # For a fixed normal n_k, the max projection of the polygon is the max dot product with its vertices.
        # So for each outer normal n_k, find max dot product with all inner vertices.
        # Then take the max over all k.
        
        # Vectorized:
        # proj: 11x6 (inner hex) x 6 (outer normals) -> 11x6x6?
        # Let's do it per outer normal.
        for k in range(6):
            n_k = outer_normals[k]
            # Dot product of all vertices of all inner hexagons with n_k
            # centers: 11x2, angles: 11
            # We already have vertices_list.
            # Flatten all vertices: 66x2
            all_v = np.vstack(vertices_list)
            dots = all_v @ n_k
            max_d = np.max(dots)
            if max_d > max_proj:
                max_proj = max_d
                
    # L = 2/sqrt(3) * max_proj
    L = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Overlap penalty
    penalty = sat_overlap_penalty(vertices_list, normals_list)
    
    # Combined objective
    # We want to minimize L, but must avoid overlap.
    # Use a penalty term that becomes very large if overlap > 0.
    # Since L is typically ~3-5, penalty should be scaled.
    # If penalty is > 0, we add a large constant.
    # Better: L + 10 * penalty.
    return L + 10.0 * penalty

def construct():
    best_score = float('inf')
    best_x = None
    
    # Initial guesses
    # 1. Hexagonal cluster 7 + 4
    # 2. Random
    
    # Generate initial configurations
    configs = []
    
    # Config 1: 7 cluster + 4 in a line
    # Centers for 7 cluster (side 1 hexes, dist sqrt(3))
    # (0,0)
    # (sqrt(3), 0), (0, sqrt(3)), (-sqrt(3), sqrt(3)), (-sqrt(3), 0), (0, -sqrt(3)), (sqrt(3), -sqrt(3))
    # Add 4 more to the right: (2*sqrt(3), 0), (3*sqrt(3), 0)...
    # This is elongated.
    
    # Config 2: Random
    np.random.seed(42)
    
    # We will run several restarts
    for restart in range(20):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Initialize x
        if restart == 0:
            # Structured init
            d = np.sqrt(3)
            cx = [0.0, d, 0.0, -d, -d, 0.0, d, 2*d, 3*d, 4*d, 5*d]
            cy = [0.0, 0.0, d, d, 0.0, -d, -d, 0.0, 0.0, 0.0, 0.0]
            # This is a line of 11? No, 7 cluster + 4 line.
            # Let's just use a random compact start
            # Place them in a circle
            for i in range(11):
                angle = i * 2 * np.pi / 11
                r = 2.0
                cx[i] = r * np.cos(angle)
                cy[i] = r * np.sin(angle)
            angles_init = np.zeros(11)
            outer_angle_init = 0.0
        else:
            # Random start
            cx = np.random.uniform(-3, 3, 11)
            cy = np.random.uniform(-3, 3, 11)
            angles_init = np.random.uniform(0, 2*np.pi, 11)
            outer_angle_init = np.random.uniform(0, 2*np.pi)
            
        x0 = np.concatenate([np.column_stack((cx, cy)).flatten(), angles_init, [outer_angle_init]])
        
        # Optimize
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < best_score:
                # Check validity (overlap should be ~0)
                # Extract L from res.fun? No, res.fun is L + penalty.
                # We need to calculate the actual L for the best solution.
                # If penalty is not 0, it's invalid.
                # We'll evaluate the final x to get the true L.
                
                # Calculate true L
                centers = res.x[0:22].reshape(11, 2)
                angles = res.x[22:33]
                outer_angle = res.x[33]
                
                # Re-calculate max_proj
                vertices_list = [get_hex_vertices(centers[i], angles[i]) for i in range(11)]
                outer_normals = get_hex_normals((0,0), outer_angle)
                
                max_proj = 0.0
                all_v = np.vstack(vertices_list)
                for k in range(6):
                    dots = all_v @ outer_normals[k]
                    max_d = np.max(dots)
                    if max_d > max_proj:
                        max_proj = max_d
                L = (2.0 / np.sqrt(3.0)) * max_proj
                
                # Check overlap
                penalty = sat_overlap_penalty(vertices_list, [get_hex_normals(centers[i], angles[i]) for i in range(11)])
                if penalty < 1e-6:
                    if L < best_score:
                        best_score = L
                        best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback to seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    centers = best_x[0:22].reshape(11, 2)
    angles_deg = np.degrees(best_x[22:33])
    outer_angle_deg = np.degrees(best_x[33])
    
    inner = []
    for i in range(11):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(angles_deg[i])])
        
    return inner, (0.0, 0.0), float(best_score), float(outer_angle_deg)

