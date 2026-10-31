import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Normals for the hexagon constraints
    # Outer hexagon normals (flat-top orientation)
    angles = np.deg2rad(np.arange(0, 360, 60))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Apothem of unit hexagon
    apothem_unit = np.sqrt(3) / 2.0
    
    # Function to compute minimal S for a set of positions
    def compute_min_S(pos):
        # pos is (N, 2)
        # For each hexagon i and each normal k, compute projection
        # proj[i, k] = dot(pos[i], normals[k])
        proj = pos @ normals.T
        # The constraint is proj[i, k] + apothem_unit <= S * apothem_outer_factor
        # S * (sqrt(3)/2) is the apothem of the outer hexagon.
        # So S = (max(proj) + apothem_unit) / (sqrt(3)/2)
        # Also need to consider the negative side? 
        # The hexagon is centered at origin.
        # The max projection on any normal direction determines the extent.
        # Since the hexagon is centrally symmetric, we just need the max over all normals and all points.
        max_proj = np.max(proj)
        # Wait, if a point is at -10,0, proj on normal (1,0) is -10.
        # The constraint is |dot| <= ...
        # So we need max over k of |dot(pos[i], n_k)|?
        # No, the hexagon is the intersection of half planes n_k . r <= A.
        # So we need n_k . r_i <= A - apothem_unit for all k, i.
        # This implies A >= max_{i,k} (n_k . r_i) + apothem_unit.
        # But we also need -n_k . r_i <= A - apothem_unit, which is covered by k+3.
        # So yes, max over all k and i.
        max_val = np.max(proj)
        S = (max_val + apothem_unit) / apothem_unit
        return S

    # Function to compute overlap penalty
    def compute_overlap_penalty(pos):
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[i] - pos[j]
                # Check if d is inside the Minkowski sum hexagon (apothem sqrt(3))
                # Overlap if max_k |d . n_k| < sqrt(3)
                projs = np.abs(d @ normals.T)
                max_proj = np.max(projs)
                if max_proj < np.sqrt(3):
                    # Distance to boundary
                    penalty += (np.sqrt(3) - max_proj) ** 2
        return penalty

    # Objective function
    def objective(pos_flat):
        pos = pos_flat.reshape(N, 2)
        # Center of mass penalty to keep it at origin
        com = np.mean(pos, axis=0)
        com_pen = np.sum(com ** 2) * 10.0
        
        S = compute_min_S(pos)
        overlap = compute_overlap_penalty(pos)
        
        # We want to minimize S, but if there is overlap, it's invalid.
        # We use a penalty method.
        # If overlap is significant, increase cost.
        return S + 100.0 * overlap + com_pen

    # Initial positions: Hexagonal grid cluster
    # Generate a grid of points
    # Axial coordinates to Cartesian
    def axial_to_cartesian(q, r, size=1.0):
        x = size * (q + r / 2.0) * np.sqrt(3)
        y = size * (r) * 1.5
        return np.array([x, y])

    # Generate points in a hexagonal pattern
    points = []
    # Radius 0
    points.append(axial_to_cartesian(0, 0))
    # Radius 1
    for q, r in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
        points.append(axial_to_cartesian(q, r))
    # Radius 2 (select 4 more to make 11)
    # We need 4 more. Let's pick symmetric ones.
    # (2,0), (0,2), (-2,1), (-2,-1) is asymmetric.
    # How about (2,0), (1,1), (-1,2), (-2,1)?
    # Let's just take the next 4 closest to center from radius 2.
    # Radius 2 points:
    r2_points = []
    for q, r in [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]:
        r2_points.append(axial_to_cartesian(q, r))
    
    # Sort by distance to origin
    r2_points.sort(key=lambda p: np.linalg.norm(p))
    # Add first 4
    for p in r2_points[:4]:
        points.append(p)
    
    # We have 11 points now.
    init_pos = np.array(points)
    
    # Randomize slightly to help optimization
    init_pos += np.random.normal(0, 0.1, init_pos.shape)
    
    best_score = float('inf')
    best_pos = init_pos.copy()
    
    # Multi-start optimization
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        current_pos = init_pos.copy()
        if i > 0:
            # Perturb initial guess
            current_pos += np.random.normal(0, 0.5, current_pos.shape)
            
        res = minimize(objective, current_pos.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < best_score:
            best_pos = res.x.reshape(N, 2)
            best_score = res.fun
            
        # Check time
        if time.time() > deadline - 5:
            break

    # Final refinement on the best position found
    # Use a more precise method if possible
    if time.time() < deadline - 5:
        res_final = minimize(objective, best_pos.flatten(), method='Powell', 
                             options={'maxiter': 1000})
        if res_final.fun < best_score:
            best_pos = res_final.x.reshape(N, 2)
            best_score = res_final.fun

    # Calculate final S
    S_final = compute_min_S(best_pos)
    
    # Construct output
    # Inner hexagons: (x, y, angle)
    inner = [[p[0], p[1], 0.0] for p in best_pos]
    
    # Outer hexagon: center (0,0), side S_final, angle 0
    center = (0.0, 0.0)
    side = S_final
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To match the interface, we just need to ensure construct is the entry point.
# The seed program had construct inside the block.
# I will move the definition here or ensure it's accessible.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program had:
# def construct():
#     ...
# return ...
# 
# My code defines construct globally. This is fine.
# EVOLVE-BLOCK-END
