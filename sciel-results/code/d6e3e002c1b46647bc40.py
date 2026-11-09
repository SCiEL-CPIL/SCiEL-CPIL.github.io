# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0
    
    n = 11
    # Initial configuration: Honeycomb lattice
    # 1 center, 6 at dist sqrt(3), 4 at dist 3
    sqrt3 = np.sqrt(3)
    centers = np.zeros((n, 2))
    centers[0] = [0, 0]
    
    # 6 neighbors
    for i in range(6):
        angle = i * np.pi / 3
        centers[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
        
    # 4 more neighbors at distance 3, angles 30, 90, 150, 210
    # We pick 4 out of the 6 available spots in the 2nd shell
    # Angles: 30, 90, 150, 210, 270, 330
    # Let's pick 30, 90, 150, 210
    outer_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]
    for i, angle in enumerate(outer_angles):
        centers[7+i] = [3 * np.cos(angle), 3 * np.sin(angle)]

    # Define the hexagonal radius function
    # Normals to the sides of the outer hexagon (angle 0)
    # Sides are at 30, 90, 150, 210, 270, 330
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    def hex_radius(centers):
        # centers: (11, 2)
        # normals: (6, 2)
        # proj[i, k] = centers[i] . normals[k]
        proj = centers @ normals.T
        # max abs over k
        max_proj = np.max(np.abs(proj), axis=1)
        return np.max(max_proj)

    def objective(x):
        c = x.reshape(11, 2)
        return hex_radius(c)

    def constraints(x):
        c = x.reshape(11, 2)
        cons = []
        for i in range(11):
            for j in range(i+1, 11):
                # Distance squared >= 3
                # dist_sq - 3 >= 0
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist_sq = dx*dx + dy*dy
                cons.append(dist_sq - 3)
        return cons

    # SLSQP optimization
    # We minimize the max radius
    # Initial guess
    x0 = centers.flatten()
    
    # Run optimization
    best_x = x0
    best_val = objective(x0)
    
    # Try a few restarts or just one good run
    # SLSQP might be slow, but n is small
    # Use a penalty method if SLSQP fails?
    # Let's try SLSQP first
    
    try:
        res = minimize(objective, x0, method='SLSQP', 
                       constraints=[{'type': 'ineq', 'fun': constraints}],
                       options={'maxiter': 200, 'ftol': 1e-9})
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # If time allows, try a local search with Nelder-Mead on the unconstrained problem with penalty
    if time.time() < deadline - safety_margin:
        def penalty_objective(x):
            c = x.reshape(11, 2)
            obj = hex_radius(c)
            # Penalty for violations
            pen = 0
            for i in range(11):
                for j in range(i+1, 11):
                    dx = c[i, 0] - c[j, 0]
                    dy = c[i, 1] - c[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 3:
                        pen += (3 - dist_sq) * 100
            return obj + pen

        # Nelder-Mead
        res2 = minimize(penalty_objective, x0, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res2.fun < best_val:
            # Check if valid
            c2 = res2.x.reshape(11, 2)
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    dx = c2[i, 0] - c2[j, 0]
                    dy = c2[i, 1] - c2[j, 1]
                    if dx*dx + dy*dy < 3 - 1e-6:
                        valid = False
                        break
                if not valid: break
            if valid:
                best_x = res2.x
                best_val = hex_radius(c2)

    # Construct result
    best_centers = best_x.reshape(11, 2)
    angles = np.zeros(11)
    
    inner_hex_data = np.column_stack((best_centers, angles))
    outer_hex_data = np.array([0, 0, 0])
    
    # Calculate S
    # M = best_val
    # S = 1 + 2/sqrt(3) * M
    S = 1 + (2 / sqrt3) * best_val
    
    # Ensure S is at least sqrt(11)
    if S < np.sqrt(11):
        S = np.sqrt(11)
        
    return inner_hex_data, outer_hex_data, S

# EVOLVE-BLOCK-END
