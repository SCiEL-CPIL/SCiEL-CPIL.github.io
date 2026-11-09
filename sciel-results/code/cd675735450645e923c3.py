import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    # Deadline guard
    start_time = time.time()
    deadline = 600  # seconds
    
    # Constants
    N_INNER = 11
    INNER_SIDE = 1.0
    EPS = 1e-6
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, angle, side):
        # angle in radians
        angles = angle + np.arange(6) * np.pi / 3
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack((x, y))

    # Helper: Get edge normals of a hexagon (outward)
    def get_hex_normals(vertices):
        # vertices shape (6, 2)
        # Edge i is vertices[i] to vertices[(i+1)%6]
        # Normal is perpendicular to edge, pointing outward
        # For CCW vertices, outward normal is (dy, -dx) normalized?
        # Let's check. Edge (0,0) to (1,0). dy=0, dx=1. (0, -1) points down. 
        # If hex is centered at (0.5, 0.5) roughly, down is outward?
        # Standard hexagon vertices CCW:
        # (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866)
        # Edge 0: (1,0) to (0.5, 0.866). Vector (-0.5, 0.866).
        # Outward normal should point roughly (1, 0.5).
        # Cross product Z-component of (Vector, Normal) should be negative for outward?
        # Let's just compute normals and check against center.
        normals = np.zeros((6, 2))
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i+1)%6]
            edge = v2 - v1
            # Normal candidates: (edge[1], -edge[0]) and (-edge[1], edge[0])
            n1 = np.array([edge[1], -edge[0]])
            n2 = -n1
            # Center of hexagon
            center = np.mean(vertices, axis=0)
            # Midpoint of edge
            mid = (v1 + v2) / 2
            # Vector from center to midpoint
            c_to_mid = mid - center
            # The outward normal should have positive dot product with c_to_mid
            if np.dot(n1, c_to_mid) > 0:
                normals[i] = n1
            else:
                normals[i] = n2
            # Normalize
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm
        return normals

    # Objective Function
    # x: [x1, y1, theta1, ..., x11, y11, theta11, S]
    def objective(x):
        if time.time() > deadline - 10:
            return 1e10 # Abort
        
        # Extract parameters
        inner_params = x[:33].reshape(11, 3)
        S = x[33]
        
        # Ensure S is valid
        if S < np.sqrt(11):
            return 1e10
            
        # Compute vertices and normals for inner hexagons
        inner_verts = []
        inner_normals = []
        for i in range(11):
            cx, cy, theta = inner_params[i]
            v = get_hex_vertices(cx, cy, theta, INNER_SIDE)
            n = get_hex_normals(v)
            inner_verts.append(v)
            inner_normals.append(n)
            
        # Outer Hexagon
        # Fixed at origin, angle 0
        outer_verts = get_hex_vertices(0, 0, 0, S)
        outer_normals = get_hex_normals(outer_verts)
        
        # Containment Violation
        # For each inner vertex, check against 6 outer half-planes
        # Outer apothem A = S * sqrt(3) / 2
        A_outer = S * np.sqrt(3) / 2
        cont_viol = 0.0
        for i in range(11):
            for p in inner_verts[i]:
                for k in range(6):
                    n_outer = outer_normals[k]
                    # Distance from origin along normal
                    dist = np.dot(p, n_outer)
                    # Must be <= A_outer
                    if dist > A_outer:
                        cont_viol += (dist - A_outer)**2
                        
        # Separation Violation
        sep_viol = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                vi = inner_verts[i]
                vj = inner_verts[j]
                ni = inner_normals[i]
                nj = inner_normals[j]
                
                # All 12 candidate axes
                axes = np.vstack([ni, nj])
                
                max_gap = -np.inf
                for axis in axes:
                    # Project vi and vj onto axis
                    pi = np.dot(vi, axis)
                    pj = np.dot(vj, axis)
                    
                    pi_min, pi_max = np.min(pi), np.max(pi)
                    pj_min, pj_max = np.min(pj), np.max(pj)
                    
                    # Gap
                    gap1 = pi_min - pj_max
                    gap2 = pj_min - pi_max
                    gap = min(gap1, gap2)
                    
                    if gap > max_gap:
                        max_gap = gap
                            
                # We want max_gap > EPS
                if max_gap < EPS:
                    sep_viol += (EPS - max_gap)**2
                    
        # Total Penalty
        penalty = 100.0 * (sep_viol + cont_viol)
        
        # Minimize S + Penalty
        return S + penalty

    # Initial Guess
    # Use the seed arrangement as a starting point, but maybe perturb it
    # Seed:
    # Center (0,0)
    # Neighbors at distance 2.5? No, unit hexes touching have center distance 2 * apothem = sqrt(3) ~ 1.732?
    # Wait. Side 1. Apothem sqrt(3)/2. Distance between centers of touching hexes is 2 * apothem = sqrt(3).
    # The seed used 2.5, which is > sqrt(3), so they are separated.
    # Let's use a tighter initial guess.
    # Hexagonal lattice:
    # dx = sqrt(3), dy = 1.5
    # (0,0)
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5)
    # (sqrt(3)/2, -1.5), (-sqrt(3)/2, -1.5)
    # (2*sqrt(3), 0) ? No, that's far.
    
    # Let's create a list of 11 positions in a hexagonal cluster
    # 1 center
    # 6 surrounding
    # 4 more?
    
    positions = []
    # Center
    positions.append((0, 0, 0))
    # Ring 1
    for k in range(6):
        angle = k * np.pi / 3
        r = np.sqrt(3)
        positions.append((r * np.cos(angle), r * np.sin(angle), 0))
    # Ring 2 (pick 4 to make 11)
    # (2*sqrt(3), 0)
    # (sqrt(3), 1.5 * 2) ? No.
    # Let's just use the seed's positions but scaled down to be touching?
    # Seed x: 0, -2.5, 2.5, -1.25, 1.25...
    # Let's use the seed positions as the initial guess for the optimizer.
    # They are valid, just not optimal.
    
    seed_inner = np.array([
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
        [3.75, -2.17, 0]
    ])
    
    # Flatten
    x0 = np.zeros(34)
    x0[:33] = seed_inner.flatten()
    x0[33] = 8.0 # Initial S
    
    # Bounds
    bounds = [(-10, 10), (-10, 10), (0, 2*np.pi)] * 11
    bounds.append((np.sqrt(11), 10.0))
    
    best_S = x0[33]
    best_x = x0
    
    # Optimization
    # Use SLSQP or L-BFGS-B
    # L-BFGS-B is faster for large N, but SLSQP handles constraints better?
    # We are using penalty, so L-BFGS-B is fine.
    
    options = {'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            x_start = x0
        else:
            # Perturb the best solution found so far
            x_start = best_x.copy()
            # Add noise to positions and angles
            noise = np.random.normal(0, 0.1, 33)
            x_start[:33] += noise
            # Keep S same or slightly larger to allow movement
            x_start[33] = best_S + 0.1
            
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, options=options)
            
            if res.success or res.fun < objective(best_x):
                # Check validity of the result
                # We need to ensure violations are effectively 0
                # The objective includes penalty, so low objective implies low penalty
                # But we must verify strictly for the final return
                if res.fun < objective(best_x):
                    best_x = res.x
                    best_S = res.x[33]
                    
        except Exception as e:
            pass

    # Final Verification and Adjustment
    # It's possible the optimizer found a local minimum with small but non-zero violations.
    # We should try to "fix" the packing by slightly increasing S if violations exist.
    # Or just return the best found.
    # The task says "Valid only if...". If invalid, score 0.
    # So we MUST ensure validity.
    
    # Let's run a final check on best_x
    # If invalid, we might need to run a tighter optimization with higher penalty weight.
    
    def check_validity(x):
        inner_params = x[:33].reshape(11, 3)
        S = x[33]
        if S < np.sqrt(11): return False
        
        inner_verts = []
        inner_normals = []
        for i in range(11):
            cx, cy, theta = inner_params[i]
            v = get_hex_vertices(cx, cy, theta, INNER_SIDE)
            n = get_hex_normals(v)
            inner_verts.append(v)
            inner_normals.append(n)
            
        outer_verts = get_hex_vertices(0, 0, 0, S)
        outer_normals = get_hex_normals(outer_verts)
        A_outer = S * np.sqrt(3) / 2
        
        # Containment
        for i in range(11):
            for p in inner_verts[i]:
                for k in range(6):
                    n_outer = outer_normals[k]
                    dist = np.dot(p, n_outer)
                    if dist > A_outer + 1e-6:
                        return False
                        
        # Separation
        for i in range(11):
            for j in range(i+1, 11):
                vi = inner_verts[i]
                vj = inner_verts[j]
                ni = inner_normals[i]
                nj = inner_normals[j]
                axes = np.vstack([ni, nj])
                
                max_gap = -np.inf
                for axis in axes:
                    pi = np.dot(vi, axis)
                    pj = np.dot(vj, axis)
                    pi_min, pi_max = np.min(pi), np.max(pi)
                    pj_min, pj_max = np.min(pj), np.max(pj)
                    gap1 = pi_min - pj_max
                    gap2 = pj_min - pi_max
                    gap = min(gap1, gap2)
                    if gap > max_gap:
                        max_gap = gap
                        
                if max_gap <= 1e-6:
                    return False
                    
        return True

    if not check_validity(best_x):
        # Try to repair by increasing S slightly and re-optimizing with high penalty
        # Or just accept the risk? No, score 0 is bad.
        # Let's try a few iterations of increasing S until valid
        current_S = best_S
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            current_S += 0.01
            x_test = best_x.copy()
            x_test[33] = current_S
            if check_validity(x_test):
                best_S = current_S
                best_x = x_test
                break
                
    # Construct output
    inner_data = best_x[:33].reshape(11, 3)
    # Convert angle to degrees
    inner_data[:, 2] = np.degrees(inner_data[:, 2])
    
    outer_data = np.array([0, 0, 0])
    
    return inner_data, outer_data, float(best_S)

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    # Deadline guard
    start_time = time.time()
    deadline = 600  # seconds
    
    # Constants
    N_INNER = 11
    INNER_SIDE = 1.0
    EPS = 1e-6
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, angle, side):
        # angle in radians
        angles = angle + np.arange(6) * np.pi / 3
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack((x, y))

    # Helper: Get edge normals of a hexagon (outward)
    def get_hex_normals(vertices):
        # vertices shape (6, 2)
        normals = np.zeros((6, 2))
        center = np.mean(vertices, axis=0)
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i+1)%6]
            edge = v2 - v1
            n1 = np.array([edge[1], -edge[0]])
            n2 = -n1
            mid = (v1 + v2) / 2
            c_to_mid = mid - center
            if np.dot(n1, c_to_mid) > 0:
                normals[i] = n1
            else:
                normals[i] = n2
            norm = np.linalg.norm(normals[i])
            if norm > 1e-12:
                normals[i] /= norm
        return normals

    # Objective Function
    def objective(x):
        if time.time() > deadline - 10:
            return 1e10
        
        inner_params = x[:33].reshape(11, 3)
        S = x[33]
        
        if S < np.sqrt(11):
            return 1e10
            
        inner_verts = []
        inner_normals = []
        for i in range(11):
            cx, cy, theta = inner_params[i]
            v = get_hex_vertices(cx, cy, theta, INNER_SIDE)
            n = get_hex_normals(v)
            inner_verts.append(v)
            inner_normals.append(n)
            
        outer_verts = get_hex_vertices(0, 0, 0, S)
        outer_normals = get_hex_normals(outer_verts)
        
        A_outer = S * np.sqrt(3) / 2
        cont_viol = 0.0
        for i in range(11):
            for p in inner_verts[i]:
                for k in range(6):
                    n_outer = outer_normals[k]
                    dist = np.dot(p, n_outer)
                    if dist > A_outer:
                        cont_viol += (dist - A_outer)**2
                        
        sep_viol = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                vi = inner_verts[i]
                vj = inner_verts[j]
                ni = inner_normals[i]
                nj = inner_normals[j]
                
                axes = np.vstack([ni, nj])
                
                max_gap = -np.inf
                for axis in axes:
                    pi = np.dot(vi, axis)
                    pj = np.dot(vj, axis)
                    
                    pi_min, pi_max = np.min(pi), np.max(pi)
                    pj_min, pj_max = np.min(pj), np.max(pj)
                    
                    gap1 = pi_min - pj_max
                    gap2 = pj_min - pi_max
                    gap = min(gap1, gap2)
                    
                    if gap > max_gap:
                        max_gap = gap
                            
                if max_gap < EPS:
                    sep_viol += (EPS - max_gap)**2
                    
        penalty = 100.0 * (sep_viol + cont_viol)
        return S + penalty

    # Initial Guess
    seed_inner = np.array([
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
        [3.75, -2.17, 0]
    ])
    
    x0 = np.zeros(34)
    x0[:33] = seed_inner.flatten()
    x0[33] = 8.0
    
    bounds = [(-10, 10), (-10, 10), (0, 2*np.pi)] * 11
    bounds.append((np.sqrt(11), 10.0))
    
    best_S = x0[33]
    best_x = x0
    
    options = {'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
    
    for restart in range(5):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            x_start = x0
        else:
            x_start = best_x.copy()
            noise = np.random.normal(0, 0.1, 33)
            x_start[:33] += noise
            x_start[33] = best_S + 0.1
            
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, options=options)
            
            if res.fun < objective(best_x):
                best_x = res.x
                best_S = res.x[33]
        except Exception as e:
            pass

    def check_validity(x):
        inner_params = x[:33].reshape(11, 3)
        S = x[33]
        if S < np.sqrt(11): return False
        
        inner_verts = []
        inner_normals = []
        for i in range(11):
            cx, cy, theta = inner_params[i]
            v = get_hex_vertices(cx, cy, theta, INNER_SIDE)
            n = get_hex_normals(v)
            inner_verts.append(v)
            inner_normals.append(n)
            
        outer_verts = get_hex_vertices(0, 0, 0, S)
        outer_normals = get_hex_normals(outer_verts)
        A_outer = S * np.sqrt(3) / 2
        
        for i in range(11):
            for p in inner_verts[i]:
                for k in range(6):
                    n_outer = outer_normals[k]
                    dist = np.dot(p, n_outer)
                    if dist > A_outer + 1e-6:
                        return False
                        
        for i in range(11):
            for j in range(i+1, 11):
                vi = inner_verts[i]
                vj = inner_verts[j]
                ni = inner_normals[i]
                nj = inner_normals[j]
                axes = np.vstack([ni, nj])
                
                max_gap = -np.inf
                for axis in axes:
                    pi = np.dot(vi, axis)
                    pj = np.dot(vj, axis)
                    pi_min, pi_max = np.min(pi), np.max(pi)
                    pj_min, pj_max = np.min(pj), np.max(pj)
                    gap1 = pi_min - pj_max
                    gap2 = pj_min - pi_max
                    gap = min(gap1, gap2)
                    if gap > max_gap:
                        max_gap = gap
                        
                if max_gap <= 1e-6:
                    return False
                    
        return True

    if not check_validity(best_x):
        current_S = best_S
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            current_S += 0.01
            x_test = best_x.copy()
            x_test[33] = current_S
            if check_validity(x_test):
                best_S = current_S
                best_x = x_test
                break
                
    inner_data = best_x[:33].reshape(11, 3)
    inner_data[:, 2] = np.degrees(inner_data[:, 2])
    
    outer_data = np.array([0, 0, 0])
    
    return inner_data, outer_data, float(best_S)

# EVOLVE-BLOCK-END
