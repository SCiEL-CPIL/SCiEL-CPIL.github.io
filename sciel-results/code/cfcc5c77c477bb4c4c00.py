import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    n_inner = 11
    # Target: Minimize S (side length of outer hexagon)
    # Variables: S, x0, y0, ..., x10, y10
    # Total variables: 1 + 2*n_inner = 23
    
    # Normals for the outer hexagon boundaries (apothem directions)
    # The outer hexagon has angle 0.
    # Vertices at 0, 60, ...
    # Edges are perpendicular to directions 30, 90, 150, 210, 270, 330.
    # Normal vectors n_k
    normals = []
    for k in range(6):
        angle = np.pi/6 + k * np.pi/3
        normals.append((np.cos(angle), np.sin(angle)))
    normals = np.array(normals) # shape (6, 2)
    
    # Constraint: C_i . n_k <= (sqrt(3)/2) * (S - 1)
    # Distance constraint: ||C_i - C_j||^2 >= 3
    
    def objective(vars):
        S = vars[0]
        return S

    def constraints(vars):
        S = vars[0]
        coords = vars[1:].reshape(n_inner, 2)
        cons = []
        
        # Boundary constraints
        # C_i . n_k <= (sqrt(3)/2) * (S - 1)
        # (sqrt(3)/2) * (S - 1) - C_i . n_k >= 0
        max_val = (np.sqrt(3)/2) * (S - 1)
        # Compute dot products for all points and all normals
        # coords shape (11, 2), normals shape (6, 2)
        # We need C_i . n_k for each i, k
        # Use broadcasting: coords[:, np.newaxis, :] @ normals.T -> (11, 6)
        dots = coords @ normals.T
        # Constraint: max_val - dots >= 0
        # Flatten to list of constraints
        boundary_cons = max_val - dots.flatten()
        cons.extend(boundary_cons.tolist())
        
        # Distance constraints
        # ||C_i - C_j||^2 >= 3
        # ||C_i - C_j||^2 - 3 >= 0
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                diff = coords[i] - coords[j]
                dist_sq = np.dot(diff, diff)
                cons.append(dist_sq - 3.0)
                
        return cons

    # Initial guess
    # S guess: 4.0
    S_init = 4.0
    # Place points in a hexagonal grid
    # Center at (0,0)
    # Neighbors at distance sqrt(3)
    r = np.sqrt(3)
    points = []
    # Center
    points.append([0, 0])
    # 6 neighbors
    for k in range(6):
        angle = k * np.pi/3
        points.append([r * np.cos(angle), r * np.sin(angle)])
    # 4 more points? 
    # We have 7 points. Need 4 more.
    # Add them further out?
    # This might violate S=4.
    # Let's just randomize slightly or use a larger S for init.
    
    # Better init: Use a known good configuration if possible, or random.
    # Let's try to construct a valid config for S=4.5
    # If S=4.5, max_val = sqrt(3)/2 * 3.5 ~ 3.03
    # Points must be within radius ~3.
    
    # Let's use differential_evolution to find a good starting point for SLSQP
    # Bounds for S: [sqrt(11), 6]
    # Bounds for x,y: [-S, S] -> [-6, 6]
    
    bounds = [(np.sqrt(11), 6)] + [(-6, 6)] * (2 * n_inner)
    
    # Differential Evolution is slow. Let's try a few random restarts with SLSQP directly,
    # or use DE with small population if time permits.
    # Given 600s, DE on 23 vars might take a while.
    # Let's try SLSQP with multiple random starts first.
    
    best_S = np.inf
    best_vars = None
    best_coords = None
    
    # Define a helper to check validity and compute S
    def check_and_get_S(vars):
        S = vars[0]
        if S < np.sqrt(11) - 1e-6:
            return None
        coords = vars[1:].reshape(n_inner, 2)
        max_val = (np.sqrt(3)/2) * (S - 1)
        dots = coords @ normals.T
        if np.any(max_val - dots < -1e-6):
            return None
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                diff = coords[i] - coords[j]
                if np.dot(diff, diff) < 3.0 - 1e-6:
                    return None
        return S

    # Try random restarts with SLSQP
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random S between 3.5 and 5.0
        S_rand = np.random.uniform(3.5, 5.0)
        # Random coords within the hexagon of side S_rand
        # Simple rejection sampling or just random box and clip?
        # Just random box [-S_rand, S_rand]
        coords_rand = np.random.uniform(-S_rand, S_rand, (n_inner, 2))
        
        vars_init = np.concatenate(([S_rand], coords_rand.flatten()))
        
        # SLSQP
        try:
            res = minimize(
                objective, 
                vars_init, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success:
                S_opt = check_and_get_S(res.x)
                if S_opt is not None and S_opt < best_S:
                    best_S = S_opt
                    best_vars = res.x
                    best_coords = res.x[1:].reshape(n_inner, 2)
        except Exception:
            pass

    # If no valid solution found with random restarts, try a more structured init
    if best_S == np.inf:
        # Try the "flower" + extra points structure
        # S=4.5
        S_init = 4.5
        r = np.sqrt(3)
        points = [[0,0]]
        for k in range(6):
            angle = k * np.pi/3
            points.append([r * np.cos(angle), r * np.sin(angle)])
        # Add 4 more at distance 2*r?
        for k in range(4):
            angle = (k * np.pi/3) + np.pi/6
            points.append([2*r * np.cos(angle), 2*r * np.sin(angle)])
        coords_init = np.array(points)
        vars_init = np.concatenate(([S_init], coords_init.flatten()))
        
        try:
            res = minimize(
                objective, 
                vars_init, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraints},
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.success:
                S_opt = check_and_get_S(res.x)
                if S_opt is not None and S_opt < best_S:
                    best_S = S_opt
                    best_vars = res.x
                    best_coords = res.x[1:].reshape(n_inner, 2)
        except Exception:
            pass

    # Fallback: Use a very large S if optimization failed
    if best_S == np.inf:
        best_S = 8.0
        # Grid arrangement
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
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, best_S

    # Construct output
    inner_hex_data = np.zeros((n_inner, 3))
    inner_hex_data[:, 0] = best_coords[:, 0]
    inner_hex_data[:, 1] = best_coords[:, 1]
    inner_hex_data[:, 2] = 0.0  # Angle 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_S

if __name__ == "__main__":
    inner, outer, side = hexagon_packing_11()
    print(f"Side: {side}")
    print(inner)
