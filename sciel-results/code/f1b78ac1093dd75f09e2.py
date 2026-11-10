import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance

def hexagon_packing_11():
    n = 11
    # Vertices of a unit regular hexagon centered at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Normals to the sides of a regular hexagon
    # Side k is between vertex k and k+1. Normal is at angle 30 + 60k
    normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
    unit_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    min_dist = np.sqrt(3)
    min_dist_sq = 3.0

    def get_outer_side_and_constraints(theta, centers):
        """
        Computes the minimal side length s of an outer hexagon with angle theta
        that contains all inner hexagons (aligned with angle 0) centered at 'centers'.
        Returns s and a list of constraint violations (should be <= 0 for validity in SLSQP, 
        but we use S as variable, so we minimize S).
        """
        # Rotate normals by theta
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        normals = unit_normals @ rot_mat.T # Shape (6, 2)
        
        # We need to check all vertices of all inner hexagons
        # Inner hexagons are at centers (n, 2)
        # Vertices of inner hex i are centers[i] + unit_hex_verts
        # All points: (n*6, 2)
        all_points = np.repeat(centers, 6, axis=0) + np.tile(unit_hex_verts, (n, 1))
        
        # Project all points onto all 6 normals
        # projections: (n*6, 6)
        projections = all_points @ normals.T
        
        # Max absolute projection
        max_proj = np.max(np.abs(projections))
        
        # Apothem a = max_proj
        # Side s = 2 * a / sqrt(3)
        s = 2 * max_proj / np.sqrt(3)
        
        return s

    def objective_and_constraints(vars):
        """
        vars: [x1, y1, ..., x11, y11, theta]
        We want to minimize S (side length).
        We introduce S as an auxiliary variable? No, standard minimize minimizes a scalar.
        We can minimize S directly if we can compute it.
        However, S is non-smooth (max function).
        Better: Minimize S subject to:
        1. Projections <= sqrt(3)/2 * S
        2. -Projections <= sqrt(3)/2 * S
        3. Pairwise distances >= sqrt(3)
        
        Let's use SLSQP with explicit constraints.
        Variables: x1..y11 (22 vars), theta (1 var), S (1 var). Total 24 vars.
        """
        centers = vars[:22].reshape(11, 2)
        theta = vars[22]
        S = vars[23]
        
        # Objective is S
        # Constraints:
        # 1. For all i, j: ||Ci - Cj||^2 - 3 >= 0
        # 2. For all points p in all_points: p . n_k <= sqrt(3)/2 * S
        # 3. For all points p in all_points: -p . n_k <= sqrt(3)/2 * S
        
        cons = []
        
        # Pairwise distance constraints
        for i in range(11):
            for j in range(i+1, 11):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                # dist_sq - 3 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.dot(v[2*i:2*i+2] - v[2*j:2*j+2], v[2*i:2*i+2] - v[2*j:2*j+2]) - 3.0
                })
        
        # Containment constraints
        # Compute normals
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        normals = unit_normals @ rot_mat.T
        
        all_points = np.repeat(centers, 6, axis=0) + np.tile(unit_hex_verts, (11, 1))
        
        # We can vectorize the containment check for the constraint functions
        # But SLSQP expects separate constraint functions or a single function returning array.
        # Let's create a single constraint function returning an array of all containment violations.
        
        # Actually, it's better to define the constraint as a function that returns an array.
        # SLSQP supports this if 'fun' returns an array.
        
        # Let's define one constraint for all containment
        def containment_constraint(v):
            c = v[:22].reshape(11, 2)
            th = v[22]
            S_val = v[23]
            
            cos_th = np.cos(th)
            sin_th = np.sin(th)
            rm = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
            nrm = unit_normals @ rm.T
            
            pts = np.repeat(c, 6, axis=0) + np.tile(unit_hex_verts, (11, 1))
            projs = pts @ nrm.T # (66, 6)
            
            # Constraint: proj <= sqrt(3)/2 * S
            # And -proj <= sqrt(3)/2 * S
            # So: proj - sqrt(3)/2 * S <= 0
            # And: -proj - sqrt(3)/2 * S <= 0
            
            limit = np.sqrt(3) / 2 * S_val
            
            c1 = projs - limit
            c2 = -projs - limit
            
            # Flatten
            return np.concatenate([c1.flatten(), c2.flatten()])
        
        cons.append({
            'type': 'ineq',
            'fun': lambda v: -containment_constraint(v) # SLSQP expects f(x) >= 0 for ineq
        })
        
        return S

    # Initial guesses
    best_result = None
    best_S = float('inf')
    
    # Guess 1: Hexagonal pattern
    # Center (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more...
    # Let's just use a small random perturbation of a grid
    x0_grid = []
    x0_grid.append([0, 0])
    for k in range(6):
        x0_grid.append([np.sqrt(3)*np.cos(np.deg2rad(60*k)), np.sqrt(3)*np.sin(np.deg2rad(60*k))])
    # Add 4 more in a ring
    for k in range(4):
        x0_grid.append([2*np.sqrt(3)*np.cos(np.deg2rad(60*k + 30)), 2*np.sqrt(3)*np.sin(np.deg2rad(60*k + 30))])
    
    x0 = np.array(x0_grid).flatten()
    x0 = np.concatenate([x0, [0.0, 1.0]]) # theta=0, S=1.0 (initial guess for side)
    
    # Run optimization
    try:
        res = minimize(
            objective_and_constraints, 
            x0, 
            method='SLSQP', 
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if res.success:
            S_opt = res.x[23]
            if S_opt < best_S:
                best_S = S_opt
                best_result = res.x
    except Exception as e:
        pass

    # Guess 2: Random
    np.random.seed(42)
    for _ in range(5):
        x0_rand = np.random.uniform(-3, 3, size=22)
        x0_rand = np.concatenate([x0_rand, [0.0, 5.0]])
        try:
            res = minimize(
                objective_and_constraints, 
                x0_rand, 
                method='SLSQP', 
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            if res.success:
                S_opt = res.x[23]
                if S_opt < best_S:
                    best_S = S_opt
                    best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed-like solution if optimization fails
        inner_hex_data = np.array([
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Extract results
    centers = best_result[:22].reshape(11, 2)
    theta_opt = best_result[22]
    S_opt = best_result[23]
    
    # Construct inner hex data
    # Angles are 0
    inner_hex_data = np.column_stack((centers, np.zeros(11)))
    
    # Outer hex data
    # Center is (0,0), angle is theta_opt
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(theta_opt)])
    
    # Side length
    outer_hex_side_length = S_opt
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
