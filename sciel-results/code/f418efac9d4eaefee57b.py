import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    # Constants
    sqrt3 = np.sqrt(3)
    # Outer hexagon normals (angle 0)
    # Vertices at 0, 60, ...
    # Normals at 30, 90, ...
    outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])

    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon at (cx, cy) with given angle."""
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def get_all_vertices(params):
        """
        params: (33,) array [x1, y1, a1, x2, y2, a2, ...]
        returns: (11, 6, 2) array of vertices
        """
        verts = np.zeros((11, 6, 2))
        for i in range(11):
            cx = params[i*3]
            cy = params[i*3 + 1]
            ang = params[i*3 + 2]
            verts[i] = get_hex_vertices(cx, cy, ang)
        return verts

    def calc_S(verts):
        """Calculate minimum side length S of outer hexagon (angle 0, center 0,0) to contain verts."""
        # Project all vertices onto outer normals
        # verts shape: (11, 6, 2) -> flatten to (66, 2)
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        global_max = np.max(max_proj)
        # S * sqrt(3)/2 = global_max  => S = 2 * global_max / sqrt(3)
        return 2.0 * global_max / sqrt3

    def overlap_penalty(params):
        """
        Fast overlap penalty based on center distances.
        Min center distance for non-overlap is sqrt(3).
        """
        centers = params[0:33:3].reshape(-1, 1) # This is wrong, need x and y
        # Reshape params to (11, 3)
        p = params.reshape(11, 3)
        xs = p[:, 0]
        ys = p[:, 1]
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < sqrt3:
                    penalty += (sqrt3 - d)**2
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        verts = get_all_vertices(params)
        S = calc_S(verts)
        pen = overlap_penalty(params)
        return S + 100.0 * pen

    def exact_check(params):
        """
        Perform exact validation.
        Returns (is_valid, S, min_gap)
        """
        verts = get_all_vertices(params)
        S = calc_S(verts)
        
        # Check containment
        # Already done in calc_S, but let's verify strict inequality or equality
        # The task says "cross(p2 - p1, p - p1) >= -1e-6"
        # This is equivalent to being inside or on the boundary.
        # calc_S ensures S is large enough.
        
        # Check separation with SAT
        min_gap = np.inf
        for i in range(11):
            v1 = verts[i]
            for j in range(i+1, 11):
                v2 = verts[j]
                # SAT check
                # Get normals for v1
                edges1 = np.roll(v1, -1, axis=0) - v1
                normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
                normals1 /= np.linalg.norm(normals1, axis=1, keepdims=True)
                
                # Get normals for v2
                edges2 = np.roll(v2, -1, axis=0) - v2
                normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
                normals2 /= np.linalg.norm(normals2, axis=1, keepdims=True)
                
                all_normals = np.vstack([normals1, normals2])
                
                proj1 = v1 @ all_normals.T
                proj2 = v2 @ all_normals.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                gaps = np.maximum(min2 - max1, min1 - max2)
                gap = np.min(gaps)
                if gap < min_gap:
                    min_gap = gap
        
        if min_gap < 1e-6:
            return False, S, min_gap
            
        # Check edge lengths
        for i in range(11):
            for j in range(6):
                p1 = verts[i, j]
                p2 = verts[i, (j+1)%6]
                if abs(np.linalg.norm(p2-p1) - 1.0) > 1e-8:
                    return False, S, min_gap
                    
        return True, S, min_gap

    # Initial Guess Generation
    def get_initial_guesses():
        guesses = []
        
        # Guess 1: Symmetric 1+6+4
        # Center
        p = np.zeros(33)
        p[0:3] = [0, 0, 0]
        
        # Ring 1: 6 hexagons
        d1 = sqrt3
        for k in range(6):
            ang = np.deg2rad(k * 60)
            idx = (k+1)*3
            p[idx] = d1 * np.cos(ang)
            p[idx+1] = d1 * np.sin(ang)
            p[idx+2] = 0 # Angle 0
            
        # Ring 2: 4 hexagons at (2,0), (0,2), (-2,0), (0,-2) in axial coords?
        # Let's use Cartesian positions corresponding to those axial coords.
        # Axial (q, r) to Cartesian:
        # x = sqrt3 * (q + r/2)
        # y = 1.5 * r
        
        # (2,0) -> x=2*sqrt3, y=0
        # (0,2) -> x=sqrt3, y=3
        # (-2,0) -> x=-2*sqrt3, y=0
        # (0,-2) -> x=-sqrt3, y=-3
        
        positions_r2 = [
            (2*sqrt3, 0),
            (sqrt3, 3),
            (-2*sqrt3, 0),
            (-sqrt3, -3)
        ]
        
        for k in range(4):
            idx = (k+7)*3
            p[idx] = positions_r2[k][0]
            p[idx+1] = positions_r2[k][1]
            p[idx+2] = 0
            
        guesses.append(p)
        
        # Guess 2: Jittered
        p2 = p.copy()
        np.random.seed(42)
        p2[0:33:3] += np.random.uniform(-0.1, 0.1, 11)
        p2[1:33:3] += np.random.uniform(-0.1, 0.1, 11)
        p2[2:33:3] += np.random.uniform(-10, 10, 11)
        guesses.append(p2)
        
        # Guess 3: Different rotation for ring 2
        p3 = p.copy()
        # Rotate ring 2 by 15 degrees?
        # Just add some randomness to angles
        p3[2:33:3] += np.random.uniform(-5, 5, 11)
        guesses.append(p3)
        
        return guesses

    best_S = 1e10
    best_params = None
    best_valid = False

    guesses = get_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline - 60:
            break
            
        try:
            # Use Powell method, robust for non-smooth
            res = minimize(objective, guess, method='Powell', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
                
            # Check if this result is valid
            valid, S_val, gap_val = exact_check(res.x)
            if valid:
                if S_val < best_S: # Update best_S if valid and smaller
                    best_S = S_val
                    best_params = res.x.copy()
                    best_valid = True
        except Exception as e:
            pass

    # If no valid solution found, try a more aggressive optimization on the best candidate
    if best_params is not None and not best_valid:
        # Try L-BFGS-B with a smoother penalty?
        # Or just return the best invalid one? No, must be valid.
        # Let's try to repair the best candidate
        pass

    # Fallback to a known valid packing if optimization failed
    if best_params is None or not best_valid:
        # Use the initial guess 1 which is likely valid or close
        p = guesses[0]
        valid, S_val, gap_val = exact_check(p)
        if valid:
            best_params = p
            best_S = S_val
            best_valid = True
        else:
            # Construct a very safe, large hexagon
            # Just place them in a line? No, must fit in a hexagon.
            # Place 11 hexagons in a 5x3 grid?
            # Width 5*2 = 10? Height 3*1.732 = 5.2?
            # Side length of outer hex must be large.
            # Let's use the grid from inspiration 1
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
            outer_hex_side_length = 8.0
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Format output
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
