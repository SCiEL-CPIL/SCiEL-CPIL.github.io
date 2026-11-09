import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 8
    # Regular octagon on unit circle (diameter 2)
    angles = 2 * np.pi * np.arange(n) / n
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])

    def get_score(points):
        """
        Calculate the score for a set of points.
        Points are scaled to have diameter 2.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        diam = np.max(dists)
        if diam < 1e-9:
            return 0.0
        scaled_points = points / (diam / 2.0)
        
        diff_s = scaled_points[:, np.newaxis, :] - scaled_points[np.newaxis, :, :]
        dists_s = np.linalg.norm(diff_s, axis=2)
        # Mask diagonal
        dists_s = np.where(np.eye(n, dtype=bool), 1.0, dists_s)
        
        # Check for zero distances (distinct points requirement)
        if np.any(dists_s < 1e-9):
            return 0.0
            
        log_prod = np.sum(np.log(dists_s))
        score = np.exp(log_prod - n * np.log(n))
        return score

    best_score = get_score(pts_reg)
    best_points = pts_reg

    def objective(params):
        # params: 12 floats for points 2..7
        p_full = np.zeros((n, 2))
        p_full[0] = [0.0, 0.0]
        p_full[1] = [2.0, 0.0]
        p_full[2:] = params.reshape((n-2, 2))
        
        # Calculate distances
        diff = p_full[:, np.newaxis, :] - p_full[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Penalty for diameter > 2
        # We want to maximize log product, so minimize -log product + penalty
        violations = np.maximum(0, dists - 2.0)
        penalty = 1000.0 * np.sum(violations**2)
        
        # Log product
        # Avoid log(0)
        dists_masked = np.where(np.eye(n, dtype=bool), 1.0, dists)
        # If points coincide, distance is 0, log is -inf. 
        # We add a small penalty for close points to keep them distinct.
        close_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i,j]
                if d < 0.01:
                    close_penalty += 100.0 * (0.01 - d)**2
        
        log_prod = np.sum(np.log(dists_masked + 1e-12))
        
        return -log_prod + penalty + close_penalty

    bounds = [(-2.0, 2.0)] * (2 * (n - 2))
    
    # 1. Try Differential Evolution
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=30, 
            popsize=15, 
            seed=42,
            tol=1e-5,
            polish=False
        )
        
        if time.time() > deadline:
            raise TimeoutError()
            
        p_full = np.zeros((n, 2))
        p_full[0] = [0.0, 0.0]
        p_full[1] = [2.0, 0.0]
        p_full[2:] = de_result.x.reshape((n-2, 2))
        
        score = get_score(p_full)
        if score > best_score:
            best_score = score
            best_points = p_full
            
    except Exception:
        pass

    # 2. Local Refinement with L-BFGS-B from multiple starts
    # Start points:
    # 1. Regular octagon (adjusted to frame)
    # 2. DE result
    # 3. Random points
    
    start_points_list = []
    
    # Regular octagon in our frame:
    # Rotate/scale so that p1=(0,0) and p2=(2,0)
    # In regular octagon, distance between adjacent vertices is $2 \sin(\pi/8) \approx 0.765$.
    # Distance between opposite is 2.
    # So we can pick p1 and p3 (opposite) as the diameter pair?
    # Or p1 and p2 (adjacent)?
    # If we pick adjacent, the diameter is NOT 2.
    # The diameter of the regular octagon is the distance between opposite vertices.
    # So we should fix p1 and p5 (index 4) as (0,0) and (2,0).
    # Let's construct this specific start point.
    
    # Indices in regular octagon: 0,1,2,3,4,5,6,7
    # Opposite to 0 is 4.
    # Let's set p_new[0] = p_reg[0], p_new[4] = p_reg[4].
    # Translate and rotate.
    
    p_reg = pts_reg
    p0 = p_reg[0]
    p4 = p_reg[4]
    
    # Vector from p0 to p4
    v = p4 - p0
    norm_v = np.linalg.norm(v)
    # Scale so norm_v becomes 2
    scale = 2.0 / norm_v
    p_scaled = p_reg * scale
    
    p0_s = p_scaled[0]
    p4_s = p_scaled[4]
    
    # Translate p0_s to (0,0)
    p_trans = p_scaled - p0_s
    
    # Rotate so p4_s is on x-axis
    # p4_s is now at some angle.
    # We want to rotate the whole set.
    # p4_s is currently at vector v_scaled.
    # Angle of v_scaled
    angle = np.arctan2(v_scaled[1], v_scaled[0])
    # Rotation matrix to rotate by -angle
    c = np.cos(-angle)
    s = np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    
    p_rot = (R @ p_trans.T).T
    
    # Now p_rot[0] is (0,0) and p_rot[4] is (2,0).
    # But our objective function assumes p[0]=(0,0) and p[1]=(2,0).
    # So we need to permute the points so that the point at (2,0) is at index 1.
    # Currently it is at index 4.
    # We can just permute the array.
    # New index 1 should be old index 4.
    # New index 0 is old index 0.
    # The rest can be in any order? No, the objective function doesn't care about order, 
    # but the parameterization `params` maps to indices 2..7.
    # So we need to map the 6 free points to indices 2..7.
    # Let's create a permutation.
    # We want the point at (2,0) to be at index 1.
    # Let's just build the array `p_full` directly for the start point.
    
    p_start_reg = np.zeros((n, 2))
    p_start_reg[0] = [0.0, 0.0]
    p_start_reg[1] = [2.0, 0.0]
    # The other 6 points are p_rot[1], p_rot[2], p_rot[3], p_rot[5], p_rot[6], p_rot[7]
    # We just need to place them in p_start_reg[2:].
    # The order doesn't matter for the score, but it matters for the optimizer's continuity.
    # Let's just take them in order.
    remaining_indices = [1, 2, 3, 5, 6, 7]
    p_start_reg[2:] = p_rot[remaining_indices]
    
    start_points_list.append(p_start_reg[2:].flatten())
    
    if 'de_result' in locals() and de_result.x is not None:
        start_points_list.append(de_result.x)
        
    # Add some random starts
    for _ in range(5):
        start_points_list.append(np.random.uniform([-2.0] * 12, [2.0] * 12))

    for start_params in start_points_list:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective, 
                start_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200}
            )
            
            p_full = np.zeros((n, 2))
            p_full[0] = [0.0, 0.0]
            p_full[1] = [2.0, 0.0]
            p_full[2:] = res.x.reshape((n-2, 2))
            
            score = get_score(p_full)
            if score > best_score:
                best_score = score
                best_points = p_full
                
        except Exception:
            continue

    return best_points.tolist()
