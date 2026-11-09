import numpy as np
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback
        d = np.sqrt(3) * 1.001
        centers = []
        for i in range(3):
            for j in range(4):
                c = np.array([i * d, j * d * np.sqrt(3)/2 + (i % 2) * d * np.sqrt(3)/4])
                centers.append(c)
        centers = np.array(centers)
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        # Simple S estimate
        n0 = np.array([1.0, 0.0])
        n1 = np.array([0.5, np.sqrt(3)/2])
        n2 = np.array([-0.5, np.sqrt(3)/2])
        projs0 = np.max(np.abs(centers @ n0))
        projs1 = np.max(np.abs(centers @ n1))
        projs2 = np.max(np.abs(centers @ n2))
        A = max(projs0, projs1, projs2) + np.sqrt(3)/2
        S = 2 * A / np.sqrt(3)
        inner = [[c_pt[0], c_pt[1], 0.0] for c_pt in centers]
        return inner, (0.0, 0.0), S, 0.0

    # 1. Initialize Grid
    # Use slightly larger spacing to ensure strict disjointness
    d = np.sqrt(3) * 1.001
    e1 = np.array([d, 0])
    e2 = np.array([d/2, d*np.sqrt(3)/2])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * e1 + j * e2
            centers.append(c)
            
    centers = np.array(centers)
    # Center the cluster
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Normals of the outer hexagon
    n0 = np.array([1.0, 0.0])
    n1 = np.array([0.5, np.sqrt(3)/2])
    n2 = np.array([-0.5, np.sqrt(3)/2])
    normals = np.array([n0, n1, n2])
    normal_angles = np.array([0.0, np.pi/3, 2*np.pi/3])

    def support_hex(alpha):
        alpha = alpha % (np.pi / 3)
        if alpha > np.pi / 6:
            alpha = np.pi / 3 - alpha
        return (np.sqrt(3) / 2) / np.cos(alpha)

    def calc_A(theta, pts):
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_centers = pts @ R.T
        
        projs = rotated_centers @ normals.T
        max_abs_projs = np.max(np.abs(projs), axis=0)
        
        h_vals = np.array([support_hex(normal_angles[j] - theta) for j in range(3)])
        
        A_vals = max_abs_projs + h_vals
        return np.max(A_vals)

    # 2. Optimize Global Rotation
    thetas = np.linspace(0, np.pi, 1000)
    A_vals = np.array([calc_A(t, centers) for t in thetas])
    min_idx = np.argmin(A_vals)
    theta_best = thetas[min_idx]
    
    # Refine rotation
    thetas_fine = np.linspace(theta_best - 0.01, theta_best + 0.01, 500)
    A_vals_fine = np.array([calc_A(t, centers) for t in thetas_fine])
    min_idx_fine = np.argmin(A_vals_fine)
    theta_opt = thetas_fine[min_idx_fine]
    A_prime_opt = A_vals_fine[min_idx_fine]
    
    # Apply rotation to centers
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    R = np.array([[c, -s], [s, c]])
    current_centers = centers @ R.T
    current_S = 2 * A_prime_opt / np.sqrt(3)
    
    # 3. Local Search (Relaxation)
    # Try to move centers slightly to reduce S while maintaining non-overlap
    # Constraint: dist(p_i, p_j) >= sqrt(3) + eps
    min_dist = np.sqrt(3) + 1e-5
    
    def is_valid(pts):
        # Check all pairs
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < min_dist:
                    return False
        return True

    def get_S(pts):
        # Find best rotation for this set of points? 
        # For local search, we assume the global rotation is fixed (theta_opt) 
        # and we only move points in the rotated frame? 
        # Actually, it's better to work in the original frame and re-evaluate rotation? 
        # That's expensive. 
        # Let's work in the frame where the outer hexagon is fixed (0 deg) 
        # and the inner points are at current_centers. 
        # We want to minimize the enclosing hexagon side length.
        # The enclosing hexagon side length depends on the positions.
        # We can calculate the required apothem A for the current points 
        # assuming the outer hexagon is aligned with axes (0 deg).
        # But the inner hexagons are rotated by theta_opt.
        # Wait, the "angle" returned for inner hexagons is theta_opt.
        # So the inner hexagons are rotated.
        # The enclosing hexagon is at 0 deg.
        # So we just need to check if the rotated inner hexagons fit in the outer hexagon.
        # This is exactly what calc_A does, but calc_A takes theta and unrotated centers.
        # Here, current_centers are already rotated by theta_opt.
        # So we can just project them onto the normals and add the support function 
        # of the inner hexagon in its current orientation (theta_opt).
        
        # Support of inner hexagon (rotated by theta_opt) along normal angle phi:
        # support = support_hex(phi - theta_opt)
        
        projs = current_centers @ normals.T # Wait, current_centers are in global frame.
        # The normals are in global frame (0, 60, 120).
        # So projs[i] is projection of center i on normal i.
        max_abs_projs = np.max(np.abs(projs), axis=0)
        
        h_vals = np.array([support_hex(normal_angles[j] - theta_opt) for j in range(3)])
        A_vals = max_abs_projs + h_vals
        A = np.max(A_vals)
        return 2 * A / np.sqrt(3)

    # Local Search Loop
    step_size = 0.05
    max_iterations = 200
    current_S_val = get_S(current_centers)
    
    for it in range(max_iterations):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        improved = False
        # Try moving each center slightly in random directions
        for i in range(N):
            # Generate a few random moves
            for _ in range(5):
                delta = np.random.normal(0, step_size, 2)
                new_pts = current_centers.copy()
                new_pts[i] = new_pts[i] + delta
                
                if not is_valid(new_pts):
                    continue
                
                new_S_val = get_S(new_pts)
                if new_S_val < current_S_val - 1e-6:
                    current_centers = new_pts
                    current_S_val = new_S_val
                    improved = True
                    break # Restart random moves for this i? Or just accept and continue?
                    # Let's accept and try next i
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-4:
                break

    # Final Calculation
    # We have current_centers in the global frame.
    # The inner hexagons are rotated by theta_opt.
    # The outer hexagon is at 0 deg.
    # We need to return:
    # inner: (x, y, angle_deg) for each hexagon.
    # center: (0,0)
    # side: current_S_val
    # angle: 0.0
    
    # Verify validity one last time
    if not is_valid(current_centers):
        # Fallback to rigid grid
        current_centers = centers @ R.T
        current_S_val = current_S

    inner = []
    angle_deg = np.degrees(theta_opt)
    for c_pt in current_centers:
        inner.append([c_pt[0], c_pt[1], angle_deg])
        
    return inner, (0.0, 0.0), current_S_val, 0.0
