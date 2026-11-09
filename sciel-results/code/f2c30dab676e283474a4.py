import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute unit hexagon vertices (flat-top)
    # Angles 0, 60, ..., 300
    angles = np.array([i * 60 for i in range(6)]) * math.pi / 180.0
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Outer hexagon normals (flat-top)
    # Normals at 30, 90, ..., 330
    normal_angles = np.array([30 + i * 60 for i in range(6)]) * math.pi / 180.0
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

    def get_required_L(centers):
        """
        centers: (12, 2) array
        Returns the minimum L such that all hexagons fit in outer hex of side L.
        """
        max_proj = 0.0
        # Vectorized calculation
        # centers: (12, 2)
        # unit_verts: (6, 2)
        # We need to broadcast centers + unit_verts
        # verts: (12, 6, 2)
        verts = centers[:, np.newaxis, :] + unit_verts[np.newaxis, :, :]
        
        # Project onto normals
        # normals: (6, 2)
        # proj: (12, 6, 6) -> dot(verts[i, v], normals[k])
        # proj[i, v, k] = sum(verts[i, v, d] * normals[k, d])
        proj = np.einsum('ivd,kd->ivk', verts, normals)
        
        # Max over vertices and normals
        local_max = proj.max(axis=(1, 2)) # (12,)
        max_proj = local_max.max()
        
        return 2.0 * max_proj / SQRT3

    def objective(x):
        """
        x: (24,) array of centers
        """
        centers = x.reshape(N, 2)
        L = get_required_L(centers)
        
        # Add penalty for overlap to help gradient? 
        # SLSQP handles constraints. We just return L.
        return L

    def constraints(x):
        """
        Constraints: distance between centers >= sqrt(3)
        """
        centers = x.reshape(N, 2)
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = np.sum((centers[i] - centers[j]) ** 2)
                # dist_sq >= 3
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.sum((x[2*i:2*i+2] - x[2*j:2*j+2]) ** 2) - 3.0})
        return cons

    # Initial guess: 7-cluster + 5 ring-2 hexagons
    # 7-cluster centers
    centers_7 = [
        (0, 0),
        (SQRT3, 0),
        (SQRT3/2, 1.5),
        (-SQRT3/2, 1.5),
        (-SQRT3, 0),
        (-SQRT3/2, -1.5),
        (SQRT3/2, -1.5)
    ]
    
    # Ring 2 centers (12 of them)
    # We pick 5 that are "symmetric" or just the first 5
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2 and (abs(q) == 2 or abs(r) == 2 or abs(q+r) == 2):
                # Exclude ring 1 and center
                if not (q == 0 and r == 0):
                    if not (abs(q) == 1 and abs(r) == 1 and abs(q+r) == 1) and not (abs(q)==1 and r==0) and not (q==0 and abs(r)==1):
                         # This logic is messy. Let's just generate all 19 and pick.
                         pass
    
    # Simpler: Generate all 19 axial coords
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = SQRT3 * (r + q / 2.0)
                all_centers.append((x, y))
                
    # Sort by distance from origin
    all_centers_sorted = sorted(all_centers, key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12 (closest to center)
    init_centers = all_centers_sorted[:12]
    x0 = np.array([coord for c in init_centers for coord in c])
    
    best_x = x0
    best_L = get_required_L(x0.reshape(N, 2))
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints(x0), options={'maxiter': 500, 'ftol': 1e-8})
        if res.success:
            L_opt = res.fun
            if L_opt < best_L:
                best_L = L_opt
                best_x = res.x
    except Exception as e:
        pass

    # Try a few more restarts if time permits
    # Add random noise to initial guess
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        x0_noise = x0 + np.random.normal(0, 0.1, size=24)
        try:
            res = minimize(objective, x0_noise, method='SLSQP', constraints=constraints(x0_noise), options={'maxiter': 200, 'ftol': 1e-8})
            if res.success:
                L_opt = res.fun
                if L_opt < best_L:
                    best_L = L_opt
                    best_x = res.x
        except:
            pass

    # Extract final centers
    final_centers = best_x.reshape(N, 2)
    
    inner = []
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], 0.0])
        
    # Outer hexagon centered at 0,0, side best_L, rotation 0
    return inner, (0.0, 0.0), best_L, 0.0

