import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import itertools

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s kill
    
    n = 14
    
    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms
    
    def compute_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # 1. Generate Dodecahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            verts.append([0, a, b])
            verts.append([a, b, 0])
            verts.append([b, 0, a])
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 2. Find best subset of 14
    best_score = 0
    best_subset_idx = None
    n_verts = 20
    
    for combo in itertools.combinations(range(n_verts), 14):
        pts = verts[list(combo)]
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        score = (d_min / d_max) ** 2
        if score > best_score:
            best_score = score
            best_subset_idx = combo
            
    initial_points = verts[list(best_subset_idx)].copy()
    
    best_final_points = initial_points.copy()
    best_final_score = best_score
    
    # 3. Gradient Descent / Fekete Point Optimization
    # We maximize the product of distances (Fekete points), which is equivalent to 
    # maximizing the geometric mean. This is a smooth objective and often leads to 
    # configurations with high min/max ratio.
    # Objective: log(prod(dist)) = sum(log(dist))
    # Gradient of log(dist_ij) w.r.t p_i is (p_i - p_j) / ||p_i - p_j||^2
    
    def fekete_grad(points):
        # points: (n, 3)
        # diff[i, j] = p_i - p_j
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist2, 1.0)
        
        # grad[i] = sum_j (p_i - p_j) / dist2[i, j]
        inv_dist2 = 1.0 / dist2
        np.fill_diagonal(inv_dist2, 0.0)
        
        grads = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
        
        # Project gradient onto tangent plane of sphere
        # p_i is unit vector
        dots = np.sum(grads * points, axis=1, keepdims=True)
        grads_tangent = grads - dots * points
        
        return grads_tangent

    def run_gradient_descent(init_pts, max_steps=5000, lr=0.05, min_lr=1e-6, decay=0.999):
        pts = init_pts.copy()
        current_lr = lr
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            grads = fekete_grad(pts)
            
            # Adaptive learning rate or simple decay
            pts = pts + current_lr * grads
            pts = project_to_sphere(pts)
            
            current_lr *= decay
            if current_lr < min_lr:
                break
                
        return pts

    # Run multiple restarts with gradient descent
    # Start from the best dodecahedron subset and some random perturbations
    np.random.seed(42)
    
    restarts = [initial_points]
    for i in range(10):
        noise = np.random.randn(n, 3) * 0.5
        pts = project_to_sphere(initial_points + noise)
        restarts.append(pts)
        
        # Also try completely random starts
        if i % 2 == 0:
            random_pts = np.random.randn(n, 3)
            random_pts = project_to_sphere(random_pts)
            restarts.append(random_pts)

    for i, init_pts in enumerate(restarts):
        if time.time() > deadline - 60:
            break
            
        # Run gradient descent
        # Use a slightly larger learning rate initially
        result_pts = run_gradient_descent(init_pts, max_steps=3000, lr=0.1, decay=0.9995)
        
        score = compute_score(result_pts)
        if score > best_final_score:
            best_final_score = score
            best_final_points = result_pts.copy()

    # 4. Local refinement with Nelder-Mead on the raw ratio
    # This helps fine-tune the min/max ratio specifically, as Fekete optimizes geometric mean
    def objective_ratio(x):
        pts = x.reshape(n, 3)
        pts = project_to_sphere(pts)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0
        return -((d_min / d_max) ** 2)
    
    # Try local optimization from the best found so far
    if time.time() < deadline - 30:
        try:
            x0 = best_final_points.flatten()
            res = minimize(objective_ratio, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            opt_pts = project_to_sphere(res.x.reshape(n, 3))
            score = compute_score(opt_pts)
            if score > best_final_score:
                best_final_score = score
                best_final_points = opt_pts.copy()
        except Exception:
            pass

    # 5. Final check and return
    # Ensure points are on the sphere and finite
    best_final_points = project_to_sphere(best_final_points)
    if not np.all(np.isfinite(best_final_points)):
        return initial_points
        
    return best_final_points
