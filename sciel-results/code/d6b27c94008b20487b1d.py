import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit
    
    # Helper to compute exact covering radius for scoring (High Precision)
    def compute_covering_radius(pts):
        try:
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return np.inf
            if np.all(np.dot(pts, pts[0]) > 0.9999): # Hemisphere check (rough)
                return np.inf
                
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    m12 = v1
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0
                    m23 = v2

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0
                    m31 = v3

                max_d_edge = max(d_e12, d_e23, d_e31)

                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                    center = c_s
                else:
                    local_rad = max_d_edge
                    if d_e12 == max_d_edge:
                        center = m12
                    elif d_e23 == max_d_edge:
                        center = m23
                    else:
                        center = m31

                if local_rad > max_rad:
                    max_rad = local_rad
            return max_rad
        except Exception:
            return np.inf

    # Vectorized facet radius computation for gradient steps
    # Returns: radii (array of len F), centers (F, 3), simplex_indices (F, 3)
    def compute_facet_data(pts):
        try:
            hull = ConvexHull(pts)
            simplices = hull.simplices
            F = len(simplices)
            radii = np.zeros(F)
            centers = np.zeros((F, 3))
            
            for i, simplex in enumerate(simplices):
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    radii[i] = np.inf
                    centers[i] = v1
                    continue
                c_s = normal / norm
                
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    m12 = v1
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0
                    m23 = v2

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0
                    m31 = v3

                max_d_edge = max(d_e12, d_e23, d_e31)

                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                    center = c_s
                else:
                    local_rad = max_d_edge
                    if d_e12 == max_d_edge:
                        center = m12
                    elif d_e23 == max_d_edge:
                        center = m23
                    else:
                        center = m31

                radii[i] = local_rad
                centers[i] = center
                
            return radii, centers, simplices
        except Exception:
            return None, None, None

    def riemannian_gradient_descent(pts, max_time=20.0):
        """
        Deterministic Riemannian Gradient Descent using a smooth surrogate.
        Objective: sum(exp(r_i / T)) approximates max(r_i).
        Gradient of r_i w.r.t v_j is towards center_i if j is in facet i.
        """
        pts = pts.copy()
        T = 0.05  # Temperature for soft-max
        lr = 0.02
        patience = 0
        best_pts = pts.copy()
        best_val = np.inf
        
        # Precompute gradient accumulation buffer
        grad_acc = np.zeros_like(pts)
        
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            radii, centers, simplices = compute_facet_data(pts)
            if radii is None:
                break
                
            # Smooth surrogate objective
            # We want to minimize max(r_i). Using log-sum-exp or sum-exp.
            # Let f = sum(exp(r_i / T)). Grad f = sum((1/T) * exp(r_i/T) * grad r_i)
            # grad r_i w.r.t v_j (where j in facet i) is roughly (c_i - (c_i.v_j)v_j) / |...| ?
            # Actually, we want to move v_j towards c_i to reduce distance.
            # The direction is (c_i - proj_{v_j} c_i).
            
            # Compute weights
            # To avoid overflow, subtract max radius
            r_max = np.max(radii)
            weights = np.exp((radii - r_max) / T)
            sum_weights = np.sum(weights)
            
            if sum_weights == 0:
                break
                
            # Normalize weights to get a probability distribution over facets
            p = weights / sum_weights
            
            grad_acc.fill(0.0)
            
            for i in range(len(radii)):
                if p[i] < 1e-6:
                    continue
                center = centers[i]
                simplex = simplices[i]
                
                # For each vertex in this facet, add contribution to gradient
                for j in simplex:
                    v = pts[j]
                    # Project center onto tangent space at v
                    # Direction from v to center on sphere
                    # grad_direction = center - (center . v) v
                    dot_cv = np.dot(center, v)
                    grad_dir = center - dot_cv * v
                    
                    # Normalize to unit step direction?
                    norm_gd = np.linalg.norm(grad_dir)
                    if norm_gd > 1e-10:
                        grad_dir /= norm_gd
                        
                    # Accumulate weighted gradient
                    # Note: We are minimizing radius, so we move in direction of grad_dir
                    # which points from v towards center.
                    # Wait, if we move v towards center, the distance decreases.
                    # So the "descent" direction for v is +grad_dir.
                    # Standard gradient descent: v_new = v - lr * grad_obj.
                    # Here grad_obj w.r.t v points in direction of increasing radius.
                    # Increasing radius means moving AWAY from center.
                    # So grad_obj ~ -grad_dir.
                    # v_new = v - lr * (-grad_dir) = v + lr * grad_dir.
                    grad_acc[j] += p[i] * grad_dir
                    
            # Apply step
            # Normalize gradient to avoid large steps if multiple facets contribute
            grad_norm = np.linalg.norm(grad_acc, axis=1, keepdims=True)
            mask = grad_norm > 1e-10
            grad_acc[mask] /= grad_norm[mask]
            
            new_pts = pts + lr * grad_acc
            # Re-normalize to sphere
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            # Evaluate exact radius for tracking
            new_rad = compute_covering_radius(new_pts)
            
            if new_rad < best_val:
                best_val = new_rad
                best_pts = new_pts.copy()
                patience = 0
            else:
                patience += 1
                if patience % 10 == 0:
                    lr *= 0.95
                    if lr < 0.001:
                        break
                        
            pts = new_pts
            
        return best_pts, best_val

    # Multi-start strategy
    best_points = None
    best_radius = np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(pts_fib)
    
    # 2. Random Uniform
    np.random.seed(42)
    for _ in range(9):
        pts_rand = np.random.randn(N, 3)
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        initializations.append(pts_rand)
        
    # 3. Clustered (to test escape)
    # Not really useful for covering, but adds diversity
    
    for init_idx, pts_init in enumerate(initializations):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Run RGD
        pts_opt, rad_opt = riemannian_gradient_descent(pts_init, max_time=10.0)
        
        # Final polish with exact local search (worst facet move)
        # This helps fine-tune the result
        pts_final = pts_opt
        rad_final = compute_covering_radius(pts_final)
        
        # Quick local refinement
        for _ in range(50):
            if time.time() - start_time > time_limit - 2.0:
                break
            radii, centers, simplices = compute_facet_data(pts_final)
            if radii is None: break
            worst_idx = np.argmax(radii)
            worst_center = centers[worst_idx]
            worst_simplex = simplices[worst_idx]
            
            new_pts = pts_final.copy()
            for j in worst_simplex:
                v = pts_final[j]
                dot_cv = np.dot(worst_center, v)
                grad_dir = worst_center - dot_cv * v
                norm_gd = np.linalg.norm(grad_dir)
                if norm_gd > 1e-10:
                    grad_dir /= norm_gd
                    new_pts[j] = v + 0.05 * grad_dir
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_rad = compute_covering_radius(new_pts)
            if new_rad < rad_final:
                pts_final = new_pts
                rad_final = new_rad
            else:
                break
                
        if rad_final < best_radius:
            best_radius = rad_final
            best_points = pts_final.copy()
            
    if best_points is None:
        return initializations[0]
        
    return best_points

# EVOLVE-BLOCK-END
