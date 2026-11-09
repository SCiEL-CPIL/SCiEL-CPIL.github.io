import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = np.inf  # We minimize the score (covering radius proxy)

    # Helper to compute exact covering radius for final validation
    def compute_exact_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                # Distance from circumcenter to vertices
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                # Edge midpoints
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                # Robust check for circumcenter inside/outside
                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                else:
                    local_rad = max_d_edge

                if local_rad > max_rad:
                    max_rad = local_rad
            return max_rad
        except Exception:
            return np.inf

    # Repulsive Energy Minimization Strategy
    # We define a potential energy that is high when points are close, 
    # specifically designed to push the minimum distance up.
    # E = sum_{i<j} exp(-k * d_{ij})
    # Minimizing this maximizes the minimum distance d_min.
    
    def energy_and_grad(pts):
        # pts shape (N, 3)
        # Compute pairwise distances
        # dist^2 = 2 - 2 * dot
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        
        # We want to avoid self-interaction
        np.fill_diagonal(dots, 2.0) # cos(0)=1 -> dist^2=0. But we want to ignore self.
        # Actually, let's just mask the diagonal in the sum.
        
        # dist = arccos(dots)
        # To make it differentiable and smooth, we use the exponential kernel.
        # k controls the "steepness". Large k focuses on the closest pair.
        
        # Let's use a fixed k that is appropriate for N=50.
        # Typical min distance for 50 points is ~ 20-30 degrees.
        # We want the potential to be significant for d < ~40 degrees.
        # exp(-k * d). If d=0.5 rad (~30 deg), we want exp(-k*0.5) ~ 0.1 => k ~ 4.6
        # Let's try k = 5.0
        
        k = 5.0
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)
        
        # Mask diagonal for energy sum
        mask = ~np.eye(N, dtype=bool)
        
        # Energy
        E = np.sum(np.exp(-k * dists)[mask])
        
        # Gradient
        # dE/dp_i = sum_{j != i} dE/d d_ij * d d_ij / d p_i
        # E = sum exp(-k d)
        # dE/dd = -k exp(-k d)
        # d d / d p_i = - (p_i - p_j) / ||p_i - p_j|| * (1/sqrt(1-dot^2))? 
        # Actually, on the sphere, the gradient of distance w.r.t p_i is - (p_j - p_i projected tangent) / d
        # Let's use the vector form.
        # grad_d_ij w.r.t p_i = - (p_j - p_i) / ||p_j - p_i||
        # Note: ||p_j - p_i|| = 2 sin(d/2).
        
        # Vectorized gradient computation
        # diff = p_i - p_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
        norms = np.linalg.norm(diff, axis=2) # (N, N)
        np.fill_diagonal(norms, 1.0) # avoid div by zero
        
        # Unit vectors u_ij = (p_i - p_j) / ||p_i - p_j||
        u = diff / norms[:, :, np.newaxis]
        
        # Weights w_ij = -k * exp(-k * d_ij)
        # We only need off-diagonal
        weights = -k * np.exp(-k * dists)
        np.fill_diagonal(weights, 0.0)
        
        # grad_p_i = sum_j w_ij * u_ij
        # u_ij is direction from j to i? No, diff is p_i - p_j.
        # d(d_ij)/d(p_i) = (p_i - p_j) / ||p_i - p_j|| = u_ij
        # So grad_E = sum_j w_ij * u_ij
        # Wait, E = sum exp(-k d). dE/dd = -k exp(-k d).
        # dE/dp_i = sum_j (dE/dd_ij) * (dd_ij/dp_i)
        # dd_ij/dp_i = (p_i - p_j)/||p_i - p_j||
        # So grad = sum_j (-k exp(-k d_ij)) * u_ij
        
        grad = np.sum(weights[:, :, np.newaxis] * u, axis=1)
        
        # Project gradient onto tangent plane (since points are on sphere)
        # p_i is normal. Tangent grad = grad - (grad . p_i) p_i
        dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - dot_prod * pts
        
        return E, grad

    # Optimization Loop
    # We will run multiple restarts with different initializations and k values.
    
    configs = [
        {'k': 5.0, 'init': 'fib'},
        {'k': 8.0, 'init': 'fib'},
        {'k': 5.0, 'init': 'rand'},
        {'k': 10.0, 'init': 'rand'},
    ]

    for cfg in configs:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        if cfg['init'] == 'fib':
            x0 = points.copy()
        else:
            # Random init
            x0 = np.random.randn(N, 3)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            
        # Use L-BFGS-B with bounds? No, sphere constraint.
        # We will use a custom step or project back to sphere after each step.
        # scipy.optimize.minimize doesn't handle sphere constraints directly well without penalty.
        # Instead, we do a manual gradient descent with projection.
        
        x = x0.copy()
        E, g = energy_and_grad(x)
        
        # Adaptive learning rate
        lr = 0.01
        patience = 0
        
        for step in range(500):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Update
            x_new = x - lr * g
            
            # Project to sphere
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            
            E_new, g_new = energy_and_grad(x_new)
            
            if E_new < E:
                x = x_new
                E = E_new
                g = g_new
                patience = 0
                # Decay lr slightly if improving
                if step % 50 == 0 and step > 0:
                    lr *= 0.95
            else:
                patience += 1
                lr *= 0.9
                if patience > 10:
                    break
        
        # Evaluate this result
        # The energy is a proxy. We need to check the actual covering radius.
        # However, minimizing the sum of exp(-k d) strongly correlates with maximizing min d.
        # We will keep the configuration with the lowest energy, and also check the exact radius periodically.
        
        # To save time, we only compute exact radius if energy is promising.
        # Or just compute it now since it's fast enough for one config.
        current_exact_rad = compute_exact_covering_radius(x)
        
        if current_exact_rad < best_score:
            best_score = current_exact_rad
            best_points = x.copy()

    # Final Polish: Local Refinement on the best point found
    # Use the exact covering radius to guide small moves if time permits.
    remaining_time = time_limit - (time.time() - start_time) - 2.0
    if remaining_time > 5.0:
        # Simple local search using exact radius
        pts = best_points.copy()
        best_local_rad = best_score
        step_size = 0.05
        
        t_start = time.time()
        while time.time() - t_start < remaining_time:
            # Find worst facet
            try:
                hull = ConvexHull(pts)
                max_rad = 0.0
                worst_simplex = None
                worst_center = None
                
                for simplex in hull.simplices:
                    v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                    normal = np.cross(v2 - v1, v3 - v1)
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    c_s = normal / norm
                    
                    d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                    d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                    d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                    max_d_circ = max(d1, d2, d3)
                    
                    e12 = v1 + v2; e12_norm = np.linalg.norm(e12)
                    m12 = e12 / e12_norm if e12_norm > 1e-10 else v1
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0)) if e12_norm > 1e-10 else 0
                    
                    e23 = v2 + v3; e23_norm = np.linalg.norm(e23)
                    m23 = e23 / e23_norm if e23_norm > 1e-10 else v2
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0)) if e23_norm > 1e-10 else 0

                    e31 = v3 + v1; e31_norm = np.linalg.norm(e31)
                    m31 = e31 / e31_norm if e31_norm > 1e-10 else v3
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0)) if e31_norm > 1e-10 else 0

                    max_d_edge = max(d_e12, d_e23, d_e31)

                    if max_d_circ <= np.pi / 2.0:
                        local_rad = max_d_circ
                        center = c_s
                    else:
                        local_rad = max_d_edge
                        if d_e12 == max_d_edge: center = m12
                        elif d_e23 == max_d_edge: center = m23
                        else: center = m31

                    if local_rad > max_rad:
                        max_rad = local_rad
                        worst_simplex = simplex
                        worst_center = center
                        
                if worst_simplex is None: break
                
                # Move vertices towards center
                v1, v2, v3 = pts[worst_simplex[0]], pts[worst_simplex[1]], pts[worst_simplex[2]]
                ss = step_size * 0.5
                
                new_v1 = v1 + ss * (worst_center - v1)
                new_v2 = v2 + ss * (worst_center - v2)
                new_v3 = v3 + ss * (worst_center - v3)
                
                new_v1 /= np.linalg.norm(new_v1)
                new_v2 /= np.linalg.norm(new_v2)
                new_v3 /= np.linalg.norm(new_v3)
                
                pts[worst_simplex[0]] = new_v1
                pts[worst_simplex[1]] = new_v2
                pts[worst_simplex[2]] = new_v3
                
                new_rad = compute_exact_covering_radius(pts)
                if new_rad < best_local_rad:
                    best_local_rad = new_rad
                    best_points = pts.copy()
                else:
                    # Revert
                    pts = best_points.copy()
                    # Small random kick to escape local minimum
                    pts += np.random.normal(0, 0.005, pts.shape)
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    step_size *= 0.95 # Shrink step if stuck

            except Exception:
                break

    return best_points

# EVOLVE-BLOCK-END
