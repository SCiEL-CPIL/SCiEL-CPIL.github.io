import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(points):
    """
    Compute the covering radius in radians via convex hull facets.
    The covering radius is the maximum circumradius of the spherical triangles
    formed by the facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        # Compute the normal vector to the plane of the triangle
        # This normal points towards the circumcenter on the sphere
        n1 = v1 - v2
        n2 = v1 - v3
        cross = np.cross(n1, n2)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            continue
            
        c = cross / norm
        
        # Ensure c points in the same hemisphere as the triangle (outward from origin)
        if c @ v1 < 0:
            c = -c
            
        cos_R = np.clip(c @ v1, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # The circumradius of the spherical triangle is R if the circumcenter
        # is inside the triangle. If not, the covering radius for that facet
        # is determined by the farthest edge midpoint.
        # For a small triangle (which is the case for N=130), the circumcenter
        # is usually inside. To be rigorous, we check edge midpoints too.
        
        # Edge midpoints on the sphere
        m12 = v1 + v2
        n12 = np.linalg.norm(m12)
        if n12 > 1e-10:
            m12 /= n12
            d12 = np.arccos(np.clip(m12 @ v1, -1.0, 1.0))
        else:
            d12 = np.pi
            
        m23 = v2 + v3
        n23 = np.linalg.norm(m23)
        if n23 > 1e-10:
            m23 /= n23
            d23 = np.arccos(np.clip(m23 @ v2, -1.0, 1.0))
        else:
            d23 = np.pi
            
        m31 = v3 + v1
        n31 = np.linalg.norm(m31)
        if n31 > 1e-10:
            m31 /= n31
            d31 = np.arccos(np.clip(m31 @ v3, -1.0, 1.0))
        else:
            d31 = np.pi
            
        # The local covering radius is the max of the circumradius and the edge midpoints
        # Note: If the circumcenter is outside, R > d_edge usually doesn't hold in the same way,
        # but max(R, d12, d23, d31) is a safe upper bound for the distance to the nearest point
        # within that facet's Voronoi cell on the sphere.
        local_r = max(R, d12, d23, d31)
        if local_r > max_radius:
            max_radius = local_r
            
    return max_radius

def _soft_max_surrogate(points, beta=50.0):
    """
    Fast O(N^2) surrogate for covering radius optimization.
    We want to maximize the minimum angular distance between any pair of points.
    Equivalently, minimize the maximum angular distance to the nearest neighbor.
    
    Let d_ij be the angular distance. We want to minimize max_i min_j d_ij.
    This is equivalent to maximizing min_i min_j cos(d_ij) is NOT correct because cos is decreasing.
    We want to MAXIMIZE the minimum separation.
    
    Let's define the objective to MINIMIZE as:
    max_i ( - min_j dot(p_i, p_j) )
    Wait, maximizing separation means maximizing the angle, which means minimizing the dot product.
    So we want to MINIMIZE the max over i of ( - min_j dot(p_i, p_j) ).
    Let f_i = - min_j dot(p_i, p_j). We want to minimize max_i f_i.
    
    Soft-max approximation of max_i f_i:
    log( sum_i exp(beta * f_i) ) / beta
    
    f_i = - min_j dot(p_i, p_j).
    min_j dot(p_i, p_j) is the minimum dot product.
    - min_j dot(p_i, p_j) = max_j (-dot(p_i, p_j)).
    
    So we are minimizing the soft-max of (max_j (-dot(p_i, p_j))).
    
    Let's compute g_i = max_j (-dot(p_i, p_j)).
    We use soft-max to approximate max_j:
    g_i ≈ (1/beta) * log( sum_j exp(-beta * dot(p_i, p_j)) )
    
    Then we use soft-max to approximate max_i:
    Objective ≈ (1/beta) * log( sum_i exp(beta * g_i) )
    
    This is getting complex to differentiate. Let's use a simpler proxy:
    Minimize the sum of squared "deficits" from a target separation?
    
    Actually, the inspiration used:
    objective = -np.mean(log_sum_exp) where log_sum_exp approximates max_j dots[i,j].
    Minimizing -max_j dots[i,j] is equivalent to maximizing max_j dots[i,j].
    max_j dots[i,j] is the dot product with the CLOSEST neighbor.
    Maximizing the dot product with the closest neighbor means MINIMIZING the angle to the closest neighbor.
    This makes points cluster! We want the OPPOSITE. We want to MAXIMIZE the angle to the closest neighbor.
    
    Correction:
    We want to MAXIMIZE the minimum angular distance.
    Let theta_min_i = min_j angle(p_i, p_j).
    We want to maximize min_i theta_min_i.
    
    Since cos(theta) is decreasing, maximizing theta is minimizing cos(theta).
    Let c_min_i = min_j cos(theta_ij) = min_j dot(p_i, p_j).
    We want to maximize min_i c_min_i.
    
    This is a "max-min" problem.
    We can use a soft-min for the inner min and soft-max for the outer max?
    Or just use a penalty on small distances.
    
    Let's stick to the inspiration's logic but fix the direction.
    The inspiration code:
    objective = -np.mean(log_sum_exp) 
    where log_sum_exp approximates max_j dots[i,j].
    Minimizing this objective means MAXIMIZING max_j dots[i,j].
    max_j dots[i,j] is the closest neighbor dot product.
    Maximizing the closest neighbor dot product brings points CLOSER.
    This minimizes the covering radius? No, it creates clusters.
    
    Wait, the covering radius is the distance to the NEAREST point for a point Y on the sphere.
    If we cluster points, the gaps between clusters become huge. The covering radius will be large.
    So the inspiration's objective might be flawed for covering radius, OR I am misinterpreting it.
    
    Let's re-read the inspiration score. It improved from 1.046 to 1.072.
    Maybe the "Soft Max" in the inspiration was actually minimizing the *maximum* distance to the nearest neighbor?
    
    Let's try a different surrogate:
    Minimize the sum of exp(-beta * min_j dot(p_i, p_j)).
    If min_j dot(p_i, p_j) is small (large angle), exp is large.
    If min_j dot(p_i, p_j) is large (small angle), exp is small.
    So minimizing this sum encourages points to be CLOSE to their neighbors? No.
    We want points to be FAR from their neighbors? No, we want the GAPS to be small.
    
    The covering radius is determined by the empty caps.
    The largest empty cap is centered at the circumcenter of the largest facet.
    
    Let's use a direct gradient on the covering radius? No, it's non-differentiable.
    
    Let's use the "Repulsion" idea from the current program but refine it.
    The current program uses Lloyd's.
    
    Let's try this surrogate:
    Minimize the sum of (1 - min_j dot(p_i, p_j))^2.
    This minimizes the squared chord distance to the nearest neighbor.
    This makes points cluster. BAD.
    
    We want to MAXIMIZE the minimum distance.
    Maximize min_i min_j distance(p_i, p_j).
    
    Let's use:
    Minimize - min_i min_j distance(p_i, p_j).
    
    Surrogate:
    Minimize log( sum_i exp(-beta * min_j distance(p_i, p_j)) )
    
    Let's implement a simple "Push-Apart" gradient.
    For each point i, find its nearest neighbor j.
    Apply a force away from j.
    Also, apply a force towards the centroid of the "Voronoi cell" (Lloyd's).
    
    Combining Lloyd's and Repulsion:
    1. Compute Voronoi cells (nearest neighbor assignment).
    2. Compute the centroid of each cell (Lloyd's step).
    3. Compute the nearest neighbor for each point.
    4. Apply a repulsive force away from the nearest neighbor.
    5. Blend the two forces.
    
    This hybrid "Lloyd-Repulsion" is a known heuristic for sphere packing/covering.
    """
    pass # Placeholder, will implement hybrid in _optimize_hybrid

def _optimize_hybrid(points, deadline):
    """
    Hybrid optimization: Lloyd's Algorithm + Nearest Neighbor Repulsion.
    """
    start_time = time.time()
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _covering_radius(current_points)
    
    # Parameters
    lloyd_weight = 1.0
    repulsion_weight = 0.5
    step_size = 0.05
    no_improve = 0
    max_iters = 300
    
    for i in range(max_iters):
        if time.time() > deadline - 10.0:
            break
            
        dots = current_points @ current_points.T
        np.fill_diagonal(dots, -1.0) # Exclude self
        
        # 1. Lloyd's Step: Move towards centroid of Voronoi cell
        nearest_indices = np.argmax(dots, axis=1)
        
        lloyd_targets = np.zeros_like(current_points)
        for j in range(N):
            neighbors = np.where(nearest_indices == j)[0]
            if len(neighbors) > 0:
                cell_points = current_points[neighbors]
                mean_vec = np.sum(cell_points, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-10:
                    lloyd_targets[j] = mean_vec / norm
                else:
                    lloyd_targets[j] = current_points[j]
            else:
                lloyd_targets[j] = current_points[j]
                
        # 2. Repulsion Step: Move away from nearest neighbor
        repulsion_targets = np.zeros_like(current_points)
        for j in range(N):
            nn_idx = nearest_indices[j]
            # Vector from NN to current point
            vec = current_points[j] - current_points[nn_idx]
            # Project onto tangent plane? Or just normalize?
            # We want to move in the direction of vec.
            # To stay on sphere, we project vec onto the tangent space at current_points[j]
            p_j = current_points[j]
            proj = vec - np.dot(vec, p_j) * p_j
            norm = np.linalg.norm(proj)
            if norm > 1e-10:
                repulsion_targets[j] = p_j + 0.1 * proj # Small step away
                # Normalize to sphere
                repulsion_targets[j] /= np.linalg.norm(repulsion_targets[j])
            else:
                repulsion_targets[j] = current_points[j]
                
        # 3. Blend
        # We want to move current_points towards a blend of lloyd_targets and repulsion_targets
        # Actually, it's better to compute the vector field.
        
        # Vector towards Lloyd target
        v_lloyd = lloyd_targets - current_points
        # Project to tangent
        v_lloyd = v_lloyd - np.dot(v_lloyd, current_points, axis=1)[:, np.newaxis] * current_points
        
        # Vector towards Repulsion target (which is already a direction)
        v_rep = repulsion_targets - current_points
        v_rep = v_rep - np.dot(v_rep, current_points, axis=1)[:, np.newaxis] * current_points
        
        # Combine
        v_total = lloyd_weight * v_lloyd + repulsion_weight * v_rep
        
        # Step
        new_points = current_points + step_size * v_total
        
        # Project to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        # Evaluate
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = new_points.copy()
            no_improve = 0
            # Adaptive step size: if improved, keep or increase slightly
            # step_size = min(step_size * 1.05, 0.1)
        else:
            no_improve += 1
            # Reduce step size
            step_size *= 0.9
            
        current_points = new_points
        
        if no_improve > 50:
            # Perturb to escape local minima
            noise = np.random.normal(0, 0.05, size=current_points.shape)
            perturbed = current_points + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            current_points = perturbed / norms
            no_improve = 0
            step_size = 0.05
            
    # Final Polish with L-BFGS on the best point found
    if time.time() < deadline - 5.0:
        def objective(params):
            p = params.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p = p / norms
            return _covering_radius(p)
            
        try:
            res = minimize(objective, best_points.flatten(), method='L-BFGS-B', 
                          options={'maxiter': 20, 'maxfun': 500, 'disp': False})
            if res.success:
                p_opt = res.x.reshape(N, 3)
                norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
                p_opt = p_opt / norms
                r_opt = _covering_radius(p_opt)
                if r_opt < best_radius:
                    best_points = p_opt
                    best_radius = r_opt
        except Exception:
            pass
            
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    inits.append(fibonacci_sphere(N))
    
    # 2. Random
    rng = np.random.default_rng(42)
    pts_rand = rng.standard_normal((N, 3))
    pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
    inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    pts_pert = fibonacci_sphere(N) + 0.1 * rng.standard_normal((N, 3))
    pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
    inits.append(pts_pert)
    
    # 4. Another Random
    pts_rand2 = rng.standard_normal((N, 3))
    pts_rand2 /= np.linalg.norm(pts_rand2, axis=1, keepdims=True)
    inits.append(pts_rand2)
    
    for init in inits:
        if time.time() > deadline - 25.0:
            break
            
        pts, rad = _optimize_hybrid(init, deadline)
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
            
    if best_points is None:
        best_points = fibonacci_sphere(N)
        
    return best_points

# Run construct to verify syntax/logic if needed, but the evaluator will call it.
# For local testing:
# if __name__ == "__main__":
#     pts = construct()
#     print("Covering Radius (deg):", np.degrees(_covering_radius(pts)))
