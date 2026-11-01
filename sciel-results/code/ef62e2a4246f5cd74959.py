"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate a dense grid of points on the sphere to approximate the max distance.
    # We need enough points to accurately estimate the covering radius.
    # The optimal radius is ~18.3 degrees. We want grid spacing < 1 degree to be safe.
    # Number of points ~ 4*pi / (delta_theta)^2.
    # Let's use a Fibonacci spiral for the grid points as well, or a lat/lon grid.
    # Lat/lon is easier to control density.
    n_grid = 50000 # ~ 0.15 degrees spacing, should be sufficient for 18 deg radius
    # Actually, let's use a Fibonacci sphere for the grid to avoid pole clustering issues
    # which might bias the "max" towards the poles if not handled carefully.
    # But the objective is max over ALL points. A uniform grid is good.
    
    # Let's use a deterministic Fibonacci spiral for the grid points to ensure uniformity
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize grid points (already normalized by construction, but just in case)
    grid_points /= np.linalg.norm(grid_points, axis=1, keepdims=True)
    
    def soft_max_covering_radius(pts_flat, beta):
        """
        Compute the soft-max covering radius.
        pts_flat: flattened array of N*3 parameters (coordinates).
        beta: sharpness parameter for log-sum-exp.
        Returns: (objective_value, gradient)
        """
        pts = pts_flat.reshape(N, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Compute squared distances from each grid point to each site point
        # grid_points: (M, 3), pts: (N, 3)
        # dist_sq[i, j] = ||grid_points[i] - pts[j]||^2
        # = ||grid_points[i]||^2 + ||pts[j]||^2 - 2 * grid_points[i] . pts[j]
        # Since both are on unit sphere, dist_sq = 2 - 2 * dot
        # We want the MIN distance from each grid point to the NEAREST site.
        # So for each grid point i, we find min_j dist_sq[i, j].
        # Then we want the MAX over i of min_j dist_sq[i, j].
        
        # Compute dot products: (M, N)
        dots = grid_points @ pts.T # (M, N)
        dist_sq = 2.0 - 2.0 * dots # (M, N)
        
        # Clip to avoid negative values due to floating point errors
        dist_sq = np.clip(dist_sq, 0.0, None)
        
        # For each grid point, find the min squared distance to any site
        min_dist_sq = np.min(dist_sq, axis=1) # (M,)
        
        # We want to maximize the minimum distance? No, we want to MINIMIZE the covering radius.
        # The covering radius is max_i (sqrt(min_j dist_sq[i,j])).
        # So we want to minimize max_i min_j dist_sq[i,j].
        
        # Using log-sum-exp to approximate the max.
        # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
        # Here x = min_dist_sq.
        # But wait, log-sum-exp approximates the max.
        # We want to minimize the max.
        # So objective = (1/beta) * log(sum(exp(beta * min_dist_sq)))
        
        # To avoid overflow, use the trick:
        # log(sum(exp(beta * x))) = beta * max_x + log(sum(exp(beta * (x - max_x))))
        max_min_dist_sq = np.max(min_dist_sq)
        # If beta is large, this can still overflow if beta * max_min_dist_sq is large.
        # But max_min_dist_sq is at most 4 (diameter squared). beta will be moderate (e.g., 10-100).
        # So beta * 4 is at most 400, which is fine for exp.
        
        scaled = beta * min_dist_sq
        # Subtract max for stability
        scaled_max = np.max(scaled)
        sum_exp = np.sum(np.exp(scaled - scaled_max))
        log_sum_exp = scaled_max + np.log(sum_exp)
        
        obj_val = log_sum_exp / beta
        
        # Gradient computation
        # obj = (1/beta) * log(sum_i exp(beta * m_i)) where m_i = min_j d_ij
        # d_obj/d_m_i = exp(beta * m_i) / sum_k exp(beta * m_k)
        # This is the softmax probability distribution over the grid points.
        probs = np.exp(scaled - scaled_max) / sum_exp # (M,)
        
        # Now, m_i = min_j d_ij.
        # The gradient of min is 0 for all j except the argmin, where it is the gradient of d_ij.
        # However, the min is non-differentiable. We can use a soft-min approximation or just
        # attribute the gradient to the argmin. For L-BFGS, using the argmin gradient is a common
        # subgradient approach, but it might be unstable.
        # Alternatively, we can use a soft-min:
        # soft_min_j d_ij ≈ - (1/beta2) * log(sum_j exp(-beta2 * d_ij))
        # This makes the whole thing differentiable.
        
        # Let's re-implement using soft-min for differentiability.
        # m_i = - (1/beta2) * log(sum_j exp(-beta2 * d_ij))
        # Let beta2 = beta (same sharpness).
        
        # Recompute with soft-min
        # d_ij = 2 - 2 * dots[i,j]
        # exp(-beta2 * d_ij) = exp(-beta2 * (2 - 2*dots[i,j])) = exp(-2*beta2) * exp(2*beta2*dots[i,j])
        # The constant exp(-2*beta2) cancels out in the log-sum-exp.
        # So log(sum_j exp(2*beta2*dots[i,j])) is the key part.
        
        # Let A = 2 * beta
        A = 2.0 * beta
        exp_dots = np.exp(A * dots) # (M, N)
        
        # Sum over sites j
        sum_exp_dots = np.sum(exp_dots, axis=1) # (M,)
        
        # m_i = 2 - (1/beta) * log(sum_j exp(2*beta*dots[i,j]))
        # Wait, the formula for soft min is:
        # min_j x_j ≈ - (1/T) log sum exp(-T x_j)
        # Here x_j = d_ij = 2 - 2 dots_ij
        # -T x_j = -T(2 - 2 dots_ij) = -2T + 2T dots_ij
        # sum exp(-T x_j) = exp(-2T) sum exp(2T dots_ij)
        # log(sum) = -2T + log(sum exp(2T dots_ij))
        # min ≈ - (1/T) (-2T + log(sum exp(2T dots_ij))) = 2 - (1/T) log(sum exp(2T dots_ij))
        
        # So m_i = 2 - (1/beta) * log(sum_j exp(2*beta*dots[i,j]))
        
        # Now we want to minimize max_i m_i.
        # Use log-sum-exp again for the max over i.
        # Obj = (1/beta) * log(sum_i exp(beta * m_i))
        
        # Let's compute this directly.
        # m_i = 2 - (1/beta) * log(S_i) where S_i = sum_j exp(2*beta*dots[i,j])
        # beta * m_i = 2*beta - log(S_i)
        # exp(beta * m_i) = exp(2*beta) * exp(-log(S_i)) = exp(2*beta) / S_i
        
        # Sum_i exp(beta * m_i) = exp(2*beta) * sum_i (1 / S_i)
        
        # Obj = (1/beta) * log( exp(2*beta) * sum_i (1/S_i) )
        #     = (1/beta) * ( 2*beta + log(sum_i (1/S_i)) )
        #     = 2 + (1/beta) * log(sum_i (1/S_i))
        
        sum_inv_S = np.sum(1.0 / sum_exp_dots) # (scalar)
        obj_val = 2.0 + (1.0 / beta) * np.log(sum_inv_S)
        
        # Gradient
        # d_obj / d_dots[i,j]
        # Obj = 2 + (1/beta) * log(sum_k (1/S_k))
        # d_obj/d_S_k = (1/beta) * (1/sum_inv_S) * (-1/S_k^2)
        # d_S_k / d_dots[i,j] = 0 if k != i.
        # If k = i: d_S_i / d_dots[i,j] = 2*beta * exp(2*beta*dots[i,j])
        
        # So d_obj/d_dots[i,j] = (1/beta) * (-1/(sum_inv_S * S_i^2)) * 2*beta * exp(2*beta*dots[i,j])
        #                      = -2 * exp(2*beta*dots[i,j]) / (sum_inv_S * S_i^2)
        
        # Let's define weights w_i = (1/S_i) / sum_inv_S. This is the probability distribution.
        # Then d_obj/d_dots[i,j] = -2 * w_i * (exp(2*beta*dots[i,j]) / S_i)
        # Note that exp(2*beta*dots[i,j]) / S_i is the softmax probability of site j for grid point i.
        # Let's call it P(j|i).
        
        # So d_obj/d_dots[i,j] = -2 * w_i * P(j|i)
        
        # Now, d_dots[i,j] / d_pts[j] = grid_points[i]
        # d_obj / d_pts[j] = sum_i d_obj/d_dots[i,j] * grid_points[i]
        #                   = sum_i (-2 * w_i * P(j|i)) * grid_points[i]
        
        # We need to accumulate this for all j.
        # grad_pts[j] = -2 * sum_i (w_i * P(j|i) * grid_points[i])
        
        # Compute P(j|i) = exp(2*beta*dots[i,j]) / S_i
        # P is (M, N)
        P = exp_dots / sum_exp_dots[:, np.newaxis] # (M, N)
        
        # w is (M,)
        w = (1.0 / sum_exp_dots) / sum_inv_S # (M,)
        
        # We need sum_i (w_i * P[i,j] * grid_points[i]) for each j.
        # This is (grid_points.T * w) @ P
        # grid_points.T is (3, M), w is (M,), so (3, M) * (M,) -> (3, M)
        # Then (3, M) @ (M, N) -> (3, N)
        
        grad_pts = -2.0 * (grid_points.T * w) @ P # (3, N)
        
        # Now we need to map gradient from pts to pts_flat (unnormalized).
        # pts = pts_flat / ||pts_flat||
        # d_pts / d_pts_flat = (I - pts pts^T) / ||pts_flat||
        # grad_flat = (I - pts pts^T) @ grad_pts / ||pts_flat||
        
        # Actually, it's easier to project the gradient back to the tangent space.
        # grad_tangent = grad_pts - (grad_pts . pts) pts
        # Then grad_flat = grad_tangent * ||pts_flat||
        
        # Let's compute dot products grad_pts . pts
        dots_grad = np.sum(grad_pts * pts, axis=0) # (N,)
        grad_tangent = grad_pts - dots_grad[np.newaxis, :] * pts # (3, N)
        
        # Scale by norms
        grad_flat = grad_tangent * norms.T # (3, N)
        
        # Flatten
        grad_flat = grad_flat.flatten()
        
        return obj_val, grad_flat

    # Objective function for scipy.optimize.minimize
    def objective(pts_flat):
        val, grad = soft_max_covering_radius(pts_flat, beta=20.0)
        return val, grad

    # Initializations
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = None
    best_score = np.inf

    # Try multiple initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            init_pts = fibonacci_sphere(N)
        elif restart == 1:
            init_pts = fibonacci_sphere(N, offset=0.0)
        else:
            init_pts = fibonacci_sphere(N) + np.random.randn(N, 3) * 0.1
            init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)

        x0 = init_pts.flatten()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'maxfun': 200})
            if res.success or res.status == 0:
                opt_pts = res.x.reshape(N, 3)
                opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
                
                # Calculate true covering radius using convex hull for accuracy
                # This is expensive, so we only do it for the final best candidate
                # For now, use the soft-max value as a proxy, but we need a true metric for comparison.
                # Let's compute the true radius for the result.
                
                try:
                    hull = ConvexHull(opt_pts)
                    max_radius = 0
                    for simplex in hull.simplices:
                        v1 = opt_pts[simplex[0]]
                        v2 = opt_pts[simplex[1]]
                        v3 = opt_pts[simplex[2]]
                        edge1 = v2 - v1
                        edge2 = v3 - v1
                        n = np.cross(edge1, edge2)
                        norm_n = np.linalg.norm(n)
                        if norm_n < 1e-10:
                            continue
                        c = n / norm_n
                        if np.dot(c, v1) < 0:
                            c = -c
                        d = np.linalg.norm(c - v1)
                        radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
                        if radius > max_radius:
                            max_radius = radius
                except:
                    max_radius = np.pi # Fallback to large value if hull fails

                if max_radius < best_score:
                    best_score = max_radius
                    best_points = opt_pts
                    
        except Exception as e:
            pass

    # Final Polish: Use the best points found so far to do a few more iterations with higher beta
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        
        # Increase beta for sharper approximation
        def objective_sharp(pts_flat):
            val, grad = soft_max_covering_radius(pts_flat, beta=50.0)
            return val, grad

        try:
            res = minimize(objective_sharp, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'maxfun': 100})
            if res.success or res.status == 0:
                opt_pts = res.x.reshape(N, 3)
                opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
                
                try:
                    hull = ConvexHull(opt_pts)
                    max_radius = 0
                    for simplex in hull.simplices:
                        v1 = opt_pts[simplex[0]]
                        v2 = opt_pts[simplex[1]]
                        v3 = opt_pts[simplex[2]]
                        edge1 = v2 - v1
                        edge2 = v3 - v1
                        n = np.cross(edge1, edge2)
                        norm_n = np.linalg.norm(n)
                        if norm_n < 1e-10:
                            continue
                        c = n / norm_n
                        if np.dot(c, v1) < 0:
                            c = -c
                        d = np.linalg.norm(c - v1)
                        radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
                        if radius > max_radius:
                            max_radius = radius
                except:
                    max_radius = np.pi

                if max_radius < best_score:
                    best_score = max_radius
                    best_points = opt_pts
        except:
            pass

    # Fallback if no valid points were found
    if best_points is None:
        best_points = fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
