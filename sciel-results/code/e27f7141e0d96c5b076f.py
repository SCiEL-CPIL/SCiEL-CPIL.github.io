import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _soft_covering_radius(points, probes, beta):
    # points: (N, 3)
    # probes: (M, 3)
    # Calculate distances from all probes to all points
    # dist[i, j] = arccos(points[i] . probes[j])
    
    # To avoid arccos numerical issues, we can use 1 - dot product or similar,
    # but arccos is fine if we clamp.
    
    dots = np.dot(points, probes.T) # (N, M)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # (N, M)
    
    # Soft-min over points for each probe
    # w = softmax(-beta * dists)
    # D = sum(w * dists)
    
    # For stability, subtract max
    scaled = -beta * dists
    max_scaled = np.max(scaled, axis=0, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    w = exp_scaled / np.sum(exp_scaled, axis=0, keepdims=True)
    
    D = np.sum(w * dists, axis=0) # (M,)
    
    # Log-Sum-Exp of D
    # J = 1/beta * log(sum(exp(beta * D)))
    scaled_D = beta * D
    max_D = np.max(scaled_D)
    J = (max_D + np.log(np.sum(np.exp(scaled_D - max_D)))) / beta
    
    return J

def _soft_covering_radius_grad(points, probes, beta):
    # Returns gradient w.r.t points (N, 3)
    # We need dJ/dpoints
    
    # Recalculate terms
    dots = np.dot(points, probes.T) # (N, M)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # (N, M)
    
    # We need d dists / d points
    # d/dp arccos(p.y) = -(y - (p.y)p) / sqrt(1 - (p.y)^2)
    # Let c = p.y
    # grad_dist = -(y - c*p) / sqrt(1-c^2)
    
    # To avoid division by zero when c is close to +/- 1, we add epsilon
    c = dots
    sin_dists = np.sqrt(np.clip(1.0 - c*c, 1e-12, 1.0))
    
    # grad_dist[i, j, k] = -(probes[j, k] - c[i, j]*points[i, k]) / sin_dists[i, j]
    # This is (N, M, 3)
    
    # This might be memory heavy if M is large.
    # N=130, M=5000. 130*5000*3 = 1.95M floats. ~15MB. OK.
    
    # Calculate weights w
    scaled = -beta * dists
    max_scaled = np.max(scaled, axis=0, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    w = exp_scaled / np.sum(exp_scaled, axis=0, keepdims=True) # (N, M)
    
    # D = sum_i w_i D_i? No, D_j = sum_i w_ij dist_ij
    D = np.sum(w * dists, axis=0) # (M,)
    
    # J = 1/beta * log(sum_j exp(beta D_j))
    scaled_D = beta * D
    max_D = np.max(scaled_D)
    exp_D = np.exp(scaled_D - max_D)
    sum_exp_D = np.sum(exp_D)
    
    # dJ/dD_j = (1/beta) * (beta * exp_D_j / sum_exp_D) = exp_D_j / sum_exp_D
    dJ_dD = exp_D / sum_exp_D # (M,)
    
    # dD_j / d dist_ij = w_ij
    # d dist_ij / d p_i = -(probes_j - c_ij p_i) / sin_ij
    
    # dJ/dp_i = sum_j dJ_dD_j * w_ij * d dist_ij / d p_i
    
    # Let's compute this efficiently
    # grad_dist is (N, M, 3)
    # We want sum over j of (dJ_dD_j * w_ij) * grad_dist_ij
    
    # coeff[i, j] = dJ_dD[j] * w[i, j]
    coeff = w * dJ_dD[None, :] # (N, M)
    
    # grad_dist[i, j, k] = -(probes[j, k] - c[i, j]*points[i, k]) / sin_dists[i, j]
    # = (c[i, j]*points[i, k] - probes[j, k]) / sin_dists[i, j]
    
    # Let A[i, j] = coeff[i, j] / sin_dists[i, j]
    A = coeff / sin_dists # (N, M)
    
    # grad_p[i, k] = sum_j A[i, j] * (c[i, j]*points[i, k] - probes[j, k])
    # = points[i, k] * sum_j A[i, j]*c[i, j] - sum_j A[i, j]*probes[j, k]
    
    # term1[i] = sum_j A[i, j]*c[i, j]
    term1 = np.sum(A * c, axis=1) # (N,)
    
    # term2[i, k] = sum_j A[i, j]*probes[j, k]
    # A is (N, M), probes is (M, 3)
    term2 = np.dot(A, probes) # (N, 3)
    
    grad_points = term1[:, None] * points - term2
    
    # Project gradient to tangent space?
    # The optimization is on R^3, but we normalize in the objective.
    # The gradient of the normalized objective w.r.t unnormalized points is:
    # If f(P) is the objective of normalized points P = x/||x||,
    # then grad_x f(x/||x||) = (I - xx^T/||x||^2) grad_P f(P) / ||x||
    # Since we start with normalized points and keep them near the sphere, ||x|| ~ 1.
    # The projection (I - xx^T) ensures we move tangentially.
    # If we use L-BFGS-B, it might move points off the sphere.
    # It is safer to re-normalize points at the end of the step or use a constrained optimizer.
    # However, simply projecting the gradient is often sufficient if the optimizer respects the manifold implicitly by the normalization in the objective.
    # Actually, if we just return grad_points, the optimizer will update x.
    # If x moves off the sphere, the normalization in the next objective call will pull it back.
    # This can be unstable.
    
    # Better: Return the projected gradient so the optimizer stays on the sphere?
    # No, the optimizer works in Euclidean space.
    # If we return the projected gradient, the step will be tangential, keeping the norm constant (to first order).
    
    # Project grad_points onto tangent space
    # proj_grad = grad_points - np.sum(grad_points * points, axis=1, keepdims=True) * points
    proj_grad = grad_points - np.sum(grad_points * points, axis=1, keepdims=True) * points
    
    return proj_grad

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Probes for soft objective
    # Use a spherical grid or random points. Random is easier.
    # 5000 points should be enough for a smooth landscape.
    np.random.seed(42)
    probes = np.random.randn(5000, 3)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_score = float('inf')
    
    # Phases: increasing beta
    betas = [10.0, 50.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Objective and Gradient for L-BFGS-B
        def obj(x):
            p = _normalize(x.reshape(N, 3))
            return _soft_covering_radius(p, probes, beta)
            
        def grad(x):
            p = _normalize(x.reshape(N, 3))
            g = _soft_covering_radius_grad(p, probes, beta)
            return g.flatten()
            
        x0 = points.flatten()
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Check if it's better
                # We should evaluate the *true* covering radius to be sure,
                # but the soft objective is a good proxy.
                # Let's just trust the soft objective for now to save time,
                # but we will do a final check.
                
                new_points = _normalize(res.x.reshape(N, 3))
                
                # Quick check: if points are valid
                if np.all(np.isfinite(new_points)):
                    # Optional: Calculate true radius to track improvement
                    # This is expensive, so maybe skip in inner loop
                    best_points = new_points
                    # best_score = res.fun # This is soft score, not comparable across betas easily
        except Exception as e:
            pass # Keep old points if optimization fails

        points = best_points
        
        # Check time
        if time.time() > deadline:
            break
            
    # Final Polish: Use a very high beta and tight tolerance
    if time.time() < deadline - 5.0:
        beta_final = 1000.0
        def obj_final(x):
            p = _normalize(x.reshape(N, 3))
            return _soft_covering_radius(p, probes, beta_final)
            
        def grad_final(x):
            p = _normalize(x.reshape(N, 3))
            g = _soft_covering_radius_grad(p, probes, beta_final)
            return g.flatten()
            
        x0 = points.flatten()
        try:
            res = minimize(obj_final, x0, jac=grad_final, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.fun < float('inf'):
                new_points = _normalize(res.x.reshape(N, 3))
                if np.all(np.isfinite(new_points)):
                    best_points = new_points
        except Exception:
            pass

    return best_points
