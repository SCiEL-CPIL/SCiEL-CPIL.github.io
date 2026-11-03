import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50

def compute_score(points):
    """
    Compute the packing radius for the given points.
    Returns 0 if invalid.
    """
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check if points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin > 1.0):
        return 0.0
    
    # Compute pairwise distances
    # Using broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_dist_to_border = np.min(1.0 - dists_to_origin)
    
    r = min(min_pair_dist / 2.0, min_dist_to_border)
    
    if r < 0:
        return 0.0
        
    return r

def smooth_objective_and_grad(x, beta):
    """
    x: (N, 2) array of points.
    Returns objective value (negative soft-min radius) and gradient.
    We want to maximize the radius, so we minimize -radius.
    Soft-min of u_ij (pair dist/2) and v_i (dist to border).
    """
    points = x.reshape(N, 2)
    
    # Distances to origin
    r_orig = np.linalg.norm(points, axis=1)
    v = 1.0 - r_orig
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    u = dists / 2.0
    
    # Flatten u and v for sum
    u_flat = u.flatten()
    v_flat = v.flatten()
    
    # Filter out very small values to avoid log(0) or overflow issues?
    # Soft min: -1/beta * log(sum(exp(-beta * val)))
    # To avoid overflow, we can shift by min_val
    all_vals = np.concatenate([u_flat, v_flat])
    min_val = np.min(all_vals)
    
    # exp(-beta * (val - min_val))
    # This is stable.
    exp_u = np.exp(-beta * (u_flat - min_val))
    exp_v = np.exp(-beta * (v_flat - min_val))
    
    sum_exp = np.sum(exp_u) + np.sum(exp_v)
    
    # Objective: -soft_min = -(-1/beta * log(sum_exp)) - min_val * (1 - 1/sum_exp?) 
    # Wait, soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # Let x' = x - min_val. sum(exp(-beta*x)) = exp(-beta*min_val) * sum(exp(-beta*x'))
    # soft_min(x) = -1/beta * ( -beta*min_val + log(sum(exp(-beta*x'))) ) = min_val - 1/beta * log(sum(exp(-beta*x')))
    # We want to maximize soft_min, so minimize -soft_min.
    # -soft_min(x) = -min_val + 1/beta * log(sum(exp(-beta*x')))
    
    obj_val = -min_val + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # d/dx (-soft_min) = - d/dx (soft_min)
    # soft_min = min_val - 1/beta * log(S)
    # d(soft_min)/dx = d(min_val)/dx - 1/beta * (1/S) * d(S)/dx
    # d(S)/dx = sum( -beta * exp(-beta*(val-min_val)) * d(val)/dx )  (chain rule, note d(min_val)/dx term in exp is tricky if min_val changes)
    # Actually, simpler:
    # Let f(x) = -1/beta * log( sum exp(-beta x_i) )
    # grad f = -1/beta * 1/S * sum ( -beta exp(-beta x_i) grad x_i ) = 1/S * sum ( exp(-beta x_i) grad x_i )
    # Here x_i are the values u_ij and v_k.
    # grad u_ij w.r.t points:
    # u_ij = 0.5 * ||p_i - p_j||
    # grad_{p_i} u_ij = 0.5 * (p_i - p_j) / ||p_i - p_j||
    # grad_{p_j} u_ij = 0.5 * (p_j - p_i) / ||p_i - p_j||
    # grad v_k w.r.t p_k:
    # v_k = 1 - ||p_k||
    # grad_{p_k} v_k = - p_k / ||p_k||
    
    # We need to accumulate gradients for each point.
    grads = np.zeros_like(points)
    
    # Weights for each term in the sum
    w_u = exp_u / sum_exp
    w_v = exp_v / sum_exp
    
    # Process pairwise terms
    # w_u is (N, N). 
    # For each pair (i, j), i < j, the weight is w_u[i,j].
    # Note: u is symmetric, so w_u is symmetric.
    # We sum contributions.
    for i in range(N):
        for j in range(i+1, N):
            w = w_u[i, j]
            if w == 0: continue
            diff_ij = points[i] - points[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-8:
                # Handle singularity
                diff_ij = np.array([1e-8, 0])
                dist_ij = 1e-8
            grad_dir = diff_ij / dist_ij
            grads[i] += w * 0.5 * grad_dir
            grads[j] += w * 0.5 * (-grad_dir)
            
    # Process boundary terms
    for i in range(N):
        w = w_v[i]
        if w == 0: continue
        r_i = r_orig[i]
        if r_i < 1e-8:
            grad_dir = np.array([0, 0]) # Gradient of norm at 0 is undefined, but v=1. 
            # Actually if p=0, v=1. grad v = 0? No, grad (1-r) = -p/r. At 0 it's 0 vector limit?
            # Let's just set to 0 to avoid NaN.
        else:
            grad_dir = -points[i] / r_i
        grads[i] += w * grad_dir
        
    # The gradient of the objective to MINIMIZE is - grad(soft_min)
    # grad(soft_min) = 1/S * sum ( exp(-beta x_i) grad x_i )
    # So grad(obj) = - (1/S * sum ... )
    # Wait, my derivation above:
    # f = -soft_min
    # grad f = - grad soft_min
    # grad soft_min = 1/S * sum ( w_i_unnorm * grad x_i ) where w_i_unnorm = exp(-beta x_i)
    # So grad f = - (1/S * sum ( exp(-beta x_i) grad x_i ))
    # My calculation of grads above computed: sum ( (exp/S) * grad x_i ) = 1/S * sum ( exp * grad x_i )
    # So grad f = - grads
    
    return obj_val, -grads

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    inits = []
    
    # 1. Hexagonal lattice
    spacing = 0.25 # Start with a reasonable guess
    # Generate hex points
    hex_points = []
    k = 10
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            x = (i + 0.5*(j%2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x*x+y*y) < 1.0:
                hex_points.append([x, y])
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        # Pick N points, maybe centered ones?
        # Sort by distance to origin
        dists = np.linalg.norm(hex_points, axis=1)
        idx = np.argsort(dists)
        inits.append(hex_points[idx[:N]])
        
    # 2. Random inits
    for _ in range(20):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
        
    # 3. Perturbed hex
    if len(inits) > 0:
        base = inits[0]
        noise = np.random.normal(0, 0.05, base.shape)
        inits.append(base + noise)

    beta = 30.0
    
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B
            res = minimize(
                smooth_objective_and_grad,
                init,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            res_pts = res.x.reshape(N, 2)
            score = compute_score(res_pts)
            
            if score > best_score:
                best_score = score
                best_points = res_pts.copy()
                
        except Exception as e:
            pass # Ignore errors in individual restarts

    # If we have time, do a polish step on the best result
    if best_points is not None and time.time() < deadline - 10:
        try:
            # Increase beta for higher precision
            beta_polish = 100.0
            res = minimize(
                smooth_objective_and_grad,
                best_points,
                args=(beta_polish,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            res_pts = res.x.reshape(N, 2)
            score = compute_score(res_pts)
            if score > best_score:
                best_score = score
                best_points = res_pts.copy()
        except:
            pass
            
    if best_points is None:
        # Fallback: return hex points if any valid, else random
        if len(inits) > 0:
            return inits[0]
        else:
            return np.zeros((N, 2))
            
    return best_points

if __name__ == "__main__":
    construct()
