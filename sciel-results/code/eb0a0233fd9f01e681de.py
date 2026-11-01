"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_init():
    """Generate a hexagonal lattice initialization."""
    # Approximate optimal spacing
    # Area per circle ~ 1/N. Hex cell area = sqrt(3)/2 * s^2.
    # s ~ sqrt(2 / (sqrt(3) * N))
    s = np.sqrt(2 / (np.sqrt(3) * N))
    
    # Generate a grid of hex points
    # Row spacing: s * sqrt(3) / 2
    # Col spacing: s
    # Offset alternating rows by s/2
    
    points = []
    # Range of x, y
    # We need enough points to fill [-0.5, 0.5]
    # Let's generate a larger grid and filter
    
    max_range = 1.5
    x_vals = np.arange(-max_range, max_range, s)
    y_spacing = s * np.sqrt(3) / 2
    y_vals = np.arange(-max_range, max_range, y_spacing)
    
    for i, y in enumerate(y_vals):
        offset = 0 if i % 2 == 0 else s / 2
        for x in x_vals:
            px = x + offset
            py = y
            # Check if inside square with margin
            # We want points that can fit a circle of radius r ~ s/2
            # So center must be in [-0.5 + r, 0.5 - r]
            # But for init, just keep points inside [-0.5, 0.5]
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                points.append((px, py))
                
    points = np.array(points)
    
    # If we have too many points, select the ones closest to the center
    if len(points) > N:
        dists = np.sum(points**2, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:N]]
    elif len(points) < N:
        # If too few, add random points or adjust spacing
        # For N=200, s ~ 0.076.
        # Grid size ~ 1/0.076 ~ 13x13 ~ 169 points.
        # We might be short.
        # Let's reduce s slightly to get more points.
        s = s * 0.9
        points = []
        max_range = 1.5
        x_vals = np.arange(-max_range, max_range, s)
        y_spacing = s * np.sqrt(3) / 2
        y_vals = np.arange(-max_range, max_range, y_spacing)
        for i, y in enumerate(y_vals):
            offset = 0 if i % 2 == 0 else s / 2
            for x in x_vals:
                px = x + offset
                py = y
                if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                    points.append((px, py))
        points = np.array(points)
        if len(points) > N:
            dists = np.sum(points**2, axis=1)
            indices = np.argsort(dists)
            points = points[indices[:N]]
        else:
            # Pad with random points if still not enough (unlikely)
            while len(points) < N:
                points = np.vstack([points, np.random.uniform(-0.4, 0.4, 2)])
            
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    
    # Clip to valid region
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _calc_score(pos):
    """Calculate the exact score (radius)."""
    if len(pos) == 0:
        return 0.0
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # dist to boundary = 0.5 - max(|x|, |y|)
    max_coord = np.max(np.abs(pos), axis=1)
    boundary_dists = 0.5 - max_coord
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _soft_min_obj_and_grad(x_flat, pos_full, indices, beta):
    """
    Objective: -soft_min(distances)
    Gradient: -grad(soft_min)
    """
    pos = pos_full.copy()
    pos[indices] = x_flat.reshape(-1, 2)
    
    N = len(pos)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    max_coord = np.max(np.abs(pos), axis=1)
    boundary_dists = 0.5 - max_coord
    
    # Terms for soft min: v_k
    # v_pair = dists / 2
    # v_bound = boundary_dists
    
    # We only care about pairs i<j to avoid double counting in sum
    # But for gradient, we need to handle indices carefully.
    
    # Vectorized soft min calculation
    # Z = sum(exp(-beta * v))
    # To avoid overflow/underflow, we can just compute it directly for N=200.
    # exp(-beta * v) where v > 0.
    
    # Create array of all pairwise distances (i<j)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    v_pairs = pair_dists / 2.0
    
    v_bounds = boundary_dists
    
    all_v = np.concatenate([v_pairs, v_bounds])
    
    # Compute weights w_k = exp(-beta * v_k) / Z
    exp_terms = np.exp(-beta * all_v)
    Z = np.sum(exp_terms)
    weights = exp_terms / Z
    
    # Objective value: -1/beta * ln(Z)
    val = -1.0 / beta * np.log(Z)
    
    # Gradient
    # d(val)/dx = -1/beta * (1/Z) * dZ/dx
    # dZ/dx = sum( w_k * Z * (-beta) * dv_k/dx )  -> wait
    # Z = sum(e_k). dZ/dx = sum( e_k * (-beta) * dv_k/dx )
    # d(val)/dx = -1/beta * (1/Z) * sum( e_k * (-beta) * dv_k/dx )
    #            = 1/Z * sum( e_k * dv_k/dx )
    #            = sum( (e_k/Z) * dv_k/dx )
    #            = sum( w_k * dv_k/dx )
    
    # So grad_val = sum( w_k * dv_k/dx )
    # We want to minimize val, so we return grad_val.
    # Wait, val = -soft_min. We want to maximize soft_min.
    # So we minimize -soft_min.
    # grad(-soft_min) = - grad(soft_min).
    # Let's re-derive.
    # S = -1/beta ln(Z).
    # dS/dx = -1/beta * (1/Z) * dZ/dx
    # dZ/dx = sum( d(exp(-beta v_k))/dx ) = sum( exp(-beta v_k) * (-beta) * dv_k/dx )
    # dS/dx = -1/beta * (1/Z) * (-beta) * sum( exp(-beta v_k) * dv_k/dx )
    #        = (1/Z) * sum( exp(-beta v_k) * dv_k/dx )
    #        = sum( w_k * dv_k/dx )
    
    # So gradient of S is sum( w_k * dv_k/dx ).
    # We want to maximize S.
    # scipy minimize minimizes f.
    # So let f = -S.
    # grad f = - grad S = - sum( w_k * dv_k/dx ).
    
    grad_pos = np.zeros_like(pos)
    
    # 1. Boundary terms
    # v_bound_i = 0.5 - max(|x_i|, |y_i|)
    # dv/dx_i = -sign(x_i) if |x_i| > |y_i|, else 0
    # dv/dy_i = -sign(y_i) if |y_i| > |x_i|, else 0
    
    abs_x = np.abs(pos[:, 0])
    abs_y = np.abs(pos[:, 1])
    
    # Weights for boundary terms are the last N weights in 'weights'
    w_bounds = weights[N*(N-1)//2:] # Number of pairs is N*(N-1)/2
    
    # Calculate gradients for boundary
    # Mask where |x| > |y|
    mask_x = abs_x > abs_y
    mask_y = abs_y > abs_x
    
    # d(v_bound)/dx = -sign(x) * mask_x
    grad_x_bound = -np.sign(pos[:, 0]) * mask_x * w_bounds
    grad_y_bound = -np.sign(pos[:, 1]) * mask_y * w_bounds
    
    grad_pos[:, 0] += grad_x_bound
    grad_pos[:, 1] += grad_y_bound
    
    # 2. Pair terms
    # v_pair_ij = 0.5 * dist_ij
    # dv/dx_i = 0.5 * (x_i - x_j) / dist_ij
    # dv/dx_j = 0.5 * (x_j - x_i) / dist_ij
    
    w_pairs = weights[:N*(N-1)//2]
    
    # We need to accumulate forces for each i.
    # For each pair (i, j), weight w.
    # Force on i: w * 0.5 * (pos[i] - pos[j]) / dist
    # Force on j: w * 0.5 * (pos[j] - pos[i]) / dist
    
    # Vectorized approach:
    # Create matrix of weights for pairs
    # This is tricky to do purely vectorized without loops for N=200?
    # N=200 is small enough for a loop over pairs if vectorized over coordinates?
    # Or just use a loop over i?
    
    # Let's use a loop over i for the pair part, it's simpler and N=200 is small.
    for i in range(N):
        # Indices j > i
        js = np.arange(i+1, N)
        if len(js) == 0: continue
        
        # Get weights for these pairs
        # The index in w_pairs for pair (i, j) where j>i
        # The pairs are stored in row-major order of the upper triangle.
        # Number of pairs for row k is N - 1 - k.
        # Offset for row i is sum(N-1-k for k=0 to i-1) = i*(N-1) - i*(i-1)/2
        
        offset = i * (N - 1) - (i * (i - 1)) // 2
        w_ij = w_pairs[offset : offset + len(js)]
        
        # Distances
        diffs = pos[i] - pos[js] # Shape (len(js), 2)
        dists_ij = np.linalg.norm(diffs, axis=1)
        
        # Avoid division by zero
        dists_ij[dists_ij == 0] = 1e-10
        
        # Gradients: 0.5 * diff / dist
        grad_dists = 0.5 * diffs / dists_ij[:, np.newaxis]
        
        # Weighted sum
        force_i = np.sum(w_ij[:, np.newaxis] * grad_dists, axis=0)
        
        # Add to grad_pos[i]
        grad_pos[i] += force_i
        
        # Subtract from grad_pos[js]
        # We need to subtract the force from each j
        # force_j = -force_i (for the pair) ? No, force on j is w * 0.5 * (pos[j]-pos[i])/dist
        # Which is - (w * 0.5 * (pos[i]-pos[j])/dist)
        # So we subtract force_i from each j?
        # Wait, force_i is the sum of contributions to i.
        # The contribution to j from i is - (contribution to i from j).
        # So we can just do:
        grad_pos[js] -= (w_ij[:, np.newaxis] * grad_dists)
        
    # Total gradient for f = -S
    grad_f = -grad_pos
    
    # Extract gradient for the free variables (indices)
    grad_flat = grad_f[indices].flatten()
    
    return val, grad_flat

def construct():
    t_start = time.time()
    
    # 1. Initialization
    pos = _hex_init()
    best_score = _calc_score(pos)
    best_pos = pos.copy()
    
    # 2. Optimization
    # Use L-BFGS-B on all variables
    indices = np.arange(N)
    
    # Adaptive beta
    beta = 30.0
    
    while time.time() - t_start < 120:
        x0 = best_pos[indices].flatten()
        
        # Run optimization
        # maxiter limited to prevent long runs per step
        res = minimize(_soft_min_obj_and_grad, x0, 
                       args=(best_pos, indices, beta), 
                       method='L-BFGS-B', jac=True, 
                       options={'maxiter': 50, 'gtol': 1e-5, 'disp': False})
        
        if res.success or res.fun < 0:
            new_pos = best_pos.copy()
            new_pos[indices] = res.x.reshape(-1, 2)
            
            # Validate
            if np.all(np.isfinite(new_pos)):
                new_score = _calc_score(new_pos)
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos
                    # Increase beta to sharpen
                    beta *= 1.2
                    if beta > 100:
                        beta = 100
                else:
                    # If no improvement, maybe beta is too high or stuck
                    beta *= 0.9
                    if beta < 10:
                        beta = 10
        
        # Check time
        if time.time() - t_start > 115:
            break
        
        # Optional: Jitter to escape local minima
        if time.time() - t_start < 100:
             # Small random perturbation
             jitter = np.random.normal(0, 0.001, best_pos.shape)
             perturbed = best_pos + jitter
             perturbed = np.clip(perturbed, -0.49, 0.49)
             pert_score = _calc_score(perturbed)
             if pert_score > best_score * 0.99: # Allow small drop to escape?
                 # Don't update best_pos if it drops, but maybe use it as start?
                 # For now, keep best_pos.
                 pass

    return best_pos

# EVOLVE-BLOCK-END
