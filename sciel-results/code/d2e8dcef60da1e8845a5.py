import numpy as np
import scipy.optimize
import scipy.special
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # We want to fit N points.
    # Approximate side length of hexagonal grid to fit N points in unit circle.
    # Area of circle = pi. Area per point ~ pi/N.
    # Spacing s ~ sqrt(pi/N).
    # Let's use the seed's logic but slightly adjusted.
    
    def get_lattice_points(n):
        # Generate a hexagonal lattice of points
        # spacing = 2.0 / (np.sqrt(n) + 1.0)
        # This might be too sparse or dense.
        # Let's try to generate a grid that covers the circle.
        s = 0.05  # Initial guess for spacing, will be refined
        k = int(1.0 / s) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    rows.append((x, y))
        points = np.array(rows)
        # Select N points closest to center? Or just first N?
        # To break symmetry and ensure spread, we should probably pick a subset.
        # The seed picks the first N (which are ordered by radius in the seed, but here we just generated them).
        # Let's sort by radius and take the first N to ensure they are inside and central.
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:n]]
        return points

    points = get_lattice_points(N)
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    points = points + np.random.normal(0, 0.005, points.shape)
    
    # Ensure points are strictly inside (add margin)
    # The optimizer will handle the boundary, but starting inside is good.
    
    def objective_and_grad(flat_c):
        c = flat_c.reshape(N, 2)
        
        # Compute pairwise distances
        # d_ij = ||c_i - c_j||
        # We need d_ij / 2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        # Values for soft-min:
        # 1. d_ij / 2 for i < j
        # 2. 1 - |c_i| for all i
        
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_vals = dist[i_idx, j_idx] / 2.0
        
        # Boundary distances
        c_norm = np.linalg.norm(c, axis=1)
        b_vals = 1.0 - c_norm
        
        # Combine into a single vector of values V
        # Note: We are maximizing min(V).
        # Soft min: S(V) = -1/tau * log(sum(exp(-tau * V)))
        # We want to maximize S(V), so minimize -S(V).
        
        V = np.concatenate([d_vals, b_vals])
        
        # Handle numerical issues: if any V is very small, exp(-tau*V) is large.
        # If V is negative (overlap), exp is huge.
        # Clip V to avoid overflow? No, we want the gradient to be strong if V is small/negative.
        
        tau = 20.0  # Sharpness of the soft-min
        
        # Calculate log-sum-exp for stability
        # log_sum_exp(-tau * V) = max(-tau*V) + log(sum(exp(-tau*V - max(-tau*V))))
        tv = -tau * V
        m = np.max(tv)
        if m > 700: # Prevent overflow in exp
             # This happens if V is very negative (large overlap).
             # In this case, the objective is dominated by the most negative term.
             # We can just return a large penalty and a gradient pointing to fix the worst overlap.
             # For simplicity, let's just clip m to 700 for the logsumexp calculation, 
             # but the gradient weights will be dominated by the max term.
             pass
        
        exp_tv = np.exp(tv - m)
        sum_exp = np.sum(exp_tv)
        log_sum_exp = m + np.log(sum_exp)
        
        soft_min_val = -log_sum_exp / tau
        obj_val = -soft_min_val
        
        # Gradient
        # p_k = exp(tv_k) / sum_exp
        p = exp_tv / sum_exp
        
        # Gradient of soft_min w.r.t V is -p (since S = -1/tau log(sum exp(-tau V)))
        # dS/dV_k = -1/tau * (1/sum) * exp(-tau V_k) * (-tau) = p_k
        # Wait. S(V) = -1/tau * LSE(-tau V).
        # dS/dV_k = -1/tau * (1/LSE_sum) * exp(-tau V_k) * (-tau) = p_k.
        # So dS/dV = p.
        # We are minimizing -S, so grad_obj = -grad_S = -p.
        
        # We need grad_obj w.r.t c.
        # grad_obj = - sum_k p_k * grad_V_k
        
        grad_c = np.zeros((N, 2))
        
        # Contribution from d_vals (pairs i, j)
        # V_k = d_ij / 2
        # grad_V_k w.r.t c_i = (c_i - c_j) / (2 * d_ij)
        # grad_V_k w.r.t c_j = (c_j - c_i) / (2 * d_ij)
        
        # p_d corresponds to the first part of V (d_vals)
        p_d = p[:len(d_vals)]
        
        # We need to accumulate gradients
        # For each pair (i, j) with weight w = p_d[idx]:
        # grad_c[i] -= w * (c_i - c_j) / (2 * d_ij)  (because we minimize -S, and dS/dV = p)
        # Wait, let's re-verify signs.
        # Minimize f(c) = -S(V(c)).
        # df/dc = - dS/dV * dV/dc = - p * dV/dc.
        
        # dV_k/dc_i for V_k = d_ij/2 is (c_i - c_j) / d_ij * 1/2?
        # d/dx ||x|| = x/||x||.
        # d/d c_i ||c_i - c_j|| = (c_i - c_j) / ||c_i - c_j||.
        # So d/d c_i (d_ij/2) = 0.5 * (c_i - c_j) / d_ij.
        
        # Vectorized update for pairs:
        # This is tricky to do fully vectorized with accumulation into grad_c without loops or advanced indexing sums.
        # N=300, pairs=45k. A loop over pairs in Python is too slow (45k iterations).
        # We need to use NumPy operations.
        
        # Let's create a matrix of gradients for the distance part.
        # G[i, j] = (c_i - c_j) / (2 * d_ij)
        # We have weights p_d for each pair (i, j).
        # grad_c[i] = - sum_{j != i} p_d[(i,j)] * G[i, j]
        
        # Construct G matrix (N x N x 2)
        # diff is already (N, N, 2)
        # dist is (N, N)
        # Safe division
        dist_safe = np.where(dist > 1e-8, dist, 1.0)
        G = diff / (2.0 * dist_safe[:, :, np.newaxis])
        # Mask diagonal
        G[np.eye(N, dtype=bool)] = 0.0
        
        # p_d is a flat array of length M. We need to map it to (N, N) matrix P_d.
        P_d = np.zeros((N, N))
        P_d[i_idx, j_idx] = p_d
        # Symmetric part? No, p_d is for i<j.
        # The gradient contribution from pair (i,j) to c_i is -p_d * G[i,j]
        # The gradient contribution from pair (i,j) to c_j is -p_d * G[j,i] = -p_d * (-G[i,j]) = p_d * G[i,j]
        
        # So grad_c[i] gets - sum_j P_d[i,j] * G[i,j]
        # And grad_c[j] gets + sum_i P_d[i,j] * G[i,j] (since P_d is upper triangular)
        
        # Let's compute term = P_d * G (element-wise)
        term = P_d * G
        
        # grad_c[i] = - sum_j term[i, j]
        grad_c -= np.sum(term, axis=1)
        
        # For the lower triangle part (j < i), the weight is in P_d[j, i] which is 0 in P_d.
        # But the pair (j, i) is the same as (i, j).
        # The weight p_d for pair (i,j) applies to both c_i and c_j.
        # In our construction:
        # term[i, j] = p_d * G[i, j]
        # term[j, i] = 0 * G[j, i] = 0
        # Sum over j for c_i: - sum_j term[i, j]. This includes the term from pair (i,j) if j>i.
        # What about pair (k,i) where k<i?
        # The weight is p_d[k, i] (stored in P_d[k, i]).
        # G[i, k] = -G[k, i].
        # The contribution to c_i from pair (k,i) should be -p_d[k,i] * G[i,k].
        # In our sum `sum(term, axis=1)`, we sum over columns.
        # term[i, k] is 0 because P_d is upper triangular (k<i).
        # So we are missing the contributions from pairs where the index is smaller.
        
        # We need to add the symmetric contributions.
        # grad_c[i] = - sum_{j>i} p_d[i,j] G[i,j] - sum_{j<i} p_d[j,i] G[i,j]
        # Note G[i,j] = -G[j,i].
        # So - sum_{j<i} p_d[j,i] G[i,j] = - sum_{j<i} p_d[j,i] (-G[j,i]) = sum_{j<i} p_d[j,i] G[j,i]
        
        # Let's just use the full matrix of weights.
        # P_full[i, j] = p_d[i, j] if i<j, p_d[j, i] if j<i, 0 if i=j.
        # Actually, the weight for the pair {i,j} is the same.
        # Let's create P_sym.
        P_sym = P_d + P_d.T
        # Now P_sym[i, j] is the weight for pair (i,j).
        term_sym = P_sym * G
        # grad_c[i] = - sum_j term_sym[i, j]
        # Wait, does this double count?
        # For pair (i,j), weight w.
        # term_sym[i, j] = w * G[i, j]
        # term_sym[j, i] = w * G[j, i] = -w * G[i, j]
        # Sum for c_i: - (w * G[i, j] + ... )
        # Sum for c_j: - (w * G[j, i] + ... ) = - (-w * G[i, j] + ... ) = w * G[i, j]
        # This looks correct. The force on i is towards j (if G[i,j] is vector from j to i? No, c_i-c_j is i-j. G is i-j. Force is -w(i-j). Pushes i towards j? No, if they are too close, we want to push them apart.
        # Let's check sign.
        # We want to MAXIMIZE distance.
        # V = d/2.
        # dS/dV = p > 0.
        # df/dc = -p * dV/dc.
        # dV/dc_i = 0.5 * (c_i - c_j) / d.
        # grad_c_i = -p * 0.5 * (c_i - c_j) / d.
        # Vector (c_i - c_j) points from j to i.
        # So grad_c_i points from i to j? No, (c_i - c_j) is vector from j to i.
        # So - (c_i - c_j) points from i to j.
        # So the gradient points from i to j.
        # Minimizing f means moving in direction -grad_f.
        # -grad_f points from j to i.
        # So we move i away from j. Correct.
        
        grad_c -= np.sum(term_sym, axis=1)
        
        # Contribution from boundary b_vals = 1 - |c_i|
        p_b = p[len(d_vals):]
        # V_k = 1 - |c_i|
        # dV_k/dc_i = - c_i / |c_i|
        # grad_c_i += -p_b[i] * (-c_i / |c_i|) = p_b[i] * c_i / |c_i|
        
        c_norm_safe = np.where(c_norm > 1e-8, c_norm, 1.0)
        grad_boundary = (c / c_norm_safe[:, np.newaxis]) * p_b[:, np.newaxis]
        grad_c += grad_boundary
        
        return obj_val, grad_c.flatten()

    # Optimization
    best_points = points
    best_score = -np.inf
    
    # We can try a few iterations or just one run with L-BFGS-B
    # L-BFGS-B is good for large scale.
    
    def time_check():
        return time.time() > deadline

    # Run minimize
    try:
        res = scipy.optimize.minimize(
            objective_and_grad, 
            points.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'maxfun': 1000, 'gtol': 1e-5}
        )
        if res.success or res.status in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            # Check if the new points are better
            # Calculate the actual min radius
            c_new = res.x.reshape(N, 2)
            # Verify validity
            if np.all(np.isfinite(c_new)):
                # Compute score
                diff = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist) / 2.0
                c_norm = np.linalg.norm(c_new, axis=1)
                min_bnd = np.min(1.0 - c_norm)
                r = min(min_dist, min_bnd)
                
                # Compare with current best
                # We should also check the current best score
                diff_old = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
                dist_old = np.linalg.norm(diff_old, axis=2)
                np.fill_diagonal(dist_old, np.inf)
                min_dist_old = np.min(dist_old) / 2.0
                c_norm_old = np.linalg.norm(best_points, axis=1)
                min_bnd_old = np.min(1.0 - c_norm_old)
                r_old = min(min_dist_old, min_bnd_old)
                
                if r > r_old:
                    best_points = c_new
                    best_score = r
    except Exception as e:
        pass

    # If time permits, do a second pass with higher tau (sharper)
    if time.time() < deadline - 20:
        # Use the best points found so far as initial guess
        points = best_points
        # Increase tau? The objective function uses a fixed tau. 
        # We could re-run with a different tau, but we'd need to redefine the function or use a closure.
        # For now, just one pass.
        
    return best_points

# EVOLVE-BLOCK-END
