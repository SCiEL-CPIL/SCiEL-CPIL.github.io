import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # 10s margin for safety
    
    best_score = 0.0
    best_points = np.random.randn(n, d)
    
    # Helper to calculate the actual metric score
    def get_score(pts):
        dists_sq = pdist(pts, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
        
        return d_min_sq / d_max_sq

    # Objective: Minimize -log(d_min_sq) + log(d_max_sq) + barrier
    # We want to maximize (d_min/d_max)^2, which is equivalent to maximizing log(d_min_sq) - log(d_max_sq)
    # So we minimize -log(d_min_sq) + log(d_max_sq)
    # To ensure d_max is the true max and d_min is the true min, we use a barrier that penalizes
    # any distance being too close to 0 or too large.
    # Actually, a simpler formulation:
    # Let f(x) = -log(min(d_sq)) + log(max(d_sq))
    # This is non-smooth. 
    # Alternative: Use a smooth approximation or just use the exact min/max with subgradients?
    # L-BFGS-B requires gradients. 
    # We can use the fact that min is achieved at some pair (p, q) and max at (r, s).
    # The gradient of -log(d_pq_sq) is -2(x_p - x_q) / d_pq_sq.
    # The gradient of log(d_rs_sq) is 2(x_r - x_s) / d_rs_sq.
    # This is valid if we identify the correct pairs.
    # To make it robust, we can use a "soft" min/max or just trust the local min/max pairs 
    # if the landscape is well-behaved.
    # However, switching pairs causes discontinuities in the gradient.
    # A better approach for smooth optimization is to use the log-sum-exp trick for min and max.
    # min(d_sq) ≈ -1/k * log(sum(exp(-k*d_sq)))
    # max(d_sq) ≈ 1/k * log(sum(exp(k*d_sq)))
    # Objective: - [ -1/k log(sum exp(-k d_sq)) ] + [ 1/k log(sum exp(k d_sq)) ]
    #            = 1/k log(sum exp(-k d_sq)) + 1/k log(sum exp(k d_sq))
    #            = 1/k [ log(sum exp(-k d_sq)) + log(sum exp(k d_sq)) ]
    
    def objective_and_grad(x, k=50.0, mu=1.0):
        pts = x.reshape(n, d)
        
        # Compute pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        
        # We only need upper triangle (i < j)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d_sq = dists_sq[mask]
        
        # Term 1: log-sum-exp for min
        # L_min = log(sum(exp(-k * d_sq)))
        # To avoid overflow: shift by min
        min_d_sq = np.min(d_sq)
        exp_neg = np.exp(-k * (d_sq - min_d_sq))
        L_min = -k * min_d_sq + np.log(np.sum(exp_neg))
        
        # Term 2: log-sum-exp for max
        # L_max = log(sum(exp(k * d_sq)))
        # To avoid overflow: shift by max
        max_d_sq = np.max(d_sq)
        exp_pos = np.exp(k * (d_sq - max_d_sq))
        L_max = k * max_d_sq + np.log(np.sum(exp_pos))
        
        obj_val = (1.0/k) * (L_min + L_max)
        
        # Gradients
        # d/dx (1/k L_min) = (1/k) * d/dx log(sum exp(-k(d_ij - min)))
        # = (1/k) * (1/S_neg) * sum ( d/dx exp(-k(d_ij - min)) )
        # = (1/k) * (1/S_neg) * sum ( exp(-k(d_ij - min)) * (-k) * d(d_ij)/dx )
        # = -(1/S_neg) * sum ( exp(-k(d_ij - min)) * d(d_ij)/dx )
        
        # d/dx (1/k L_max) = (1/k) * d/dx log(sum exp(k(d_ij - max)))
        # = (1/k) * (1/S_pos) * sum ( exp(k(d_ij - max)) * k * d(d_ij)/dx )
        # = (1/S_pos) * sum ( exp(k(d_ij - max)) * d(d_ij)/dx )
        
        S_neg = np.sum(exp_neg)
        S_pos = np.sum(exp_pos)
        
        # We need to compute the sum of weights * gradient of distance for each point.
        # Let's construct the weight matrices.
        # W_neg[i, j] = exp(-k(d_ij - min)) if i!=j else 0
        # W_pos[i, j] = exp(k(d_ij - max)) if i!=j else 0
        
        # Note: The sums above were over unique pairs (mask).
        # The gradient calculation needs to account for the fact that each pair (i,j) contributes
        # to the gradient of both x_i and x_j.
        # d(d_ij)/dx_i = 2(x_i - x_j)
        # d(d_ij)/dx_j = 2(x_j - x_i) = -2(x_i - x_j)
        
        # Let's create full matrices for W_neg and W_pos.
        W_neg_full = np.exp(-k * (dists_sq - min_d_sq))
        np.fill_diagonal(W_neg_full, 0.0)
        
        W_pos_full = np.exp(k * (dists_sq - max_d_sq))
        np.fill_diagonal(W_pos_full, 0.0)
        
        # The sum over unique pairs in L_min corresponds to half the sum over all i!=j in the full matrix,
        # because exp(-k(d_ij - min)) = exp(-k(d_ji - min)).
        # So S_neg_unique = 0.5 * sum(W_neg_full).
        # Let's verify.
        # L_min = log(sum_{i<j} exp(-k(d_ij - min))).
        # Sum_{i<j} = 0.5 Sum_{i!=j}.
        # So S_neg = 0.5 * np.sum(W_neg_full).
        
        S_neg_full = np.sum(W_neg_full)
        S_pos_full = np.sum(W_pos_full)
        
        # Gradient of Term 1:
        # G1 = -(1/S_neg_unique) * sum_{i<j} w_ij * d(d_ij)/dx
        # sum_{i<j} w_ij * d(d_ij)/dx_i = 0.5 * sum_{j!=i} w_ij * 2(x_i - x_j) ?
        # Let T1_i = sum_{j!=i} W_neg_full[i,j] * 2(x_i - x_j).
        # Then sum_{i<j} w_ij * d(d_ij)/dx_i = 0.5 * T1_i.
        # So G1_i = -(1/S_neg_unique) * 0.5 * T1_i = -(1/(0.5 S_neg_full)) * 0.5 * T1_i = -T1_i / S_neg_full.
        
        W_neg_row_sums = np.sum(W_neg_full, axis=1)
        W_neg_pts = W_neg_full @ pts
        T1 = 2.0 * (W_neg_row_sums[:, np.newaxis] * pts - W_neg_pts)
        
        grad_term1 = -T1 / S_neg_full
        
        # Gradient of Term 2:
        # G2 = (1/S_pos_unique) * sum_{i<j} w_ij * d(d_ij)/dx
        # Similarly, sum_{i<j} ... = 0.5 * T2_i.
        # S_pos_unique = 0.5 S_pos_full.
        # G2_i = (1/(0.5 S_pos_full)) * 0.5 * T2_i = T2_i / S_pos_full.
        
        W_pos_row_sums = np.sum(W_pos_full, axis=1)
        W_pos_pts = W_pos_full @ pts
        T2 = 2.0 * (W_pos_row_sums[:, np.newaxis] * pts - W_pos_pts)
        
        grad_term2 = T2 / S_pos_full
        
        grad_total = (grad_term1 + grad_term2).flatten()
        
        return obj_val, grad_total

    def run_optimization(x0, k):
        x0 = x0.flatten()
        try:
            res = minimize(objective_and_grad, x0, args=(k,), method='L-BFGS-B', jac=True,
                           options={'maxiter': 100, 'gtol': 1e-8})
            return res.x.reshape(n, d)
        except Exception:
            return x0.reshape(n, d)

    # Initializations
    # 1. Truncated Icosahedron (12 vertices of icosahedron + 2 random)
    # 2. Geodesic Sphere (Subdivide icosahedron faces, pick 14 points)
    # 3. Random on sphere
    
    # Generate Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    # Generate 14 points on sphere using Fibonacci sphere for even distribution
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    # Pre-generate some good initializations
    inits = []
    
    # 1. Fibonacci
    inits.append(fibonacci_sphere(n))
    
    # 2. Icosahedron + 2 random
    for _ in range(5):
        v1 = np.random.randn(3)
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(3)
        v2 /= np.linalg.norm(v2)
        pts = np.vstack([ico_verts, [v1, v2]])
        # Normalize to unit sphere
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 3. Random on sphere
    for _ in range(5):
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)

    num_restarts = 0
    while time.time() < deadline:
        # Choose initialization
        init_idx = num_restarts % len(inits)
        pts = inits[init_idx].copy()
        
        # Perturb slightly
        pts += np.random.randn(n, d) * 0.05
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Anneal k
        # Start with small k (smoother), increase to sharp min/max
        k_values = [5.0, 10.0, 20.0, 50.0, 100.0]
        k_idx = num_restarts % len(k_values)
        k = k_values[k_idx]
        
        try:
            local_pts = run_optimization(pts, k)
            local_score = get_score(local_pts)
            
            if local_score > best_score:
                best_score = local_score
                best_points = local_pts
                
        except Exception:
            pass
        
        num_restarts += 1
        
        # If we are very good, break
        if best_score > 0.99:
            break

    return best_points
