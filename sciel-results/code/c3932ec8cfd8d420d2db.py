"""Pack 300 equal circles in the unit disk via differentiable refinement."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    # 1) Generate a good initial hexagonal lattice configuration
    #    We want exactly N points, so we generate a dense hex lattice and pick the best N.
    #    Start with a spacing that yields roughly N points in the disk.
    #    Area of disk = pi. Area per hex point ~ (sqrt(3)/2) * s^2 where s is spacing.
    #    N * (sqrt(3)/2) * s^2 ~ pi  =>  s ~ sqrt(2*pi/(N*sqrt(3)))
    s0 = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate hex lattice points
    def gen_hex(spacing, k):
        pts = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                pts.append((x, y))
        return np.array(pts)

    # Find k such that we get enough points
    k = int(1.0 / s0) + 3
    while True:
        pts = gen_hex(s0, k)
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - s0 / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            break
        k += 1
        if k > 200:
            break
    
    # Select the N points closest to center (stable ordering)
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(r, kind='stable')
    x = pts[idx[:N]].copy()
    
    # 2) Differentiable refinement: maximize minimum distance
    #    Use a soft-min objective: f(x) = -log-sum-exp(-d_ij / tau) approximates -min(d_ij)
    #    Also enforce boundary: g(x) = -log(1 - |x_i|) as a penalty
    #    We'll use simple gradient descent with adaptive step size.
    
    def compute_score(x):
        # Compute all pairwise distances
        dx = x[:, 0:1] - x[0:1, 0]  # N x N
        dy = x[:, 1:2] - x[0:1, 1]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        # Boundary constraint
        r = np.hypot(x[:, 0], x[:, 1])
        min_boundary = 1.0 - np.max(r)
        return min(min_d / 2.0, min_boundary)
    
    current_score = compute_score(x)
    
    # Gradient-based optimization
    # Objective: maximize min over all pairs and boundaries of the "slack"
    # We use a log-sum-exp based soft minimum for differentiability
    # For pair i,j: slack = d_ij - 2*r  (we want this > 0)
    # For boundary i: slack = 1 - |x_i| - r
    # We maximize the minimum slack. Use soft-min via log-sum-exp.
    
    def soft_min_slack(x, tau=0.01):
        """Return soft-min of all slacks (pairwise and boundary)."""
        # Pairwise slacks: d_ij - 2*r_est, but we don't know r_est yet.
        # Instead, we directly maximize the minimum distance and boundary.
        # Let's use a different approach: maximize a soft version of the score.
        
        # Compute pairwise distances
        dx = x[:, 0:1] - x[0:1, 0]
        dy = x[:, 1:2] - x[0:1, 1]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r = np.hypot(x[:, 0], x[:, 1])
        b = 1.0 - r
        
        # Combine: we want min over all d_ij and b_i
        # Use log-sum-exp for soft min: -log(sum(exp(-v/tau))) * tau approximates min(v)
        # But we have two sets. Let's stack them.
        # d is NxN, take upper triangle
        iu = np.triu_indices(N, 1)
        d_flat = d[iu]
        # b is N
        v = np.concatenate([d_flat, b])
        
        # Soft min
        v_min = np.min(v)
        v_shifted = v - v_min
        soft_min_val = v_min - tau * np.log(np.sum(np.exp(-v_shifted / tau)) / len(v))
        
        # Gradient: d(soft_min)/dx = -sum(exp(-v_i/tau) * dv_i/dx) / sum(exp(-v_i/tau))
        weights = np.exp(-v_shifted / tau)
        weights /= weights.sum()
        
        grad = np.zeros_like(x)
        
        # Gradient from pairwise distances
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # For each pair (i,j), weight w_ij, contributes w_ij * (x_i - x_j)/d_ij to grad[i]
        # and -w_ij * (x_i - x_j)/d_ij to grad[j]
        
        # Build weight matrix
        w_mat = np.zeros((N, N))
        w_flat = weights[:len(d_flat)]
        w_mat[iu] = w_flat
        w_mat = w_mat + w_mat.T
        # Note: w_mat[i,j] = weight for pair (i,j), symmetric
        
        # grad[i] = sum_j w_mat[i,j] * (x_i - x_j) / d_ij
        # = sum_j w_mat[i,j] * (x_i - x_j) / d_ij
        # We can compute this as: for each i, sum over j of w_mat[i,j] * (x_i - x_j)/d_ij
        
        # Efficient computation:
        # grad[i] = x_i * sum_j w_mat[i,j]/d_ij - sum_j w_mat[i,j]*x_j/d_ij
        
        w_over_d = w_mat / d  # Note: d has inf on diagonal, but w_mat has 0 on diagonal
        np.fill_diagonal(w_over_d, 0)
        
        row_sum = w_over_d.sum(axis=1)  # sum_j w_ij / d_ij
        weighted_x = (w_over_d * x[:, 0:1]).sum(axis=1)  # sum_j w_ij/d_ij * x_j[:,0]
        weighted_y = (w_over_d * x[:, 1:2]).sum(axis=1)  # sum_j w_ij/d_ij * x_j[:,1]
        
        grad[:, 0] = x[:, 0] * row_sum - weighted_x
        grad[:, 1] = x[:, 1] * row_sum - weighted_y
        
        # Boundary gradient
        # b_i = 1 - |x_i|, db_i/dx_i = -x_i / |x_i|
        b_weights = weights[len(d_flat):]
        r_safe = np.where(r > 1e-12, r, 1e-12)
        grad[:, 0] += b_weights * (-x[:, 0] / r_safe)
        grad[:, 1] += b_weights * (-x[:, 1] / r_safe)
        
        return soft_min_val, grad
    
    # Optimize
    tau = 0.02
    lr = 0.001
    best_x = x.copy()
    best_score = current_score
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        val, grad = soft_min_slack(x, tau)
        
        # Update
        x_new = x + lr * grad
        
        # Check if valid (all inside disk)
        r_new = np.hypot(x_new[:, 0], x_new[:, 1])
        if np.any(r_new > 1.0):
            # Project back
            mask = r_new > 1.0
            x_new[mask] = x_new[mask] / r_new[mask, None] * 0.999
            x_new = x_new[mask, None] * 0.999  # This is wrong, let me fix
        
        # Actually, let's just reject if invalid
        if np.any(r_new > 1.0):
            lr *= 0.5
            continue
        
        score_new = compute_score(x_new)
        
        if score_new > best_score:
            best_score = score_new
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.999  # Slow decay
    
    return best_x
# EVOLVE-BLOCK-END
