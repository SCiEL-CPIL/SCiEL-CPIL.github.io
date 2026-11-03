import numpy as np
import time
from scipy.optimize import minimize

N = 200
HALF = 0.5
TIME_LIMIT = 145.0

def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))

def _hex_points(s, ox, oy):
    """Hexagonal lattice points with spacing s, offset (ox, oy)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 3
    rows = int(np.ceil(1.0 / dy)) + 3
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    # offset every other column
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x + 0.0, y + yoff))
    arr = np.array(pts)
    # keep inside square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]

def _greedy_select(cands, k, r_min):
    """Greedy farthest-point selection of k points maximizing min distance."""
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    selected = []
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    for _ in range(k - 1):
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        j = int(np.argmax(min_dist))
        if min_dist[j] < r_min - 1e-9:
            break
        selected.append(j)
    return cands[selected]

def _soft_min_obj_and_grad(x_flat, beta):
    """
    Objective: maximize soft-min of distances and margins.
    x_flat: shape (2N,)
    Returns: (objective_value, gradient) for minimization (negative of max).
    """
    x = x_flat.reshape(N, 2)
    
    # Compute pairwise distances
    # This is O(N^2). For N=200, 40000 pairs.
    # We can vectorize.
    diff = x[:, None, :] - x[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Margins
    margins = HALF - np.max(np.abs(x), axis=1)
    
    # Soft min: -1/beta * log( sum exp(-beta * val) )
    # We want to maximize this. So minimize -soft_min.
    # To avoid overflow, use log-sum-exp trick.
    # vals = concat(upper_triangle(dists), margins)
    # But we need to be careful with the gradient.
    
    # Let's just compute the sum of exps directly if beta is not too large.
    # If beta is large, exp(-beta * d) underflows for large d, which is good.
    # But for small d, it's large.
    
    # We only care about the "active" constraints, i.e., those close to the min.
    # However, for a smooth gradient, we sum all.
    
    # Optimization: Only compute for pairs with dist < some threshold?
    # No, soft-min is global. But exp(-beta * d) is very small for d >> 1/beta.
    
    # Let's compute the weights.
    # d_flat = dists[upper_tri_indices]
    # m_flat = margins
    
    # To save time, we can restrict the sum to pairs with d < 1.0 (since r is ~0.03, 1.0 is huge).
    # Actually, r is ~0.03, so 1/beta should be comparable to r.
    # If beta = 100, 1/beta = 0.01.
    # exp(-100 * 0.5) = exp(-50) ~ 1e-22.
    # So we can ignore pairs with d > 1.0.
    
    mask = dists < 1.0
    # Also include margins
    # We need to handle the gradient carefully.
    
    # Let's use a simpler approach:
    # Just compute the full sum. N=200 is small enough.
    
    d_flat = dists[mask]
    m_flat = margins
    
    # Combine
    # We want -1/beta * log( sum exp(-beta * d) + sum exp(-beta * m) )
    # Let S = sum exp(-beta * d) + sum exp(-beta * m)
    # log_S = log(S)
    # obj = -1/beta * log_S
    # grad_obj = -1/beta * (1/S) * sum ( -beta * exp(-beta * val) * grad_val )
    #          = 1/S * sum ( exp(-beta * val) * grad_val )
    
    # We want to MINIMIZE -obj = 1/beta * log_S
    # grad_min = 1/beta * (1/S) * sum ( -beta * exp(-beta * val) * grad_val )
    #           = -1/S * sum ( exp(-beta * val) * grad_val )
    
    # Let's compute weights w_val = exp(-beta * val) / S
    # Then grad_min = - sum w_val * grad_val
    
    # Compute S
    exp_d = np.exp(-beta * d_flat)
    exp_m = np.exp(-beta * m_flat)
    S = np.sum(exp_d) + np.sum(exp_m)
    
    if S == 0:
        return 0.0, np.zeros_like(x_flat)
        
    w_d = exp_d / S
    w_m = exp_m / S
    
    # Gradient of dists w.r.t x
    # grad_dists[i, j] = (x_i - x_j) / dists[i, j]
    # We need to sum over all pairs.
    # grad_x_i = sum_j w_d[i, j] * (x_i - x_j) / d_ij
    
    # This is still O(N^2).
    # We can compute it efficiently.
    
    # Let's construct the gradient array.
    grad_x = np.zeros_like(x)
    
    # Only process pairs in mask
    # This is tricky with vectorization if mask is sparse.
    # For N=200, N^2 = 40000. It's fast enough in Python if we loop? No, 40000 loops is fine.
    # But we can do better.
    
    # Let's just loop over the non-zero weights.
    # Actually, most weights will be zero due to exp underflow.
    
    # Better: Vectorized update.
    # We have diff = x[:, None, :] - x[None, :, :]
    # We have w_d matrix (N, N) but only upper triangle? No, dists is symmetric.
    # Let's create a weight matrix W_d of shape (N, N).
    W_d = np.zeros((N, N))
    W_d[mask] = w_d  # This is wrong, mask is (N, N) but w_d is 1D.
    
    # Let's re-extract indices.
    i_idx, j_idx = np.where(mask)
    # Exclude diagonal
    valid = i_idx != j_idx
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    w_d_valid = w_d  # w_d was computed from d_flat which was from mask.
    # Wait, d_flat = dists[mask]. mask includes diagonal?
    # np.fill_diagonal(dists, np.inf) was done.
    # So mask will be False for diagonal if inf < 1.0 is False.
    # So d_flat does NOT include diagonal.
    # So i_idx, j_idx are correct pairs.
    
    # We need to sum contributions to grad_x.
    # For each pair (i, j) with weight w:
    # grad_x[i] += w * (x_i - x_j) / d_ij
    # grad_x[j] += w * (x_j - x_i) / d_ij
    
    # We can vectorize this using advanced indexing.
    # diff_vec = x[i_idx] - x[j_idx]  # shape (M, 2)
    # d_vec = dists[i_idx, j_idx]     # shape (M,)
    # unit_vec = diff_vec / d_vec[:, None]
    
    # contrib_i = w_d_valid[:, None] * unit_vec
    # contrib_j = -contrib_i
    
    # grad_x[i_idx] += contrib_i
    # grad_x[j_idx] += contrib_j
    
    # Numpy doesn't support in-place addition with repeated indices easily without np.add.at.
    # np.add.at is slow.
    
    # Alternative: Use sparse matrix or just loop if M is small.
    # M is number of pairs with d < 1.0.
    # With r ~ 0.03, and N=200, the number of close pairs is roughly N * (1/r^2) * area?
    # In a hex packing, each circle has 6 neighbors.
    # So M ~ 6 * N / 2 = 300.
    # 300 iterations is very fast.
    
    # Let's loop.
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = w_d_valid[k]
        if w < 1e-10: continue
        d_ij = dists[i, j]
        if d_ij < 1e-10: continue
        v = (x[i] - x[j]) / d_ij
        grad_x[i] += w * v
        grad_x[j] -= w * v
        
    # Margins
    # grad of margin_i = - sign(x_i) / ?
    # margin_i = 0.5 - max(|x_ix|, |x_iy|)
    # d margin_i / d x_ix = - sign(x_ix) if |x_ix| > |x_iy| else 0
    # d margin_i / d x_iy = - sign(x_iy) if |x_iy| > |x_ix| else 0
    
    # w_m are the weights for margins.
    for i in range(N):
        w = w_m[i]
        if w < 1e-10: continue
        ax = np.abs(x[i, 0])
        ay = np.abs(x[i, 1])
        if ax > ay:
            grad_x[i, 0] -= w * np.sign(x[i, 0])
        else:
            grad_x[i, 1] -= w * np.sign(x[i, 1])
            
    # Objective value for minimization
    obj_val = (1.0 / beta) * np.log(S)
    
    # Gradient for minimization
    grad_flat = -grad_x.flatten()
    
    return obj_val, grad_flat

def _gradient_polish(pos, time_budget=60.0):
    """
    Local search using gradient ascent on soft-min objective.
    """
    t0 = time.time()
    pos = pos.copy().astype(float)
    
    # Start with a small beta and increase it (annealing)
    beta = 10.0
    
    bounds = [(-HALF, HALF)] * (2 * N)
    
    best_r = _score(pos)
    best_pos = pos.copy()
    
    while time.time() - t0 < time_budget:
        x_flat = pos.flatten()
        
        # Define objective and gradient for this beta
        def obj_grad(x):
            val, grad = _soft_min_obj_and_grad(x, beta)
            return val, grad
            
        # Use L-BFGS-B
        try:
            res = minimize(
                obj_grad, 
                x_flat, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            pos = res.x.reshape(N, 2)
        except Exception:
            break
            
        r = _score(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
        # Increase beta to sharpen the minimum
        beta *= 1.5
        if beta > 1000:
            break
            
    return best_pos, best_r

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_v = 0.0
    
    # Target spacing estimate
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)
    
    # Phase 1: Generate diverse initial configurations
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > DEADLINE - 30.0:  # Leave time for polish
            break
        
        # Vary spacing and offset
        s = s0 * (0.98 + 0.04 * rng.random())
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        cands = _hex_points(s, ox, oy)
        
        if cands.shape[0] < N:
            continue
        
        r_min = 0.5 * s * 0.9
        sel = _greedy_select(cands, N, r_min)
        if sel.shape[0] < N:
            continue
        
        v = _score(sel)
        if v > best_v:
            best_v = v
            best_c = sel.copy()
            
    # Phase 2: Gradient Polish
    if best_c is not None:
        remaining = DEADLINE - time.time()
        if remaining > 10.0:
            polished, pv = _gradient_polish(best_c, time_budget=min(remaining - 5.0, 120.0))
            if pv > best_v:
                best_c = polished
                best_v = pv
    else:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_v = _score(best_c)
        
    # Final validity clamp
    best_c = np.clip(best_c, -HALF, HALF)
    
    return best_c
