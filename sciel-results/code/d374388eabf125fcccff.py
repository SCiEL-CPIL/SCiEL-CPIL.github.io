import time
import numpy as np
from numpy.linalg import eigh

def _compute_soft_gap(cell, k_points, pw_radius=14, beta=10.0):
    """
    Compute a smooth surrogate for the gap-midgap ratio using log-sum-exp.
    Also returns gradients w.r.t the continuous field c (0..1) if needed.
    """
    N = len(cell)
    c = np.array(cell, dtype=float)
    inv_eps = 1.0 / (1.0 + 11.0 * c)
    
    # FFT of inv_eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    # Plane waves
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    # Build inv_eps matrix
    size_d = 2 * pw_radius + 1
    offset_d = pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    # Precompute indices for the matrix
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Vectorized fill
    # We need to map (dqx, dqy) to fft_inv[dqx % N, dqy % N]
    # Only valid if -pw_radius <= dq <= pw_radius
    valid_mask = (np.abs(Gx_diff) <= pw_radius) & (np.abs(Gy_diff) <= pw_radius)
    
    # Create a flat index for valid entries
    valid_indices = np.where(valid_mask)
    dqx_vals = Gx_diff[valid_indices]
    dqy_vals = Gy_diff[valid_indices]
    
    # Map to fft indices
    fft_idx_x = dqx_vals % N
    fft_idx_y = dqy_vals % N
    fft_vals = fft_inv[fft_idx_x, fft_idx_y]
    
    # Assign to inv_eps_full
    inv_eps_full[dqx_vals + offset_d, dqy_vals + offset_d] = fft_vals
    
    # Precompute kG and dots for all k-points? No, k-points vary.
    # But we can precompute the structure of M.
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    # For soft gap, we want to approximate:
    # Score = 2 * (min_k w5(k) - max_k w4(k)) / (min_k w5(k) + max_k w4(k))
    # Let A = -log(sum(exp(-beta * w4(k))))  (soft min of w4? No, we want max w4)
    # Let B = log(sum(exp(beta * w5(k)))) / beta (soft max of w5? No, we want min w5)
    
    # Actually, to maximize the gap, we want to maximize min_w5 and minimize max_w4.
    # Let's define:
    # S4 = soft_max(w4) = (1/beta) * log(sum(exp(beta * w4)))
    # S5 = soft_min(w5) = -(1/beta) * log(sum(exp(-beta * w5)))
    # Objective = 2 * (S5 - S4) / (S5 + S4)
    
    sum_exp_beta_w4 = 0.0
    sum_exp_neg_beta_w5 = 0.0
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        # M = dots * inv_eps_mat
        # Note: inv_eps_mat is the same for all k, just dots changes.
        # We can reuse inv_eps_full.
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                
                # Accumulate for soft min/max
                sum_exp_beta_w4 += np.exp(beta * w4)
                sum_exp_neg_beta_w5 += np.exp(-beta * w5)
        except:
            continue
    
    if sum_exp_beta_w4 == 0 or sum_exp_neg_beta_w5 == 0:
        return -1.0, None
        
    S4 = (1.0 / beta) * np.log(sum_exp_beta_w4)
    S5 = -(1.0 / beta) * np.log(sum_exp_neg_beta_w5)
    
    # If S5 <= S4, gap is negative
    if S5 <= S4:
        return -1.0, None
        
    score = 2.0 * (S5 - S4) / (S5 + S4)
    return score, None

def _compute_gap_fast(cell, n_k=14, pw_radius=12):
    """Compute a fast estimate of the gap-midgap ratio between TM bands 4 and 5."""
    N = len(cell)
    cell_arr = np.array(cell, dtype=float)
    inv_eps = 1.0 / (1.0 + 11.0 * cell_arr)
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 2 * pw_radius + 1
    offset_d = pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            if -pw_radius <= dqx <= pw_radius and -pw_radius <= dqy <= pw_radius:
                inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    # k-points in the irreducible BZ (first quadrant, kx >= ky)
    # Using a denser grid similar to evaluator
    k_points = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.3), (0.4, 0.2), (0.2, 0.2), (0.3, 0.1), (0.4, 0.4),
        (0.1, 0.1), (0.2, 0.1), (0.3, 0.2), (0.4, 0.1), (0.45, 0.45)
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)

def _make_symmetric_from_fundamental(c_fund):
    """
    Expand a 16x16 fundamental domain (top-left quadrant including diagonals)
    to a full 32x32 symmetric cell.
    The symmetry group is D4. The fundamental domain is 16x16.
    Indices i, j in [0..15].
    Symmetries:
    1. Identity: (i, j)
    2. Transpose: (j, i)
    3. Row reverse: (31-i, j)
    4. Col reverse: (i, 31-j)
    5. 180 rot: (31-i, 31-j)
    6. Anti-diag: (31-j, 31-i)
    7. Row rev + Trans: (31-j, i)
    8. Col rev + Trans: (j, 31-i)
    """
    size = 32
    c_full = np.zeros((size, size), dtype=float)
    for i in range(16):
        for j in range(16):
            val = c_fund[i, j]
            c_full[i, j] = val
            c_full[j, i] = val
            c_full[31-i, j] = val
            c_full[i, 31-j] = val
            c_full[31-i, 31-j] = val
            c_full[31-j, 31-i] = val
            c_full[31-j, i] = val
            c_full[j, 31-i] = val
    return c_full

def _compute_gradient_fundamental(c_fund, k_points, pw_radius=12, beta=10.0, step=0.01):
    """
    Compute gradient of the soft gap w.r.t the fundamental domain c_fund.
    Uses central differences on the fundamental domain, expanding to full cell.
    This is expensive but accurate enough for a few steps.
    """
    N_fund = 16
    grad = np.zeros((N_fund, N_fund), dtype=float)
    
    # We only need to differentiate w.r.t independent pixels in the fundamental domain.
    # However, due to symmetry, changing c_fund[i,j] affects 8 pixels in the full cell.
    # The gradient w.r.t c_fund[i,j] is the sum of gradients w.r.t those 8 pixels.
    
    # To save time, we can compute the gradient w.r.t the full cell c_full,
    # then map it back.
    
    c_full = _make_symmetric_from_fundamental(c_fund)
    
    # Compute base score
    score_base, _ = _compute_soft_gap(c_full, k_points, pw_radius, beta)
    if score_base == -1.0:
        # If invalid, try a small perturbation or return zero grad
        return np.zeros_like(c_fund)
        
    # This will be very slow if we do full finite differences for 256 pixels.
    # Instead, we can use a simpler heuristic or reduce the number of k-points/pw.
    # Let's use a very small number of k-points for the gradient step.
    k_points_grad = k_points[:4]
    pw_radius_grad = 10
    
    score_base_grad, _ = _compute_soft_gap(c_full, k_points_grad, pw_radius_grad, beta)
    
    if score_base_grad == -1.0:
        return np.zeros_like(c_fund)
        
    for i in range(N_fund):
        for j in range(N_fund):
            # Perturb c_fund[i, j]
            c_fund_p = c_fund.copy()
            c_fund_p[i, j] += step
            if c_fund_p[i, j] > 1.0: continue # Skip if out of bounds, or clamp
            
            c_full_p = _make_symmetric_from_fundamental(c_fund_p)
            score_p, _ = _compute_soft_gap(c_full_p, k_points_grad, pw_radius_grad, beta)
            
            if score_p == -1.0:
                grad[i, j] = 0
            else:
                grad[i, j] = (score_p - score_base_grad) / step
                
    return grad

def design(problem):
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin (230s limit)
    
    size = problem["pixels"]
    assert size == 32
    
    # Initialize fundamental domain
    # Start with a simple symmetric pattern, e.g., a disk in the fundamental domain
    c_fund = np.zeros((16, 16), dtype=float)
    for i in range(16):
        for j in range(16):
            x = (i + 0.5) / 32.0 - 0.5
            y = (j + 0.5) / 32.0 - 0.5
            # A disk of radius 0.15 in the full cell
            if x*x + y*y < 0.15*0.15:
                c_fund[i, j] = 1.0
            else:
                c_fund[i, j] = 0.0
            
    # Ensure symmetry in fundamental domain itself?
    # The fundamental domain is just the top-left 16x16.
    # The symmetry is enforced by _make_symmetric_from_fundamental.
    # But we should ensure that pixels on the boundaries of the fundamental domain
    # (i=15, j=15, i=j) are handled correctly. They are mapped to themselves or pairs.
    # The expansion handles this correctly.
    
    # k-points for optimization
    # Use a subset of the evaluator's 30 points
    k_points_opt = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.3), (0.4, 0.2), (0.2, 0.2), (0.3, 0.1), (0.4, 0.4),
        (0.1, 0.1), (0.2, 0.1), (0.3, 0.2), (0.4, 0.1), (0.45, 0.45)
    ]
    
    best_score = -1.0
    best_c_fund = c_fund.copy()
    
    # Phase 1: Gradient Descent on Continuous Field
    # Use a smaller number of k-points and pw_radius for speed
    k_points_fast = k_points_opt[:8]
    pw_radius_fast = 10
    beta = 10.0
    
    for step_idx in range(30):
        if time.time() > deadline - 60:
            break
            
        # Compute gradient
        # This is the bottleneck. 256 pixels * (1 eval) * (cost of eval)
        # Cost of eval with 8 k-points, pw=10 (141 PWs) is ~10-20ms?
        # 256 * 15ms = ~4s per step. 30 steps = 120s. Might be too slow.
        # Let's reduce to 10 steps or fewer pixels.
        # Or use a coarser fundamental domain? No, must be 16x16.
        
        # Let's try to compute gradient only for a subset of pixels or use a cheaper metric.
        # Actually, let's just do 5 steps of this expensive gradient.
        
        grad = _compute_gradient_fundamental(c_fund, k_points_fast, pw_radius_fast, beta, step=0.05)
        
        # Update
        lr = 0.2
        c_fund += lr * grad
        c_fund = np.clip(c_fund, 0.0, 1.0)
        
        # Evaluate current state
        c_full = _make_symmetric_from_fundamental(c_fund)
        score = _compute_gap_fast(c_full, n_k=10, pw_radius=10)
        
        if score > best_score:
            best_score = score
            best_c_fund = c_fund.copy()
            
    # Phase 2: Local Search on Binary Cell
    if time.time() < deadline - 30:
        c_full = _make_symmetric_from_fundamental(best_c_fund)
        # Threshold to binary
        c_bin = (c_full > 0.5).astype(int).tolist()
        
        current_score = _compute_gap_fast(c_bin, n_k=10, pw_radius=10)
        if current_score > best_score:
            best_score = current_score
            best_c_fund = (np.array(c_bin) > 0.5).copy() # Store binary as float 0/1
            
        # Local search: flip pixels in fundamental domain
        for iter in range(50):
            if time.time() > deadline - 10:
                break
                
            # Pick a random pixel in fundamental domain
            i = np.random.randint(0, 16)
            j = np.random.randint(0, 16)
            
            c_fund_new = best_c_fund.copy()
            c_fund_new[i, j] = 1.0 - c_fund_new[i, j]
            
            c_full_new = _make_symmetric_from_fundamental(c_fund_new)
            c_bin_new = (c_full_new > 0.5).astype(int).tolist()
            
            score_new = _compute_gap_fast(c_bin_new, n_k=10, pw_radius=10)
            
            if score_new > best_score:
                best_score = score_new
                best_c_fund = c_fund_new
            else:
                # Backtrack
                pass
                
    # Final Polish: High precision check
    if time.time() < deadline - 5:
        c_full = _make_symmetric_from_fundamental(best_c_fund)
        c_bin = (c_full > 0.5).astype(int).tolist()
        # Verify symmetry
        c_arr = np.array(c_bin)
        if not (np.allclose(c_arr, c_arr.T) and np.allclose(c_arr, c_arr[::-1, :]) and np.allclose(c_arr, c_arr[:, ::-1])):
            # Force symmetry
            c_avg = (c_arr + c_arr.T + c_arr[::-1, :] + c_arr[:, ::-1] + c_arr[::-1, ::-1] + c_arr.T[::-1, :] + c_arr.T[:, ::-1] + c_arr[::-1, ::-1].T) / 8.0
            c_bin = (c_avg > 0.5).astype(int).tolist()
            
        return c_bin
    
    # Fallback
    return [[0]*32 for _ in range(32)]

