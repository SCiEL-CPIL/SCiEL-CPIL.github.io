import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 24  # number of coefficients
    
    # Use a moderate grid for screening
    M_screen = 1024
    angles = np.linspace(0, 2 * np.pi, M_screen, endpoint=False)
    z = np.exp(1j * angles)
    
    # Precompute Z[k, m] = z[m]^k
    Z = np.zeros((N, M_screen), dtype=complex)
    Z[0, :] = 1.0
    for k in range(1, N):
        Z[k, :] = Z[k-1, :] * z
    
    # We fix c[0] = 1 (symmetry: flipping all signs doesn't change |P(z)|)
    # So we enumerate 2^23 candidates for c[1:24]
    
    # Larger batch size to reduce per-batch overhead
    B = 16384
    
    best_score = np.inf
    best_coeffs = np.ones(N)  # default
    
    # Total candidates: 2^23
    total_candidates = 1 << 23
    num_batches = (total_candidates + B - 1) // B
    
    # base = sum_{k=0}^{23} z^k (all +1)
    base = np.sum(Z, axis=0)  # shape (M_screen,)
    
    Z_flip = Z[1:24]  # shape (23, M_screen)
    
    for b in range(num_batches):
        if time.time() > deadline - 5.0:
            break
        
        idx_start = b * B
        idx_end = min(idx_start + B, total_candidates)
        B_actual = idx_end - idx_start
        
        indices = np.arange(idx_start, idx_end, dtype=np.int64)
        
        # Vectorized bit extraction: build (B_actual, 23) matrix
        # Use bitwise operations on the full array at once
        bit_positions = np.arange(23, dtype=np.int64)
        # (B_actual, 1) >> (1, 23) & 1
        flips = ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.float64)
        
        # Compute batch_P = base[None,:] - 2 * (flips @ Z_flip)
        batch_P = base[None, :] - 2.0 * (flips @ Z_flip)
        
        # Get max |P(z)| for each candidate
        scores = np.max(np.abs(batch_P), axis=1)
        
        # Find the best in this batch
        local_best_idx = np.argmin(scores)
        local_best_score = scores[local_best_idx]
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_idx = idx_start + local_best_idx
            # Reconstruct coefficients
            best_coeffs = np.ones(N)
            for i in range(23):
                if (best_idx >> i) & 1:
                    best_coeffs[i + 1] = -1.0
    
    # Refine: use a finer grid to get accurate score for the best candidate
    M_fine = 8192
    angles_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    z_fine = np.exp(1j * angles_fine)
    Z_fine = np.zeros((N, M_fine), dtype=complex)
    Z_fine[0, :] = 1.0
    for k in range(1, N):
        Z_fine[k, :] = Z_fine[k-1, :] * z_fine
    
    vals_fine = best_coeffs @ Z_fine
    fine_score = np.max(np.abs(vals_fine))
    
    # Golden-section refinement of the peak
    abs_vals = np.abs(vals_fine)
    peak_idx = np.argmax(abs_vals)
    
    angle_center = angles_fine[peak_idx]
    delta_angle = 2 * np.pi / M_fine
    a = angle_center - delta_angle
    c = angle_center + delta_angle
    
    def P_at_angle(angle):
        z_val = np.exp(1j * angle)
        powers = np.ones(N, dtype=complex)
        for k in range(1, N):
            powers[k] = powers[k-1] * z_val
        return np.abs(np.sum(best_coeffs * powers))
    
    # Golden section search
    gr = (np.sqrt(5) - 1) / 2
    b_pt = a + gr * (c - a)
    d_pt = a + (1 - gr) * (c - a)
    f_b = P_at_angle(b_pt)
    f_d = P_at_angle(d_pt)
    
    for _ in range(50):
        if f_b < f_d:
            c = d_pt
            d_pt = b_pt
            f_d = f_b
            b_pt = a + gr * (c - a)
            f_b = P_at_angle(b_pt)
        else:
            a = b_pt
            b_pt = d_pt
            f_b = f_d
            d_pt = a + (1 - gr) * (c - a)
            f_d = P_at_angle(d_pt)
    
    refined_score = max(f_b, f_d)
    
    return [int(c) for c in best_coeffs]
