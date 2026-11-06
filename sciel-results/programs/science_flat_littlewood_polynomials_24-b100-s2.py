import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    
    # Batch size
    B = 2000
    
    best_score = np.inf
    best_coeffs = np.ones(N)  # default
    
    # Total candidates: 2^23
    total_candidates = 1 << 23
    num_batches = (total_candidates + B - 1) // B
    
    # Precompute: for each bit position 1..23, the contribution to Z
    # When bit i is set (c_i = -1 instead of +1), the change is -2 * z^i
    # So P(z) = sum_{k=0}^{23} c_k z^k = sum_{k=0}^{23} z^k - 2 * sum_{i in flipped} z^i
    # base = sum_{k=0}^{23} z^k (all +1)
    base = np.sum(Z, axis=0)  # shape (M_screen,)
    
    # For a candidate with flip mask f (bits 1..23), P(z) = base - 2 * sum_{i: f_i=1} z^i
    # = base - 2 * (Z[1:24] @ f_vector)
    
    # For batched evaluation:
    # batch_flips: (B, 23) binary matrix
    # batch_P: (B, M_screen) = base[None,:] - 2 * (batch_flips @ Z[1:24])
    
    Z_flip = Z[1:24]  # shape (23, M_screen)
    
    for b in range(num_batches):
        if time.time() > deadline - 5.0:
            break
        
        idx_start = b * B
        idx_end = min(idx_start + B, total_candidates)
        B_actual = idx_end - idx_start
        
        # Generate flip masks for this batch
        # Each candidate is an integer from idx_start to idx_end-1
        # The flip mask is the bits 0..22 of the integer (corresponding to c_1..c_23)
        indices = np.arange(idx_start, idx_end, dtype=np.int64)
        
        # Extract 23 bits from each index
        # bit i of the integer corresponds to flip of c_{i+1}
        # We want a (B_actual, 23) matrix where entry (j, i) = bit i of indices[j]
        
        # Efficient bit extraction
        # For each bit position i (0..22), check if (indices >> i) & 1
        # This gives 23 arrays of shape (B_actual,)
        
        # Build flip matrix
        flips = np.zeros((B_actual, 23), dtype=np.float64)
        for i in range(23):
            flips[:, i] = (indices >> i) & 1
        
        # Compute batch_P = base[None,:] - 2 * (flips @ Z_flip)
        # flips @ Z_flip: (B_actual, 23) @ (23, M_screen) = (B_actual, M_screen)
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
    # Find the angle where |P(z)| is maximized
    abs_vals = np.abs(vals_fine)
    peak_idx = np.argmax(abs_vals)
    
    # Refine around the peak using golden section
    # The peak is at angles_fine[peak_idx]
    # Search in a small neighborhood
    angle_center = angles_fine[peak_idx]
    # Search range: one grid spacing
    delta_angle = 2 * np.pi / M_fine
    a = angle_center - delta_angle
    c = angle_center + delta_angle
    
    def P_at_angle(angle):
        z_val = np.exp(1j * angle)
        # Compute P(z_val) = sum c_k z_val^k
        powers = np.ones(N, dtype=complex)
        for k in range(1, N):
            powers[k] = powers[k-1] * z_val
        return np.abs(np.sum(best_coeffs * powers))
    
    # Golden section search
    gr = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
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
    
    # The score is refined_score / sqrt(24)
    # But we just need to return the coefficients
    return [int(c) for c in best_coeffs]
