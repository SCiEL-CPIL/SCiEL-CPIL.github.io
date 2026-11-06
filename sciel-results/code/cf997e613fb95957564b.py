import numpy as np
import time

def _supnorm(coeffs, n_pts=4096, refine_rounds=5, refine_points=10):
    """
    Estimate max |P(z)| for |z|=1 using dense sampling + golden-section refinement.
    coeffs: list/array of length n
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    # Dense sampling
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    if n_pts > refine_points:
        top_idx = np.argpartition(magnitudes, -refine_points)[-refine_points:]
    else:
        top_idx = np.arange(n_pts)
    
    best_mag = float(np.max(magnitudes))
    
    # Refine around the top peaks
    window = 2.0 * np.pi / n_pts * 2
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    inv_golden = 1.0 - 1.0 / golden_ratio
    
    for idx in top_idx:
        center = angles[idx]
        a = max(0.0, center - window)
        b = min(2.0 * np.pi, center + window)
        
        if b <= a:
            continue
            
        c = b - inv_golden * (b - a)
        d = a + inv_golden * (b - a)
        
        def _eval(angle):
            return np.abs(np.sum(coeffs * np.exp(1j * k * angle)))
        
        fc = _eval(c)
        fd = _eval(d)
        
        for _ in range(refine_rounds):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + inv_golden * (b - a)
                fd = _eval(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - inv_golden * (b - a)
                fc = _eval(c)
        
        fa = _eval(a)
        fb = _eval(b)
        
        local_best = max(fa, fb, fc, fd)
        if local_best > best_mag:
            best_mag = local_best
    
    return best_mag

def _score(coeffs):
    """Score = supnorm / sqrt(n)"""
    n = len(coeffs)
    return _supnorm(coeffs) / np.sqrt(n)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 24
    
    # Precompute sampling matrix for fast batch evaluation
    # Use a moderate number of points for the search phase
    n_pts_search = 1024
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts_search, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(N)
    # M[i, j] = z_i^j
    M = np.power(z[:, None], k[None, :])
    
    best_coeffs = np.ones(N, dtype=np.float64)
    best_score = np.inf
    best_fast_score = np.inf
    
    # Batch size for exhaustive search
    batch_size = 50000
    total_candidates = 1 << N
    
    # We will iterate through all 2^24 possibilities
    # To save memory and time, we process in batches
    # We use a random permutation of indices to avoid bias, or just sequential?
    # Sequential is fine, but random might find good ones earlier if we have a time limit.
    # Given the time limit, let's do random sampling first, then fill in with sequential if time permits?
    # No, exhaustive is better. Let's just do sequential batches.
    
    # Actually, generating 50k random indices is faster than sequential if we don't cover all, 
    # but we want to cover all. 
    # Let's stick to sequential batches to ensure full coverage if time allows.
    
    for start in range(0, total_candidates, batch_size):
        if time.time() > deadline - 10:
            break
        
        end = min(start + batch_size, total_candidates)
        n_batch = end - start
        
        # Generate indices for this batch
        indices = np.arange(start, end, dtype=np.uint32)
        
        # Convert indices to binary coefficients
        # We want bits to be 1 or -1.
        # Let's map bit 0 to -1 if 0, 1 if 1? Or vice versa.
        # It doesn't matter as long as we cover all.
        
        # Efficient way to get bits:
        # Create a matrix of bits
        # For each index, we want N bits.
        # We can use broadcasting.
        
        # Instead of creating a huge index array, we can compute the values directly?
        # No, we need the coefficients to multiply with M.
        
        # Let's create the coefficient matrix C of shape (n_batch, N)
        # C[i, j] = 1 if bit j of indices[i] is 1, else -1
        
        # This is a common pattern.
        # powers = 1 << np.arange(N)
        # bits = (indices[:, None] & powers[None, :]) != 0
        # C = np.where(bits, 1.0, -1.0)
        
        # This creates a boolean array of size n_batch x N.
        # 50000 x 24 is small.
        
        powers = 1 << np.arange(N, dtype=np.uint32)
        bits = (indices[:, None] & powers[None, :]) != 0
        C = np.where(bits, 1.0, -1.0)
        
        # Evaluate supnorm for all candidates in this batch
        # V = C @ M.T
        # C is (n_batch, N), M.T is (N, n_pts_search)
        # V is (n_batch, n_pts_search)
        
        V = C @ M.T
        mags = np.max(np.abs(V), axis=1)
        
        # Find the best in this batch
        min_idx_in_batch = np.argmin(mags)
        min_val = mags[min_idx_in_batch]
        
        if min_val < best_fast_score:
            best_fast_score = min_val
            best_coeffs = C[min_idx_in_batch].copy()
            
    # After the search, refine the best candidate with high precision
    best_score = _score(best_coeffs)
    
    return [int(c) for c in best_coeffs]
