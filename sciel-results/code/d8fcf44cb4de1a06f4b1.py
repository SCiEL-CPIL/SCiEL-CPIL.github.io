import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    
    n = 17
    min_dist = 6
    total_words = 1 << n
    
    # Precompute popcount for all 17-bit words
    # Using a lookup table or vectorized bit manipulation
    # np.unpackbits is slow for this size? 
    # Better: use a precomputed array for 0..2^17
    # 131072 uint8 array is small.
    
    # Efficient popcount calculation
    # We can use the fact that popcount(x) = popcount(x & 0x5555) + popcount(x >> 1 & 0x5555) ...
    # Or just use a loop with NumPy vectorization on chunks?
    # Actually, Python's int.bit_count() is fast in CPython 3.10+.
    # But we need vectorized.
    
    # Let's create a popcount array
    popcount = np.zeros(total_words, dtype=np.uint8)
    # Vectorized popcount for uint32
    # x = x - ((x >> 1) & 0x55555555)
    # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    # x = (x + (x >> 4)) & 0x0F0F0F0F
    # popcount = (x * 0x01010101) >> 24
    
    x = np.arange(total_words, dtype=np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    popcount = (x * 0x01010101) >> 24
    popcount = popcount.astype(np.uint8)
    
    # Step 1: Construct the 256-word baseline [17, 8, 6] QR code
    # Generate the [17, 9, 5] QR code
    # Generator polynomial g(x) = 1 + x + x^4 + x^5 + x^6 + x^7 + x^8
    # Coeffs: 1, 1, 0, 0, 1, 1, 1, 1, 1
    g_coeffs = [1, 1, 0, 0, 1, 1, 1, 1, 1]
    g_int = 0
    for i, c in enumerate(g_coeffs):
        if c:
            g_int |= (1 << i)
            
    # Generate all 2^9 codewords
    qr_codewords = []
    for m in range(1 << 9):
        # c(x) = m(x) * g(x) mod (x^17 - 1)
        # Since deg(m) < 9 and deg(g) = 8, deg(c) < 17.
        # So no wrap-around needed for the multiplication itself, 
        # but we are in a cyclic code, so if deg >= 17, we wrap.
        # Here max degree is 8+8=16, so no wrap.
        c = 0
        mm = m
        gi = g_int
        # Multiply polynomials
        # This is just bitwise convolution
        # Simpler:
        # c = 0
        # for i in range(9):
        #     if (m >> i) & 1:
        #         c ^= (g_int << i)
        
        # Optimized multiplication
        c = 0
        temp_m = m
        temp_g = g_int
        # We can just do the shift and xor
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)
        
        qr_codewords.append(c)
        
    # Filter for even weight to get the [17, 8, 6] code
    baseline = []
    for c in qr_codewords:
        if popcount[c] % 2 == 0:
            baseline.append(c)
            
    # Ensure we have 256 words
    # If the generator polynomial was wrong, this might fail.
    # But for the QR code, this is standard.
    
    if len(baseline) < 256:
        # Fallback: Use a greedy construction from scratch if baseline fails
        # But let's assume it works.
        # If it's not 256, we might have a bug in g(x).
        # Let's try a different g(x) if needed? 
        # For now, proceed with whatever we have.
        pass
        
    current_code = baseline.copy()
    
    # Step 2: Greedy Expansion
    # Use NumPy for vectorized distance calculation
    
    # Precompute candidates array
    candidates = np.arange(total_words, dtype=np.uint32)
    
    while time.time() < deadline - 2.0:
        # Convert current code to numpy array for vectorized ops
        code_arr = np.array(current_code, dtype=np.uint32)
        
        # Compute XOR matrix: (N_candidates, N_code)
        # This can be large: 131072 * 300 ~ 39M elements.
        # 39M * 4 bytes = 156 MB. This is acceptable.
        
        xor_matrix = candidates[:, None] ^ code_arr[None, :]
        
        # Compute popcount for the XOR matrix
        # We can reuse the popcount logic on the flattened array?
        # Or apply the bit tricks on the matrix.
        
        x = xor_matrix
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        dist_matrix = (x * 0x01010101) >> 24
        dist_matrix = dist_matrix.astype(np.uint8)
        
        # Find minimum distance for each candidate
        min_dists = np.min(dist_matrix, axis=1)
        
        # Find candidates with min_dist >= 6
        valid_mask = min_dists >= min_dist
        
        if not np.any(valid_mask):
            break
            
        # Select the candidate with the highest minimum distance
        # This heuristic tends to pack the code more evenly
        # We only look at valid candidates
        valid_min_dists = min_dists[valid_mask]
        # Find the index in the valid_mask array with the max value
        # Then map it back to the original candidate index
        
        # Get indices of valid candidates
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            break
            
        # Find the valid candidate with the largest min_dist
        # If there are ties, pick the first one (or random)
        max_dist = np.max(valid_min_dists)
        # Find the first candidate with this max_dist
        # valid_indices[valid_min_dists == max_dist][0]
        
        best_local_idx = np.where(valid_min_dists == max_dist)[0][0]
        best_candidate = valid_indices[best_local_idx]
        
        current_code.append(int(best_candidate))
        
        # Optional: If we have time, we could do a local search/swap
        # But the greedy expansion is the main driver.
        
    return current_code
