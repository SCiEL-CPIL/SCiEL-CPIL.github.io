import numpy as np

N = 400

def construct():
    # Target size for the support.
    # Smaller M reduces the potential for large peaks, but the L2 normalization 
    # increases the amplitude of each step. 
    # Peak ~ sum of products. If M is small, the number of terms in convolution 
    # for a specific shift is small.
    # Let's try M around 20-30.
    
    M = 25
    
    # Random initialization of support
    indices = np.random.choice(N, M, replace=False)
    
    # Initialize signs randomly
    signs = np.random.choice([-1, 1], M)
    
    # Function to compute the peak of the autoconvolution
    # We compute the convolution of x with x.
    # x is sparse.
    def get_peak(support_idx, sgn):
        # Create sparse representation for fast convolution?
        # N=400 is small enough for direct convolution using FFT or direct loops.
        # Direct loops O(M^2) to compute all non-zero conv terms?
        # No, convolution is O(N^2) or O(N log N).
        # With M=25, the number of non-zero pairs is M^2 = 625.
        # We can compute the convolution array directly by summing contributions.
        
        conv = np.zeros(N) # Linear convolution length is 2N-1, but we care about the range?
        # The problem says "on [-1/4, 1/4]". This usually implies circular convolution 
        # on a torus of length N, or linear convolution truncated?
        # "400 equal intervals on [-1/4, 1/4]". 
        # If it's a function on a bounded interval, the convolution is defined on a larger interval.
        # However, "autoconvolution upper bound" in harmonic analysis contexts often 
        # implies the torus (circular) or the specific $L_p$ norms.
        # Given the "Sidon" comment in the old code, circular convolution (differences mod N) 
        # is a strong candidate. Let's assume circular convolution (differences mod N).
        
        # If circular:
        # conv[d] = sum_{i} x[i] * x[(i-d) % N]
        
        # We only have non-zeros at support_idx.
        # conv[d] = sum_{i in support} x[i] * x[(i-d) % N]
        
        # Let's use the difference method.
        # For each pair (i, j) in support, the difference d = (i - j) % N contributes 
        # x[i] * x[j] to conv[d].
        
        # Actually, (x*x)[d] = sum_i x[i] x[i-d].
        # Let i be an index in support. Then x[i] is non-zero.
        # x[i-d] is non-zero if (i-d) % N is in support.
        
        # Let's map support to values.
        x_vals = {}
        for idx, s in zip(support_idx, sgn):
            x_vals[idx] = s
        
        # We need to sum x[i] * x[j] where (i - j) % N == d.
        # This is equivalent to summing x[i] * x[j] for all pairs (i,j) and adding to bucket (i-j)%N.
        
        # Efficient way:
        # Since M is small, we can just iterate over all pairs.
        # O(M^2) operations.
        conv_vals = np.zeros(N)
        
        for i_idx in range(M):
            i_pos = support_idx[i_idx]
            i_val = sgn[i_idx]
            for j_idx in range(M):
                j_pos = support_idx[j_idx]
                j_val = sgn[j_idx]
                d = (i_pos - j_pos) % N
                conv_vals[d] += i_val * j_val
        
        return np.max(np.abs(conv_vals))
    
    # Local Search
    # 1. Compute initial peak
    current_peak = get_peak(indices, signs)
    
    # Optimization loop
    # We will try to improve by:
    # a) Flipping a sign
    # b) Swapping an active index with a random inactive index
    
    # Pre-compute inactive indices? 
    # It's easier to just pick random indices for swaps.
    
    max_iters = 5000
    np.random.seed(42) # For reproducibility in this block, though construct is called fresh?
    # Actually, construct() is the entry point. We should use a fixed seed or random?
    # The prompt implies we can use randomness.
    
    for _ in range(max_iters):
        improved = False
        
        # Try flipping signs
        # Pick a random index in support
        flip_idx = np.random.randint(M)
        new_signs = signs.copy()
        new_signs[flip_idx] *= -1
        
        new_peak = get_peak(indices, new_signs)
        if new_peak < current_peak:
            signs = new_signs
            current_peak = new_peak
            improved = True
        
        # Try swapping an active index with an inactive one
        # Pick a random active index to remove
        rem_idx = np.random.randint(M)
        # Pick a random inactive index to add
        # To avoid picking an active one, we can pick from 0..N-1 and check, 
        # or maintain a set. With M=25, N=400, collision is low.
        add_pos = np.random.randint(N)
        
        if add_pos in indices:
            continue # Skip if we picked an active one, or just retry logic
        
        new_indices = indices.copy()
        new_indices[rem_idx] = add_pos
        # Note: We keep the sign associated with the position? 
        # Or do we keep the sign associated with the "slot"?
        # Usually, the sign is a property of the function at that location.
        # If we move the "mass" from old_pos to new_pos, the sign moves with it?
        # Or do we just change the support and keep signs array aligned?
        # Let's assume the sign belongs to the *value* at that location.
        # If we replace index `indices[rem_idx]` with `add_pos`, we are changing the location of a non-zero value.
        # The sign `signs[rem_idx]` should probably stay with the "entity" being moved, 
        # or we can just treat it as: new_support has add_pos, and we keep the sign 
        # at the same *position* in the array? 
        # It's better to think of it as: we have a set of (position, sign) pairs.
        # We replace (old_pos, sign_k) with (new_pos, sign_k).
        
        # So new_indices[rem_idx] = add_pos. signs remains the same.
        
        new_peak = get_peak(new_indices, signs)
        if new_peak < current_peak:
            indices = new_indices
            current_peak = new_peak
            improved = True
            
        if not improved and _ % 100 == 0:
            # Random restart or perturbation if stuck?
            # For now, just continue.
            pass
            
    # Construct final x
    x = np.zeros(N)
    for idx, s in zip(indices, signs):
        x[idx] = s / np.sqrt(M)
        
    return x
