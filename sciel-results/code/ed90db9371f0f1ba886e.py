import numpy as np
import time

N = 100
W = 2.0 / N

def _overlap_max_binary(mask):
    """
    Compute the maximum overlap for a binary mask (1s and 0s).
    mask: boolean array of length N where True means h_i=1.
    The heights are h_i = 1 if mask[i] else 0.
    Sum of h must be N/2, so exactly N/2 ones.
    
    Overlap at shift k:
    O_k = sum_{i} h_i * (1 - h_{i+k}) * W
    Since h is binary, h_i * (1 - h_{i+k}) is 1 if h_i=1 and h_{i+k}=0, else 0.
    So O_k = W * (number of i such that h_i=1 and h_{i+k}=0)
    
    We want to minimize max_k O_k.
    """
    # Use FFT to compute cross-correlation between h and 1-h
    # h is binary. 1-h is also binary (complement).
    # Correlation C_k = sum_i h_i * (1-h)_{i+k}
    # This is equivalent to sum_i h_i - sum_i h_i * h_{i+k}
    # Sum_i h_i = N/2 (constant)
    # So minimizing max C_k is equivalent to maximizing min autocorr_k (for k != 0)
    # Wait, let's just compute it directly via FFT for safety and speed.
    
    h = mask.astype(np.float64)
    # 1-h is the complement
    # Autocorrelation of h: A_k = sum h_i h_{i+k}
    # C_k = sum h_i (1 - h_{i+k}) = sum h_i - A_k = N/2 - A_k
    # We want to minimize max_k C_k = N/2 - min_k A_k (for k=1..N-1)
    # So we want to maximize the minimum non-zero autocorrelation.
    
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] is sum_i h_i * h_{i-k} (circular)
    # Due to symmetry, autocorr[k] == autocorr[N-k]
    # We care about shifts 1 to N-1.
    # The value at k=0 is N/2 (sum of squares).
    
    # We want min_{k=1..N-1} autocorr[k] to be as large as possible.
    # Let min_ac = min(autocorr[1:])
    # Max overlap = (N/2 - min_ac) * W
    
    min_ac = np.min(autocorr[1:])
    max_overlap = (N/2.0 - min_ac) * W
    return max_overlap

def _project_to_binary_sum(h, target_ones):
    """
    Given a real vector h, project to a binary vector with exactly target_ones ones.
    This is done by thresholding. To ensure exact count, we sort by value and pick top K.
    Ties are broken randomly or by index.
    """
    # To make it a proper "projection" for local search, we just take the top K indices
    # based on the current values. This ensures we stay in the discrete manifold.
    idx = np.argsort(-h)
    mask = np.zeros(N, dtype=bool)
    mask[idx[:target_ones]] = True
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    target_ones = N // 2
    
    # Initial state: Random binary vector with N/2 ones
    # Or a structured one? Random is good for diversity.
    # Let's try a "balanced" random start.
    initial_indices = np.random.choice(N, target_ones, replace=False)
    h = np.zeros(N)
    h[initial_indices] = 1.0
    mask = h.astype(bool)
    
    current_val = _overlap_max_binary(mask)
    best_mask = mask.copy()
    best_val = current_val
    
    # Local Search: Simulated Annealing on the binary space
    # Move: Flip one 1 to 0 and one 0 to 1 (maintain sum constraint)
    
    T_start = 0.1
    T_end = 0.001
    alpha = 0.995
    
    T = T_start
    step = 0
    max_steps = 20000
    
    # Precompute indices
    indices = np.arange(N)
    
    while step < max_steps and time.time() < deadline:
        # Choose a random 1 to flip to 0
        ones_indices = np.where(mask)[0]
        zeros_indices = np.where(~mask)[0]
        
        if len(ones_indices) == 0 or len(zeros_indices) == 0:
            break
            
        idx_one = np.random.choice(ones_indices)
        idx_zero = np.random.choice(zeros_indices)
        
        # Apply move
        new_mask = mask.copy()
        new_mask[idx_one] = False
        new_mask[idx_zero] = True
        
        # Evaluate
        val_new = _overlap_max_binary(new_mask)
        
        # Acceptance
        if val_new < current_val:
            mask = new_mask
            current_val = val_new
            if current_val < best_val:
                best_val = current_val
                best_mask = mask.copy()
        else:
            delta = val_new - current_val
            # Since we are minimizing, delta > 0
            if np.random.random() < np.exp(-delta / T):
                mask = new_mask
                current_val = val_new
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            
        step += 1
        
        # Adaptive: Occasionally try a 2-flip move (swap two 1s with two 0s)
        if step % 50 == 0:
            if len(ones_indices) >= 2 and len(zeros_indices) >= 2:
                # Re-fetch indices in case mask changed
                ones_indices = np.where(mask)[0]
                zeros_indices = np.where(~mask)[0]
                idx1, idx2 = np.random.choice(ones_indices, 2, replace=False)
                idx3, idx4 = np.random.choice(zeros_indices, 2, replace=False)
                
                new_mask = mask.copy()
                new_mask[idx1] = False
                new_mask[idx2] = False
                new_mask[idx3] = True
                new_mask[idx4] = True
                
                val_new = _overlap_max_binary(new_mask)
                if val_new < current_val:
                    mask = new_mask
                    current_val = val_new
                    if current_val < best_val:
                        best_val = current_val
                        best_mask = mask.copy()
                else:
                    delta = val_new - current_val
                    if np.random.random() < np.exp(-delta / T):
                        mask = new_mask
                        current_val = val_new

    # Final result: Convert best binary mask to heights
    # The task asks for heights in [0,1]. Binary 0/1 is valid.
    # However, the sum must be exactly N/2. Binary mask with N/2 ones sums to N/2.
    # So we just return the binary vector.
    
    result = best_mask.astype(np.float64)
    
    # Ensure exact sum (it should be, but just in case of floating point weirdness in conversion)
    # Sum is exactly N/2 because it's integer count.
    
    return result
