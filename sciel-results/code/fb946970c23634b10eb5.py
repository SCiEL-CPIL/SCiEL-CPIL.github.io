import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Initialize with a random balanced sequence
    # Using a fixed seed for reproducibility in the code structure, 
    # but random restarts in the loop
    best_h = None
    best_score = float('inf')
    
    def calc_max_overlap(h):
        # h is np.array of 0/1
        # C_k = sum h_i (1 - h_{i+k})
        # This is equivalent to N_1 - sum h_i h_{i+k}
        # Or just compute directly
        # Using FFT for speed? N=100 is small, direct is fine.
        max_c = 0
        for k in range(1, N):
            # h shifted by k
            h_shifted = np.roll(h, k)
            # overlap h_i * (1 - h_shifted_i)
            c = np.sum(h * (1 - h_shifted))
            if c > max_c:
                max_c = c
        return max_c

    # Try several random initializations
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Generate random permutation of 50 ones and 50 zeros
        h = np.zeros(N, dtype=int)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1
        
        # Local search
        current_score = calc_max_overlap(h)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick a random 1 and a random 0
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            i1 = np.random.choice(ones)
            i0 = np.random.choice(zeros)
            
            # Swap
            h[i1], h[i0] = h[i0], h[i1]
            new_score = calc_max_overlap(h)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                h[i1], h[i0] = h[i0], h[i1]
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    if best_h is None:
        # Fallback to block sequence
        best_h = np.zeros(N, dtype=int)
        best_h[:N//2] = 1
        
    return best_h
