import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    # Use 110s to be safe
    deadline = start_time + 110.0
    
    best_min_r = -float('inf')
    best_seq = None
    
    # Helper to calculate min autocorrelation for k != 0
    def get_min_corr(seq):
        # seq is array of +/- 1
        min_r = float('inf')
        for k in range(1, N):
            # Autocorrelation at lag k
            r = np.dot(seq, np.roll(seq, k))
            if r < min_r:
                min_r = r
        return min_r

    # Local search function
    def local_search(init_seq, max_iters=5000):
        seq = init_seq.copy()
        current_min_r = get_min_corr(seq)
        best_local_min_r = current_min_r
        best_local_seq = seq.copy()
        
        no_improve_count = 0
        limit = 1000
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Find indices of 1s and -1s
            ones = np.where(seq == 1)[0]
            zeros = np.where(seq == -1)[0]
            
            if len(ones) == 0 or len(zeros) == 0:
                break
                
            # Pick random swap
            a = np.random.choice(ones)
            b = np.random.choice(zeros)
            
            # Perform swap
            seq[a] = -1
            seq[b] = 1
            
            new_min_r = get_min_corr(seq)
            
            if new_min_r > best_local_min_r:
                best_local_min_r = new_min_r
                best_local_seq = seq.copy()
                no_improve_count = 0
            else:
                # Revert
                seq[a] = 1
                seq[b] = -1
                no_improve_count += 1
                
            if no_improve_count > limit:
                break
                
        return best_local_min_r, best_local_seq

    # Multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Initialize random sequence with N/2 ones
        seq = np.zeros(N, dtype=int)
        seq[:N//2] = 1
        np.random.shuffle(seq)
        
        # Map to +/- 1 for autocorrelation calculation
        # 1 -> 1, 0 -> -1
        x_seq = 2 * seq - 1
        
        # Run local search
        min_r, best_x_seq = local_search(x_seq)
        
        if min_r > best_min_r:
            best_min_r = min_r
            best_seq = best_x_seq
            
        restarts += 1
        # Allow for many restarts if fast, but local search dominates time
        # If local search is too fast, we might do thousands of restarts.
        # That's fine.

    # Convert best_seq (+/-1) back to heights (0/1)
    # 1 -> 1, -1 -> 0
    if best_seq is not None:
        h = (best_seq + 1) // 2
    else:
        # Fallback to constant 0.5 if something went wrong
        h = np.full(N, 0.5)
        
    return h
