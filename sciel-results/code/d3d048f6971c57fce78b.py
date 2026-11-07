import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 150  # seconds
    
    # Initialize with a random binary sequence with 50 ones and 50 zeros
    h = np.zeros(N)
    h[:50] = 1.0
    np.random.shuffle(h)
    
    # Compute initial autocorrelation
    C = np.correlate(h, h, mode='full')
    min_C = np.min(C[N:2*N-1])
    
    best_h = h.copy()
    best_min_C = min_C
    
    # Precompute indices of 0s and 1s
    zeros = np.where(h == 0)[0]
    ones = np.where(h == 1)[0]
    
    while time.time() - start_time < time_limit - 5:
        improved = False
        # Try random swaps
        for _ in range(1000):  # Try 1000 random swaps per step
            i = zeros[np.random.randint(len(zeros))]
            j = ones[np.random.randint(len(ones))]
            
            if i == j:
                continue
                
            # Swap i and j
            h[i], h[j] = h[j], h[i]
            
            # Update zeros and ones lists
            if h[i] == 0:
                # i was 1, now 0
                ones[ones == i] = j
                zeros[zeros == j] = i
            else:
                # i was 0, now 1
                zeros[zeros == i] = j
                ones[ones == j] = i
                
            C_new = np.correlate(h, h, mode='full')
            new_min_C = np.min(C_new[N:2*N-1])
            
            if new_min_C > min_C:
                min_C = new_min_C
                best_h = h.copy()
                best_min_C = min_C
                improved = True
            else:
                # Swap back
                h[i], h[j] = h[j], h[i]
                # Revert zeros and ones lists
                if h[i] == 0:
                    ones[ones == j] = i
                    zeros[zeros == i] = j
                else:
                    zeros[zeros == j] = i
                    ones[ones == i] = j
                    
        if not improved:
            # If no improvement in 1000 random swaps, try a few more or break
            # To avoid getting stuck, we can restart with a new random sequence
            # or just break
            break
            
    return best_h
