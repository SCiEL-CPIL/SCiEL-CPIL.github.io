import time
import numpy as np
from scipy.fft import fft

N = 100
DEADLINE_S = 145.0

def _time_remaining():
    return DEADLINE_S - time.time()

def _max_overlap(h):
    """
    Calculate the maximum circular overlap O(s) = sum_i h_i (1 - h_{i-s}).
    O(s) = sum_i h_i - sum_i h_i h_{i-s} = N/2 - C(s).
    C(s) is the circular autocorrelation of h.
    DFT of C(s) is |H_k|^2.
    So C(s) = ifft(|H|^2).real.
    O(s) = N/2 - C(s).
    We want to minimize max(O(s)).
    """
    H = fft(h)
    power = np.abs(H)**2
    corr = np.real(np.fft.ifft(power))
    overlaps = N/2.0 - corr
    return float(np.max(overlaps))

def construct():
    t0 = time.time()
    best_h = None
    best_val = float('inf')
    
    # We will try a few random restarts
    for restart in range(5):
        if _time_remaining() < 10.0:
            break
            
        # Generate random binary sequence with N/2 ones
        h = np.zeros(N)
        ones_indices = np.random.choice(N, N//2, replace=False)
        h[ones_indices] = 1.0
        
        current_val = _max_overlap(h)
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        max_steps = 20000 # More steps, but break on time
        
        # Maintain lists of indices for 1s and 0s to speed up move selection
        ones_list = list(ones_indices)
        zeros_list = [i for i in range(N) if i not in set(ones_indices)]
        
        for step in range(max_steps):
            if _time_remaining() < 5.0:
                break
                
            if not ones_list or not zeros_list:
                break
                
            # Pick random indices to swap
            i_idx = np.random.randint(len(ones_list))
            j_idx = np.random.randint(len(zeros_list))
            i_val = ones_list[i_idx]
            j_val = zeros_list[j_idx]
            
            # Swap
            h[i_val] = 0.0
            h[j_val] = 1.0
            
            new_val = _max_overlap(h)
            
            # Acceptance
            delta = new_val - current_val
            if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
                # Accept
                current_val = new_val
                # Update lists
                ones_list[i_idx] = j_val
                zeros_list[j_idx] = i_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Revert
                h[i_val] = 1.0
                h[j_val] = 0.0
            
            T *= cooling
            if T < T_min:
                T = T_min
                
    # Final polish: Hill Climbing on the best solution found
    if best_h is not None and _time_remaining() > 5.0:
        h = best_h.copy()
        current_val = best_val
        ones_list = list(np.where(h == 1.0)[0])
        zeros_list = list(np.where(h == 0.0)[0])
        
        # Simple local search: try all single swaps? No, that's 2500 checks.
        # Just random swaps until no improvement found in X tries.
        no_improve = 0
        max_no_improve = 500
        while no_improve < max_no_improve and _time_remaining() > 2.0:
            if not ones_list or not zeros_list:
                break
            i_idx = np.random.randint(len(ones_list))
            j_idx = np.random.randint(len(zeros_list))
            i_val = ones_list[i_idx]
            j_val = zeros_list[j_idx]
            
            h[i_val] = 0.0
            h[j_val] = 1.0
            new_val = _max_overlap(h)
            
            if new_val < current_val:
                current_val = new_val
                ones_list[i_idx] = j_val
                zeros_list[j_idx] = i_val
                no_improve = 0
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i_val] = 1.0
                h[j_val] = 0.0
                no_improve += 1
                
    return best_h
