import numpy as np
import math
import time
import random

def construct():
    # Time limit
    start_time = time.time()
    time_limit = 115.0
    
    def evaluate(U):
        if not U or 0 not in U:
            return -np.inf
        U = sorted(set(U))
        M = max(U)
        if M == 0:
            return 1.0
        
        # FFT size
        n_fft = 1
        while n_fft < 2 * M + 1:
            n_fft *= 2
        
        v = np.zeros(n_fft, dtype=np.float64)
        # Vectorized assignment
        v[U] = 1.0
        
        # Sumset
        v_fft = np.fft.fft(v)
        sum_conv = np.fft.ifft(v_fft * v_fft).real
        sumset_size = np.sum(sum_conv[:2*M+1] > 1e-9)
        
        # Difference set (Autocorrelation)
        auto_corr = np.fft.ifft(v_fft * np.conj(v_fft)).real
        
        # Lags 1..M and -1..-M
        pos_lags = auto_corr[1:M+1]
        neg_lags = auto_corr[n_fft-1:n_fft-M-1:-1]
        
        diffset_size = np.sum(pos_lags > 1e-9) + np.sum(neg_lags > 1e-9) + 1
        
        if sumset_size == 0:
            return -np.inf
            
        ratio = diffset_size / sumset_size
        denom = math.log(2 * M + 1)
        if denom == 0:
            return 1.0
            
        score = 1 + math.log(ratio) / denom
        return score

    best_U = [0, 1, 3]
    best_score = evaluate(best_U)
    
    # Try different target sizes and max values
    # We want to explore sets that are not too dense to keep sumset small
    # but spread out to keep diffset large.
    
    for N in [5, 8, 12, 20, 30, 50]:
        # Initialize with a geometric-like sequence to ensure spread
        U = [0]
        current_val = 1
        for i in range(1, N):
            # Geometric growth with some noise
            growth = random.uniform(1.5, 2.5)
            new_val = int(current_val * growth)
            if new_val <= current_val:
                new_val = current_val + 1
            U.append(new_val)
            current_val = new_val
        
        U = sorted(set(U))
        # Ensure unique and sorted
        while len(U) < N:
             U.append(random.randint(1, max(U)+5))
             U = sorted(set(U))
        if len(U) > N:
             U = U[:N]
        
        current_score = evaluate(U)
        if current_score > best_score:
            best_score = current_score
            best_U = U.copy()
            
        # Simulated Annealing
        T = 0.5
        T_min = 0.01
        cooling_rate = 0.995
        max_iter = 5000 # Cap iterations per restart
        
        for i in range(max_iter):
            if (time.time() - start_time) > time_limit:
                break
                
            U_temp = U.copy()
            move_type = random.random()
            
            if move_type < 0.2:
                # Add element
                max_val = max(U_temp)
                # Try to add in gaps or extend range
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val)
                else:
                    new_val = max_val + random.randint(1, 5)
                if new_val not in U_temp:
                    U_temp.append(new_val)
            elif move_type < 0.4:
                # Remove element
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    U_temp.pop(idx)
            elif move_type < 0.7:
                # Shift element
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    # Larger shifts to escape local minima
                    shift = random.randint(-10, 10)
                    new_val = U_temp[idx] + shift
                    if new_val >= 0 and new_val not in U_temp:
                        U_temp[idx] = new_val
                        U_temp.sort()
                    else:
                        continue
            else:
                # Swap two elements
                if len(U_temp) >= 2:
                    idx1, idx2 = random.sample(range(1, len(U_temp)), 2)
                    U_temp[idx1], U_temp[idx2] = U_temp[idx2], U_temp[idx1]
                    U_temp.sort()
            
            if not U_temp or 0 not in U_temp:
                continue
                
            new_score = evaluate(U_temp)
            
            if new_score > current_score:
                U = U_temp
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = U.copy()
            else:
                delta = current_score - new_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    U = U_temp
                    current_score = new_score
            
            T *= cooling_rate
            if T < T_min:
                break
                
    return best_U
