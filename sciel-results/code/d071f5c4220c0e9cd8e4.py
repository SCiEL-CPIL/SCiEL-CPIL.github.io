import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1]
    ]
    
    M = 2**15
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def evaluate(coeffs):
        F = D @ coeffs
        return np.max(np.abs(F)) / np.sqrt(24)
        
    best_score = float('inf')
    best_c = None
    
    for c in candidates:
        c_arr = np.array(c, dtype=float)
        score = evaluate(c_arr)
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()
            
    current_c = best_c
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f))
    
    # Iterated Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Try 1-bit flips
        best_flip_1 = -1
        best_max_1 = current_max
        
        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_1:
                best_max_1 = max_val
                best_flip_1 = n
                
        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = D @ current_c
            current_max = best_max_1
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            improved = True
            continue # Restart loop with new state
            
        # 2. Try 2-bit flips if 1-bit failed
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                # Flip n1 and n2
                # f_new = f - 2*c[n1]*D[:,n1] - 2*c[n2]*D[:,n2]
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
                    
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            improved = True
            continue # Restart loop with new state
            
        # 3. If stuck, random restart (perturb)
        # Flip 3 random bits
        if np.random.rand() < 0.5:
            indices = np.random.choice(24, 3, replace=False)
            current_c[indices] *= -1.0
            current_f = D @ current_c
            current_max = np.max(np.abs(current_f))
            # Don't update best_score here, let the search try to recover
            
    return [int(x) for x in best_c]
