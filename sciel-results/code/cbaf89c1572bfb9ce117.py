import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 11
    # Seed solution for n=11
    n_seed = 11
    best_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    # Try n from 27 down to 12
    for n in range(27, best_n, -1):
        if time.time() > deadline - 5:
            break
            
        # Time allocation: give more time to larger n?
        # Or equal time.
        # Let's give 5s to 27, 5s to 26, etc.
        # But if 27 is impossible, we waste time.
        # Let's try 27 for 5s. If fail, try 26 for 10s.
        
        time_limit = time.time() + 5.0
        if n == 26:
            time_limit = time.time() + 15.0 # Give 26 a chance
            
        colors = local_search_sa(n, time_limit)
        
        if colors is not None:
            best_n = n
            best_colors = colors
            # If we found 27, we can stop?
            # If we found 26, we can stop?
            # We want max n.
            # If we find 27, we return it.
            # If we find 26, we return it (unless we try 27 first and fail).
            # Since we iterate 27, 26, 25..., if we find 27, we break.
            # If we find 26, we break?
            # Yes, because we are going downwards.
            break
            
    return {'n': best_n, 'colors': best_colors}

def local_search_sa(n, time_limit):
    M = [np.zeros((n, n), dtype=bool) for _ in range(5)]
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, 4)
            M[c][i, j] = True
            M[c][j, i] = True
            
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < time_limit and T > min_T:
        # Calculate current energy
        E_curr = 0
        for c in range(5):
            common = M[c] @ M[c].T
            E_curr += np.sum(common > 1)
            
        if E_curr == 0:
            # Valid
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    for c in range(5):
                        if M[c][i, j]:
                            colors.append(c)
                            break
            return colors
            
        # Pick random edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j: continue
        if i > j: i, j = j, i
        
        old_c = -1
        for c in range(5):
            if M[c][i, j]:
                old_c = c
                break
                
        new_c = random.randint(0, 4)
        if new_c == old_c: continue
        
        M[old_c][i, j] = False
        M[old_c][j, i] = False
        M[new_c][i, j] = True
        M[new_c][j, i] = True
        
        E_new = 0
        for c in range(5):
            common = M[c] @ M[c].T
            E_new += np.sum(common > 1)
            
        delta_E = E_new - E_curr
        
        if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
            pass
        else:
            M[new_c][i, j] = False
            M[new_c][j, i] = False
            M[old_c][i, j] = True
            M[old_c][j, i] = True
            
        T *= cooling
        
    return None
