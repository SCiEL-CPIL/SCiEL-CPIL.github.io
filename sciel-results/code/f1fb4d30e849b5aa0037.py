import numpy as np
import time

N = 100

def _calc_max_overlap(h):
    g = 1 - h
    max_ov = 0.0
    for k in range(N):
        ov = np.dot(h[:N-k], g[k:])
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _run_sa(h_init, time_limit):
    start = time.time()
    h = h_init.copy()
    current_score = _calc_max_overlap(h)
    best_score = current_score
    best_h = h.copy()
    
    T = 0.5
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start < time_limit:
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        delta = np.random.uniform(-0.05, 0.05)
        
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        
        if h_new[i] < 0 or h_new[i] > 1 or h_new[j] < 0 or h_new[j] > 1:
            continue
            
        new_score = _calc_max_overlap(h_new)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            
        if new_score < current_score:
            current_score = new_score
            h = h_new
        else:
            delta_score = new_score - current_score
            if np.random.rand() < np.exp(-delta_score / T):
                current_score = new_score
                h = h_new
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_h, best_score

def construct():
    global_start = time.time()
    deadline = 115.0
    
    best_h_global = np.full(N, 0.5)
    best_score_global = _calc_max_overlap(best_h_global)
    
    h_init = np.full(N, 0.5)
    h_res, score_res = _run_sa(h_init, 100.0)
    
    if score_res < best_score_global:
        best_score_global = score_res
        best_h_global = h_res
        
    return best_h_global.astype(float)

# EVOLVE-BLOCK-END
