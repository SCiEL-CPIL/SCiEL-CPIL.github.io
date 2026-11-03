import numpy as np
import time
import random
from scipy.spatial.distance import pdist

N = 100

def _score_fast(pos):
    bdist = np.min(0.5 - np.abs(pos), axis=1)
    min_bdist = np.min(bdist)
    dists = pdist(pos)
    min_pdist = np.min(dists)
    return min(min_pdist / 2.0, min_bdist)

def _sa_optimize(pos, time_limit, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
    r = _score_fast(pos)
    best_pos = pos.copy()
    best_r = r
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.999
    
    step_size = 0.05
    accept_count = 0
    total_count = 0
    window_size = 100
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        total_count += 1
        
        n_move = np.random.randint(1, 4)
        idx = np.random.choice(N, n_move, replace=False)
        delta = np.random.normal(0, step_size, (n_move, 3))
        
        new_pos = pos.copy()
        new_pos[idx] += delta
        
        new_r = _score_fast(new_pos)
        
        if new_r > r or np.random.random() < np.exp((new_r - r) / T):
            pos = new_pos
            r = new_r
            accept_count += 1
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
        
        if total_count % window_size == 0:
            rate = accept_count / window_size
            if rate > 0.5:
                step_size *= 1.1
            elif rate < 0.1:
                step_size *= 0.9
            accept_count = 0
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_pos, best_r

def construct():
    start_time = time.time()
    total_time_limit = 145.0
    
    best_overall_pos = None
    best_overall_r = -1
    
    restarts = 0
    while time.time() - start_time < total_time_limit:
        restarts += 1
        
        if restarts % 2 == 1:
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.choice(len(grid), N, replace=False)
            pos = grid[idx].copy()
            pos += np.random.normal(0, 0.01, (N, 3))
        else:
            pos = np.random.uniform(-0.4, 0.4, (N, 3))
            
        time_spent = time.time() - start_time
        time_left = total_time_limit - time_spent
        if time_left < 1.0:
            break
            
        sa_time = max(0.5, time_left * 0.2)
        
        new_pos, new_r = _sa_optimize(pos, sa_time, seed=restarts)
        
        if new_r > best_overall_r:
            best_overall_r = new_r
            best_overall_pos = new_pos
            
    if best_overall_pos is None:
        # Fallback
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_overall_pos = grid[:N]
        
    return best_overall_pos
