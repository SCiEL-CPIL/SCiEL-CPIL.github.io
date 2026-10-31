import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    x = np.ones(N, dtype=np.int8)
    c = np.convolve(x.astype(np.int32), x.astype(np.int32), mode='full')
    s = N
    best_val = N / (N * N)
    best_x = x.copy()
    
    np.random.seed(42)
    
    for it in range(500000):
        if time.time() > deadline:
            break
        
        i = np.random.randint(N)
        v = int(x[i])
        
        n_start = max(0, i)
        n_end = min(2*N-2, i + N - 1)
        
        idx = np.arange(n_start, n_end + 1)
        deltas = -4 * v * x[idx - i].astype(np.int32)
        if 2*i >= n_start and 2*i <= n_end:
            deltas[2*i - n_start] = 0
        
        x[i] = -v
        s_new = s - 2 * v
        c[n_start:n_end+1] += deltas
        
        if s_new == 0:
            val_new = np.inf
        else:
            val_new = float(np.max(np.abs(c)) / (s_new * s_new))
        
        if val_new < best_val:
            best_val = val_new
            best_x = x.copy()
            s = s_new
        else:
            x[i] = v
            c[n_start:n_end+1] -= deltas
    
    return best_x
