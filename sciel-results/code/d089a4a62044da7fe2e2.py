import numpy as np
import time

N = 26
_START = time.time()
DEADLINE = _START + 140.0

def _compute_radii(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    
    r_walls = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d, axis=1) / 2.0
    
    r = np.minimum(r_walls, min_dist)
    r = np.maximum(r, 0.0)
    return r

def _score(x, y):
    r = _compute_radii(x, y)
    return np.sum(r)

def _run_sa(RNG, x_init, y_init, max_iters=5000):
    x = x_init.copy()
    y = y_init.copy()
    
    best_x = x.copy()
    best_y = y.copy()
    best_score = _score(best_x, best_y)
    
    T = 0.05
    T_min = 1e-5
    cool = 0.999
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
            
        i = RNG.integers(N)
        
        # Adaptive step size?
        step = 0.1 * (T / 0.05)
        
        x_new = x.copy()
        y_new = y.copy()
        
        x_new[i] += RNG.normal(0, step)
        y_new[i] += RNG.normal(0, step)
        
        x_new[i] = np.clip(x_new[i], 0.0, 1.0)
        y_new[i] = np.clip(y_new[i], 0.0, 1.0)
        
        score_new = _score(x_new, y_new)
        
        delta = score_new - best_score
        
        if delta > 0 or RNG.random() < np.exp(delta / T):
            x = x_new
            y = y_new
            if score_new > best_score:
                best_x = x.copy()
                best_y = y.copy()
                best_score = score_new
                
        T *= cool
        if T < T_min:
            T = T_min

    return best_x, best_y, best_score

def construct():
    RNG = np.random.default_rng(20240517)
    
    best_x = None
    best_y = None
    best_score = -np.inf
    
    # Multiple restarts
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) < N:
                xs.append((i + 0.5) / side)
                ys.append((j + 0.5) / side)
    while len(xs) < N:
        xs.append(RNG.uniform(0.1, 0.9))
        ys.append(RNG.uniform(0.1, 0.9))
        
    x_grid = np.array(xs)
    y_grid = np.array(ys)
    
    bx, by, bs = _run_sa(RNG, x_grid, y_grid, max_iters=2000)
    if bs > best_score:
        best_score = bs
        best_x = bx
        best_y = by
        
    # 2. Random starts
    for k in range(10):
        if time.time() > DEADLINE - 10:
            break
        x_rand = RNG.uniform(0.05, 0.95, N)
        y_rand = RNG.uniform(0.05, 0.95, N)
        
        bx, by, bs = _run_sa(RNG, x_rand, y_rand, max_iters=1000)
        if bs > best_score:
            best_score = bs
            best_x = bx
            best_y = by

    r_best = _compute_radii(best_x, best_y)
    return np.column_stack([best_x, best_y, r_best])
