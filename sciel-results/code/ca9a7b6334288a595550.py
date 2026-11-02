import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3
    
    # Initial configuration: Honeycomb grid
    q_range = range(-5, 6)
    r_range = range(-5, 6)
    points = []
    for q in q_range:
        for r in r_range:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            points.append((x, y))
    
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_points = np.array(points[:N])
    
    x = initial_points[:, 0].copy()
    y = initial_points[:, 1].copy()
    S = 3.0
    alpha = 0.0
    
    best_score = float('inf')
    best_x = x.copy()
    best_y = y.copy()
    best_S = S
    best_alpha = alpha
    
    T = 1.0
    cooling = 0.9995
    step_size = 0.5
    
    angles = np.arange(6) * np.pi / 3
    inner_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_score(x, y, S, alpha):
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        
        overlaps = np.maximum(0, min_dist - dists)
        overlap_pen = np.sum(overlaps**2)
        
        normal_angles = alpha + np.pi/6 + np.arange(6) * np.pi / 3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        R_in = S * sqrt3 / 2.0
        
        contain_pen = 0.0
        for i in range(N):
            cx = x[i]
            cy = y[i]
            verts = inner_verts + np.array([cx, cy])
            dots = verts @ normals.T
            max_violations = np.max(dots, axis=1) - R_in
            positive_violations = np.maximum(0, max_violations)
            contain_pen += np.sum(positive_violations**2)
            
        return S + 100.0 * (overlap_pen + contain_pen)

    current_score = get_score(x, y, S, alpha)
    if current_score < best_score:
        best_score = current_score
        best_x = x.copy()
        best_y = y.copy()
        best_S = S
        best_alpha = alpha

    while T > 0.01:
        if time.time() > deadline:
            break
            
        new_x = x.copy()
        new_y = y.copy()
        new_S = S
        new_alpha = alpha
        
        if np.random.random() < 0.7:
            idx = np.random.randint(0, N)
            new_x[idx] += np.random.randn() * step_size
            new_y[idx] += np.random.randn() * step_size
            
        if np.random.random() < 0.3:
            new_S += np.random.randn() * step_size * 0.5
            if new_S < 1.0: new_S = 1.0
            
        if np.random.random() < 0.2:
            new_alpha += np.random.randn() * step_size * 0.5
            
        new_score = get_score(new_x, new_y, new_S, new_alpha)
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            x = new_x
            y = new_y
            S = new_S
            alpha = new_alpha
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
                best_y = y.copy()
                best_S = S
                best_alpha = alpha
                
        T *= cooling
        
        if T < 0.1:
            step_size *= 0.95

    inner = []
    for i in range(N):
        inner.append([best_x[i], best_y[i], 0.0])
        
    return inner, (0.0, 0.0), best_S, best_alpha
# EVOLVE-BLOCK-END
